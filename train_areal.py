#!/usr/bin/env python3
"""AReaL GRPO 训练脚本。

使用 AReaL 框架训练 CUA Agent。

用法：
    # 单节点多 GPU
    python -m areal.launcher.local train_areal.py --config configs/cua_grpo.yaml
    
    # 多节点（使用 Ray）
    python -m areal.launcher.ray train_areal.py --config configs/cua_grpo.yaml
    
    # 直接运行（用于调试）
    python train_areal.py --config configs/cua_grpo.yaml
"""

import os
import sys
import warnings
from copy import deepcopy

import torch.distributed as dist
from datasets import Dataset

# ============ Patch AReaL dataset module to support CUA dataset ============
# This must be done before importing get_custom_dataset
import areal.dataset as areal_dataset_module

# Store the original function
_original_get_custom_dataset = areal_dataset_module._get_custom_dataset

def _get_cua_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    **kwargs,
):
    """Load CUA dataset for RL training.
    
    Supports paths like:
    - cua_agent.tasks:get_areal_train_dataset
    - cua_agent.tasks:get_areal_eval_dataset
    """
    # Parse the function path
    if ":" in path:
        module_path, func_name = path.split(":", 1)
    else:
        # Fallback: try to import from cua_agent.tasks
        module_path = "cua_agent.tasks"
        func_name = "get_areal_train_dataset" if split == "train" else "get_areal_eval_dataset"
    
    # Import the function dynamically
    import importlib
    module = importlib.import_module(module_path)
    get_dataset_func = getattr(module, func_name)
    
    # Get the raw dataset (list of dicts)
    raw_data = get_dataset_func()
    
    # Convert to HuggingFace Dataset format
    # AReaL expects format: {"messages": [{"role": "user", "content": "..."}]}
    def process_item(item):
        # Extract prompt from the item
        prompt = item.get("prompt", "")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            # If prompt is already in messages format, use it
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        
        # Store task info for reward function
        # AReaL will pass this through kwargs to the reward function
        task = item.get("task")
        task_dict = task.to_dict() if hasattr(task, "to_dict") else (task if task else {})
        metadata = item.get("metadata", {})
        
        # Create the dataset item
        # Note: AReaL RL datasets typically have "messages" and optionally "answer" or task info
        result = {
            "messages": messages,
        }
        
        # Store task information that can be accessed by the reward function
        # We'll store it in a way that can be passed through the workflow
        if task_dict:
            # Store task description as "answer" field (used by some AReaL workflows)
            result["answer"] = task_dict.get("description", "")
            # Also store full task info for reward function
            result["task_id"] = item.get("id", "")
            result["task_metadata"] = metadata
        
        return result
    
    processed_data = [process_item(item) for item in raw_data]
    dataset = Dataset.from_list(processed_data)
    
    # Filter by max_length if provided
    if max_length is not None and tokenizer is not None:
        def filter_length(sample):
            # Tokenize the user content to check length
            messages = sample.get("messages", [])
            if messages:
                content = messages[0].get("content", "")
                if content:
                    tokens = tokenizer.encode(content, add_special_tokens=False)
                    return len(tokens) <= max_length
            return True
        
        dataset = dataset.filter(filter_length)
    
    return dataset


def _patched_get_custom_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer=None,
    processor=None,
    **kwargs,
):
    """Patched version of _get_custom_dataset that supports CUA dataset."""
    # Check if this is a CUA dataset
    if ("cua" in path.lower() or "cua_agent" in path) and type == "rl":
        return _get_cua_rl_dataset(
            path=path,
            split=split or "train",
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    
    # Otherwise, use the original function
    return _original_get_custom_dataset(
        path=path,
        type=type,
        split=split,
        max_length=max_length,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )

# Apply the patch
areal_dataset_module._get_custom_dataset = _patched_get_custom_dataset
# Update VALID_DATASETS to include cua
if "cua" not in areal_dataset_module.VALID_DATASETS:
    areal_dataset_module.VALID_DATASETS.append("cua")
# ============ End of patch ============

# AReaL imports - 使用官方 API
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

# 项目 imports
from cua_agent.reward import simple_reward_function, CUARolloutResult

warnings.filterwarnings("ignore", category=DeprecationWarning)


def cua_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    """CUA 奖励函数，适配 AReaL 的奖励函数接口。
    
    Args:
        prompt: 提示文本
        completions: 模型生成的完成文本（字符串或列表）
        prompt_ids: 提示 token IDs
        completion_ids: 完成 token IDs
        answer: 期望的答案（任务数据，来自数据集的 "answer" 字段）
        **kwargs: 其他参数（可能包含任务信息，如 task_id, task_metadata 等）
        
    Returns:
        奖励值（float）
    """
    # 如果 completions 是列表，取第一个
    if isinstance(completions, list):
        completion = completions[0] if completions else ""
    else:
        completion = completions
    
    # 从 kwargs 中提取任务信息
    task_id = kwargs.get("task_id", "")
    task_metadata = kwargs.get("task_metadata", {})
    task_completed = kwargs.get("task_completed", False)
    task_success = kwargs.get("task_success", False)
    num_turns = kwargs.get("num_turns", 0)
    max_turns = kwargs.get("max_turns", task_metadata.get("max_steps", 15))
    errors = kwargs.get("errors", [])
    
    # 构建 CUARolloutResult
    rollout_result = CUARolloutResult(
        task_id=task_id,
        task_completed=task_completed,
        task_success=task_success,
        num_turns=num_turns,
        max_turns=max_turns,
        errors=errors,
    )
    
    # 使用现有的奖励函数
    # 注意：simple_reward_function 需要 task 对象，这里我们创建一个简单的任务对象
    class SimpleTask:
        def __init__(self, task_id, description):
            self.id = task_id
            self.description = description
    
    task = SimpleTask(
        task_id=task_id,
        description=answer if isinstance(answer, str) else str(answer)
    )
    
    reward = simple_reward_function(rollout_result, task)
    return reward


def main(args):
    """主函数。"""
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK", "0"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset and dataloaders
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )

    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemotevLLMEngine(config.rollout)
    eval_rollout = RemotevLLMEngine(deepcopy(config.rollout))
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    if config.actor.weight_update_mode == "xccl":
        weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
            allocation_mode,
            use_lora=config.actor.use_lora,
            lora_name=getattr(config.gconfig, "lora_name", None),
            lora_int_id=1,  # hard coded for the single lora example
            base_model_name=config.actor.path,
        )
    elif config.actor.weight_update_mode == "disk":
        weight_update_meta = WeightUpdateMeta.from_disk(
            config.saver.experiment_name,
            config.saver.trial_name,
            config.saver.fileroot,
            use_lora=config.actor.use_lora,
            lora_name=getattr(config.gconfig, "lora_name", None),
            lora_int_id=1,  # hard coded for the single lora example
            base_model_name=config.actor.path,
        )
    else:
        raise ValueError(
            f"Invalid weight_update_mode: {config.actor.weight_update_mode}. Expected 'xccl' or 'disk'."
        )

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    workflow = RLVRWorkflow(
        reward_fn=cua_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = RLVRWorkflow(
        reward_fn=cua_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                if actor.is_data_parallel_head():
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats = actor.export_stats()
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
