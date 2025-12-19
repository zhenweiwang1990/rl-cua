#!/usr/bin/env python3
"""AReaL GRPO 训练脚本 - Actor 驱动环境版本。

使用 AReaL 框架训练 CUA Agent。
训练中的 actor（通过 AReaL 管理的 vLLM）直接驱动 GBox 环境交互。

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
import logging
import asyncio
import json
from copy import deepcopy
from datetime import datetime

import torch
import torch.distributed as dist
import numpy as np
from datasets import Dataset
from openai import AsyncOpenAI

# Configure logging for CUA Agent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,
)

logging.getLogger('cua_agent').setLevel(logging.INFO)
logging.getLogger('gbox_cua').setLevel(logging.INFO)
logging.getLogger('areal').setLevel(logging.INFO)

# ============ Patch AReaL dataset module to support CUA dataset ============
from cua_agent.dataset import patch_areal_dataset_module
patch_areal_dataset_module()
# ============ End of patch ============

# AReaL imports
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

# 项目 imports
from cua_agent.areal_workflow import CUAEnvRolloutWorkflow
from cua_agent.reward import cua_reward_fn

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _print_training_progress(
    global_step: int,
    epoch: int,
    step: int,
    steps_per_epoch: int,
    max_steps: int,
    stats: dict,
):
    """Print a concise training progress summary."""
    # Extract key stats
    reward = stats.get("reward", stats.get("rollout/reward", 0.0))
    if isinstance(reward, (list, tuple)):
        reward = sum(reward) / len(reward) if reward else 0.0
    
    pg_loss = stats.get("grpo_actor/pg_loss", stats.get("pg_loss", 0.0))
    kl = stats.get("grpo_actor/kl", stats.get("kl", 0.0))
    lr = stats.get("grpo_actor/lr", stats.get("lr", 0.0))
    
    # Calculate progress
    progress = (global_step + 1) / max_steps * 100
    
    # Timing stats
    timing = stats.get("timing", {})
    rollout_time = timing.get("rollout", 0)
    train_time = timing.get("train_step", 0)
    
    # Format output
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(
        f"[{timestamp}] Step {global_step + 1}/{max_steps} "
        f"({progress:.1f}%) | "
        f"Epoch {epoch + 1} Step {step + 1}/{steps_per_epoch} | "
        f"Reward: {reward:.3f} | "
        f"PG Loss: {pg_loss:.4f} | "
        f"KL: {kl:.4f} | "
        f"LR: {lr:.2e}",
        flush=True
    )


def main(args):
    """主函数。"""
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 1:
            device_id = 0
        else:
            device_id = local_rank % num_gpus
        
        torch.cuda.set_device(device_id)
        
        if rank == 0:
            print(f"[Rank {rank}] CUDA device assignment: device_id={device_id}, "
                  f"local_rank={local_rank}, world_size={world_size}, num_gpus={num_gpus}")
    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    
    actual_world_size = dist.get_world_size() if dist.is_initialized() else 1
    actual_rank = dist.get_rank() if dist.is_initialized() else 0
    
    if actual_rank == 0:
        print(f"[Rank {actual_rank}] Detected world_size={actual_world_size}, "
              f"data_parallel_world_size={actor.data_parallel_world_size}")

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
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    # Determine weight update mode
    effective_world_size = actual_world_size if dist.is_initialized() else 1
    effective_dp_size = actor.data_parallel_world_size if hasattr(actor, 'data_parallel_world_size') else 1
    num_visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if effective_world_size == 1 or effective_dp_size == 1 or num_visible_gpus == 1:
        if actual_rank == 0:
            print(f"[Rank {actual_rank}] Single GPU mode detected, using 'disk' mode for weight updates")
        weight_update_mode = "disk"
    else:
        weight_update_mode = config.actor.weight_update_mode
        if actual_rank == 0:
            print(f"[Rank {actual_rank}] Multi-GPU mode detected, using '{weight_update_mode}' mode")
    
    if weight_update_mode == "xccl":
        weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
            allocation_mode,
            use_lora=config.actor.use_lora,
            lora_name=getattr(config.gconfig, "lora_name", None),
            lora_int_id=1,
            base_model_name=config.actor.path,
        )
    elif weight_update_mode == "disk":
        weight_update_meta = WeightUpdateMeta.from_disk(
            config.saver.experiment_name,
            config.saver.trial_name,
            config.saver.fileroot,
            use_lora=config.actor.use_lora,
            lora_name=getattr(config.gconfig, "lora_name", None),
            lora_int_id=1,
            base_model_name=config.actor.path,
        )
    else:
        raise ValueError(f"Invalid weight_update_mode: {weight_update_mode}")

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow - 使用新的环境驱动 workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # Trace 目录
    trace_dir = os.path.join(
        StatsLogger.get_log_path(config.stats_logger), "trace"
    )
    
    workflow = CUAEnvRolloutWorkflow(
        reward_fn=cua_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=True,
        trace_dir=trace_dir,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    
    # 调试：检查 workflow 是否正确初始化
    if actual_rank == 0:
        logger = logging.getLogger(__name__)
        logger.info(f"[Workflow Init] Workflow type: {type(workflow).__name__}")
        logger.info(f"[Workflow Init] GBOX_API_KEY present: {bool(workflow.gbox_api_key)}")
    eval_workflow = CUAEnvRolloutWorkflow(
        reward_fn=cua_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        enable_thinking=True,
        trace_dir=os.path.join(trace_dir, "eval"),
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Run training
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

    if actual_rank == 0:
        # 获取 vLLM 地址
        areal_llm_addrs = os.getenv("AREAL_LLM_SERVER_ADDRS", "")
        vllm_api_base = "N/A"
        if areal_llm_addrs:
            first_addr = areal_llm_addrs.split(",")[0].strip()
            if ":" in first_addr:
                host, port = first_addr.rsplit(":", 1)
                vllm_api_base = f"http://{host}:{port}/v1"
            else:
                vllm_api_base = f"http://{first_addr}/v1"
        
        print(f"\n{'='*60}")
        print(f"CUA GRPO Training - Actor Driven Environment")
        print(f"{'='*60}")
        print(f"  vLLM API: {vllm_api_base} (AReaL managed)")
        print(f"  AREAL_LLM_SERVER_ADDRS: {areal_llm_addrs or 'NOT SET'}")
        print(f"  GBox API Key: {'***' + os.getenv('GBOX_API_KEY', '')[-4:] if os.getenv('GBOX_API_KEY') else 'NOT SET'}")
        print(f"  Max Turns: {int(os.getenv('CUA_MAX_TURNS', '15'))}")
        print(f"  Context Window: {int(os.getenv('CUA_CONTEXT_WINDOW', '5'))}")
        print(f"  Trace Dir: {trace_dir}")
        print(f"  Total Steps: {max_steps}")
        print(f"{'='*60}\n")

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
            if actual_rank == 0 and global_step == start_step:
                logger = logging.getLogger(__name__)
                logger.info(f"[Prepare Batch] Calling prepare_batch with workflow: {type(workflow).__name__}")
                logger.info(f"[Prepare Batch] Workflow has GBOX_API_KEY: {bool(workflow.gbox_api_key)}")
            
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )
            
            if actual_rank == 0 and global_step == start_step:
                logger = logging.getLogger(__name__)
                logger.info(f"[Prepare Batch] Batch prepared, keys: {list(batch.keys()) if isinstance(batch, dict) else 'not a dict'}")
                if isinstance(batch, dict) and "reward" in batch:
                    rewards = batch.get("reward", [])
                    logger.info(f"[Prepare Batch] Sample rewards: {rewards[:5] if len(rewards) > 5 else rewards}")

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
            
            # Log group-wise statistics for GRPO
            if actual_rank == 0 and isinstance(batch, dict):
                # Extract group information from batch
                rewards = batch.get("reward", [])
                advantages = batch.get("advantage", [])
                
                if rewards and len(rewards) > 0:
                    group_size = actor.config.group_size
                    num_groups = len(rewards) // group_size if group_size > 0 else 1
                    
                    logger = logging.getLogger(__name__)
                    logger.info(f"[GRPO Groups] Total samples: {len(rewards)}, Group size: {group_size}, Num groups: {num_groups}")
                    
                    # Log each group's statistics
                    for group_idx in range(num_groups):
                        start_idx = group_idx * group_size
                        end_idx = start_idx + group_size
                        group_rewards = rewards[start_idx:end_idx]
                        group_advantages = advantages[start_idx:end_idx] if advantages and len(advantages) >= end_idx else []
                        
                        if group_rewards:
                            mean_reward = float(np.mean(group_rewards))
                            std_reward = float(np.std(group_rewards))
                            min_reward = float(np.min(group_rewards))
                            max_reward = float(np.max(group_rewards))
                            
                            logger.info(
                                f"[GRPO Group {group_idx}] "
                                f"Rewards: {[f'{r:.3f}' for r in group_rewards]} | "
                                f"Mean: {mean_reward:.3f} | Std: {std_reward:.3f} | "
                                f"Range: [{min_reward:.3f}, {max_reward:.3f}]"
                            )
                            
                            if group_advantages:
                                mean_adv = float(np.mean(group_advantages))
                                logger.info(
                                    f"[GRPO Group {group_idx}] "
                                    f"Advantages: {[f'{a:.3f}' for a in group_advantages]} | "
                                    f"Mean: {mean_adv:.3f}"
                                )

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

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

        stats = actor.export_stats()
        stats_logger.commit(epoch, step, global_step, stats)

        # Print training progress
        if actual_rank == 0:
            _print_training_progress(
                global_step=global_step,
                epoch=epoch,
                step=step,
                steps_per_epoch=steps_per_epoch,
                max_steps=max_steps,
                stats=stats,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
