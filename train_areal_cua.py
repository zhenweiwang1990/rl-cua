#!/usr/bin/env python3
"""AReaL CUA GRPO Training Script.

Train CUA Agent with GRPO using AReaL framework.
Supports both HuggingFace and Unsloth model loaders.

Usage:
    # Single GPU with HF loader (validation mode)
    python train_areal_cua.py --config configs/areal_cua_single_gpu.yaml --loader hf

    # Single GPU with Unsloth loader (acceleration mode)
    python train_areal_cua.py --config configs/areal_cua_single_gpu.yaml --loader unsloth

    # Multi-GPU with DDP
    torchrun --nproc_per_node=4 train_areal_cua.py --config configs/areal_cua_multi_gpu.yaml

    # With AReaL launcher
    python -m areal.launcher.local train_areal_cua.py --config configs/areal_cua_single_gpu.yaml

Architecture:
    AReaL Trainer
        ↓
    Model Loader (HF or Unsloth)
        ↓
    Qwen3-VL-32B-Instruct + LoRA
        ↓
    CUAEnvRolloutWorkflow
        ↓
    GBox CUA Agent
        ↓
    GRPO Update
"""

import os
import sys
import warnings
import logging
from datetime import datetime
from copy import deepcopy

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.distributed as dist
import numpy as np

# Configure logging
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

# Local model engine for PyTorch/Unsloth
from cua_agent.local_model_engine import LocalModelEngine
from cua_agent.model_loader import create_model_loader
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

# Project imports
from cua_agent.areal_cua_workflow import CUAEnvRolloutWorkflow
from cua_agent.reward import cua_reward_fn

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


def _parse_loader_arg(args: list) -> tuple[str, list]:
    """Parse --loader argument from command line and return filtered args.
    
    Returns:
        tuple: (loader_type, filtered_args)
    """
    loader = "hf"  # Default
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == "--loader" and i + 1 < len(args):
            loader = args[i + 1].lower()
            i += 2  # Skip both --loader and its value
        elif args[i].startswith("--loader="):
            loader = args[i].split("=", 1)[1].lower()
            i += 1  # Skip this argument
        else:
            filtered_args.append(args[i])
            i += 1
    return loader, filtered_args


def _print_training_header(config, loader_type: str, rank: int):
    """Print training header with configuration summary."""
    if rank != 0:
        return
    
    print("\n" + "=" * 80)
    print("🚀 AReaL CUA GRPO Training")
    print("=" * 80)
    print(f"  Model:        {config.actor.path}")
    print(f"  Loader:       {loader_type.upper()}")
    print(f"  LoRA:         {'Enabled' if config.actor.use_lora else 'Disabled'}")
    print(f"  n_samples:    {config.gconfig.n_samples} (GRPO group size)")
    print(f"  Batch size:   {config.train_dataset.batch_size}")
    print(f"  Max turns:    {os.getenv('CUA_MAX_TURNS', '15')}")
    print(f"  GBox API:     {'***' + os.getenv('GBOX_API_KEY', '')[-4:] if os.getenv('GBOX_API_KEY') else 'NOT SET'}")
    print(f"  Epochs:       {config.total_train_epochs}")
    print("=" * 80 + "\n")


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
    
    # Color based on reward
    if reward >= 0.5:
        color = "\033[92m"  # Green
    elif reward >= 0.2:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[91m"  # Red
    reset = "\033[0m"
    
    print(
        f"[{timestamp}] Step {global_step + 1}/{max_steps} "
        f"({progress:.1f}%) | "
        f"Epoch {epoch + 1} Step {step + 1}/{steps_per_epoch} | "
        f"{color}Reward: {reward:.3f}{reset} | "
        f"PG Loss: {pg_loss:.4f} | "
        f"KL: {kl:.4f} | "
        f"LR: {lr:.2e}",
        flush=True
    )


def _log_grpo_group_stats(batch: dict, actor, rank: int):
    """Log group-wise statistics for GRPO."""
    if rank != 0:
        return
    
    if not isinstance(batch, dict):
        return
    
    rewards = batch.get("reward", [])
    advantages = batch.get("advantage", [])
    
    if not rewards or len(rewards) == 0:
        return
    
    group_size = actor.config.group_size
    num_groups = len(rewards) // group_size if group_size > 0 else 1
    
    logger.info(f"[GRPO Groups] Total samples: {len(rewards)}, "
               f"Group size: {group_size}, Num groups: {num_groups}")
    
    for group_idx in range(min(num_groups, 3)):  # Log first 3 groups
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        group_rewards = rewards[start_idx:end_idx]
        group_advantages = advantages[start_idx:end_idx] if advantages and len(advantages) >= end_idx else []
        
        if group_rewards:
            mean_reward = float(np.mean(group_rewards))
            std_reward = float(np.std(group_rewards))
            
            logger.info(
                f"[GRPO Group {group_idx}] "
                f"Rewards: {[f'{r:.3f}' for r in group_rewards]} | "
                f"Mean: {mean_reward:.3f} | Std: {std_reward:.3f}"
            )


def main(args):
    """Main training function."""
    # Parse loader type and filter args
    loader_type, filtered_args = _parse_loader_arg(args)
    
    # Load config with filtered args (without --loader)
    config, _ = load_expr_config(filtered_args, GRPOConfig)
    config: GRPOConfig

    # Get rank information
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    # Setup CUDA device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 1:
            device_id = 0
        else:
            device_id = local_rank % num_gpus
        
        torch.cuda.set_device(device_id)
        
        if rank == 0:
            logger.info(f"CUDA device assignment: device_id={device_id}, "
                       f"local_rank={local_rank}, world_size={world_size}, num_gpus={num_gpus}")
    
    # Print header
    _print_training_header(config, loader_type, rank)
    
    # Load tokenizer
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Set random seed
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    
    # Setup allocation mode
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine (actor)
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    
    actual_world_size = dist.get_world_size() if dist.is_initialized() else 1
    actual_rank = dist.get_rank() if dist.is_initialized() else 0
    
    if actual_rank == 0:
        logger.info(f"Detected world_size={actual_world_size}, "
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

    # Initialize inference engine (Local PyTorch/Unsloth instead of vLLM)
    if loader_type == "unsloth":
        logger.info("Using Unsloth local model engine for rollouts")
    else:
        logger.info("Using HuggingFace local model engine for rollouts")
    
    # Create model loader
    from cua_agent.model_loader import ModelLoaderConfig, LoRAConfig
    loader_config = ModelLoaderConfig(
        model_name=config.actor.path,
        lora=LoRAConfig(enabled=config.actor.use_lora),
    )
    model_loader = create_model_loader(loader_type, loader_config)
    
    # Create local inference engines
    rollout = LocalModelEngine(model_loader=model_loader)
    eval_rollout = LocalModelEngine(model_loader=model_loader)

    # Determine weight update mode
    effective_world_size = actual_world_size if dist.is_initialized() else 1
    effective_dp_size = actor.data_parallel_world_size if hasattr(actor, 'data_parallel_world_size') else 1
    num_visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if effective_world_size == 1 or effective_dp_size == 1 or num_visible_gpus == 1:
        if actual_rank == 0:
            logger.info("Single GPU mode detected, using 'disk' mode for weight updates")
        weight_update_mode = "disk"
    else:
        weight_update_mode = config.actor.weight_update_mode
        if actual_rank == 0:
            logger.info(f"Multi-GPU mode detected, using '{weight_update_mode}' mode")
    
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

    # Initialize reference model for KL divergence
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Ensure stop tokens are configured
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # Create trace directory
    trace_dir = os.path.join(
        StatsLogger.get_log_path(config.stats_logger), "trace"
    )
    
    # Create rollout workflow
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
    
    # Create eval workflow with lower temperature
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

    if actual_rank == 0:
        logger.info(f"Workflow initialized: {type(workflow).__name__}")
        logger.info(f"GBOX_API_KEY present: {bool(workflow.gbox_api_key)}")

    # Initialize utilities
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    # Recovery handler
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

    # Calculate training schedule
    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    if actual_rank == 0:
        logger.info(f"Training schedule: {max_steps} steps "
                   f"({total_epochs} epochs × {steps_per_epoch} steps/epoch)")
        logger.info(f"Starting from step {start_step}")

    # ============ Main Training Loop ============
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        # Rollout phase
        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        # Recompute logprob
        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        # Reference logprob for KL
        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        # Compute advantages (GRPO group-wise)
        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")
            
            # Log group-wise stats
            _log_grpo_group_stats(batch, actor, actual_rank)

        # PPO/GRPO update
        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        rollout.pause()

        # Update weights
        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        # Save checkpoint
        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        # Checkpoint for recovery
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

        # Evaluation phase
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

        with stats_tracker.record_timing("eval"):
            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Log and print progress
        stats = actor.export_stats()
        stats_logger.commit(epoch, step, global_step, stats)

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

    # ============ Training Complete ============
    if actual_rank == 0:
        print("\n" + "=" * 80)
        print("✅ Training Complete!")
        print("=" * 80)
        print(f"  Total steps: {max_steps}")
        print(f"  Model saved to: {config.saver.fileroot}")
        print("=" * 80 + "\n")

    # Cleanup
    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])

