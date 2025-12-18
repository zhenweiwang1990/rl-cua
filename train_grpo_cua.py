#!/usr/bin/env python3
"""GRPO Training Script for CUA Agent.

This script trains a CUA (Computer Use Agent) model using GRPO
(Group Relative Policy Optimization) with AReaL-inspired patterns.

Features:
- Multi-GPU vLLM inference for rollout collection
- Dynamic LoRA switching during training
- Checkpoint saving and resume support
- Detailed logging with wandb integration

Usage:
    # Basic training
    python train_grpo_cua.py
    
    # Resume from checkpoint
    python train_grpo_cua.py --resume
    
    # Resume from specific checkpoint
    python train_grpo_cua.py --resume_from_checkpoint outputs/grpo_cua/checkpoints/checkpoint-50
    
    # With custom configuration
    BATCH_SIZE=8 MAX_STEPS=500 python train_grpo_cua.py
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Configure logging with immediate flushing
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[log_handler]
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except:
        pass


def patch_qwen3_gradient_checkpointing(model):
    """Ensure Qwen3 decoder layers expose `_gradient_checkpointing_func`."""
    import torch.utils.checkpoint

    checkpoint_fn = torch.utils.checkpoint.checkpoint
    candidate_layer_lists = []

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidate_layer_lists.append(model.model.layers)
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "layers")
    ):
        candidate_layer_lists.append(model.base_model.model.layers)

    patched = 0
    for layers in candidate_layer_lists:
        if not layers:
            continue
        for layer in layers:
            if not hasattr(layer, "_gradient_checkpointing_func"):
                layer._gradient_checkpointing_func = checkpoint_fn
                patched += 1
        if patched:
            break
    return patched


def load_model_with_unsloth(model_name: str, max_seq_length: int, load_in_4bit: bool):
    """Load model with Unsloth for efficient training."""
    from unsloth import FastLanguageModel
    
    try:
        return FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
            device_map="auto",
        )
    except Exception as e:
        if "No config file found" in str(e):
            print(f"⚠ Failed to load {model_name}: {e}", flush=True)
            print(f"🔄 Attempting to download from ModelScope...", flush=True)
            try:
                from modelscope import snapshot_download
                model_dir = snapshot_download(model_name)
                print(f"✓ Model downloaded to: {model_dir}", flush=True)
                return FastLanguageModel.from_pretrained(
                    model_name=model_dir,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                    dtype=None,
                    device_map="auto",
                )
            except Exception as ms_error:
                print(f"❌ ModelScope download failed: {ms_error}", flush=True)
                raise e
        raise e


def find_latest_checkpoint(output_dir: str) -> Path:
    """Find the latest checkpoint in output directory."""
    output_path = Path(output_dir) / "checkpoints"
    if not output_path.exists():
        return None
    
    checkpoints = []
    for path in output_path.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            try:
                step = int(path.name.split("-")[1])
                checkpoints.append((step, path))
            except (IndexError, ValueError):
                continue
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def find_best_checkpoint(output_dir: str):
    """Find the best checkpoint based on accuracy."""
    output_path = Path(output_dir) / "checkpoints"
    if not output_path.exists():
        return None
    
    best_checkpoint = None
    best_accuracy = -1.0
    
    for path in output_path.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            metadata_file = path / "training_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    accuracy = metadata.get("accuracy", 0.0)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_checkpoint = path
                except (json.JSONDecodeError, KeyError):
                    continue
    
    return (best_checkpoint, best_accuracy) if best_checkpoint else None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="GRPO Training for CUA Agent")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint to resume from"
    )
    parser.add_argument(
        "--resume_best",
        action="store_true",
        help="Resume from best checkpoint (highest accuracy) instead of latest"
    )
    parser.add_argument(
        "--enable-detailed-logging",
        action="store_true",
        help="Enable detailed rollout logging"
    )
    args = parser.parse_args()
    
    # Import after argument parsing
    from cua_agent.grpo_config import GRPOConfig, get_env_int, get_env_float, get_env_bool
    from cua_agent.grpo_trainer import CUAGRPOTrainer
    from cua_agent.tasks import get_training_tasks, get_eval_tasks
    
    # Load configuration from environment
    config = GRPOConfig.from_env()
    
    # Override with command line flags
    if args.enable_detailed_logging:
        config.enable_detailed_logging = True
    
    # Print configuration
    print("=" * 80, flush=True)
    print("CUA GRPO Training", flush=True)
    print("=" * 80, flush=True)
    print(f"Model: {config.model_name}", flush=True)
    print(f"Max sequence length: {config.max_seq_length}", flush=True)
    print(f"Load in 4-bit: {config.load_in_4bit}", flush=True)
    print(f"Learning rate: {config.learning_rate}", flush=True)
    print(f"Batch size: {config.batch_size}", flush=True)
    print(f"Max steps: {config.max_steps}", flush=True)
    print(f"Rollouts per task: {config.rollout.num_rollouts}", flush=True)
    print(f"Max turns per rollout: {config.rollout.max_turns}", flush=True)
    print(f"Target accuracy: {config.target_accuracy * 100:.1f}%", flush=True)
    print(f"Output dir: {config.output_dir}", flush=True)
    print(f"vLLM API: {config.vllm.api_base}", flush=True)
    print(f"Dynamic LoRA: {config.vllm.enable_dynamic_lora}", flush=True)
    print("=" * 80, flush=True)
    
    # Determine checkpoint to resume from
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        resume_from_checkpoint = args.resume_from_checkpoint
        print(f"\n📂 Resuming from specified checkpoint: {resume_from_checkpoint}", flush=True)
    elif args.resume_best:
        result = find_best_checkpoint(config.output_dir)
        if result:
            resume_from_checkpoint, accuracy = result
            print(f"\n📂 Resuming from best checkpoint (accuracy: {accuracy:.2%}): {resume_from_checkpoint}", flush=True)
    elif args.resume:
        resume_from_checkpoint = find_latest_checkpoint(config.output_dir)
        if resume_from_checkpoint:
            print(f"\n📂 Resuming from latest checkpoint: {resume_from_checkpoint}", flush=True)
    else:
        # Auto-resume: check for existing checkpoints
        resume_from_checkpoint = find_latest_checkpoint(config.output_dir)
        if resume_from_checkpoint:
            print(f"\n📂 Auto-resuming from checkpoint: {resume_from_checkpoint}", flush=True)
        else:
            print("\n📝 No checkpoints found, starting from scratch", flush=True)
    
    # Load model and tokenizer
    print("\n📦 Loading model and tokenizer...", flush=True)
    
    from unsloth import FastLanguageModel
    
    if resume_from_checkpoint:
        # Load base model first
        print(f"Loading base model: {config.model_name}", flush=True)
        model, tokenizer = load_model_with_unsloth(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=False,  # Don't load in 4-bit when resuming
        )
        
        # Apply LoRA configuration
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.r,
            target_modules=config.lora.target_modules,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
            max_seq_length=config.max_seq_length,
        )
        
        # Load LoRA weights from checkpoint
        print(f"Loading LoRA weights from checkpoint: {resume_from_checkpoint}", flush=True)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        
        checkpoint_path = Path(resume_from_checkpoint)
        safetensors_path = checkpoint_path / "adapter_model.safetensors"
        bin_path = checkpoint_path / "adapter_model.bin"
        
        if safetensors_path.exists():
            adapter_weights = load_file(str(safetensors_path))
        elif bin_path.exists():
            adapter_weights = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter weights found in {resume_from_checkpoint}")
        
        set_peft_model_state_dict(model, adapter_weights)
        print("✓ Base model and LoRA adapter loaded", flush=True)
    else:
        # Load fresh model
        print(f"Loading model: {config.model_name}", flush=True)
        model, tokenizer = load_model_with_unsloth(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
        )
        
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.r,
            target_modules=config.lora.target_modules,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
            max_seq_length=config.max_seq_length,
        )
    
    # Patch gradient checkpointing
    patched_layers = patch_qwen3_gradient_checkpointing(model)
    if patched_layers:
        print(f"🩹 Patched {patched_layers} decoder layers for gradient checkpointing", flush=True)
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    print("✓ Model loaded successfully", flush=True)
    
    # Load tasks
    print("\n📚 Loading tasks...", flush=True)
    train_tasks = get_training_tasks()
    eval_tasks = get_eval_tasks()
    print(f"✓ Loaded {len(train_tasks)} training tasks", flush=True)
    print(f"✓ Loaded {len(eval_tasks)} evaluation tasks", flush=True)
    
    # Create trainer
    print("\n📋 Initializing CUA GRPO Trainer...", flush=True)
    trainer = CUAGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
    )
    
    # Enable logits return for unsloth
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    # Start training
    trainer.train()
    
    print("\n✅ Training complete!", flush=True)


if __name__ == "__main__":
    main()

