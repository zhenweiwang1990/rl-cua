#!/usr/bin/env python3
"""
Initialize a LoRA adapter directory for dynamic vLLM rollout.

This script:
  - Reads GRPO / LoRA / vLLM config from environment variables.
  - Checks LORA_PATH for existing adapter_model.safetensors/bin.
  - If none exists, loads the base model with Unsloth + LoRA config and
    saves the adapter weights into LORA_PATH.

It is safe to run multiple times; if an adapter already exists, it does nothing.
"""

import os
import sys
from pathlib import Path

from cua_agent.grpo_config import GRPOConfig


def main() -> None:
    cfg = GRPOConfig.from_env()
    lora_path_str = cfg.vllm.lora_path

    if not lora_path_str:
        print("[init_lora_adapter] LORA_PATH not set in environment or config; skipping init.", flush=True)
        return

    lora_path = Path(lora_path_str)
    adapter_safetensors = lora_path / "adapter_model.safetensors"
    adapter_bin = lora_path / "adapter_model.bin"

    if adapter_safetensors.exists() or adapter_bin.exists():
        print(f"[init_lora_adapter] LoRA adapter already exists at {lora_path}; skipping init.", flush=True)
        return

    print(f"[init_lora_adapter] No existing adapter found in {lora_path}.", flush=True)
    print(f"[init_lora_adapter] Initializing LoRA adapter for model={cfg.model_name} ...", flush=True)

    lora_path.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid overhead when not needed
    try:
        from unsloth import FastLanguageModel
    except Exception as e:
        print(f"[init_lora_adapter] Failed to import unsloth: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        # Load base model with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            dtype=None,
            device_map="auto",
        )

        # Apply LoRA configuration
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            target_modules=cfg.lora.target_modules,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
            max_seq_length=cfg.max_seq_length,
        )

        # Save adapter weights into LORA_PATH (this will create adapter_model.safetensors)
        model.save_pretrained(str(lora_path))
        tokenizer.save_pretrained(str(lora_path))

        print(f"[init_lora_adapter] LoRA adapter initialized at {lora_path}.", flush=True)
    except Exception as e:
        print(f"[init_lora_adapter] Failed to initialize LoRA adapter: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


