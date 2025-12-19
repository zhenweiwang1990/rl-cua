"""GRPO Configuration for CUA Agent training.

This module defines configurations for GRPO training with LoRA,
vLLM inference, and checkpoint management.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


def get_env_int(key: str, default: str) -> int:
    """Get integer from environment variable."""
    value = os.environ.get(key, default)
    value = value.split('#')[0].strip()
    return int(value)


def get_env_float(key: str, default: str) -> float:
    """Get float from environment variable."""
    value = os.environ.get(key, default)
    value = value.split('#')[0].strip()
    return float(value)


def get_env_bool(key: str, default: str) -> bool:
    """Get boolean from environment variable."""
    value = os.environ.get(key, default)
    value = value.split('#')[0].strip().lower()
    return value in ("true", "1", "yes")


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    
    # LoRA hyperparameters
    r: int = 16  # LoRA rank
    alpha: int = 32  # LoRA alpha (scaling factor)
    dropout: float = 0.0  # LoRA dropout
    
    # Target modules for LoRA
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Bias configuration
    bias: str = "none"  # "none", "all", or "lora_only"
    
    # Task type
    task_type: str = "CAUSAL_LM"
    
    @classmethod
    def from_env(cls) -> "LoRAConfig":
        """Create LoRA config from environment variables."""
        return cls(
            r=get_env_int("LORA_R", "16"),
            alpha=get_env_int("LORA_ALPHA", "32"),
            dropout=get_env_float("LORA_DROPOUT", "0.0"),
        )


@dataclass
class VLLMConfig:
    """vLLM inference server configuration."""
    
    # Server connection
    api_base: str = "http://localhost:8000/v1"
    api_key: Optional[str] = None
    
    # Model names
    base_model: str = "unsloth/Qwen3-VL-32B-Instruct"
    lora_name: str = "cua_agent_lora"  # Name registered with vLLM
    # Shared LoRA adapter directory (host path), mounted into vLLM container.
    # Trainer will sync latest LoRA weights into this directory for dynamic rollout.
    lora_path: Optional[str] = None
    
    # Inference settings
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Dynamic LoRA switching
    enable_dynamic_lora: bool = True
    lora_update_steps: int = 10  # Update LoRA every N training steps
    
    # Connection settings
    timeout: float = 120.0
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> "VLLMConfig":
        """Create vLLM config from environment variables."""
        return cls(
            api_base=os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY"),
            base_model=os.environ.get("MODEL_NAME", "unsloth/Qwen3-VL-32B-Instruct"),
            lora_name=os.environ.get("LORA_NAME", "cua_agent_lora"),
            lora_path=os.environ.get("LORA_PATH"),
            max_tokens=get_env_int("MAX_TOKENS", "2048"),
            temperature=get_env_float("TEMPERATURE", "0.7"),
            top_p=get_env_float("TOP_P", "0.9"),
            enable_dynamic_lora=get_env_bool("ENABLE_DYNAMIC_LORA", "true"),
            lora_update_steps=get_env_int("LORA_UPDATE_STEPS", "10"),
            timeout=get_env_float("VLLM_TIMEOUT", "120.0"),
        )


@dataclass
class RolloutConfig:
    """Rollout collection configuration."""
    
    # Rollout settings
    num_rollouts: int = 4  # Number of rollouts per task (GRPO group size)
    max_turns: int = 15  # Maximum turns per rollout
    
    # Concurrency
    rollout_concurrency: int = 4  # Number of parallel rollouts
    
    # Environment
    gbox_api_key: Optional[str] = None
    box_type: str = "android"
    box_timeout: str = "60s"
    box_expires_in: str = "15m"
    
    # Timing
    action_delay: float = 0.5
    screenshot_delay: float = 0.3
    
    # Temperature variation for diverse rollouts
    enable_temperature_variation: bool = True
    base_temperature: float = 0.6
    temperature_increment: float = 0.1
    
    @classmethod
    def from_env(cls) -> "RolloutConfig":
        """Create rollout config from environment variables."""
        return cls(
            num_rollouts=get_env_int("NUM_ROLLOUTS", "4"),
            max_turns=get_env_int("MAX_TURNS", "15"),
            rollout_concurrency=get_env_int("ROLLOUT_CONCURRENCY", "4"),
            gbox_api_key=os.environ.get("GBOX_API_KEY"),
            box_type=os.environ.get("BOX_TYPE", "android"),
            action_delay=get_env_float("ACTION_DELAY", "0.5"),
            screenshot_delay=get_env_float("SCREENSHOT_DELAY", "0.3"),
        )


@dataclass
class GRPOConfig:
    """Main GRPO training configuration."""
    
    # Model settings
    model_name: str = "unsloth/Qwen3-VL-32B-Instruct"
    max_seq_length: int = 16384
    load_in_4bit: bool = True
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    beta: float = 0.01  # KL penalty coefficient
    clip_epsilon: float = 0.2  # PPO clip parameter
    
    # Batch settings
    batch_size: int = 4  # Number of tasks per batch
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Training schedule
    max_steps: int = 200
    warmup_steps: int = 10
    
    # Evaluation and checkpointing
    eval_steps: int = 10
    save_steps: int = 10
    save_total_limit: int = 5
    
    # Early stopping
    target_accuracy: float = 0.80
    patience: int = 5
    
    # GRPO specific
    min_group_std: float = 0.05  # Minimum std to keep a group
    advantage_normalization: bool = True
    
    # Output
    output_dir: str = "outputs/grpo_cua"
    seed: int = 42
    
    # Logging
    verbose: bool = False
    enable_wandb: bool = True
    wandb_project: str = "cua-grpo"
    wandb_entity: Optional[str] = None
    enable_detailed_logging: bool = False
    
    # Sub-configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    
    @classmethod
    def from_env(cls) -> "GRPOConfig":
        """Create GRPO config from environment variables."""
        return cls(
            model_name=os.environ.get("MODEL_NAME", "unsloth/Qwen3-VL-32B-Instruct"),
            max_seq_length=get_env_int("MAX_SEQ_LENGTH", "16384"),
            load_in_4bit=get_env_bool("LOAD_IN_4BIT", "true"),
            learning_rate=get_env_float("LEARNING_RATE", "1e-5"),
            beta=get_env_float("BETA", "0.01"),
            clip_epsilon=get_env_float("CLIP_EPSILON", "0.2"),
            batch_size=get_env_int("BATCH_SIZE", "4"),
            gradient_accumulation_steps=get_env_int("GRADIENT_ACCUMULATION_STEPS", "1"),
            max_grad_norm=get_env_float("MAX_GRAD_NORM", "1.0"),
            max_steps=get_env_int("MAX_STEPS", "200"),
            warmup_steps=get_env_int("WARMUP_STEPS", "10"),
            eval_steps=get_env_int("EVAL_STEPS", "10"),
            save_steps=get_env_int("SAVE_STEPS", "10"),
            save_total_limit=get_env_int("SAVE_TOTAL_LIMIT", "5"),
            target_accuracy=get_env_float("TARGET_ACCURACY", "0.80"),
            patience=get_env_int("PATIENCE", "5"),
            min_group_std=get_env_float("MIN_GROUP_STD", "0.05"),
            output_dir=os.environ.get("OUTPUT_DIR", "outputs/grpo_cua"),
            seed=get_env_int("SEED", "42"),
            verbose=get_env_bool("VERBOSE", "false"),
            enable_wandb=get_env_bool("ENABLE_WANDB", "true"),
            wandb_project=os.environ.get("WANDB_PROJECT", "cua-grpo"),
            wandb_entity=os.environ.get("WANDB_ENTITY"),
            enable_detailed_logging=get_env_bool("ENABLE_DETAILED_LOGGING", "false"),
            lora=LoRAConfig.from_env(),
            vllm=VLLMConfig.from_env(),
            rollout=RolloutConfig.from_env(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "learning_rate": self.learning_rate,
            "beta": self.beta,
            "clip_epsilon": self.clip_epsilon,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "target_accuracy": self.target_accuracy,
            "patience": self.patience,
            "num_rollouts": self.rollout.num_rollouts,
            "max_turns": self.rollout.max_turns,
            "lora_r": self.lora.r,
            "lora_alpha": self.lora.alpha,
            "output_dir": self.output_dir,
            "seed": self.seed,
        }


__all__ = [
    "LoRAConfig",
    "VLLMConfig",
    "RolloutConfig",
    "GRPOConfig",
    "get_env_int",
    "get_env_float",
    "get_env_bool",
]

