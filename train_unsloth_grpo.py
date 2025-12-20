#!/usr/bin/env python3
"""Unsloth GRPO 训练脚本

使用 Unsloth + HuggingFace 进行 GRPO 训练，替代 vLLM 推理后端。

架构：
- RL: AReaL-style 异步 Rollout 模式
- Rollout: HuggingFace + Unsloth 本地推理
- LoRA: Vision + Cross-modal 目标模块
- RL 算法: GRPO (Group Relative Policy Optimization)
- Reference: HuggingFace 冻结模型

用法：
    # 基本训练
    python train_unsloth_grpo.py
    
    # 指定配置
    python train_unsloth_grpo.py --config configs/unsloth_grpo.yaml
    
    # 从检查点恢复
    python train_unsloth_grpo.py --resume outputs/unsloth_grpo/checkpoints/checkpoint-100

环境变量：
    MODEL_NAME: 模型名称 (默认: Qwen/Qwen3-VL-8B-Instruct)
    GBOX_API_KEY: GBox API 密钥
    OUTPUT_DIR: 输出目录 (默认: outputs/unsloth_grpo)
    MAX_STEPS: 最大训练步数 (默认: 200)
    BATCH_SIZE: 每批任务数 (默认: 2)
    NUM_ROLLOUTS: 每任务 rollout 数 (默认: 4)
    LEARNING_RATE: 学习率 (默认: 1e-5)
    LORA_R: LoRA rank (默认: 16)
    LORA_ALPHA: LoRA alpha (默认: 32)
    USE_LORA: 是否使用 LoRA (默认: true)
    LOAD_IN_4BIT: 是否 4-bit 量化 (默认: true)
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,
)

# Set log levels
logging.getLogger('cua_agent').setLevel(logging.INFO)
logging.getLogger('gbox_cua').setLevel(logging.INFO)

# Suppress some warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unsloth GRPO Training for CUA Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    
    # Model settings
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (overrides env var MODEL_NAME)",
    )
    
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=None,
        help="Load model in 4-bit quantization",
    )
    
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    
    # LoRA settings
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=None,
        help="Use LoRA for training",
    )
    
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full parameter training)",
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank",
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha",
    )
    
    # Training settings
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of tasks per batch",
    )
    
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=None,
        help="Number of rollouts per task (GRPO group size)",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="KL penalty coefficient",
    )
    
    # Rollout settings
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum turns per episode",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Evaluate every N steps",
    )
    
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps",
    )
    
    # Logging settings
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name",
    )
    
    return parser.parse_args()


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Install with: pip install pyyaml")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_config(args) -> "UnslothGRPOConfig":
    """Build training configuration from args and environment."""
    from cua_agent.unsloth_grpo_trainer import UnslothGRPOConfig
    
    # Start with environment-based config
    config = UnslothGRPOConfig.from_env()
    
    # Load YAML config if provided
    if args.config:
        yaml_config = load_yaml_config(args.config)
        for key, value in yaml_config.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
    
    # Override with command line args
    if args.model_name:
        config.model_name = args.model_name
    
    if args.load_in_4bit is not None:
        config.load_in_4bit = args.load_in_4bit
    if args.no_4bit:
        config.load_in_4bit = False
    
    if args.use_lora is not None:
        config.use_lora = args.use_lora
    if args.no_lora:
        config.use_lora = False
    
    if args.lora_r is not None:
        config.lora_r = args.lora_r
    if args.lora_alpha is not None:
        config.lora_alpha = args.lora_alpha
    
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_rollouts is not None:
        config.num_rollouts = args.num_rollouts
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.beta is not None:
        config.beta = args.beta
    
    if args.max_turns is not None:
        config.max_turns = args.max_turns
    if args.temperature is not None:
        config.temperature = args.temperature
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.eval_steps is not None:
        config.eval_steps = args.eval_steps
    if args.save_steps is not None:
        config.save_steps = args.save_steps
    
    if args.verbose:
        config.verbose = True
        config.enable_detailed_logging = True
    
    if args.wandb:
        config.enable_wandb = True
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    
    # Get GBox API key
    if not config.gbox_api_key:
        config.gbox_api_key = os.environ.get("GBOX_API_KEY", "")
    
    return config


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██╗   ██╗███╗   ██╗███████╗██╗      ██████╗ ████████╗██╗  ██╗  ║
║   ██║   ██║████╗  ██║██╔════╝██║     ██╔═══██╗╚══██╔══╝██║  ██║  ║
║   ██║   ██║██╔██╗ ██║███████╗██║     ██║   ██║   ██║   ███████║  ║
║   ██║   ██║██║╚██╗██║╚════██║██║     ██║   ██║   ██║   ██╔══██║  ║
║   ╚██████╔╝██║ ╚████║███████║███████╗╚██████╔╝   ██║   ██║  ██║  ║
║    ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚══════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝  ║
║                                                                  ║
║           GRPO Training for CUA Agent                            ║
║                                                                  ║
║   • RL: AReaL-style Async Rollout                               ║
║   • Rollout: HuggingFace + Unsloth                              ║
║   • LoRA: Vision + Cross-modal                                  ║
║   • Reference: HF Frozen Model                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point."""
    args = parse_args()
    
    print_banner()
    
    # Check for GBox API key
    gbox_api_key = os.environ.get("GBOX_API_KEY", "")
    if not gbox_api_key:
        logger.warning(
            "⚠️  GBOX_API_KEY not set! Rollouts will fail.\n"
            "   Set it with: export GBOX_API_KEY=your_api_key"
        )
    else:
        logger.info(f"✅ GBOX_API_KEY detected: ***{gbox_api_key[-4:]}")
    
    # Build configuration
    config = build_config(args)
    
    # Print configuration summary
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: {'Enabled' if config.use_lora else 'Disabled'}")
    if config.use_lora:
        print(f"    - r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  4-bit Quantization: {'Enabled' if config.load_in_4bit else 'Disabled'}")
    print(f"  Batch Size: {config.batch_size} tasks")
    print(f"  Rollouts per Task: {config.num_rollouts}")
    print(f"  Max Steps: {config.max_steps}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  KL Penalty (beta): {config.beta}")
    print(f"  Max Turns: {config.max_turns}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Output Directory: {config.output_dir}")
    print(f"  Wandb: {'Enabled' if config.enable_wandb else 'Disabled'}")
    print("=" * 60 + "\n")
    
    # Create trainer
    from cua_agent.unsloth_grpo_trainer import UnslothGRPOTrainer
    
    trainer = UnslothGRPOTrainer(
        config=config,
        resume_from_checkpoint=args.resume,
    )
    
    # Run training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        # Save checkpoint on interrupt
        from cua_agent.unsloth_grpo_trainer import TrainingMetrics
        metrics = TrainingMetrics(
            loss=0.0, policy_loss=0.0, kl_loss=0.0,
            avg_reward=0.0, max_reward=0.0, min_reward=0.0,
            accuracy=0.0,
        )
        trainer._save_checkpoint(metrics, is_best=False)
        logger.info("💾 Checkpoint saved on interrupt")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

