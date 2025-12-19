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

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model

# AReaL imports
# 注意：这些导入可能需要根据实际 AReaL API 调整
try:
    from areal.trainer import GRPOTrainer
    from areal.config import GRPOConfig
    from areal.rollout import AsyncRolloutCollector
except ImportError:
    # 如果 AReaL 未安装，提供友好的错误信息
    print("Error: AReaL is not installed. Please install it with:")
    print("  pip install areal>=0.5.0")
    print("  or")
    print("  pip install git+https://github.com/inclusionAI/AReaL.git")
    sys.exit(1)

# 项目 imports
from cua_agent.areal_env import GBoxAReaLEnv, GBoxEnvConfig, create_env_factory
from cua_agent.tasks import get_areal_train_dataset, get_areal_eval_dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config):
    """使用标准 HuggingFace + PEFT 加载模型。
    
    Args:
        config: AReaL 配置
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {config.model.name}")
    
    # 确定 dtype
    torch_dtype = getattr(torch, config.model.torch_dtype, torch.bfloat16)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    # 设置 padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 加载模型（不使用 device_map，让 FSDP 处理）
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=config.model.trust_remote_code,
        # 不使用 device_map，让 FSDP 自动处理
    )
    
    # 应用 LoRA（静态 LoRA）
    if config.model.use_lora:
        logger.info("Applying LoRA configuration")
        
        lora_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.alpha,
            lora_dropout=config.model.lora.dropout,
            target_modules=config.model.lora.target_modules,
            bias=config.model.lora.bias,
            task_type=config.model.lora.task_type,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 设置 pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info("Model and tokenizer loaded successfully")
    
    return model, tokenizer


def create_rollout_collector(config, tokenizer):
    """创建 AReaL 异步 Rollout Collector。
    
    Args:
        config: AReaL 配置
        tokenizer: Tokenizer
        
    Returns:
        AsyncRolloutCollector
    """
    # 创建环境配置
    env_config = GBoxEnvConfig(
        gbox_api_key=os.environ.get("GBOX_API_KEY", ""),
        box_type=config.rollout.gbox.box_type,
        timeout=config.rollout.gbox.timeout,
        expires_in=config.rollout.gbox.expires_in,
        action_delay=config.rollout.action_delay,
        screenshot_delay=config.rollout.screenshot_delay,
        max_turns=config.rollout.max_turns,
    )
    
    # 创建环境工厂
    env_factory = create_env_factory(env_config)
    
    # 创建 Rollout Collector
    collector = AsyncRolloutCollector(
        env_factory=env_factory,
        tokenizer=tokenizer,
        
        # 异步配置
        num_rollouts=config.rollout.num_rollouts,
        concurrency=config.rollout.concurrency,
        
        # 可中断生成
        interruptible=config.rollout.interruptible,
        
        # 数据陈旧度控制
        max_staleness=config.rollout.max_staleness,
        
        # 推理配置
        inference_backend=config.inference.backend,
        inference_config={
            "api_base": config.inference.vllm.api_base,
            "max_tokens": config.inference.vllm.max_tokens,
            "temperature": config.inference.vllm.temperature,
            "top_p": config.inference.vllm.top_p,
            "timeout": config.inference.vllm.timeout,
        },
        
        # 温度变化
        enable_temperature_variation=config.rollout.enable_temperature_variation,
        base_temperature=config.rollout.base_temperature,
        temperature_increment=config.rollout.temperature_increment,
    )
    
    logger.info("Rollout collector created")
    
    return collector


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="AReaL GRPO Training for CUA Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cua_grpo.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint to resume from",
    )
    args = parser.parse_args()
    
    # 加载配置
    logger.info(f"Loading config from: {args.config}")
    config = GRPOConfig.from_yaml(args.config)
    
    # 处理 resume
    if args.resume_from_checkpoint:
        config.checkpoint.resume_from_checkpoint = args.resume_from_checkpoint
    elif args.resume:
        # AReaL 会自动查找最新 checkpoint
        config.checkpoint.resume_from_checkpoint = "latest"
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("CUA Agent AReaL GRPO Training")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"LoRA: r={config.model.lora.r}, alpha={config.model.lora.alpha}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Max steps: {config.training.max_steps}")
    logger.info(f"Rollouts per task: {config.rollout.num_rollouts}")
    logger.info(f"Max turns: {config.rollout.max_turns}")
    logger.info(f"Async mode: {config.rollout.async_mode}")
    logger.info(f"Interruptible: {config.rollout.interruptible}")
    logger.info(f"Max staleness: {config.rollout.max_staleness}")
    logger.info(f"Output dir: {config.training.output_dir}")
    logger.info("=" * 60)
    
    # 加载模型和 tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 创建 Rollout Collector
    logger.info("Creating rollout collector...")
    rollout_collector = create_rollout_collector(config, tokenizer)
    
    # 加载任务数据集
    logger.info("Loading task datasets...")
    train_dataset = get_areal_train_dataset()
    eval_dataset = get_areal_eval_dataset()
    logger.info(f"Training tasks: {len(train_dataset)}")
    logger.info(f"Evaluation tasks: {len(eval_dataset)}")
    
    # 创建 Trainer
    logger.info("Creating GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rollout_collector=rollout_collector,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 开始训练
    logger.info("Starting training...")
    trainer.train()
    
    # 完成
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

