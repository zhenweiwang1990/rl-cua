# Unsloth GRPO Training Guide

本指南介绍如何使用 Unsloth + HuggingFace 进行 CUA Agent 的 GRPO 训练，替代原来的 AReaL + vLLM 架构。

## 架构概述

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unsloth GRPO Training                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Policy    │     │  Reference  │     │    GBox     │       │
│  │   Model     │     │   Model     │     │Environment  │       │
│  │             │     │             │     │             │       │
│  │ Unsloth +   │     │ HuggingFace │     │  Android    │       │
│  │ LoRA        │     │  Frozen     │     │   Box       │       │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │
│         │                   │                   │               │
│         │                   │                   │               │
│         v                   v                   v               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    GRPO Training Loop                       ││
│  │                                                             ││
│  │  1. Collect rollouts using Policy Model                    ││
│  │  2. Compute KL divergence with Reference Model             ││
│  │  3. Calculate group-wise advantages                        ││
│  │  4. Update Policy Model with GRPO loss                     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

| 组件 | 说明 |
|------|------|
| **RL** | AReaL-style 异步 Rollout 模式 |
| **Rollout** | HuggingFace + Unsloth 本地推理 |
| **LoRA** | Vision + Cross-modal 目标模块 |
| **RL 算法** | GRPO (Group Relative Policy Optimization) |
| **Reference** | HuggingFace 冻结模型 |

## 快速开始

### 1. 安装依赖

```bash
# 基本依赖
pip install -r requirements.txt

# 安装 Unsloth (选择适合您的 CUDA 版本)
# CUDA 12.1
pip install unsloth

# 或开发版本
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
```

### 2. 设置环境变量

```bash
# 必需
export GBOX_API_KEY=your_gbox_api_key

# 可选 (有默认值)
export MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
export OUTPUT_DIR=outputs/unsloth_grpo
export LOAD_IN_4BIT=true
export USE_LORA=true
```

### 3. 运行训练

```bash
# 使用默认配置
python train_unsloth_grpo.py

# 使用指定配置
python train_unsloth_grpo.py --config configs/unsloth_grpo.yaml

# 单 GPU 优化配置
python train_unsloth_grpo.py --config configs/unsloth_grpo_single_gpu.yaml

# 使用便捷脚本
./scripts/train_unsloth.sh
./scripts/train_unsloth.sh single_gpu
```

## 配置文件

### 标准配置: `configs/unsloth_grpo.yaml`

适用于多 GPU 或高内存 GPU (如 H200 80GB)。

### 单 GPU 配置: `configs/unsloth_grpo_single_gpu.yaml`

针对单 GPU 优化，使用更小的 batch size 和更激进的内存优化。

## 关键参数

### 模型设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | Qwen/Qwen3-VL-8B-Instruct | 基础模型 |
| `load_in_4bit` | true | 4-bit 量化 |
| `max_seq_length` | 8192 | 最大序列长度 |

### LoRA 设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_lora` | true | 是否使用 LoRA |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha |

**目标模块 (Vision + Cross-modal):**
- Language: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Vision: 根据模型架构自动检测

### 训练设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 2 | 每批任务数 |
| `num_rollouts` | 4 | 每任务 rollout 数 (GRPO 组大小) |
| `learning_rate` | 1e-5 | 学习率 |
| `beta` | 0.01 | KL penalty 系数 |
| `max_steps` | 200 | 最大训练步数 |

### Rollout 设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_turns` | 15 | 每个 episode 最大步数 |
| `temperature` | 0.7 | 采样温度 |
| `max_new_tokens` | 1024 | 最大生成 token 数 |

## 命令行参数

```bash
python train_unsloth_grpo.py --help

# 常用参数
--config CONFIG         配置文件路径
--resume CHECKPOINT     从检查点恢复
--model-name NAME       模型名称
--lora-r R              LoRA rank
--lora-alpha ALPHA      LoRA alpha
--batch-size N          每批任务数
--num-rollouts N        每任务 rollout 数
--max-steps N           最大训练步数
--learning-rate LR      学习率
--output-dir DIR        输出目录
--verbose               详细日志
--wandb                 启用 Wandb
```

## 与 vLLM 版本的对比

| 特性 | vLLM 版本 | Unsloth 版本 |
|------|----------|-------------|
| 推理后端 | vLLM 服务器 | 本地 HuggingFace/Unsloth |
| 部署复杂度 | 需要单独启动 vLLM | 单进程运行 |
| LoRA 支持 | vLLM 动态 LoRA | Unsloth/PEFT LoRA |
| 内存效率 | 高 (专业推理优化) | 高 (4-bit 量化) |
| 多 GPU | 支持 tensor parallel | 需要 FSDP |
| Reference 模型 | 状态字典复制 | 独立冻结模型 |

## 文件结构

```
rl-cua/
├── cua_agent/
│   ├── unsloth_inference.py      # Unsloth 推理模块
│   ├── unsloth_grpo_trainer.py   # GRPO Trainer
│   └── ...
├── configs/
│   ├── unsloth_grpo.yaml         # 标准配置
│   └── unsloth_grpo_single_gpu.yaml  # 单 GPU 配置
├── scripts/
│   └── train_unsloth.sh          # 训练脚本
└── train_unsloth_grpo.py         # 训练入口
```

## 常见问题

### 1. Unsloth 安装失败

确保使用正确的 CUDA 版本安装。参考 [Unsloth 官方文档](https://github.com/unslothai/unsloth)。

### 2. 内存不足 (OOM)

- 减小 `batch_size` 和 `num_rollouts`
- 启用 `load_in_4bit`
- 减小 `max_seq_length` 和 `max_new_tokens`
- 使用单 GPU 配置: `configs/unsloth_grpo_single_gpu.yaml`

### 3. LoRA 目标模块不匹配

不同模型架构可能有不同的模块名称。系统会自动过滤无效的目标模块并使用默认值。

### 4. GBox 连接失败

确保 `GBOX_API_KEY` 已正确设置，并检查网络连接。

## 迁移指南

从 vLLM 版本迁移到 Unsloth 版本：

1. **不再需要 vLLM 服务器**: 移除 vLLM 相关配置和启动脚本
2. **更新环境变量**: 移除 `VLLM_*` 变量，添加 Unsloth 相关变量
3. **使用新训练脚本**: `train_unsloth_grpo.py` 替代 `train_areal.py`
4. **更新配置文件**: 使用 `configs/unsloth_grpo.yaml`

## 参考资料

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [TRL GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)

