# AReaL 训练快速开始

## 🚀 最快启动方式（Docker）

```bash
# 1. 配置环境变量
cp env.example .env
# 编辑 .env，设置 GBOX_API_KEY

# 2. 启动训练（一键启动 vLLM + 训练）
./docker_train_areal.sh
```

## 📋 完整步骤

### 前置检查

```bash
# 检查 GPU
nvidia-smi

# 检查 Docker
docker --version
docker-compose --version

# 检查 GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 配置

```bash
# 1. 环境变量
cp env.example .env
# 编辑 .env，设置 GBOX_API_KEY

# 2. 训练配置（可选）
# 编辑 configs/cua_grpo.yaml 根据你的资源调整
```

### 启动

**方式 1: Docker Compose（推荐）**
```bash
./docker_train_areal.sh
```

**方式 2: 本地环境**
```bash
# 终端 1: 启动 vLLM
./scripts/run_vllm_base.sh

# 终端 2: 启动训练
source venv/bin/activate
python -m areal.launcher.local train_areal.py --config configs/cua_grpo.yaml
```

## 🔍 常用命令

### 查看日志
```bash
# 训练日志
tail -f outputs/grpo_cua/logs/training.log

# Rollout 日志
tail -f outputs/grpo_cua/logs/rollouts.log

# Docker 日志
docker-compose -f docker-compose.areal.yml logs -f trainer
```

### 检查服务
```bash
# vLLM 健康检查
curl http://localhost:8000/health

# 查看检查点
ls -lh outputs/grpo_cua/checkpoints/
```

### 断点续训
```bash
# 从最新检查点恢复
python -m areal.launcher.local train_areal.py \
  --config configs/cua_grpo.yaml \
  --resume

# 从指定检查点恢复
python -m areal.launcher.local train_areal.py \
  --config configs/cua_grpo.yaml \
  --resume_from_checkpoint outputs/grpo_cua/checkpoints/checkpoint-50
```

## ⚙️ 关键配置

### 最小化测试配置

编辑 `configs/cua_grpo.yaml`:
```yaml
training:
  max_steps: 2
  batch_size: 1

rollout:
  num_rollouts: 1
  concurrency: 1
```

### GPU 资源不足时

```yaml
training:
  batch_size: 2  # 减少 batch size

model:
  lora:
    r: 8  # 减少 LoRA rank

rollout:
  concurrency: 2  # 减少并发数
```

## 🐛 常见问题

### vLLM 无法连接
```bash
# 检查服务
curl http://localhost:8000/health

# 查看日志
docker logs vllm-cua-areal
```

### GPU 内存不足
- 减少 `batch_size`
- 减少 `lora.r`
- 减少 `concurrency`

### GBox API 错误
```bash
# 检查环境变量
echo $GBOX_API_KEY

# 在 .env 中设置
GBOX_API_KEY=your_actual_api_key
```

## 📚 详细文档

- [完整启动指南](./AREAL_TRAINING_GUIDE.md) - 详细步骤和说明
- [vLLM 设置](./VLLM_SETUP.md) - vLLM 详细配置
- [迁移计划](./MIGRATION_PLAN.md) - 完整迁移计划

