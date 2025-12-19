# 独立脚本使用指南

本文档介绍如何使用独立的 vLLM 和训练脚本。这种方式的好处是：
- vLLM 服务持续运行（`restart: always`），训练停止后仍可使用
- 训练脚本可以多次运行，无需重启 vLLM
- 更灵活的资源管理

## 目录结构

```
scripts/
├── start_vllm.sh              # 启动 vLLM（多 GPU）
├── start_vllm_single_gpu.sh    # 启动 vLLM（单 GPU）
├── stop_vllm.sh                # 停止 vLLM（多 GPU）
├── stop_vllm_single_gpu.sh     # 停止 vLLM（单 GPU）
├── train_areal_standalone.sh   # 训练脚本（多 GPU）
└── train_areal_single_gpu.sh   # 训练脚本（单 GPU）

docker-compose.vllm.yml              # vLLM 配置（多 GPU）
docker-compose.vllm.single-gpu.yml   # vLLM 配置（单 GPU）
docker-compose.trainer.yml           # 训练配置（多 GPU）
docker-compose.trainer.single-gpu.yml # 训练配置（单 GPU）
```

## 单 GPU 使用流程

### 1. 启动 vLLM 服务

```bash
# 启动 vLLM（单 GPU，持续运行）
./scripts/start_vllm_single_gpu.sh
```

**说明**：
- vLLM 会在后台运行，使用 `restart: always` 策略
- 即使容器崩溃也会自动重启
- 默认使用 50% GPU 内存，为训练留出空间

**自定义配置**（通过环境变量）：
```bash
export MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
export MAX_MODEL_LEN=8192
export GPU_MEMORY_UTILIZATION=0.5
./scripts/start_vllm_single_gpu.sh
```

### 2. 运行训练

```bash
# 使用默认配置
./scripts/train_areal_single_gpu.sh

# 或指定配置文件
./scripts/train_areal_single_gpu.sh --config configs/cua_grpo_single_gpu.yaml
```

**说明**：
- 脚本会自动等待 vLLM 服务就绪
- 训练完成后，vLLM 服务继续运行
- 可以多次运行训练脚本，无需重启 vLLM

### 3. 停止 vLLM 服务（可选）

```bash
# 停止 vLLM 服务
./scripts/stop_vllm_single_gpu.sh
```

## 多 GPU 使用流程

### 1. 启动 vLLM 服务

```bash
./scripts/start_vllm.sh
```

### 2. 运行训练

```bash
./scripts/train_areal_standalone.sh --config configs/cua_grpo.yaml
```

### 3. 停止 vLLM 服务（可选）

```bash
./scripts/stop_vllm.sh
```

## 环境变量配置

在 `.env` 文件中设置：

```bash
# GBox API
GBOX_API_KEY=your_gbox_api_key

# HuggingFace
HF_ENDPOINT=https://hf-mirror.com
HF_TOKEN=your_hf_token

# Weights & Biases（可选）
WANDB_API_KEY=your_wandb_key

# vLLM 配置（可选）
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.5
TENSOR_PARALLEL_SIZE=1  # 单 GPU 时设为 1
```

## 监控和管理

### 查看 vLLM 日志

```bash
# 单 GPU
docker compose -f docker-compose.vllm.single-gpu.yml -p vllm-cua-areal-single logs -f vllm

# 多 GPU
docker compose -f docker-compose.vllm.yml -p vllm-cua-areal logs -f vllm
```

### 查看 vLLM 状态

```bash
# 单 GPU
docker ps | grep vllm-cua-areal-single

# 多 GPU
docker ps | grep vllm-cua-areal
```

### 检查 vLLM 健康状态

```bash
curl http://localhost:8000/health
```

### 查看 GPU 使用情况

```bash
nvidia-smi
```

## 常见问题

### 1. vLLM 启动失败

**检查**：
- GPU 驱动是否正确安装
- Docker 是否有 GPU 访问权限
- 模型路径是否正确

**查看日志**：
```bash
docker compose -f docker-compose.vllm.single-gpu.yml -p vllm-cua-areal-single logs vllm
```

### 2. 训练无法连接到 vLLM

**检查**：
- vLLM 是否正在运行：`docker ps | grep vllm`
- vLLM 健康检查：`curl http://localhost:8000/health`
- 网络连接：确保训练容器和 vLLM 容器在同一网络

### 3. GPU 内存不足

**单 GPU 配置**：
- 降低 `GPU_MEMORY_UTILIZATION`（默认 0.5）
- 使用更小的模型（如 8B 而不是 32B）
- 减少 `MAX_MODEL_LEN`

### 4. 训练完成后 vLLM 仍占用 GPU

这是正常行为。vLLM 服务会持续运行，以便：
- 快速开始下一次训练
- 用于推理测试
- 避免重复加载模型

如需释放 GPU，运行：
```bash
./scripts/stop_vllm_single_gpu.sh
```

## 优势

1. **资源效率**：vLLM 和训练共享 GPU，充分利用资源
2. **灵活性**：可以随时启动/停止训练，不影响 vLLM
3. **稳定性**：vLLM 自动重启，训练失败不影响推理服务
4. **开发友好**：可以快速迭代训练，无需等待 vLLM 启动

## 注意事项

1. **单 GPU 内存管理**：
   - vLLM 使用 50% GPU 内存（可调整）
   - 训练使用剩余内存
   - 如果内存不足，降低 `GPU_MEMORY_UTILIZATION`

2. **网络配置**：
   - vLLM 和训练容器需要连接到同一网络
   - 脚本会自动处理网络创建

3. **端口冲突**：
   - vLLM 默认使用 8000 端口
   - 如果端口被占用，修改 `docker-compose.vllm.single-gpu.yml`

