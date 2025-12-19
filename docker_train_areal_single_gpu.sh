#!/bin/bash
# AReaL 训练 Docker 启动脚本 - 单 GPU (H200) 版本

set -e

# ============ 配置 ============
COMPOSE_FILE="docker-compose.areal.single-gpu.yml"
PROJECT_NAME="cua-areal-single"
CONFIG_FILE="configs/cua_grpo_single_gpu.yaml"

# ============ 检查 GPU ============
echo "Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Warning: Multiple GPUs detected. This script is optimized for single GPU."
    echo "Consider using docker_train_areal.sh for multi-GPU training."
    read -p "Continue with single GPU setup? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============ 加载环境变量 ============
if [ -f .env ]; then
    # 安全地加载 .env 文件，只导出格式正确的 KEY=VALUE 行
    while IFS= read -r line || [ -n "$line" ]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        if [[ "$line" =~ ^[[:alnum:]_]+= ]]; then
            export "$line"
        fi
    done < .env
fi

# ============ 检查必要的环境变量 ============
if [ -z "$GBOX_API_KEY" ]; then
    echo "Error: GBOX_API_KEY is not set"
    echo "Please set it in .env file or as environment variable"
    exit 1
fi

# ============ 检查配置文件 ============
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ============ 构建镜像 ============
echo "Building Docker images..."
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" build

# ============ 启动服务 ============
echo "Starting vLLM service (using 50% GPU memory)..."
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d vllm

# 等待 vLLM 启动
echo "Waiting for vLLM to be ready..."
MAX_WAIT=300  # 最多等待 5 分钟
WAITED=0
while ! docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" exec -T vllm curl -s http://localhost:8000/health > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Error: vLLM failed to start within $MAX_WAIT seconds"
        echo "Check logs with: docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs vllm"
        exit 1
    fi
    echo "  Waiting... ($WAITED/$MAX_WAIT seconds)"
    sleep 10
    WAITED=$((WAITED + 10))
done
echo "vLLM is ready!"

# ============ 显示 GPU 使用情况 ============
echo ""
echo "Current GPU usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# ============ 启动训练 ============
echo ""
echo "Starting AReaL training (using remaining GPU memory)..."
echo "Config: $CONFIG_FILE"
echo ""
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up trainer

# ============ 清理 ============
echo ""
echo "Training complete. Cleaning up..."
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down

echo "Done!"

