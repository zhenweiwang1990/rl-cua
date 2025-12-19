#!/bin/bash
# 启动 vLLM 服务（持续运行，always restart）

set -e

# ============ 配置 ============
COMPOSE_FILE="docker-compose.vllm.yml"
PROJECT_NAME="vllm-cua-areal"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"

# ============ 加载环境变量 ============
if [ -f .env ]; then
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

# ============ 检查 vLLM 是否已在运行 ============
if docker ps --format '{{.Names}}' | grep -q "^vllm-cua-areal$"; then
    echo "vLLM 服务已在运行中"
    echo "容器状态:"
    docker ps --filter "name=vllm-cua-areal" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    read -p "是否要重启 vLLM 服务? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "停止现有 vLLM 服务..."
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down
    else
        echo "使用现有的 vLLM 服务"
        exit 0
    fi
fi

# ============ 显示配置 ============
echo "=========================================="
echo "vLLM Service Configuration"
echo "=========================================="
echo "Model: ${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE:-1}"
echo "Max Model Length: ${MAX_MODEL_LEN:-8192}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION:-0.5}"
echo "Restart Policy: always"
echo "=========================================="
echo ""

# ============ 启动 vLLM 服务 ============
echo "Starting vLLM service (with restart: always)..."
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d

# ============ 等待 vLLM 就绪 ============
echo "Waiting for vLLM to be ready..."
MAX_WAIT=600  # 最多等待 10 分钟
WAITED=0
while ! curl -s -f "${VLLM_URL}/health" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Error: vLLM failed to start within $MAX_WAIT seconds"
        echo "Check logs with: docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs vllm"
        exit 1
    fi
    echo "  Waiting... ($WAITED/$MAX_WAIT seconds)"
    sleep 10
    WAITED=$((WAITED + 10))
done

echo "✅ vLLM is ready!"
echo ""
echo "vLLM Service Status:"
docker ps --filter "name=vllm-cua-areal" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "Health check:"
curl -s "${VLLM_URL}/health" | head -5
echo ""
echo "vLLM service is running and will auto-restart if it crashes."
echo "To stop: ./scripts/stop_vllm.sh"
echo "To view logs: docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f vllm"

