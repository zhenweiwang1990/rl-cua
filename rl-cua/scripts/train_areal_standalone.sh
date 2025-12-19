#!/bin/bash
# AReaL 训练脚本（独立运行，连接到已启动的 vLLM）

set -e

# ============ 解析命令行参数 ============
CONFIG_FILE="configs/cua_grpo.yaml"  # 默认配置文件
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MAX_WAIT=300  # 最多等待 5 分钟

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --vllm-url)
            VLLM_URL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --config FILE     Training config file (default: configs/cua_grpo.yaml)"
            echo "  --vllm-url URL         vLLM service URL (default: http://localhost:8000)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default config"
            echo "  $0 --config configs/cua_grpo_single_gpu.yaml"
            echo "  $0 -c configs/cua_grpo_single_gpu.yaml --vllm-url http://vllm-server:8000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

# ============ 检查配置文件 ============
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ============ 检查必要的环境变量 ============
if [ -z "$GBOX_API_KEY" ]; then
    echo "Error: GBOX_API_KEY is not set"
    echo "Please set it in .env file or as environment variable"
    exit 1
fi

# ============ 显示配置信息 ============
echo "=========================================="
echo "AReaL Training Configuration"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "vLLM URL: $VLLM_URL"
echo "=========================================="
echo ""

# ============ 等待 vLLM 服务就绪 ============
echo "Checking vLLM service availability..."
WAITED=0
while ! curl -s -f "${VLLM_URL}/health" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Error: vLLM service is not available at ${VLLM_URL}"
        echo "Please start vLLM service first:"
        echo "  ./scripts/start_vllm.sh"
        echo ""
        echo "Or check if vLLM is running:"
        echo "  docker ps | grep vllm"
        exit 1
    fi
    echo "  Waiting for vLLM... ($WAITED/$MAX_WAIT seconds)"
    sleep 5
    WAITED=$((WAITED + 5))
done

echo "✅ vLLM service is ready!"
echo ""

# ============ 检查 vLLM 健康状态 ============
HEALTH_RESPONSE=$(curl -s "${VLLM_URL}/health" 2>/dev/null || echo "")
if [ -n "$HEALTH_RESPONSE" ]; then
    echo "vLLM health check: OK"
else
    echo "Warning: Could not verify vLLM health, but service is responding"
fi
echo ""

# ============ 构建训练镜像（如果需要） ============
echo "Building training image (if needed)..."
docker compose -f docker-compose.trainer.yml build

# ============ 启动训练 ============
echo "Starting AReaL training..."
echo "Using config: $CONFIG_FILE"
echo "Connecting to vLLM at: $VLLM_URL"
echo ""

# 通过环境变量传递配置
export CONFIG_FILE="$CONFIG_FILE"
export VLLM_API_BASE="${VLLM_URL}/v1"

# 启动训练服务（连接到 vLLM 网络）
docker compose -f docker-compose.trainer.yml up trainer

echo ""
echo "Training complete!"

