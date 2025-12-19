#!/bin/bash
# AReaL 训练脚本（单 GPU，使用 AReaL 内置 vLLM）

set -e

# ============ 解析命令行参数 ============
CONFIG_FILE="configs/cua_grpo_single_gpu.yaml"  # 默认配置文件

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --config FILE     Training config file (default: configs/cua_grpo_single_gpu.yaml)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default config"
            echo "  $0 --config configs/cua_grpo_single_gpu.yaml"
            echo ""
            echo "Note: vLLM is managed by AReaL internally (allocation_mode: vllm:d1p1t1+d1p1t1)"
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
echo "AReaL Training Configuration (Single GPU)"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "vLLM: Managed by AReaL (built-in)"
echo "GPU: Single GPU (H200)"
echo "=========================================="
echo ""

# ============ 构建训练镜像（如果需要） ============
echo "Building training image (if needed)..."
docker compose -f docker-compose.areal.single-gpu.yml build

# ============ 启动训练 ============
echo "Starting AReaL training (single GPU)..."
echo "Using config: $CONFIG_FILE"
echo "Note: vLLM will be automatically started and managed by AReaL"
echo ""

# 通过环境变量传递配置文件路径
export CONFIG_FILE="$CONFIG_FILE"

# 启动训练服务（AReaL 会内置管理 vLLM）
docker compose -f docker-compose.areal.single-gpu.yml up trainer

echo ""
echo "Training complete!"

