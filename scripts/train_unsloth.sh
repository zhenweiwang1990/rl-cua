#!/bin/bash
# train_unsloth.sh - Unsloth GRPO 训练脚本
#
# 用法：
#   ./scripts/train_unsloth.sh [config_file]
#
# 示例：
#   ./scripts/train_unsloth.sh                              # 使用默认配置
#   ./scripts/train_unsloth.sh configs/unsloth_grpo.yaml    # 使用指定配置
#   ./scripts/train_unsloth.sh single_gpu                   # 使用单 GPU 配置

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Unsloth GRPO Training for CUA Agent               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# 检查 GBOX_API_KEY
if [ -z "$GBOX_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  Warning: GBOX_API_KEY not set${NC}"
    echo -e "${YELLOW}   Set it with: export GBOX_API_KEY=your_api_key${NC}"
else
    echo -e "${GREEN}✅ GBOX_API_KEY detected${NC}"
fi

# 确定配置文件
CONFIG_FILE=""
if [ -n "$1" ]; then
    if [ "$1" = "single_gpu" ]; then
        CONFIG_FILE="configs/unsloth_grpo_single_gpu.yaml"
    elif [ -f "$1" ]; then
        CONFIG_FILE="$1"
    else
        echo -e "${RED}❌ Config file not found: $1${NC}"
        exit 1
    fi
else
    CONFIG_FILE="configs/unsloth_grpo.yaml"
fi

echo -e "${BLUE}📋 Using config: ${CONFIG_FILE}${NC}"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

# 加载 .env 文件（如果存在）
if [ -f ".env" ]; then
    echo -e "${BLUE}📦 Loading environment from .env${NC}"
    export $(cat .env | grep -v '^#' | xargs)
fi

# 默认环境变量
export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-VL-8B-Instruct"}
export OUTPUT_DIR=${OUTPUT_DIR:-"outputs/unsloth_grpo"}
export LOAD_IN_4BIT=${LOAD_IN_4BIT:-"true"}
export USE_LORA=${USE_LORA:-"true"}
export LORA_R=${LORA_R:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}

echo -e "${BLUE}🤖 Model: ${MODEL_NAME}${NC}"
echo -e "${BLUE}📁 Output: ${OUTPUT_DIR}${NC}"
echo -e "${BLUE}🔧 4-bit: ${LOAD_IN_4BIT}, LoRA: ${USE_LORA}${NC}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行训练
echo -e "\n${GREEN}🚀 Starting training...${NC}\n"

python train_unsloth_grpo.py \
    --config "$CONFIG_FILE" \
    --verbose

echo -e "\n${GREEN}✅ Training complete!${NC}"
echo -e "${BLUE}📁 Results saved to: ${OUTPUT_DIR}${NC}"

