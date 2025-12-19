#!/bin/bash
# AReaL 训练启动脚本

set -e

# ============ 配置 ============
CONFIG_FILE="${CONFIG_FILE:-configs/cua_grpo.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_cua}"

# ============ 创建输出目录 ============
mkdir -p "$OUTPUT_DIR/logs"

# ============ 检查环境 ============
echo "Checking environment..."

if [ -z "$GBOX_API_KEY" ]; then
    echo "Error: GBOX_API_KEY is not set"
    exit 1
fi

# ============ 启动训练 ============
echo "Starting AReaL GRPO training..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

# 使用 AReaL launcher（自动处理分布式）
python -m areal.launcher.local \
    train_areal.py \
    --config "$CONFIG_FILE" \
    "$@"

echo "Training complete!"

