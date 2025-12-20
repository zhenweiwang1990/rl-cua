#!/bin/bash
# ============================================================================
# AReaL CUA GRPO Training - Multi-GPU (DDP) Launch Script
# ============================================================================
#
# Usage:
#   # 4 GPUs
#   ./scripts/train_areal_cua_multi.sh --gpus 4
#
#   # 8 GPUs with Unsloth
#   ./scripts/train_areal_cua_multi.sh --gpus 8 --loader unsloth
#
#   # Custom config
#   ./scripts/train_areal_cua_multi.sh --gpus 4 --config configs/custom.yaml
#
# Environment Variables:
#   GBOX_API_KEY     - GBox API key (required)
#   MODEL_NAME       - Model to train (default: Qwen/Qwen3-VL-32B-Instruct)
#   CUA_MAX_TURNS    - Max turns per episode (default: 15)
#   OUTPUT_DIR       - Output directory (default: ./outputs/areal_cua_multi)
#   MASTER_ADDR      - Master node address (default: localhost)
#   MASTER_PORT      - Master node port (default: 29500)
#
# ============================================================================

set -e

# ============ Configuration ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/areal_cua_multi}"
CUA_MAX_TURNS="${CUA_MAX_TURNS:-15}"
CUA_CONTEXT_WINDOW="${CUA_CONTEXT_WINDOW:-5}"
CONFIG_FILE="${CONFIG_FILE:-configs/areal_cua_multi_gpu.yaml}"
LOADER="${LOADER:-hf}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --loader)
            LOADER="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============ Validation ============
if [ -z "$GBOX_API_KEY" ]; then
    echo "❌ Error: GBOX_API_KEY environment variable is required"
    echo "   Set it with: export GBOX_API_KEY=your_api_key"
    exit 1
fi

# Check GPU count
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "❌ Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

# ============ Setup ============
cd "$PROJECT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Export environment variables
export MODEL_NAME
export CUA_MAX_TURNS
export CUA_CONTEXT_WINDOW
export PYTHONUNBUFFERED=1
export MASTER_ADDR
export MASTER_PORT

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# ============ Print Configuration ============
echo ""
echo "=============================================="
echo "🚀 AReaL CUA GRPO Training - Multi-GPU (DDP)"
echo "=============================================="
echo "  Model:        $MODEL_NAME"
echo "  Loader:       $LOADER"
echo "  Config:       $CONFIG_FILE"
echo "  Output:       $OUTPUT_DIR"
echo "  GPUs:         $NUM_GPUS"
echo "  Max turns:    $CUA_MAX_TURNS"
echo "  Context:      $CUA_CONTEXT_WINDOW"
echo "  GBox API:     ***${GBOX_API_KEY: -4}"
echo "  Master:       $MASTER_ADDR:$MASTER_PORT"
echo "=============================================="
echo ""

# ============ Launch Training ============
if [ "$LOADER" = "unsloth" ]; then
    echo "📦 Using Unsloth model loader (2x faster)"
else
    echo "📦 Using HuggingFace model loader (validation mode)"
fi

echo ""
echo "🏃 Starting multi-GPU training with $NUM_GPUS GPUs..."
echo ""

# Method 1: Using torchrun (recommended for DDP)
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nnodes=1 \
    --node_rank=0 \
    train_areal_cua.py \
    --config "$CONFIG_FILE" \
    --loader "$LOADER"

# Method 2: Using AReaL launcher (for managed vLLM with tensor parallel)
# Uncomment this and comment out the above if you want AReaL to manage vLLM
# python -m areal.launcher.local train_areal_cua.py \
#     --config "$CONFIG_FILE" \
#     --loader "$LOADER"

echo ""
echo "✅ Training completed!"
echo "   Output: $OUTPUT_DIR"
echo ""

