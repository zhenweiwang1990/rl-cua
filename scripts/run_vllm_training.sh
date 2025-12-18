#!/bin/bash
set -e

echo "=========================================="
echo "vLLM Inference Server for GRPO Training"
echo "=========================================="
echo ""
echo "This script starts a vLLM server optimized for GRPO training"
echo "with dynamic LoRA adapter support."
echo ""

# Default values - Using Qwen3-VL for CUA
# NOTE: This script now only respects MODEL_NAME for selecting the base model.
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-VL-32B-Instruct}"
LORA_PATH="${LORA_PATH:-}"  # Optional: Initial LoRA adapter path
LORA_NAME="${LORA_NAME:-cua_agent_lora}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_DEVICES="${GPU_DEVICES:-all}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-cua-training}"

# Docker image selection
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

# Display help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_vllm_training.sh"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_NAME          - Base model name (default: unsloth/Qwen3-VL-32B-Instruct)"
    echo "  LORA_PATH           - Path to initial LoRA adapter (optional)"
    echo "  LORA_NAME           - LoRA adapter name for API (default: cua_agent_lora)"
    echo "  PORT                - API server port (default: 8000)"
    echo "  HOST                - API server host (default: 0.0.0.0)"
    echo "  GPU_DEVICES         - GPU devices: 'all', '0-7', '0,1,2,3' (default: all)"
    echo "  TENSOR_PARALLEL_SIZE- Tensor parallelism (default: auto from GPU_DEVICES)"
    echo "  MAX_MODEL_LEN       - Maximum model length (default: 16384)"
    echo "  GPU_MEMORY_UTILIZATION - GPU memory usage (default: 0.85)"
    echo "  CONTAINER_NAME      - Docker container name (default: vllm-cua-training)"
    echo "  VLLM_IMAGE          - Docker image to use"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings (auto-detect GPUs)"
    echo "  ./scripts/run_vllm_training.sh"
    echo ""
    echo "  # Run with 8 GPUs"
    echo "  GPU_DEVICES=0-7 ./scripts/run_vllm_training.sh"
    echo ""
    echo "  # Run with initial LoRA adapter"
    echo "  LORA_PATH=outputs/grpo_cua/best_model ./scripts/run_vllm_training.sh"
    echo ""
    exit 0
fi

# Detect GPU count
if command -v nvidia-smi &> /dev/null; then
    TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
    
    if [ "$GPU_DEVICES" = "all" ]; then
        NUM_GPUS_TO_USE=$TOTAL_GPUS
        GPU_LIST=$(seq -s, 0 $((TOTAL_GPUS - 1)))
    elif [[ "$GPU_DEVICES" =~ ^[0-9]+$ ]]; then
        NUM_GPUS_TO_USE=1
        GPU_LIST="$GPU_DEVICES"
    elif [[ "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
        GPU_LIST="$GPU_DEVICES"
        NUM_GPUS_TO_USE=$(echo "$GPU_DEVICES" | tr ',' '\n' | wc -l)
    elif [[ "$GPU_DEVICES" =~ ^[0-9]+-[0-9]+$ ]]; then
        START_GPU=$(echo "$GPU_DEVICES" | cut -d'-' -f1)
        END_GPU=$(echo "$GPU_DEVICES" | cut -d'-' -f2)
        NUM_GPUS_TO_USE=$((END_GPU - START_GPU + 1))
        GPU_LIST=$(seq -s, $START_GPU $END_GPU)
    else
        echo "Warning: Invalid GPU_DEVICES format. Using all GPUs."
        NUM_GPUS_TO_USE=$TOTAL_GPUS
        GPU_LIST=$(seq -s, 0 $((TOTAL_GPUS - 1)))
    fi
else
    echo "Warning: nvidia-smi not found. Assuming 1 GPU."
    TOTAL_GPUS=1
    NUM_GPUS_TO_USE=1
    GPU_LIST="0"
fi

# Auto-detect tensor parallel size
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE=$NUM_GPUS_TO_USE
fi

CUDA_VISIBLE_DEVICES="$GPU_LIST"

echo "Configuration:"
echo "  - Base model: $MODEL_NAME"
echo "  - LoRA name: $LORA_NAME"
if [ -n "$LORA_PATH" ]; then
    echo "  - Initial LoRA path: $LORA_PATH"
fi
echo "  - Container name: $CONTAINER_NAME"
echo "  - Port: $PORT"
echo "  - GPUs: $GPU_LIST ($NUM_GPUS_TO_USE GPU(s))"
echo "  - Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  - Max model length: $MAX_MODEL_LEN"
echo "  - GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo ""

# Show GPU info
if [ "$TOTAL_GPUS" -gt 0 ]; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Stop existing container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo ""
fi

# Prepare GPU flags
GPU_FLAGS="--runtime nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"

# NCCL environment for multi-GPU
NCCL_ENV_VARS=""
if [ "$NUM_GPUS_TO_USE" -gt 1 ]; then
    NCCL_ENV_VARS="-e NCCL_DEBUG=INFO -e NCCL_IB_DISABLE=0 -e NCCL_P2P_DISABLE=0"
fi

# Cache directories
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

# Prepare volume mounts
VOLUME_MOUNTS="-v $HF_CACHE_DIR:/root/.cache/huggingface -v $MODELSCOPE_CACHE:/root/.cache/modelscope"

# Mount LoRA path if provided
if [ -n "$LORA_PATH" ]; then
    # Get absolute path
    if [[ "$LORA_PATH" != /* ]]; then
        LORA_ABS_PATH="$(cd "$(dirname "$LORA_PATH")" && pwd)/$(basename "$LORA_PATH")"
    else
        LORA_ABS_PATH="$LORA_PATH"
    fi
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $LORA_ABS_PATH:/workspace/lora_adapter:ro"
fi

# Build Docker command
DOCKER_CMD="docker run -d \
    --name $CONTAINER_NAME \
    --restart=always \
    $GPU_FLAGS \
    -p $PORT:$PORT \
    $VOLUME_MOUNTS \
    -e HF_HOME=/root/.cache/huggingface \
    -e MODELSCOPE_CACHE=/root/.cache/modelscope \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -e VLLM_MM_VIDEO_MAX_NUM=0 \
    -e VLLM_MM_IMAGE_MAX_NUM=16 \
    $NCCL_ENV_VARS \
    $VLLM_IMAGE"

# Build vLLM command
VLLM_CMD="--model $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization=$GPU_MEMORY_UTILIZATION \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder"

# Add trust remote code
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    VLLM_CMD="$VLLM_CMD --trust-remote-code"
fi

# Add LoRA support if path provided
if [ -n "$LORA_PATH" ]; then
    VLLM_CMD="$VLLM_CMD --enable-lora --lora-modules $LORA_NAME=/workspace/lora_adapter"
fi

echo "Starting vLLM server for GRPO training..."
echo ""

# Run the container
CONTAINER_ID=$($DOCKER_CMD $VLLM_CMD)

echo "✓ Container started!"
echo "  Container ID: ${CONTAINER_ID:0:12}"
echo "  Container name: $CONTAINER_NAME"
echo ""
echo "API available at: http://$HOST:$PORT/v1"
echo ""
echo "Available models:"
echo "  - Base model: $MODEL_NAME"
if [ -n "$LORA_PATH" ]; then
    echo "  - LoRA model: $LORA_NAME"
fi
echo ""
echo "Commands:"
echo "  - View logs: docker logs -f $CONTAINER_NAME"
echo "  - Stop: docker stop $CONTAINER_NAME"
echo ""
echo "Test with:"
echo "  curl http://localhost:$PORT/v1/models"
echo ""

# Follow logs
echo "Following logs (Ctrl+C to detach, container keeps running)..."
echo "=========================================="
docker logs -f $CONTAINER_NAME

