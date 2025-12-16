#!/bin/bash
set -e

echo "=========================================="
echo "vLLM Inference Server - Base VLM Model"
echo "=========================================="
echo ""

# Default values - Using Qwen3-VL for CUA
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-VL-32B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_DEVICES="${GPU_DEVICES:-all}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-cua-server}"

# Auto-detect number of GPUs if TENSOR_PARALLEL_SIZE is not set
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        TENSOR_PARALLEL_SIZE=$NUM_GPUS
        echo "Auto-detected $NUM_GPUS GPU(s), setting TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
    else
        TENSOR_PARALLEL_SIZE=1
        echo "No GPU detection available, defaulting to TENSOR_PARALLEL_SIZE=1"
    fi
fi

# Docker image
VLLM_IMAGE="nvcr.io/nvidia/vllm:25.11-py3"

# Display usage information
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_vllm_base.sh"
    echo ""
    echo "Environment variables:"
    echo "  BASE_MODEL          - Base model name or path (default: unsloth/Qwen3-VL-32B-Instruct)"
    echo "  PORT                - API server port (default: 8000)"
    echo "  HOST                - API server host (default: 0.0.0.0)"
    echo "  GPU_DEVICES         - GPU devices to use (default: all)"
    echo "  TENSOR_PARALLEL_SIZE - Tensor parallelism size (default: auto-detect from available GPUs)"
    echo "  MAX_MODEL_LEN       - Maximum model length (default: 32768)"
    echo "  GPU_MEMORY_UTILIZATION - GPU memory utilization ratio (default: 0.85, range: 0.0-1.0)"
    echo "  TRUST_REMOTE_CODE   - Trust remote code (default: true)"
    echo "  CONTAINER_NAME      - Docker container name (default: vllm-cua-server)"
    echo "  VLLM_IMAGE          - Docker image to use (default: vllm/vllm-openai:latest)"
    echo "  MODEL_HUB           - Model hub to use: 'huggingface' or 'modelscope' (default: huggingface)"
    echo "  HF_ENDPOINT         - Hugging Face endpoint (default: https://hf-mirror.com for China)"
    echo "  MODELSCOPE_CACHE    - ModelScope cache directory (default: ~/.cache/modelscope)"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings (Qwen3-VL-32B)"
    echo "  ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with custom port"
    echo "  PORT=8080 ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with different VLM model"
    echo "  BASE_MODEL=Qwen/Qwen2-VL-7B-Instruct ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Use ModelScope (for China users)"
    echo "  MODEL_HUB=modelscope BASE_MODEL=Qwen/Qwen2.5-VL-32B-Instruct ./scripts/run_vllm_base.sh"
    echo ""
    exit 0
fi

# Check for GPU
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
    GPU_FLAGS="--gpus $GPU_DEVICES --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
else
    echo "Warning: No GPU detected. vLLM requires GPU for optimal performance."
    echo "Continuing anyway (may fail if CUDA is not available)..."
fi

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Model Hub Selection
# MODEL_HUB: "huggingface" (default) or "modelscope"
MODEL_HUB="${MODEL_HUB:-modelscope}"

# ModelScope cache directory
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

echo "Configuration:"
echo "  - Base model: $BASE_MODEL"
echo "  - Docker image: $VLLM_IMAGE"
echo "  - Model hub: $MODEL_HUB"
echo "  - Container name: $CONTAINER_NAME"
echo "  - Port: $PORT"
echo "  - Host: $HOST"
echo "  - Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  - Max model length: $MAX_MODEL_LEN"
echo "  - GPU memory utilization: $GPU_MEMORY_UTILIZATION"
if [ "$MODEL_HUB" = "modelscope" ]; then
    echo "  - ModelScope cache: $MODELSCOPE_CACHE"
else
    echo "  - Hugging Face endpoint: $HF_ENDPOINT"
fi
echo "  - Auto restart: enabled"
echo ""

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo ""
fi

# Prepare Docker command with auto-restart
if [ "$MODEL_HUB" = "modelscope" ]; then
    # ModelScope configuration
    DOCKER_CMD="docker run -d \
        --name $CONTAINER_NAME \
        --restart=always \
        $GPU_FLAGS \
        -p $PORT:$PORT \
        -v $MODELSCOPE_CACHE:/root/.cache/modelscope \
        -v $HF_CACHE_DIR:/root/.cache/huggingface \
        -e MODELSCOPE_CACHE=/root/.cache/modelscope \
        -e HF_HOME=/root/.cache/huggingface \
        $VLLM_IMAGE"
else
    # Hugging Face configuration
    DOCKER_CMD="docker run -d \
        --name $CONTAINER_NAME \
        --restart=always \
        $GPU_FLAGS \
        -p $PORT:$PORT \
        -v $HF_CACHE_DIR:/root/.cache/huggingface \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_HUB_ENABLE_HF_TRANSFER=0 \
        $VLLM_IMAGE"
fi

# Prepare vLLM command
# Note: For VLM models, we need to specify image input capabilities
# First upgrade transformers to support Qwen3-VL, then start vLLM
# If using ModelScope, convert model path to local ModelScope cache path
# Add --enforce-eager for Qwen3-VL to avoid CUDA compilation issues with conv3d
if [ "$MODEL_HUB" = "modelscope" ]; then
    # Extract model org and name from BASE_MODEL (e.g., "unsloth/Qwen3-VL-32B-Instruct")
    MODEL_ORG=$(echo "$BASE_MODEL" | cut -d'/' -f1)
    MODEL_NAME=$(echo "$BASE_MODEL" | cut -d'/' -f2)
    # Use local ModelScope cache path
    MODEL_PATH="/root/.cache/modelscope/hub/models/$MODEL_ORG/$MODEL_NAME"
    echo "Using ModelScope local path: $MODEL_PATH"
    VLLM_BASE_CMD="vllm serve $MODEL_PATH --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes"
else
    VLLM_BASE_CMD="vllm serve $BASE_MODEL --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes"
fi

# Add trust-remote-code flag if enabled
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    VLLM_BASE_CMD="$VLLM_BASE_CMD --trust-remote-code"
fi

# Build the command string
# Check if model requires newer transformers (Qwen3-VL)
if [[ "$BASE_MODEL" == *"Qwen3-VL"* ]] || [[ "$BASE_MODEL" == *"Qwen2.5-VL"* ]]; then
    NEEDS_NEW_TRANSFORMERS=true
else
    NEEDS_NEW_TRANSFORMERS=false
fi

if [ "$MODEL_HUB" = "modelscope" ]; then

    VLLM_CMD="bash -c 'pip install modelscope -q && export MODELSCOPE_CACHE=/root/.cache/modelscope && $VLLM_BASE_CMD'"
    
else
    VLLM_CMD="$VLLM_BASE_CMD"
    
fi

echo "Starting vLLM inference server with auto-restart..."

if [ "$MODEL_HUB" = "modelscope" ]; then
    echo "Note: Installing ModelScope SDK..."
fi
echo ""

# Run the container in detached mode
if [ "$MODEL_HUB" = "modelscope" ]; then
    # Use eval for bash -c command
    CONTAINER_ID=$(eval "$DOCKER_CMD $VLLM_CMD")
else
    # Direct execution
    CONTAINER_ID=$($DOCKER_CMD $VLLM_CMD)
fi

echo "✓ Container started successfully!"
echo "  Container ID: ${CONTAINER_ID:0:12}"
echo "  Container name: $CONTAINER_NAME"
echo ""
echo "API will be available at: http://$HOST:$PORT/v1"
echo "OpenAI-compatible endpoints:"
echo "  - Chat completions: http://$HOST:$PORT/v1/chat/completions"
echo "  - Completions: http://$HOST:$PORT/v1/completions"
echo ""
echo "Model: $BASE_MODEL"
echo ""
echo "Useful commands:"
echo "  - View logs: docker logs -f $CONTAINER_NAME"
echo "  - Stop server: docker stop $CONTAINER_NAME"
echo "  - Remove container: docker rm $CONTAINER_NAME"
echo ""
echo "Example curl command (with image):"
echo '  curl http://localhost:'$PORT'/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"model": "'$BASE_MODEL'", "messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}], "max_tokens": 1000}'"'"
echo ""

# Follow logs
echo "Following container logs (Ctrl+C to detach, container will keep running)..."
echo "=========================================="
docker logs -f $CONTAINER_NAME

