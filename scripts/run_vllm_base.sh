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
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-cua-server}"

# Docker image
VLLM_IMAGE="nvcr.io/nvidia/vllm:25.10-py3"

# Display usage information
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_vllm_base.sh"
    echo ""
    echo "Environment variables:"
    echo "  BASE_MODEL          - Base model name or path (default: unsloth/Qwen3-VL-32B-Instruct)"
    echo "  PORT                - API server port (default: 8000)"
    echo "  HOST                - API server host (default: 0.0.0.0)"
    echo "  GPU_DEVICES         - GPU devices to use (default: all)"
    echo "  TENSOR_PARALLEL_SIZE - Tensor parallelism size (default: 2)"
    echo "  MAX_MODEL_LEN       - Maximum model length (default: 32768)"
    echo "  TRUST_REMOTE_CODE   - Trust remote code (default: true)"
    echo "  CONTAINER_NAME      - Docker container name (default: vllm-cua-server)"
    echo "  HF_ENDPOINT         - Hugging Face endpoint (default: https://hf-mirror.com for China)"
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
    exit 0
fi

# Check for GPU
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    GPU_FLAGS="--gpus $GPU_DEVICES --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
else
    echo "Warning: No GPU detected. vLLM requires GPU for optimal performance."
    echo "Continuing anyway (may fail if CUDA is not available)..."
fi

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Hugging Face mirror for China (can be overridden via HF_ENDPOINT env var)
# Common mirrors: https://hf-mirror.com (default), https://huggingface.co (original)
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "Configuration:"
echo "  - Base model: $BASE_MODEL"
echo "  - Container name: $CONTAINER_NAME"
echo "  - Port: $PORT"
echo "  - Host: $HOST"
echo "  - Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  - Max model length: $MAX_MODEL_LEN"
echo "  - Hugging Face endpoint: $HF_ENDPOINT"
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
DOCKER_CMD="docker run -d \
    --name $CONTAINER_NAME \
    --restart=always \
    $GPU_FLAGS \
    -p $PORT:$PORT \
    -v $HF_CACHE_DIR:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_ENDPOINT=$HF_ENDPOINT \
    -e HF_HUB_ENABLE_HF_TRANSFER=0 \
    $VLLM_IMAGE"

# Prepare vLLM command
# Note: For VLM models, we need to specify image input capabilities
# First upgrade transformers to support Qwen3-VL, then start vLLM
VLLM_BASE_CMD="vllm serve $BASE_MODEL --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN --enable-auto-tool-choice --tool-call-parser hermes"

# Add trust-remote-code flag if enabled
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    VLLM_BASE_CMD="$VLLM_BASE_CMD --trust-remote-code"
fi

# Wrap in bash -c to upgrade transformers first
# Build the full command string with proper escaping
VLLM_CMD="bash -c 'pip install --upgrade transformers>=4.50.0 -q && $VLLM_BASE_CMD'"

echo "Starting vLLM inference server with auto-restart..."
echo "Note: Upgrading transformers to support Qwen3-VL model..."
echo ""

# Run the container in detached mode
# Use eval to properly handle the command string with spaces and quotes
CONTAINER_ID=$(eval "$DOCKER_CMD $VLLM_CMD")

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

