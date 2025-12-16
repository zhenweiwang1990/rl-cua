#!/bin/bash
set -e

echo "=========================================="
echo "vLLM Inference Server - LoRA VLM Model"
echo "=========================================="
echo ""

# Default values - Using Qwen3-VL for CUA
MODEL_PATH="${MODEL_PATH:-outputs/grpo/best_model}"
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-VL-32B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LORA_NAME="${LORA_NAME:-cua_agent_lora}"
GPU_DEVICES="${GPU_DEVICES:-all}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-cua-lora-server}"

# Docker image
VLLM_IMAGE="nvcr.io/nvidia/vllm:25.10-py3"

# Display usage information
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_vllm_lora.sh"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH          - Path to LoRA adapter directory (default: outputs/grpo/best_model)"
    echo "  BASE_MODEL          - Base model name or path (default: unsloth/Qwen3-VL-32B-Instruct)"
    echo "  PORT                - API server port (default: 8000)"
    echo "  HOST                - API server host (default: 0.0.0.0)"
    echo "  LORA_NAME           - LoRA adapter name (default: cua_agent_lora)"
    echo "  GPU_DEVICES         - GPU devices to use (default: all)"
    echo "  TENSOR_PARALLEL_SIZE - Tensor parallelism size (default: 2)"
    echo "  MAX_MODEL_LEN       - Maximum model length (default: 32768)"
    echo "  TRUST_REMOTE_CODE   - Trust remote code (default: true)"
    echo "  CONTAINER_NAME      - Docker container name (default: vllm-cua-lora-server)"
    echo "  HF_ENDPOINT         - Hugging Face endpoint (default: https://hf-mirror.com for China)"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings"
    echo "  ./scripts/run_vllm_lora.sh"
    echo ""
    echo "  # Run with custom port"
    echo "  PORT=8080 ./scripts/run_vllm_lora.sh"
    echo ""
    echo "  # Run with specific model path"
    echo "  MODEL_PATH=outputs/grpo/checkpoint-60 ./scripts/run_vllm_lora.sh"
    echo ""
    exit 0
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Please check the MODEL_PATH environment variable or train a model first."
    exit 1
fi

# Check if adapter files exist
if [ ! -f "$MODEL_PATH/adapter_model.safetensors" ] && [ ! -f "$MODEL_PATH/adapter_model.bin" ]; then
    echo "Error: LoRA adapter not found in $MODEL_PATH"
    echo "Expected files: adapter_model.safetensors or adapter_model.bin"
    exit 1
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

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Convert MODEL_PATH to absolute path if it's relative
if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_ABS_PATH="$PROJECT_ROOT/$MODEL_PATH"
else
    MODEL_ABS_PATH="$MODEL_PATH"
fi

# Normalize the path (resolve .. and .)
MODEL_ABS_PATH="$(cd "$(dirname "$MODEL_ABS_PATH")" && pwd)/$(basename "$MODEL_ABS_PATH")"

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Model Hub Selection
MODEL_HUB="${MODEL_HUB:-huggingface}"

# Hugging Face mirror for China
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ModelScope cache directory
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

echo "Configuration:"
echo "  - Base model: $BASE_MODEL"
echo "  - Model hub: $MODEL_HUB"
echo "  - LoRA adapter: $MODEL_ABS_PATH"
echo "  - LoRA name: $LORA_NAME"
echo "  - Container name: $CONTAINER_NAME"
echo "  - Port: $PORT"
echo "  - Host: $HOST"
echo "  - Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  - Max model length: $MAX_MODEL_LEN"
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
    DOCKER_CMD="docker run -d \
        --name $CONTAINER_NAME \
        --restart=always \
        $GPU_FLAGS \
        -p $PORT:$PORT \
        -v $MODEL_ABS_PATH:/workspace/lora_adapter:ro \
        -v $MODELSCOPE_CACHE:/root/.cache/modelscope \
        -v $HF_CACHE_DIR:/root/.cache/huggingface \
        -e MODELSCOPE_CACHE=/root/.cache/modelscope \
        -e HF_HOME=/root/.cache/huggingface \
        $VLLM_IMAGE"
else
    DOCKER_CMD="docker run -d \
        --name $CONTAINER_NAME \
        --restart=always \
        $GPU_FLAGS \
        -p $PORT:$PORT \
        -v $MODEL_ABS_PATH:/workspace/lora_adapter:ro \
        -v $HF_CACHE_DIR:/root/.cache/huggingface \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_ENDPOINT=$HF_ENDPOINT \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        $VLLM_IMAGE"
fi

# Prepare vLLM command
# Note: For VLM models with LoRA
# First upgrade transformers to support Qwen3-VL, then start vLLM
VLLM_BASE_CMD="vllm serve $BASE_MODEL --enable-lora --lora-modules $LORA_NAME=/workspace/lora_adapter --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN --enable-auto-tool-choice --tool-call-parser hermes"

# Add trust-remote-code flag if enabled
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    VLLM_BASE_CMD="$VLLM_BASE_CMD --trust-remote-code"
fi

# Build the command string
if [ "$MODEL_HUB" = "modelscope" ]; then
    # Install modelscope library, use vLLM's built-in transformers
    VLLM_CMD="bash -c 'pip install modelscope -q && $VLLM_BASE_CMD'"
else
    # Use vLLM's built-in transformers version
    VLLM_CMD="$VLLM_BASE_CMD"
fi

echo "Starting vLLM inference server with auto-restart..."
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
echo "Available models:"
echo "  - Base model (no LoRA): \"$BASE_MODEL\""
echo "  - LoRA model: \"$LORA_NAME\""
echo ""
echo "Useful commands:"
echo "  - View logs: docker logs -f $CONTAINER_NAME"
echo "  - Stop server: docker stop $CONTAINER_NAME"
echo "  - Remove container: docker rm $CONTAINER_NAME"
echo ""
echo "Example curl command:"
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"model\": \"$LORA_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 100}'"
echo ""

# Follow logs
echo "Following container logs (Ctrl+C to detach, container will keep running)..."
echo "=========================================="
docker logs -f $CONTAINER_NAME

