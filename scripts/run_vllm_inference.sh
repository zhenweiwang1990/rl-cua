#!/bin/bash
set -e

echo "=========================================="
echo "vLLM Inference Server - LoRA VLM Model"
echo "(Interactive Mode)"
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

# Docker image
VLLM_IMAGE="nvcr.io/nvidia/vllm:25.10-py3"

# Display usage information
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_vllm_inference.sh"
    echo ""
    echo "This script runs vLLM in interactive mode (stops when you Ctrl+C)."
    echo "For background/daemon mode, use run_vllm_lora.sh instead."
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
    echo "  HF_ENDPOINT         - Hugging Face endpoint (default: https://hf-mirror.com for China)"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings"
    echo "  ./scripts/run_vllm_inference.sh"
    echo ""
    echo "  # Run with custom port"
    echo "  PORT=8080 ./scripts/run_vllm_inference.sh"
    echo ""
    echo "  # Run with specific model path"
    echo "  MODEL_PATH=outputs/grpo/checkpoint-60 ./scripts/run_vllm_inference.sh"
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
    # Relative path - make it relative to project root
    MODEL_ABS_PATH="$PROJECT_ROOT/$MODEL_PATH"
else
    # Already absolute path
    MODEL_ABS_PATH="$MODEL_PATH"
fi

# Normalize the path (resolve .. and .)
MODEL_ABS_PATH="$(cd "$(dirname "$MODEL_ABS_PATH")" && pwd)/$(basename "$MODEL_ABS_PATH")"

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Hugging Face mirror for China (can be overridden via HF_ENDPOINT env var)
# Common mirrors: https://hf-mirror.com (default), https://huggingface.co (original)
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "Configuration:"
echo "  - Base model: $BASE_MODEL"
echo "  - LoRA adapter: $MODEL_ABS_PATH"
echo "  - LoRA name: $LORA_NAME"
echo "  - Port: $PORT"
echo "  - Host: $HOST"
echo "  - Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  - Max model length: $MAX_MODEL_LEN"
echo "  - Hugging Face endpoint: $HF_ENDPOINT"
echo "  - Tool calling: enabled (auto tool choice)"
echo ""

# Create container name
CONTAINER_NAME="vllm-inference-$(date +%s)"

# Prepare Docker command
DOCKER_CMD="docker run --rm -it \
    --name $CONTAINER_NAME \
    $GPU_FLAGS \
    -p $PORT:$PORT \
    -v $MODEL_ABS_PATH:/workspace/lora_adapter:ro \
    -v $HF_CACHE_DIR:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_ENDPOINT=$HF_ENDPOINT \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    $VLLM_IMAGE"

# Prepare vLLM command
# vLLM supports LoRA through --enable-lora and --lora-modules
# Enable tool calling support for function calling
# Enable multimodal support for VLM
# First upgrade transformers to support Qwen3-VL, then start vLLM
VLLM_BASE_CMD="vllm serve $BASE_MODEL --enable-lora --lora-modules $LORA_NAME=/workspace/lora_adapter --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN --enable-auto-tool-choice --tool-call-parser hermes"

# Add trust-remote-code flag if enabled
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    VLLM_BASE_CMD="$VLLM_BASE_CMD --trust-remote-code"
fi

# Wrap in bash -c to upgrade transformers first
# Build the full command string with proper escaping
VLLM_CMD="bash -c 'pip install --upgrade transformers>=4.50.0 -q && $VLLM_BASE_CMD'"

echo "Starting vLLM inference server..."
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
echo "Example curl commands:"
echo ""
echo "  # Chat completions with LoRA"
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"model\": \"$LORA_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 100}'"
echo ""
echo "  # Chat completions with base model (no LoRA)"
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"model\": \"$BASE_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 100}'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the container
# Use eval to properly handle the command string with spaces and quotes
eval "$DOCKER_CMD $VLLM_CMD"

