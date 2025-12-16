#!/bin/bash
set -e

echo "=========================================="
echo "vLLM Inference Server - Base VLM Model"
echo "=========================================="
echo ""

# Default values - Using Qwen3-VL for CUA
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-VL-30B-A3B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_DEVICES="${GPU_DEVICES:-all}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-cua-server}"

# Function to detect GPU type and select appropriate vLLM image
detect_gpu_type() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "unknown"
        return 1
    fi
    
    # Use timeout to prevent hanging (5 seconds timeout)
    local gpu_name
    if command -v timeout &> /dev/null; then
        gpu_name=$(timeout 5 nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr '[:upper:]' '[:lower:]' || echo "")
    else
        # Fallback if timeout command is not available
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr '[:upper:]' '[:lower:]' || echo "")
    fi
    
    if [[ -z "${gpu_name}" ]]; then
        echo "unknown"
        return 1
    fi
    
    # Remove extra whitespace
    gpu_name=$(echo "${gpu_name}" | xargs)
    
    if echo "${gpu_name}" | grep -qi "h100"; then
        echo "h100"
        return 0
    elif echo "${gpu_name}" | grep -qi "gh200\|grace hopper"; then
        echo "gh200"
        return 0
    elif echo "${gpu_name}" | grep -qi "h200"; then
        echo "h200"
        return 0
    elif echo "${gpu_name}" | grep -qi "gb10"; then
        echo "gb10"
        return 0
    else
        echo "unknown"
        return 1
    fi
}

select_vllm_image() {
    local override_image="${1:-}"
    
    if [[ -n "${override_image}" ]]; then
        echo "${override_image}"
        return 0
    fi
    
    local gpu_type
    gpu_type=$(detect_gpu_type)
    
    case "${gpu_type}" in
        h100|h200)
            echo "vllm/vllm-openai:latest"
            ;;
        gh200)
            echo "rajesh550/gh200-vllm:0.11.1rc2"
            ;;
        *)
            # Default to vllm/vllm-openai:latest for unknown GPUs
            echo "vllm/vllm-openai:latest"
            ;;
    esac
}

# Multi-GPU support: Parse GPU_DEVICES to determine actual GPU count
if command -v nvidia-smi &> /dev/null; then
    TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
    
    # Parse GPU_DEVICES to get actual number of GPUs to use
    if [ "$GPU_DEVICES" = "all" ]; then
        NUM_GPUS_TO_USE=$TOTAL_GPUS
        GPU_LIST=$(seq -s, 0 $((TOTAL_GPUS - 1)))
    elif [[ "$GPU_DEVICES" =~ ^[0-9]+$ ]]; then
        # Single GPU index
        NUM_GPUS_TO_USE=1
        GPU_LIST="$GPU_DEVICES"
    elif [[ "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
        # Comma-separated GPU indices (e.g., "0,1,2,3")
        GPU_LIST="$GPU_DEVICES"
        NUM_GPUS_TO_USE=$(echo "$GPU_DEVICES" | tr ',' '\n' | wc -l)
    elif [[ "$GPU_DEVICES" =~ ^[0-9]+-[0-9]+$ ]]; then
        # Range format (e.g., "0-7")
        START_GPU=$(echo "$GPU_DEVICES" | cut -d'-' -f1)
        END_GPU=$(echo "$GPU_DEVICES" | cut -d'-' -f2)
        NUM_GPUS_TO_USE=$((END_GPU - START_GPU + 1))
        GPU_LIST=$(seq -s, $START_GPU $END_GPU)
    else
        echo "Warning: Invalid GPU_DEVICES format: $GPU_DEVICES. Using all GPUs."
        NUM_GPUS_TO_USE=$TOTAL_GPUS
        GPU_LIST=$(seq -s, 0 $((TOTAL_GPUS - 1)))
    fi
    
    # Validate GPU indices
    for gpu in $(echo "$GPU_LIST" | tr ',' ' '); do
        if [ "$gpu" -ge "$TOTAL_GPUS" ] || [ "$gpu" -lt 0 ]; then
            echo "Error: GPU index $gpu is out of range (0-$((TOTAL_GPUS - 1)))"
            exit 1
        fi
    done
else
    TOTAL_GPUS=0
    NUM_GPUS_TO_USE=1
    GPU_LIST="0"
fi

# Auto-detect tensor parallel size if not set
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE=$NUM_GPUS_TO_USE
    echo "Auto-detected $NUM_GPUS_TO_USE GPU(s), setting TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
fi

# Validate tensor parallel size matches GPU count
if [ "$TENSOR_PARALLEL_SIZE" -gt "$NUM_GPUS_TO_USE" ]; then
    echo "Warning: TENSOR_PARALLEL_SIZE ($TENSOR_PARALLEL_SIZE) > available GPUs ($NUM_GPUS_TO_USE)."
    echo "Setting TENSOR_PARALLEL_SIZE to $NUM_GPUS_TO_USE"
    TENSOR_PARALLEL_SIZE=$NUM_GPUS_TO_USE
fi

# Set CUDA_VISIBLE_DEVICES for container
# This ensures vLLM sees the correct GPU indices inside the container
CUDA_VISIBLE_DEVICES="$GPU_LIST"

# Auto-detect vLLM image based on GPU type (can be overridden by VLLM_IMAGE env var)
if [[ -z "${VLLM_IMAGE:-}" ]]; then
    echo "Detecting GPU type to select appropriate vLLM image..."
    gpu_type=$(detect_gpu_type 2>/dev/null || echo "unknown")
    if [[ "${gpu_type}" != "unknown" ]]; then
        echo "Detected GPU type: ${gpu_type}"
    else
        echo "Could not detect GPU type, using default image"
    fi
    VLLM_IMAGE=$(select_vllm_image)
    echo "Selected vLLM image: ${VLLM_IMAGE}"
else
    echo "Using manually specified VLLM_IMAGE: ${VLLM_IMAGE}"
fi

# Display usage information
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_vllm_base.sh"
    echo ""
    echo "Environment variables:"
    echo "  BASE_MODEL          - Base model name or path (default: unsloth/Qwen3-VL-8B-Instruct)"
    echo "  PORT                - API server port (default: 8000)"
    echo "  HOST                - API server host (default: 0.0.0.0)"
    echo "  GPU_DEVICES         - GPU devices to use: 'all', '0-7', '0,1,2,3', or single index (default: all)"
    echo "  TENSOR_PARALLEL_SIZE - Tensor parallelism size (default: auto-detect from GPU_DEVICES)"
    echo "  MAX_MODEL_LEN       - Maximum model length (default: 32800)"
    echo "  GPU_MEMORY_UTILIZATION - GPU memory utilization ratio (default: 0.8, range: 0.0-1.0)"
    echo "  TRUST_REMOTE_CODE   - Trust remote code (default: true)"
    echo "  CONTAINER_NAME      - Docker container name (default: vllm-cua-server)"
    echo "  VLLM_IMAGE          - Docker image to use (default: auto-detect based on GPU type)"
    echo "  MODEL_HUB           - Model hub to use: 'huggingface' or 'modelscope' (default: modelscope)"
    echo "  HF_ENDPOINT         - Hugging Face endpoint (default: https://hf-mirror.com for China)"
    echo "  MODELSCOPE_CACHE    - ModelScope cache directory (default: ~/.cache/modelscope)"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings (auto-detect all GPUs)"
    echo "  ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with 8 GPUs (GPUs 0-7)"
    echo "  GPU_DEVICES=0-7 TENSOR_PARALLEL_SIZE=8 ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with specific 4 GPUs"
    echo "  GPU_DEVICES=0,1,2,3 TENSOR_PARALLEL_SIZE=4 ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with custom port and 8 GPUs"
    echo "  PORT=8080 GPU_DEVICES=0-7 TENSOR_PARALLEL_SIZE=8 ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with different VLM model on 8 GPUs"
    echo "  BASE_MODEL=unsloth/Qwen3-VL-30B-A3B-Instruct GPU_DEVICES=0-7 TENSOR_PARALLEL_SIZE=8 ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Use ModelScope (for China users) with 8 GPUs"
    echo "  MODEL_HUB=modelscope BASE_MODEL=unsloth/Qwen3-VL-30B-A3B-Instruct GPU_DEVICES=0-7 TENSOR_PARALLEL_SIZE=8 ./scripts/run_vllm_base.sh"
    echo ""
    exit 0
fi

# Check for GPU and prepare GPU flags
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null && [ "$TOTAL_GPUS" -gt 0 ]; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
    
    # Auto-adjust GPU memory utilization based on available memory
    # Get the first GPU's memory info (for single GPU) or calculate average
    if [ "$NUM_GPUS_TO_USE" -eq 1 ]; then
        # Single GPU: get the first GPU in GPU_LIST
        FIRST_GPU=$(echo "$GPU_LIST" | cut -d',' -f1)
        GPU_MEM_INFO=$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | awk -F', ' -v gpu="$FIRST_GPU" '$1==gpu {print $2, $3}')
    else
        # Multi-GPU: use first GPU as reference
        FIRST_GPU=$(echo "$GPU_LIST" | cut -d',' -f1)
        GPU_MEM_INFO=$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | awk -F', ' -v gpu="$FIRST_GPU" '$1==gpu {print $2, $3}')
    fi
    
    if [ -n "$GPU_MEM_INFO" ]; then
        TOTAL_MEM=$(echo "$GPU_MEM_INFO" | awk '{print $1}')
        FREE_MEM=$(echo "$GPU_MEM_INFO" | awk '{print $2}')
        
        if [ -n "$TOTAL_MEM" ] && [ -n "$FREE_MEM" ] && [ "$TOTAL_MEM" -gt 0 ]; then
            # Calculate available memory ratio
            AVAILABLE_RATIO=$(awk "BEGIN {printf \"%.3f\", $FREE_MEM / $TOTAL_MEM}")
            REQUESTED_MEM=$(awk "BEGIN {printf \"%.2f\", $GPU_MEMORY_UTILIZATION * $TOTAL_MEM}")
            
            echo "GPU Memory Check:"
            echo "  - Total memory: ${TOTAL_MEM} MiB"
            echo "  - Free memory: ${FREE_MEM} MiB"
            echo "  - Available ratio: $(awk "BEGIN {printf \"%.1f\", $AVAILABLE_RATIO * 100}")%"
            echo "  - Requested utilization: ${GPU_MEMORY_UTILIZATION} (${REQUESTED_MEM} MiB)"
            
            # If requested memory exceeds available memory, adjust utilization
            if (( $(awk "BEGIN {print ($REQUESTED_MEM > $FREE_MEM)}") )); then
                # Set utilization to 90% of available memory to leave some headroom
                ADJUSTED_UTIL=$(awk "BEGIN {printf \"%.2f\", ($FREE_MEM * 0.90) / $TOTAL_MEM}")
                # Cap at 0.95 maximum
                if (( $(awk "BEGIN {print ($ADJUSTED_UTIL > 0.95)}") )); then
                    ADJUSTED_UTIL=0.95
                fi
                # Floor at 0.1 minimum
                if (( $(awk "BEGIN {print ($ADJUSTED_UTIL < 0.1)}") )); then
                    ADJUSTED_UTIL=0.1
                fi
                
                echo "  ⚠ Warning: Requested memory (${REQUESTED_MEM} MiB) exceeds available (${FREE_MEM} MiB)"
                echo "  → Auto-adjusting GPU_MEMORY_UTILIZATION from ${GPU_MEMORY_UTILIZATION} to ${ADJUSTED_UTIL}"
                GPU_MEMORY_UTILIZATION=$ADJUSTED_UTIL
            else
                echo "  ✓ Available memory is sufficient"
            fi
            echo ""
        fi
    fi
    
    echo "Multi-GPU Configuration:"
    echo "  - Total GPUs available: $TOTAL_GPUS"
    echo "  - GPUs to use: $GPU_LIST ($NUM_GPUS_TO_USE GPU(s))"
    echo "  - Tensor parallel size: $TENSOR_PARALLEL_SIZE"
    echo ""
    
    # Prepare GPU flags for Docker
    # Use --runtime nvidia instead of --gpus (matching gelato deployment)
    GPU_FLAGS="--runtime nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
    
    # For multi-GPU, add NCCL environment variables for better communication
    if [ "$NUM_GPUS_TO_USE" -gt 1 ]; then
        echo "  - Multi-GPU mode enabled (NCCL optimizations will be applied)"
    fi
else
    echo "Warning: No GPU detected. vLLM requires GPU for optimal performance."
    echo "Continuing anyway (may fail if CUDA is not available)..."
    GPU_FLAGS="--ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
fi

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Model Hub Selection
# MODEL_HUB: "huggingface" (default) or "modelscope"
MODEL_HUB="${MODEL_HUB:-huggingface}"

# ModelScope cache directory
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

echo "Configuration:"
echo "  - Base model: $BASE_MODEL"
echo "  - Docker image: $VLLM_IMAGE"
echo "  - Model hub: $MODEL_HUB"
echo "  - Container name: $CONTAINER_NAME"
echo "  - Port: $PORT"
echo "  - Host: $HOST"
echo "  - GPU devices: $GPU_LIST ($NUM_GPUS_TO_USE GPU(s))"
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

# Prepare Docker command with auto-restart and multi-GPU support
# Set NCCL environment variables for multi-GPU communication
NCCL_ENV_VARS=""
if [ "$NUM_GPUS_TO_USE" -gt 1 ]; then
    NCCL_ENV_VARS="-e NCCL_DEBUG=INFO \
        -e NCCL_IB_DISABLE=0 \
        -e NCCL_IB_GID_INDEX=3 \
        -e NCCL_SOCKET_IFNAME=^docker0,lo \
        -e NCCL_P2P_DISABLE=0 \
        -e NCCL_SHM_DISABLE=0"
fi

# Determine if we need to set entrypoint for GH200 image
VLLM_ENTRYPOINT_FLAG=""
if [[ "$VLLM_IMAGE" == *"gh200"* ]] || [[ "$VLLM_IMAGE" == *"rajesh550"* ]] || [[ "$VLLM_IMAGE" == *"gb10"* ]]; then
    # GH200/GB10 image may need explicit entrypoint
    VLLM_ENTRYPOINT_FLAG="--entrypoint python3"
fi

if [ "$MODEL_HUB" = "modelscope" ]; then
    # ModelScope configuration
    DOCKER_CMD="docker run -d \
        --name $CONTAINER_NAME \
        --restart=always \
        $GPU_FLAGS \
        $VLLM_ENTRYPOINT_FLAG \
        -p $PORT:$PORT \
        -v $MODELSCOPE_CACHE:/root/.cache/modelscope \
        -v $HF_CACHE_DIR:/root/.cache/huggingface \
        -e MODELSCOPE_CACHE=/root/.cache/modelscope \
        -e HF_HOME=/root/.cache/huggingface \
        -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        -e VLLM_MM_VIDEO_MAX_NUM=0 \
        -e VLLM_MM_IMAGE_MAX_NUM=16 \
        $NCCL_ENV_VARS \
        $VLLM_IMAGE"
else
    # Hugging Face configuration
    DOCKER_CMD="docker run -d \
        --name $CONTAINER_NAME \
        --restart=always \
        $GPU_FLAGS \
        $VLLM_ENTRYPOINT_FLAG \
        -p $PORT:$PORT \
        -v $HF_CACHE_DIR:/root/.cache/huggingface \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_HUB_ENABLE_HF_TRANSFER=0 \
        -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        -e VLLM_MM_VIDEO_MAX_NUM=0 \
        -e VLLM_MM_IMAGE_MAX_NUM=16 \
        $NCCL_ENV_VARS \
        $VLLM_IMAGE"
fi

# Prepare vLLM command arguments
# Note: Following gelato deployment pattern - simpler configuration without enforce-eager
# Video is disabled via environment variables (VLLM_MM_VIDEO_MAX_NUM=0)
# For vllm/vllm-openai image, the entrypoint is already set, so we just pass arguments
# For GH200 image, we may need to set entrypoint explicitly
if [ "$MODEL_HUB" = "modelscope" ]; then
    # Extract model org and name from BASE_MODEL (e.g., "unsloth/Qwen3-VL-30B-A3B-Instruct")
    MODEL_ORG=$(echo "$BASE_MODEL" | cut -d'/' -f1)
    MODEL_NAME=$(echo "$BASE_MODEL" | cut -d'/' -f2)
    # Use local ModelScope cache path
    MODEL_PATH="/root/.cache/modelscope/hub/models/$MODEL_ORG/$MODEL_NAME"
    echo "Using ModelScope local path: $MODEL_PATH"
    VLLM_BASE_CMD="--model $MODEL_PATH --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization=$GPU_MEMORY_UTILIZATION"
else
    VLLM_BASE_CMD="--model $BASE_MODEL --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization=$GPU_MEMORY_UTILIZATION"
fi

# Add trust-remote-code flag if enabled
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    VLLM_BASE_CMD="$VLLM_BASE_CMD --trust-remote-code"
fi

# Enable auto tool choice for function calling support
# This is required when using tool_choice="auto" in API requests
VLLM_BASE_CMD="$VLLM_BASE_CMD --enable-auto-tool-choice"
# Use openai parser for tool calls (tools are in OpenAI format)
# For Qwen3 models, can also try: qwen3_coder or qwen3_xml
VLLM_BASE_CMD="$VLLM_BASE_CMD --tool-call-parser openai"

# Build the command string
# Check if model requires newer transformers (Qwen3-VL)
if [[ "$BASE_MODEL" == *"Qwen3-VL"* ]] || [[ "$BASE_MODEL" == *"Qwen2.5-VL"* ]]; then
    NEEDS_NEW_TRANSFORMERS=true
else
    NEEDS_NEW_TRANSFORMERS=false
fi

# Build vLLM command based on image type and model hub
if [[ "$VLLM_IMAGE" == *"gh200"* ]] || [[ "$VLLM_IMAGE" == *"rajesh550"* ]] || [[ "$VLLM_IMAGE" == *"gb10"* ]]; then
    # For GH200/GB10 image, entrypoint is set to python3, so we need to call the module
    if [ "$MODEL_HUB" = "modelscope" ]; then
        VLLM_CMD="bash -c 'pip install modelscope -q && export MODELSCOPE_CACHE=/root/.cache/modelscope && python3 -m vllm.entrypoints.openai.api_server $VLLM_BASE_CMD'"
    else
        VLLM_CMD="python3 -m vllm.entrypoints.openai.api_server $VLLM_BASE_CMD"
    fi
else
    # For vllm/vllm-openai image, entrypoint is already set, just pass arguments
    if [ "$MODEL_HUB" = "modelscope" ]; then
        VLLM_CMD="bash -c 'pip install modelscope -q && export MODELSCOPE_CACHE=/root/.cache/modelscope && $VLLM_BASE_CMD'"
    else
        VLLM_CMD="$VLLM_BASE_CMD"
    fi
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
echo "Multi-GPU: $NUM_GPUS_TO_USE GPU(s) (Tensor Parallel: $TENSOR_PARALLEL_SIZE)"
echo ""
echo "Useful commands:"
echo "  - View logs: docker logs -f $CONTAINER_NAME"
echo "  - Stop server: docker stop $CONTAINER_NAME"
echo "  - Remove container: docker rm $CONTAINER_NAME"
echo "  - Check GPU usage: nvidia-smi"
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

