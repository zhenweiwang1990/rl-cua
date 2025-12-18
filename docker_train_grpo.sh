#!/bin/bash
#
# Docker training runner for CUA GRPO.
# - Runs training inside the cua-agent image
# - Container uses --restart=always for long-running / background training
# - Supports resume / resume_best via CLI flags
# - Supports GPU separation from vLLM via TRAIN_GPU_DEVICES
#
# Usage:
#   ./docker_train_grpo.sh                # start training (auto-resume if checkpoints exist)
#   ./docker_train_grpo.sh --resume       # force resume from latest checkpoint
#   ./docker_train_grpo.sh --resume_best  # resume from best checkpoint
#
# Environment variables:
#   TRAIN_GPU_DEVICES  - GPU devices for training (e.g., "4", "4-7", "4,5,6,7")
#                        Default: "all" (uses all GPUs, may conflict with vLLM)
#                        Recommended: Use different GPUs than vLLM (e.g., vLLM uses 0-3, training uses 4-7)
#

set -e

IMAGE_NAME="cua-agent:latest"
CONTAINER_NAME="${CONTAINER_NAME:-cua-agent-train}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load .env if present (for GRPO/VLLM/GBOX config)
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$SCRIPT_DIR/.env"
  set +a
  echo "✓ Loaded .env"
else
  echo "⚠ No .env file found, using environment defaults"
fi

MODE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      MODE_ARGS+=(--resume)
      shift
      ;;
    --resume_best)
      MODE_ARGS+=(--resume_best)
      shift
      ;;
    --resume_from_checkpoint)
      MODE_ARGS+=(--resume_from_checkpoint "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--resume | --resume_best | --resume_from_checkpoint PATH]"
      exit 1
      ;;
  esac
done

# Basic checks
if [ -z "$GBOX_API_KEY" ]; then
  echo "❌ GBOX_API_KEY is not set. Set it in .env or export it before running."
  exit 1
fi

if [ -z "$VLLM_API_BASE" ]; then
  echo "⚠ VLLM_API_BASE is not set. Make sure your vLLM server is running and set VLLM_API_BASE in .env."
fi

# GPU configuration for training
TRAIN_GPU_DEVICES="${TRAIN_GPU_DEVICES:-all}"

# Parse GPU devices for Docker --gpus flag
# Use --gpus all for better compatibility (especially for newer GPUs like B200)
# Then use CUDA_VISIBLE_DEVICES to limit which GPUs are visible to the application
if [ "$TRAIN_GPU_DEVICES" = "all" ]; then
  GPU_FLAG="all"
  CUDA_VISIBLE_DEVICES=""
else
  # Always use --gpus all for Docker, then limit via CUDA_VISIBLE_DEVICES
  GPU_FLAG="all"
  # Convert to CUDA_VISIBLE_DEVICES format
  if [[ "$TRAIN_GPU_DEVICES" =~ ^[0-9]+$ ]]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU_DEVICES"
  elif [[ "$TRAIN_GPU_DEVICES" =~ ^[0-9]+-[0-9]+$ ]]; then
    START_GPU=$(echo "$TRAIN_GPU_DEVICES" | cut -d'-' -f1)
    END_GPU=$(echo "$TRAIN_GPU_DEVICES" | cut -d'-' -f2)
    GPU_LIST=$(seq -s, $START_GPU $END_GPU)
    CUDA_VISIBLE_DEVICES="$GPU_LIST"
  elif [[ "$TRAIN_GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU_DEVICES"
  else
    echo "⚠ Invalid TRAIN_GPU_DEVICES format: $TRAIN_GPU_DEVICES. Using all GPUs."
    CUDA_VISIBLE_DEVICES=""
  fi
fi

echo "GPU Configuration:"
echo "  - Training GPUs: $TRAIN_GPU_DEVICES"
echo "  - Docker --gpus flag: $GPU_FLAG"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
  echo "  - CUDA_VISIBLE_DEVICES: (not set, will use all available GPUs)"
fi

# Check if single GPU and warn about memory
if [ "$TRAIN_GPU_DEVICES" = "all" ] || [[ "$TRAIN_GPU_DEVICES" =~ ^[0-9]+$ ]]; then
  echo ""
  echo "⚠️  Single GPU detected. Make sure vLLM uses low memory utilization:"
  echo "    GPU_MEMORY_UTILIZATION=0.5 GPU_DEVICES=0 ./scripts/run_vllm_training.sh"
  echo ""
fi
echo ""

# Prepare optional CUDA env args for docker (only set if non-empty to avoid hiding GPUs)
CUDA_ENV_ARGS=()
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  CUDA_ENV_ARGS+=(-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi

# Build image if needed
if ! docker images | grep -q "${IMAGE_NAME%%:*}"; then
  echo "Docker image '${IMAGE_NAME}' not found, building..."
  "$SCRIPT_DIR/build_docker.sh" "${IMAGE_NAME%%:*}" "${IMAGE_NAME##*:}"
fi

echo "🐳 Starting CUA GRPO training in Docker (container: $CONTAINER_NAME)..."

# Stop existing container (if any)
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container: $CONTAINER_NAME"
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# Run training container with auto-restart
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart=always \
  --gpus "$GPU_FLAG" \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$SCRIPT_DIR:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$SCRIPT_DIR/logs:/workspace/logs" \
  -v "$SCRIPT_DIR/outputs:/workspace/outputs" \
  -e GBOX_API_KEY="$GBOX_API_KEY" \
  -e VLLM_API_BASE="${VLLM_API_BASE:-}" \
  -e LORA_PATH="${LORA_PATH:-}" \
  -e LORA_NAME="${LORA_NAME:-cua_agent_lora}" \
  -e MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-VL-32B-Instruct}" \
  -e BOX_TYPE="${BOX_TYPE:-android}" \
  -e MAX_TURNS="${MAX_TURNS:-20}" \
  -e OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_cua}" \
  "${CUDA_ENV_ARGS[@]}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE_NAME" \
  bash -lc "cd /workspace && python init_lora_adapter.py && python train_grpo_cua.py ${MODE_ARGS[*]}"

echo ""
echo "✓ Training container started."
echo "  - Container name : $CONTAINER_NAME"
echo "  - Auto-restart   : enabled (--restart=always)"
echo "  - Training GPUs  : $TRAIN_GPU_DEVICES"
echo ""
echo "To monitor logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "To stop training but keep checkpoints:"
echo "  docker stop $CONTAINER_NAME"
echo ""
echo "To resume training later (auto-resume from latest checkpoint):"
echo "  ./docker_train_grpo.sh"
echo ""
echo "💡 GPU Configuration Tips:"
if [ "$TRAIN_GPU_DEVICES" = "all" ] || [[ "$TRAIN_GPU_DEVICES" =~ ^[0-9]+$ ]]; then
  echo "  - Single GPU: Make sure vLLM uses low memory (GPU_MEMORY_UTILIZATION=0.45)"
else
  echo "  - Multi-GPU: Recommended to use different GPUs than vLLM"
  echo "    Example: vLLM uses 0-3, Training uses 4-7"
fi
echo "  - For 8-GPU setup (e.g., B200 x8):"
echo "    vLLM: GPU_DEVICES=0-3 TENSOR_PARALLEL_SIZE=4 ./scripts/run_vllm_training.sh"
echo "    Training: TRAIN_GPU_DEVICES=4-7 ./docker_train_grpo.sh"
echo ""

docker logs -f $CONTAINER_NAME


