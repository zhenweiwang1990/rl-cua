#!/bin/bash
#
# Docker training runner for CUA GRPO.
# - Runs training inside the cua-agent image
# - Container uses --restart=always for long-running / background training
# - Supports resume / resume_best via CLI flags
#
# Usage:
#   ./docker_train_grpo.sh                # start training (auto-resume if checkpoints exist)
#   ./docker_train_grpo.sh --resume       # force resume from latest checkpoint
#   ./docker_train_grpo.sh --resume_best  # resume from best checkpoint
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
  --gpus all \
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
  -e MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-VL-32B-Instruct}" \
  -e BOX_TYPE="${BOX_TYPE:-android}" \
  -e MAX_TURNS="${MAX_TURNS:-20}" \
  -e OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_cua}" \
  "$IMAGE_NAME" \
  bash -lc "cd /workspace && python train_grpo_cua.py ${MODE_ARGS[*]}"

echo ""
echo "✓ Training container started."
echo "  - Container name : $CONTAINER_NAME"
echo "  - Auto-restart   : enabled (--restart=always)"
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

docker logs -f $CONTAINER_NAME


