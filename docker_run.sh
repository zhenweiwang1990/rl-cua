#!/bin/bash
# Docker run script for CUA Agent
# Usage: ./docker_run.sh "Task description" [options]

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="cua-agent:latest"
CONTAINER_NAME="cua-agent-run"

# Check if task is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <task_description> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 'Open the Settings app'"
    echo "  $0 'Search for weather on Google' --verbose"
    echo ""
    echo "Options:"
    echo "  --box-type, -b     Box type (android, linux). Default: android"
    echo "  --model            Model name. Default: unsloth/Qwen3-VL-32B-Instruct"
    echo "  --max-turns        Maximum turns. Default: 20"
    echo "  --verbose, -v      Enable verbose output"
    echo "  --image            Docker image name (default: cua-agent:latest)"
    echo "  --container-name   Container name (default: cua-agent-run)"
    echo ""
    echo "Environment variables:"
    echo "  GBOX_API_KEY       GBox API key (required)"
    echo "  VLLM_API_BASE      vLLM server URL (optional)"
    exit 1
fi

TASK="$1"
shift

# Parse arguments
BOX_TYPE="${BOX_TYPE:-android}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-VL-32B-Instruct}"
MAX_TURNS="${MAX_TURNS:-20}"
VERBOSE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --box-type|-b)
            BOX_TYPE="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"

# Load .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Check for GBOX_API_KEY
if [ -z "$GBOX_API_KEY" ]; then
    echo -e "${RED}❌ Error: GBOX_API_KEY environment variable is not set${NC}"
    echo "Get your API key from https://gbox.ai and:"
    echo "  1. Export it: export GBOX_API_KEY=your_api_key"
    echo "  2. Or add it to .env file in the project root"
    exit 1
fi

# Check if Docker image exists
if ! docker images | grep -q "${IMAGE_NAME%%:*}"; then
    echo -e "${YELLOW}⚠️  Docker image '${IMAGE_NAME}' not found.${NC}"
    echo "Building it now..."
    "$SCRIPT_DIR/build_docker.sh" "${IMAGE_NAME%%:*}" "${IMAGE_NAME##*:}"
fi

echo -e "${BLUE}🐳 Running CUA Agent in Docker...${NC}"
echo "=================================================="
echo "Task: $TASK"
echo "Box Type: $BOX_TYPE"
echo "Model: $MODEL_NAME"
echo "Max Turns: $MAX_TURNS"
echo ""

# Build docker run command
DOCKER_CMD="docker run --rm --gpus all"
DOCKER_CMD+=" -it"
DOCKER_CMD+=" --name $CONTAINER_NAME"
# Network and DNS settings for proper connectivity
DOCKER_CMD+=" --network=host"
# Shared memory for PyTorch
DOCKER_CMD+=" --ipc=host"
DOCKER_CMD+=" --ulimit memlock=-1"
DOCKER_CMD+=" --ulimit stack=67108864"
DOCKER_CMD+=" -v \"$SCRIPT_DIR:/workspace\""
DOCKER_CMD+=" -v \"$HOME/.cache/huggingface:/root/.cache/huggingface\""
DOCKER_CMD+=" -e GBOX_API_KEY=\"$GBOX_API_KEY\""
DOCKER_CMD+=" -e VLLM_API_BASE=\"${VLLM_API_BASE:-}\""
DOCKER_CMD+=" -e MODEL_NAME=\"$MODEL_NAME\""
DOCKER_CMD+=" -e BOX_TYPE=\"$BOX_TYPE\""
DOCKER_CMD+=" -e MAX_TURNS=\"$MAX_TURNS\""
DOCKER_CMD+=" $IMAGE_NAME"
DOCKER_CMD+=" ./run_agent.sh \"$TASK\""
[ -n "$VERBOSE" ] && DOCKER_CMD+=" $VERBOSE"
[ -n "$EXTRA_ARGS" ] && DOCKER_CMD+=" $EXTRA_ARGS"

# Execute docker run
eval $DOCKER_CMD

