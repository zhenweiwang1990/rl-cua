#!/bin/bash
# CUA Agent Runner Script
# Usage: ./run_agent.sh "Task description"
# 
# Environment variables:
#   GBOX_API_KEY         - GBox API key (required)
#   VLM_PROVIDER         - VLM provider: "vllm" or "openrouter" (default: vllm)
#   VLLM_API_BASE        - vLLM server URL (optional, for server mode)
#   OPENROUTER_API_KEY   - OpenRouter API key (required if VLM_PROVIDER=openrouter)
#   MODEL_NAME           - Model name (default: unsloth/Qwen3-VL-30B-A3B-Instruct)
#   BOX_TYPE             - Box type (default: android)
#   MAX_TURNS            - Maximum turns (default: 20)

set -e

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
    echo "  --model            Model name. Default: unsloth/Qwen3-VL-30B-A3B-Instruct"
    echo "  --max-turns        Maximum turns. Default: 20"
    echo "  --verbose, -v      Enable verbose output"
    echo ""
    echo "Environment variables:"
    echo "  GBOX_API_KEY         GBox API key (required)"
    echo "  VLM_PROVIDER         VLM provider: 'vllm' or 'openrouter' (default: vllm)"
    echo "  VLLM_API_BASE        vLLM server URL (optional, for vllm provider)"
    echo "  OPENROUTER_API_KEY   OpenRouter API key (required if VLM_PROVIDER=openrouter)"
    exit 1
fi

TASK="$1"
shift

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Check for GBOX_API_KEY
if [ -z "$GBOX_API_KEY" ]; then
    echo "Error: GBOX_API_KEY environment variable is not set"
    echo "Get your API key from https://gbox.ai and run:"
    echo "  export GBOX_API_KEY=your_api_key"
    echo "Or add it to .env file in the project root"
    exit 1
fi

# Default values
BOX_TYPE="${BOX_TYPE:-android}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-VL-30B-A3B-Instruct}"
MAX_TURNS="${MAX_TURNS:-20}"

# Build command
CMD="python3 $SCRIPT_DIR/run_agent.py"
CMD+=" --task \"$TASK\""
CMD+=" --box-type $BOX_TYPE"
CMD+=" --model \"$MODEL_NAME\""
CMD+=" --max-turns $MAX_TURNS"

if [ -n "$VLLM_API_BASE" ]; then
    CMD+=" --vllm-api-base $VLLM_API_BASE"
fi

if [ -n "$VLM_PROVIDER" ]; then
    CMD+=" --vlm-provider $VLM_PROVIDER"
fi

if [ -n "$OPENROUTER_API_KEY" ]; then
    CMD+=" --openrouter-api-key $OPENROUTER_API_KEY"
fi

# Pass through remaining arguments
CMD+=" $@"

echo "Starting CUA Agent..."
echo "Task: $TASK"
echo "Box Type: $BOX_TYPE"
echo "Model: $MODEL_NAME"
echo "VLM Provider: ${VLM_PROVIDER:-vllm}"
[ -n "$VLLM_API_BASE" ] && echo "vLLM API Base: $VLLM_API_BASE" || echo "vLLM API Base: (not set, will use local model)"
[ -n "$OPENROUTER_API_KEY" ] && echo "OpenRouter API Key: ${OPENROUTER_API_KEY:0:10}..." || echo "OpenRouter API Key: (not set)"
echo ""

eval $CMD

