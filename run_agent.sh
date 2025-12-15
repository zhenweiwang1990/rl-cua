#!/bin/bash
# CUA Agent Runner Script
# Usage: ./run_agent.sh "Task description"
# 
# Environment variables:
#   GBOX_API_KEY      - GBox API key (required)
#   VLLM_API_BASE     - vLLM server URL (optional, for server mode)
#   MODEL_NAME        - Model name (default: unsloth/Qwen3-VL-32B-Instruct)
#   BOX_TYPE          - Box type (default: android)
#   MAX_TURNS         - Maximum turns (default: 20)

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
    echo "  --model            Model name. Default: unsloth/Qwen3-VL-32B-Instruct"
    echo "  --max-turns        Maximum turns. Default: 20"
    echo "  --verbose, -v      Enable verbose output"
    echo ""
    echo "Environment variables:"
    echo "  GBOX_API_KEY       GBox API key (required)"
    echo "  VLLM_API_BASE      vLLM server URL (optional)"
    exit 1
fi

TASK="$1"
shift

# Check for GBOX_API_KEY
if [ -z "$GBOX_API_KEY" ]; then
    echo "Error: GBOX_API_KEY environment variable is not set"
    echo "Get your API key from https://gbox.ai and run:"
    echo "  export GBOX_API_KEY=your_api_key"
    exit 1
fi

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
BOX_TYPE="${BOX_TYPE:-android}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-VL-32B-Instruct}"
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

# Pass through remaining arguments
CMD+=" $@"

echo "Starting CUA Agent..."
echo "Task: $TASK"
echo "Box Type: $BOX_TYPE"
echo "Model: $MODEL_NAME"
echo ""

eval $CMD

