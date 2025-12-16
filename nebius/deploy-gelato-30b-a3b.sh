#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/deploy-vllm-common.sh"

CONTAINER_NAME="vllm-gelato-30b-a3b"
MODEL_NAME="mlfoundations/Gelato-30B-A3B"

# Auto-detect image based on GPU type (H100 -> vllm/vllm-openai:nightly, GH200 -> rajesh550/gh200-vllm:0.11.1rc2)
# Can be overridden by setting VLLM_IMAGE environment variable
if [[ -z "${VLLM_IMAGE:-}" ]]; then
    echo "Detecting GPU type to select appropriate vLLM image..."
    gpu_type=$(detect_gpu_type)
    if [[ "${gpu_type}" != "unknown" ]]; then
        echo "Detected GPU type: ${gpu_type}"
    fi
    VLLM_IMAGE=$(select_vllm_image)
    echo "Selected vLLM image: ${VLLM_IMAGE}"
else
    echo "Using manually specified VLLM_IMAGE: ${VLLM_IMAGE}"
fi
HOST_PORT="${HOST_PORT:-8001}"

# For vllm/vllm-openai image, use default entrypoint (leave empty)
# For GH200 image (rajesh550/gh200-vllm), may need explicit entrypoint
# Only set entrypoint if VLLM_ENTRYPOINT is explicitly provided, or if using GH200 image
if [[ -z "${VLLM_ENTRYPOINT:-}" ]]; then
    if [[ "${VLLM_IMAGE}" == *"gh200"* ]] || [[ "${VLLM_IMAGE}" == *"rajesh550"* ]]; then
        # GH200 image may need explicit entrypoint
        VLLM_ENTRYPOINT="python3 -m vllm.entrypoints.openai.api_server"
    else
        # vllm/vllm-openai image has correct entrypoint, leave it empty
        VLLM_ENTRYPOINT=""
    fi
fi
VLLM_COMMAND="${VLLM_COMMAND:-}"

deploy_vllm_container "${CONTAINER_NAME}" "${VLLM_IMAGE}" "${MODEL_NAME}" "${HOST_PORT}" \
    "${VLLM_ENTRYPOINT}" \
    "${VLLM_COMMAND}" \
    "" \
    --model "${MODEL_NAME}" \
    --gpu-memory-utilization=0.85 \
    --max-model-len 32800

