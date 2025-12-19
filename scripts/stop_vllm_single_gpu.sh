#!/bin/bash
# 停止 vLLM 服务（单 GPU）

set -e

COMPOSE_FILE="docker-compose.vllm.single-gpu.yml"
PROJECT_NAME="vllm-cua-areal-single"

echo "Stopping vLLM service (single GPU)..."
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down

echo "✅ vLLM service stopped"

