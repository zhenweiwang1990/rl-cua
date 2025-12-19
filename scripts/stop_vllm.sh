#!/bin/bash
# 停止 vLLM 服务

set -e

COMPOSE_FILE="docker-compose.vllm.yml"
PROJECT_NAME="vllm-cua-areal"

echo "Stopping vLLM service..."
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down

echo "✅ vLLM service stopped"
