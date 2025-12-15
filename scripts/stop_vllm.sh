#!/bin/bash
# Stop vLLM containers

echo "Stopping vLLM containers..."

# Stop base server
if docker ps -a --format '{{.Names}}' | grep -q "^vllm-cua-server$"; then
    echo "Stopping vllm-cua-server..."
    docker stop vllm-cua-server 2>/dev/null || true
    docker rm vllm-cua-server 2>/dev/null || true
fi

# Stop LoRA server
if docker ps -a --format '{{.Names}}' | grep -q "^vllm-cua-lora-server$"; then
    echo "Stopping vllm-cua-lora-server..."
    docker stop vllm-cua-lora-server 2>/dev/null || true
    docker rm vllm-cua-lora-server 2>/dev/null || true
fi

# Stop any inference containers
for container in $(docker ps -a --format '{{.Names}}' | grep "^vllm-inference-"); do
    echo "Stopping $container..."
    docker stop $container 2>/dev/null || true
    docker rm $container 2>/dev/null || true
done

echo "Done!"

