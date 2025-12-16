#!/bin/bash
# Build Docker image for CUA Agent
# Usage: ./build_docker.sh [image_name] [tag]

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default image name and tag
IMAGE_NAME="${1:-cua-agent}"
TAG="${2:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo -e "${BLUE}🐳 Building Docker image for CUA Agent...${NC}"
echo "=================================================="
echo -e "Image: ${GREEN}${FULL_IMAGE_NAME}${NC}"
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
cd "$SCRIPT_DIR"

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}❌ Error: Dockerfile not found!${NC}"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: requirements.txt not found!${NC}"
    exit 1
fi

# Build with BuildKit for better caching
echo -e "${BLUE}Building Docker image...${NC}"
DOCKER_BUILDKIT=1 docker build \
    -t "${FULL_IMAGE_NAME}" \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Docker image built successfully!${NC}"
    echo "=================================================="
    echo -e "Image: ${GREEN}${FULL_IMAGE_NAME}${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run container: ./docker_run.sh \"Task description\""
    echo "  2. Or use docker-compose: docker-compose up"
    echo "  3. Or run manually:"
    echo "     docker run --gpus all -it --rm \\"
    echo "       -v \$(pwd):/workspace \\"
    echo "       -v ~/.cache/huggingface:/root/.cache/huggingface \\"
    echo "       -e GBOX_API_KEY=\$GBOX_API_KEY \\"
    echo "       ${FULL_IMAGE_NAME} \\"
    echo "       ./run_agent.sh \"Your task\""
    echo "=================================================="
else
    echo ""
    echo -e "${RED}❌ Docker build failed!${NC}"
    exit 1
fi

