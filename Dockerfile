FROM nvcr.io/nvidia/pytorch:25.11-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install GBox SDK from GitHub first (not available on PyPI)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install git+https://github.com/babelcloud/gbox-sdk-py.git

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/logs /workspace/outputs

# Set environment variables
ENV PYTHONPATH=/workspace
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_CACHE=/root/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub

# Default command - show help message
CMD ["bash", "-c", "echo '=== CUA Agent - Computer Use Agent ==='; echo ''; echo 'Usage:'; echo '  ./run_agent.sh \"Task description\"'; echo '  python run_agent.py \"Task description\" --verbose'; echo ''; echo 'Required environment variables:'; echo '  - GBOX_API_KEY: Your GBox API key (from https://gbox.ai)'; echo '  - VLLM_API_BASE: (optional) vLLM server URL'; echo '  - MODEL_NAME: (optional) Model name'; echo '  - BOX_TYPE: (optional) Box type (android/linux)'; echo ''; echo 'Examples:'; echo '  ./run_agent.sh \"Open the Settings app\"'; echo '  ./run_agent.sh \"Search for weather on Google\" --verbose'; echo ''; exec bash"]

