FROM python:3.11-slim

# ========== basic ==========
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ========== python ==========
RUN pip install --upgrade pip setuptools wheel

# ========== GBox ==========
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/babelcloud/gbox-sdk-py.git

# ========== GBox CUA (shared code) ==========
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/babelcloud/gbox-cua.git

# ========== project ==========
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ========== Vision deps for Unsloth / torchvision ==========
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torchvision

# ========== Unsloth (for GRPO LoRA training) ==========
# Note: Unsloth is installed from GitHub. Requires a GPU-enabled environment.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

COPY . .

RUN mkdir -p /workspace/logs /workspace/outputs

# ========== env ==========
ENV PYTHONPATH=/workspace

# ========== entry ==========
CMD ["bash", "-c", "\
echo '=== CUA Agent - Computer Use Agent ==='; \
echo ''; \
echo 'Usage:'; \
echo '  ./run_agent.sh \"Task description\"'; \
echo '  python run_agent.py \"Task description\" --verbose'; \
echo ''; \
echo 'Env:'; \
echo '  - GBOX_API_KEY'; \
echo '  - VLM_PROVIDER (vllm or openrouter)'; \
echo '  - VLLM_API_BASE (for vllm server mode)'; \
echo '  - OPENROUTER_API_KEY (for openrouter mode)'; \
echo '  - MODEL_NAME'; \
echo '  - BOX_TYPE'; \
echo ''; \
exec bash"]
