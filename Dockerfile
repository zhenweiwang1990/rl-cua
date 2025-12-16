FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

# ========== basic ==========
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# ========== python ==========
RUN pip3 install --upgrade pip setuptools wheel

# ---------- PyTorch ----------
# 官方 wheel，不走 NVIDIA 魔改
RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------- vLLM ----------
# Qwen-VL / Conv3D / Vision 最稳版本
RUN pip install \
    vllm==0.10.3 \
    transformers>=4.45.0 \
    accelerate \
    pillow \
    safetensors \
    sentencepiece

# ========== GBox ==========
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/babelcloud/gbox-sdk-py.git

# ========== project ==========
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p /workspace/logs /workspace/outputs

# ========== env ==========
ENV PYTHONPATH=/workspace
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_CACHE=/root/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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
echo '  - VLLM_API_BASE'; \
echo '  - MODEL_NAME'; \
echo '  - BOX_TYPE'; \
echo ''; \
exec bash"]
