# CUA Agent - Computer Use Agent for GRPO Training

A Computer Use Agent that interacts with device screens (Android/Linux) to complete tasks using vision-language models.

## Overview

The CUA Agent:
1. Creates a GBox environment (Android or Linux virtual device)
2. Takes screenshots and sends them to a Vision-Language Model (VLM)
3. Receives action instructions from the VLM
4. Converts element descriptions to coordinates using `gbox-handy-1` model
5. Executes UI actions on the device
6. Repeats until task completion or max turns reached

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CUA Agent                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   VLM        │    │   Actions    │    │   GBox       │       │
│  │   Inference  │───▶│   Parser     │───▶│   Client     │       │
│  │   (vLLM)     │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         ▲                                        │               │
│         │                                        ▼               │
│    Screenshot ◀───────────────────────── UI Actions             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Option 1: Docker (Recommended)

Using Docker ensures a consistent environment with all dependencies:

```bash
# Build the Docker image
./build_docker.sh

# Run agent in Docker
./docker_run.sh "Open the Settings app"

# Or use docker-compose
docker-compose up
```

### Option 2: Local Installation

```bash
# Clone or navigate to the project
cd rl-cua

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the environment template and fill in your API keys:

```bash
cp env.example .env
# Edit .env with your GBOX_API_KEY
```

## Usage

### Basic Usage

```bash
# Using the shell script
./run_agent.sh "Open the Settings app"

# Using Python directly
python run_agent.py "Open the Settings app" --verbose
```

### Options

```bash
./run_agent.sh "Task description" [options]

Options:
  --box-type, -b     Box type (android, linux). Default: android
  --model            Model name. Default: unsloth/Qwen3-VL-32B-Instruct
  --max-turns        Maximum turns. Default: 20
  --verbose, -v      Enable verbose output
  --vllm-api-base    vLLM server URL (for server mode)
```

### Examples

```bash
# Open settings on Android
./run_agent.sh "Open the Settings app and navigate to Wi-Fi settings"

# Search on Google
./run_agent.sh "Open Chrome and search for 'weather today'" --verbose

# With custom server
VLLM_API_BASE=http://gpu-server:8000/v1 ./run_agent.sh "Install Twitter app"

# Using Docker
./docker_run.sh "Open the Settings app" --verbose
./docker_run.sh "Search for weather" --box-type android --model unsloth/Qwen3-VL-32B-Instruct
```

## Docker Usage

### Building the Image

```bash
# Build with default name (cua-agent:latest)
./build_docker.sh

# Build with custom name and tag
./build_docker.sh my-cua-agent v1.0
```

### Running with Docker

```bash
# Simple usage
./docker_run.sh "Task description"

# With options
./docker_run.sh "Task description" --verbose --box-type android

# Using docker-compose (set GBOX_API_KEY in .env first)
docker-compose up
```

### Docker Environment Variables

You can set environment variables in `.env` file or pass them when running:

- `GBOX_API_KEY` (required): Your GBox API key
- `VLLM_API_BASE` (optional): vLLM server URL
- `MODEL_NAME` (optional): Model name
- `BOX_TYPE` (optional): Box type (android/linux)
- `MAX_TURNS` (optional): Maximum turns

## Action Types

The agent supports these action types:

| Action | Description | Parameters |
|--------|-------------|------------|
| `click` | Click on an element | `option` (left/right/double), `target` |
| `swipe` | Swipe between two points | `start_target`, `end_target` |
| `scroll` | Scroll in a direction | `direction`, `distance`, `target` |
| `input` | Type text | `text`, `target` |
| `key_press` | Press keys | `keys` |
| `button_press` | Press device button | `button` (back/home/menu) |

### Target Description

Targets are described with these fields:

```json
{
  "element": "login button",     // Required: what the element is
  "label": "Sign In",            // Text on the element
  "color": "blue",               // Element color
  "size": "large",               // Size (small/medium/large)
  "location": "center of screen", // Location on screen
  "shape": "rectangle"           // Shape
}
```

## Project Structure

```
rl-cua/
├── cua_agent/
│   ├── __init__.py      # Package exports
│   ├── actions.py       # Action types and parsing
│   ├── agent.py         # Main CUA Agent
│   ├── config.py        # Configuration dataclasses
│   ├── gbox_client.py   # GBox API client
│   ├── prompts.py       # Prompt templates
│   ├── tools.py         # Tool definitions
│   └── vlm_inference.py # VLM inference module
├── run_agent.py         # CLI entry point
├── run_agent.sh         # Shell script wrapper
├── requirements.txt     # Python dependencies
├── env.example          # Environment template
└── README.md            # This file
```

## 中国大陆用户 - 使用 Hugging Face 镜像

如果你在中国大陆，访问 Hugging Face 可能会遇到网络问题。所有脚本已默认配置使用镜像站点。

### 自动配置

所有 vLLM 脚本已默认使用 `https://hf-mirror.com` 作为镜像站点。无需额外配置即可使用。

### Transformers 版本问题

**重要**：如果遇到 `qwen3_vl` 模型类型不被识别的错误，脚本会自动在 Docker 容器中升级 Transformers 到最新版本（>=4.50.0）以支持 Qwen3-VL 模型。首次启动时会自动执行升级，可能需要几分钟时间。

### 手动配置

如果需要使用其他镜像或恢复原始站点，可以通过环境变量设置：

```bash
# 使用默认镜像（hf-mirror.com）
./scripts/run_vllm_base.sh

# 使用其他镜像
HF_ENDPOINT=https://your-mirror.com ./scripts/run_vllm_base.sh

# 使用原始 Hugging Face 站点（需要代理）
HF_ENDPOINT=https://huggingface.co ./scripts/run_vllm_base.sh
```

### 常见镜像站点

- `https://hf-mirror.com` - 默认镜像（推荐）
- `https://huggingface.co` - 原始站点（需要代理）

### 预下载模型（可选）

如果网络仍然不稳定，可以预先下载模型到本地：

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 使用 huggingface-cli 下载
huggingface-cli download unsloth/Qwen3-VL-32B-Instruct --local-dir ~/.cache/huggingface/hub/models--unsloth--Qwen3-VL-32B-Instruct

# 或者使用 Python
python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Qwen3-VL-32B-Instruct', cache_dir='~/.cache/huggingface')"
```

## vLLM Server Mode

For better performance with multiple agents or larger models, use the provided scripts:

### Start vLLM Server (Base Model)

```bash
# Start with default settings (Qwen3-VL-32B-Instruct from Hugging Face)
./scripts/run_vllm_base.sh

# With custom port
PORT=8080 ./scripts/run_vllm_base.sh

# With different model
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct ./scripts/run_vllm_base.sh

# Use ModelScope (for users in China)
MODEL_HUB=modelscope MODEL_NAME=Qwen/Qwen2.5-VL-32B-Instruct ./scripts/run_vllm_base.sh
```

### Start vLLM Server (with LoRA)

```bash
# Start with trained LoRA adapter
MODEL_PATH=outputs/grpo/best_model ./scripts/run_vllm_lora.sh

# Interactive mode (stops when Ctrl+C)
./scripts/run_vllm_inference.sh
```

### Stop vLLM Server

```bash
./scripts/stop_vllm.sh
```

### Run Agent with vLLM Server

```bash
export VLLM_API_BASE=http://localhost:8000/v1
./run_agent.sh "Open Gmail and compose a new email"
```

### Available Scripts

| Script | Description |
|--------|-------------|
| `run_vllm_base.sh` | Start vLLM with base VLM model (daemon mode) |
| `run_vllm_lora.sh` | Start vLLM with LoRA adapter (daemon mode) |
| `run_vllm_inference.sh` | Start vLLM with LoRA (interactive mode) |
| `stop_vllm.sh` | Stop all vLLM containers |

### ModelScope Support (for China Users)

All scripts support loading models from ModelScope:

```bash
# Set MODEL_HUB environment variable
export MODEL_HUB=modelscope

# Start with ModelScope
MODEL_HUB=modelscope MODEL_NAME=Qwen/Qwen2.5-VL-32B-Instruct ./scripts/run_vllm_base.sh

# Or with LoRA
MODEL_HUB=modelscope ./scripts/run_vllm_lora.sh
```

**Environment Variables for ModelScope:**
- `MODEL_HUB=modelscope` - Use ModelScope instead of Hugging Face
- `MODELSCOPE_CACHE` - Cache directory (default: `~/.cache/modelscope`)

**Note:** When using ModelScope, the scripts will automatically install the `modelscope` Python package.

## AReaL GRPO Training

The project now supports training using the AReaL framework for Group Relative Policy Optimization (GRPO).

### Quick Start

#### 1. Install Dependencies

**使用 Docker（推荐）：**
依赖已经集成在 `Dockerfile.areal` 中，无需单独安装。Dockerfile 基于 AReaL 官方 Dockerfile，并添加了项目特定的依赖。

**本地安装（不使用 Docker）：**
如果需要本地安装，请参考 `Dockerfile.areal` 中的依赖安装步骤，主要包括：
- 安装 AReaL（包含 vllm 和 sglang 支持）
- 安装项目特定依赖（safetensors, Pillow, python-dotenv 等）
- 安装 GBox

```bash
# 安装 AReaL
uv pip install --no-build-isolation --system --no-cache-dir \
    "git+https://github.com/inclusionAI/AReaL.git@v0.5.1#egg=areal[vllm,sglang]"

# 安装项目特定依赖
uv pip install --no-build-isolation --system --no-cache-dir \
    safetensors>=0.4.5 \
    Pillow>=10.4.0 \
    python-dotenv>=1.0.1

# 安装 GBox
uv pip install --no-build-isolation --system --no-cache-dir \
    git+https://github.com/babelcloud/gbox-cua.git
```

#### 2. Configuration

Edit `configs/cua_grpo.yaml` and `.env` files.

#### 3. Start Training

```bash
# Using Docker (recommended)
./docker_train_areal.sh

# Or directly
python -m areal.launcher.local train_areal.py --config configs/cua_grpo.yaml
```

#### 4. Resume Training

```bash
python -m areal.launcher.local train_areal.py --config configs/cua_grpo.yaml --resume
```

### AReaL Training Features

- **Asynchronous Rollout Collection**: Parallel rollout collection with configurable concurrency
- **Interruptible Generation**: Stop generation early when task is completed
- **Staleness Control**: Manage data freshness during async rollouts
- **FSDP Support**: Automatic GPU allocation with FSDP
- **Static LoRA**: LoRA adapters attached to the model (no dynamic switching)
- **Built-in Checkpointing**: Automatic checkpoint saving and resuming

### Configuration

Main configuration is in `configs/cua_grpo.yaml`:

- Model settings (name, LoRA parameters)
- Training hyperparameters (learning rate, batch size, etc.)
- GRPO algorithm parameters (beta, clip_epsilon, etc.)
- Rollout configuration (num_rollouts, concurrency, etc.)
- Inference settings (vLLM backend configuration)
- Distributed training (FSDP settings)

Environment variables (in `.env`):
- `GBOX_API_KEY`: GBox API key (required)
- `HF_ENDPOINT`: HuggingFace endpoint (default: https://hf-mirror.com)
- `WANDB_API_KEY`: Weights & Biases API key (optional)

### Legacy GRPO Training

The original custom GRPO trainer is still available but deprecated. See `GRPO_TRAINING.md` for details.

## API Reference

### CUAAgent

```python
from cua_agent import CUAAgent, CUAConfig, GBoxConfig

config = CUAConfig(
    model_name="unsloth/Qwen3-VL-32B-Instruct",
    max_turns=20,
    gbox=GBoxConfig(
        api_key="your_key",
        box_type="android",
    )
)

async with CUAAgent(config) as agent:
    rubric, history = await agent.run_task(
        task_description="Open the Settings app",
        verbose=True,
    )
```

### GBoxClient

```python
from gbox_cua.gbox_client import GBoxClient
from cua_agent.config import GBoxConfig

config = GBoxConfig(api_key="your_key")

async with GBoxClient(
    api_key=config.api_key,
    model=config.model,
    box_type=config.box_type,
) as client:
    # Create box
    await client.create_box("android")
    
    # Take screenshot
    image_bytes, data_uri = await client.take_screenshot()
    
    # Execute actions
    await client.click(x=100, y=200)
    await client.type_text("Hello world")
    await client.press_button("home")
    
    # Generate coordinates
    result = await client.generate_coordinates(
        screenshot_uri=data_uri,
        action_type="click",
        target="the search button",
    )
    coords = result["response"]["coordinates"]
```

## License

MIT License

