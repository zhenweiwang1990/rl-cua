# ModelScope 使用指南

ModelScope 是阿里云推出的模型托管平台，为国内用户提供更快的模型下载速度。

## 快速开始

### 1. 使用 ModelScope 启动 vLLM

```bash
# 基础模型
MODEL_HUB=modelscope BASE_MODEL=Qwen/Qwen2.5-VL-32B-Instruct ./scripts/run_vllm_base.sh

# 带 LoRA 的模型
MODEL_HUB=modelscope ./scripts/run_vllm_lora.sh
```

### 2. 环境变量配置

在 `.env` 文件中设置：

```bash
# 使用 ModelScope
MODEL_HUB=modelscope

# ModelScope 缓存目录（可选）
MODELSCOPE_CACHE=/path/to/cache
```

## ModelScope vs Hugging Face

| 特性 | Hugging Face | ModelScope |
|------|-------------|-----------|
| **默认设置** | ✓ | |
| **国内速度** | 慢（需要镜像） | 快 |
| **自动安装** | transformers | transformers + modelscope |
| **缓存位置** | `~/.cache/huggingface` | `~/.cache/modelscope` |
| **环境变量** | `HF_ENDPOINT` | `MODELSCOPE_CACHE` |

## 模型命名对应关系

ModelScope 上的模型名称通常与 Hugging Face 保持一致：

| Hugging Face | ModelScope |
|--------------|-----------|
| `unsloth/Qwen3-VL-32B-Instruct` | `Qwen/Qwen3-VL-32B-Instruct` |
| `Qwen/Qwen2-VL-7B-Instruct` | `Qwen/Qwen2-VL-7B-Instruct` |
| `Qwen/Qwen2.5-VL-32B-Instruct` | `Qwen/Qwen2.5-VL-32B-Instruct` |

## 常见问题

### Q: 如何切换回 Hugging Face？

不设置 `MODEL_HUB` 或设置为 `huggingface`：

```bash
MODEL_HUB=huggingface ./scripts/run_vllm_base.sh
# 或者不设置（默认就是 huggingface）
./scripts/run_vllm_base.sh
```

### Q: ModelScope 和 HF Mirror 哪个更快？

- **ModelScope**: 阿里云官方服务，在国内速度最快
- **HF Mirror** (`https://hf-mirror.com`): 社区镜像，速度也不错

推荐优先使用 ModelScope。

### Q: 可以同时缓存两个平台的模型吗？

可以！两个平台使用不同的缓存目录：
- Hugging Face: `~/.cache/huggingface`
- ModelScope: `~/.cache/modelscope`

## 示例

### 示例 1: 使用 ModelScope 运行 Agent

```bash
# 1. 启动 vLLM 服务器
MODEL_HUB=modelscope BASE_MODEL=Qwen/Qwen2.5-VL-32B-Instruct ./scripts/run_vllm_base.sh

# 2. 在另一个终端运行 Agent
export VLLM_API_BASE=http://localhost:8000/v1
./run_agent.sh "打开设置应用" --verbose
```

### 示例 2: 使用 ModelScope 训练 LoRA

```bash
# 训练后使用 ModelScope 启动推理服务器
MODEL_HUB=modelscope \
MODEL_PATH=outputs/grpo/best_model \
BASE_MODEL=Qwen/Qwen2.5-VL-32B-Instruct \
./scripts/run_vllm_lora.sh
```

## 性能对比

在国内网络环境下的下载速度对比（参考值）：

| 平台 | Qwen3-VL-32B (约60GB) | 下载时间 |
|------|---------------------|---------|
| Hugging Face (原始) | 很慢，经常超时 | > 2小时 |
| HF Mirror | 中等速度 | 30-60分钟 |
| **ModelScope** | **快速** | **15-30分钟** |

## 技术细节

### 自动安装

脚本会自动安装 ModelScope SDK：

```bash
pip install --upgrade transformers>=4.50.0 modelscope -q
```

### Docker 挂载

使用 ModelScope 时，Docker 容器会挂载：

```bash
-v $MODELSCOPE_CACHE:/root/.cache/modelscope \
-e MODELSCOPE_CACHE=/root/.cache/modelscope
```

### 兼容性

- ✅ 与 vLLM 完全兼容
- ✅ 支持所有 Qwen VLM 模型
- ✅ 支持 LoRA 适配器
- ✅ 支持工具调用（Tool Calling）

## 更多信息

- [ModelScope 官网](https://modelscope.cn/)
- [ModelScope GitHub](https://github.com/modelscope/modelscope)
- [Qwen 模型库](https://modelscope.cn/organization/qwen)

