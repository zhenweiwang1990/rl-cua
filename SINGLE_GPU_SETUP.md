# Single GPU Setup Guide

如果你只有**一块 GPU**，需要让 vLLM 和训练共享同一块 GPU。这需要仔细管理内存。

## 快速开始

### 1. 配置环境变量

编辑 `.env` 文件，设置单 GPU 优化参数：

```bash
# vLLM 配置（降低内存占用）
GPU_MEMORY_UTILIZATION=0.45  # 只用 45% GPU 内存，留空间给训练
MAX_MODEL_LEN=8192           # 降低最大序列长度

# 训练配置（使用 4-bit 量化）
LOAD_IN_4BIT=true            # 必须启用
BATCH_SIZE=2                 # 减小 batch size
NUM_ROLLOUTS=2               # 减少每个任务的 rollouts

# 可选：使用更小的模型（如果可用）
MODEL_NAME=unsloth/Qwen3-VL-8B-Instruct  # 8B 而不是 30B/32B
```

### 2. 启动 vLLM（使用 GPU 0，低内存占用）

```bash
# 只使用 45% GPU 内存，留空间给训练
GPU_MEMORY_UTILIZATION=0.45 \
GPU_DEVICES=0 \
MAX_MODEL_LEN=8192 \
./scripts/run_vllm_training.sh
```

### 3. 启动训练（共享 GPU 0）

```bash
# 训练会使用剩余的 GPU 内存
TRAIN_GPU_DEVICES=0 ./docker_train_grpo.sh
```

## 内存优化建议

### vLLM 端优化

1. **降低 GPU 内存利用率**：
   ```bash
   GPU_MEMORY_UTILIZATION=0.4  # 只用 40%，更保守
   ```

2. **减少最大序列长度**：
   ```bash
   MAX_MODEL_LEN=4096  # 进一步降低（如果任务允许）
   ```

3. **使用更小的模型**（如果可用）：
   ```bash
   MODEL_NAME=unsloth/Qwen3-VL-8B-Instruct  # 8B 模型
   ```

### 训练端优化

1. **确保 4-bit 量化已启用**（默认已启用）：
   ```bash
   LOAD_IN_4BIT=true
   ```

2. **减小 batch size**：
   ```bash
   BATCH_SIZE=1  # 最小 batch size
   ```

3. **减少 rollouts**：
   ```bash
   NUM_ROLLOUTS=2  # 每个任务只收集 2 个 rollouts
   ```

4. **减少最大序列长度**：
   ```bash
   MAX_SEQ_LENGTH=8192  # 降低训练时的序列长度
   ```

## 故障排除

### CUDA OOM 错误

如果看到 `CUDA out of memory` 错误：

1. **进一步降低 vLLM 内存**：
   ```bash
   GPU_MEMORY_UTILIZATION=0.35  # 降到 35%
   ```

2. **使用更小的模型**：
   ```bash
   MODEL_NAME=unsloth/Qwen3-VL-8B-Instruct
   ```

3. **减少训练 batch size**：
   ```bash
   BATCH_SIZE=1
   NUM_ROLLOUTS=1  # 临时测试用
   ```

### 性能优化

单 GPU 下训练会较慢，因为 vLLM 和训练会竞争 GPU 资源。建议：

1. **监控 GPU 使用率**：
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **如果 GPU 利用率不高**，可以适当增加：
   ```bash
   GPU_MEMORY_UTILIZATION=0.5  # 增加到 50%
   ```

3. **考虑升级到多 GPU**（如果可能）：
   - 4+ GPUs 可以显著提升性能
   - vLLM 和训练可以完全并行

## 示例配置

### 最小内存配置（适合 24GB GPU）

```bash
# .env
GPU_MEMORY_UTILIZATION=0.35
MAX_MODEL_LEN=4096
MODEL_NAME=unsloth/Qwen3-VL-8B-Instruct
BATCH_SIZE=1
NUM_ROLLOUTS=2
MAX_SEQ_LENGTH=4096
LOAD_IN_4BIT=true
```

### 平衡配置（适合 40GB+ GPU）

```bash
# .env
GPU_MEMORY_UTILIZATION=0.45
MAX_MODEL_LEN=8192
MODEL_NAME=unsloth/Qwen3-VL-30B-A3B-Instruct
BATCH_SIZE=2
NUM_ROLLOUTS=3
MAX_SEQ_LENGTH=8192
LOAD_IN_4BIT=true
```

## 注意事项

1. **训练速度会较慢**：vLLM 和训练共享 GPU，无法完全并行
2. **内存管理很重要**：需要仔细平衡两者的内存占用
3. **建议监控**：使用 `nvidia-smi` 监控 GPU 使用情况
4. **考虑升级**：如果可能，多 GPU 会显著提升性能

