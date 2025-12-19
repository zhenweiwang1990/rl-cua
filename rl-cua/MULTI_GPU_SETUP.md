# Multi-GPU Setup Guide (8x B200)

针对 **8 卡 B200** 的优化配置指南。B200 单卡有 192GB HBM3e 内存，非常适合大模型训练和推理。

## 推荐配置方案

### 方案 A：7+1 分离

**vLLM 使用 7 卡，训练使用 1 卡**，完全并行运行：

```bash
# Terminal 1: 启动 vLLM（GPU 0-6）
GPU_DEVICES=0-6 \
TENSOR_PARALLEL_SIZE=7 \
GPU_MEMORY_UTILIZATION=0.85 \
MAX_MODEL_LEN=16384 \
./scripts/run_vllm_training.sh

# Terminal 2: 启动训练（GPU 7）
TRAIN_GPU_DEVICES=7 ./docker_train_grpo.sh
```
