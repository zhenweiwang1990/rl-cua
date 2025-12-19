# 单 GPU (H200) 训练指南

## H200 GPU 规格

- **显存**: 141GB HBM3
- **计算能力**: 非常强大，理论上可以训练大模型
- **限制**: 只有一块 GPU，需要同时运行 vLLM 和训练

## 可行性分析

### ✅ 可以训练，但需要优化配置

H200 有 141GB 显存，足够大，但单 GPU 的限制是：
1. **vLLM 和训练共享 GPU**：需要合理分配显存
2. **无法并行处理**：rollout 需要串行执行
3. **训练速度较慢**：相比多 GPU 会慢很多

### 推荐方案

#### 方案 A: 小模型 + 共享 GPU（推荐）

使用 **Qwen3-VL-8B** 模型，vLLM 和训练共享 H200：

- **vLLM**: 使用 50% GPU 内存（~70GB）
- **训练**: 使用剩余 50% GPU 内存（~70GB）
- **优势**: 可以完整训练流程
- **劣势**: 速度较慢，需要串行处理

#### 方案 B: 大模型 + 时间分片

使用 **Qwen3-VL-32B** 模型，但需要：
- 训练时暂停 vLLM（或使用外部 vLLM 服务）
- 推理时暂停训练
- **优势**: 可以使用更大的模型
- **劣势**: 需要手动切换，流程复杂

#### 方案 C: 使用外部 vLLM 服务

如果有其他机器运行 vLLM，H200 只用于训练：
- **优势**: 充分利用 H200 进行训练
- **劣势**: 需要额外的 vLLM 服务

## 配置步骤

### 1. 使用单 GPU 配置文件

```bash
# 使用单 GPU 优化配置
cp configs/cua_grpo_single_gpu.yaml configs/cua_grpo.yaml
```

### 2. 调整配置参数

编辑 `configs/cua_grpo_single_gpu.yaml`：

**关键参数**：
```yaml
model:
  name: "Qwen/Qwen3-VL-8B-Instruct"  # 使用 8B 模型
  max_seq_length: 8192  # 减少序列长度

training:
  batch_size: 1  # 单 GPU 小 batch
  gradient_accumulation_steps: 4  # 通过累积模拟大 batch

rollout:
  num_rollouts: 2  # 减少 rollout 数量
  concurrency: 1  # 单 GPU 串行处理
  async_mode: false  # 关闭异步
```

### 3. 启动训练

#### 方式 1: Docker Compose（单 GPU 版本）

```bash
# 使用单 GPU Docker Compose 配置
docker compose -f docker-compose.areal.single-gpu.yml up
```

#### 方式 2: 手动启动（推荐，更灵活）

**终端 1: 启动 vLLM（使用部分 GPU）**

```bash
docker run -d \
  --name vllm-cua-single \
  --gpus '"device=0"' \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_ENDPOINT=https://hf-mirror.com \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --enable-lora \
  --max-lora-rank 16 \
  --trust-remote-code \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048
```

**终端 2: 启动训练（使用剩余 GPU）**

```bash
# 设置环境变量
export GBOX_API_KEY=your_gbox_api_key
export VLLM_API_BASE=http://localhost:8000/v1
export CUDA_VISIBLE_DEVICES=0

# 启动训练
python train_areal.py --config configs/cua_grpo_single_gpu.yaml
```

### 4. 监控 GPU 使用

```bash
# 实时监控 GPU 使用情况
watch -n 1 nvidia-smi
```

## 性能优化建议

### 1. GPU 内存分配

**vLLM 配置**：
- `--gpu-memory-utilization 0.5`: vLLM 使用 50% 显存（~70GB）
- `--max-num-seqs 4`: 限制并发序列数
- `--max-num-batched-tokens 2048`: 限制批处理 token 数

**训练配置**：
- 使用梯度检查点（如果支持）
- 减少 batch size
- 使用梯度累积

### 2. 模型选择

| 模型 | 显存需求 | 推荐度 | 说明 |
|------|---------|--------|------|
| Qwen3-VL-8B | ~30GB | ⭐⭐⭐⭐⭐ | 最适合单 GPU |
| Qwen3-VL-30B | ~60GB | ⭐⭐⭐ | 需要仔细调优 |
| Qwen3-VL-32B | ~65GB | ⭐⭐ | 可能内存不足 |

### 3. 训练策略

**减少计算量**：
- 减少 `num_rollouts`（2-4 个）
- 减少 `max_turns`（10-15 回合）
- 减少 `max_seq_length`（8192）

**提高效率**：
- 使用 `gradient_accumulation_steps` 模拟大 batch
- 减少评估频率
- 使用更小的 LoRA rank（r=8）

## 预期性能

### 训练速度

- **单 GPU H200**: 约 2-5 步/小时（取决于配置）
- **多 GPU (8x)**: 约 10-20 步/小时

### 内存使用

- **vLLM**: ~70GB
- **训练**: ~60GB
- **系统**: ~10GB
- **总计**: ~140GB（接近 H200 的 141GB 限制）

## 故障排查

### 问题 1: GPU 内存不足 (OOM)

**解决方案**：
1. 减少 vLLM 的 `gpu-memory-utilization`（如 0.4）
2. 减少训练的 batch size
3. 使用更小的模型（8B）
4. 减少序列长度

### 问题 2: 训练速度太慢

**解决方案**：
1. 减少 rollout 数量
2. 减少最大回合数
3. 使用更小的模型
4. 考虑使用外部 vLLM 服务

### 问题 3: vLLM 和训练冲突

**解决方案**：
1. 确保 vLLM 使用 `gpu-memory-utilization 0.5` 或更少
2. 使用 `CUDA_VISIBLE_DEVICES=0` 确保使用同一 GPU
3. 监控 GPU 使用：`watch -n 1 nvidia-smi`

## 替代方案

### 如果单 GPU 训练太慢

1. **使用云服务**: 租用多 GPU 实例进行训练
2. **分阶段训练**: 
   - 阶段 1: 在其他机器上收集 rollout 数据
   - 阶段 2: 在 H200 上训练模型
3. **使用外部 vLLM**: 如果有其他 GPU 运行 vLLM，H200 只用于训练

## 总结

✅ **可以训练**，但需要：
- 使用较小的模型（8B 推荐）
- 合理分配 GPU 内存（vLLM 50%, 训练 50%）
- 接受较慢的训练速度
- 仔细监控 GPU 使用情况

🚀 **推荐配置**：
- 模型: Qwen3-VL-8B
- vLLM GPU 利用率: 0.5
- Batch size: 1
- Gradient accumulation: 4
- Rollouts: 2
- Concurrency: 1

祝训练顺利！💪

