# vLLM 设置说明

## AReaL 训练中的 vLLM 使用

### 是否需要独立启动 vLLM？

**是的，AReaL 训练仍然需要独立启动 vLLM 服务。**

原因：
1. **AReaL 的 AsyncRolloutCollector** 需要外部推理服务来生成 rollouts
2. **训练和推理分离**：训练过程使用 FSDP 在训练 GPU 上，推理在独立的 vLLM 服务上
3. **资源优化**：可以灵活分配 GPU 资源（部分 GPU 用于推理，部分用于训练）

### 架构

```
┌─────────────────┐         ┌─────────────────┐
│   vLLM 服务     │◄────────┤  AReaL Trainer  │
│  (推理 GPU)     │  HTTP   │  (训练 GPU)     │
│  Port: 8000     │         │                 │
└─────────────────┘         └─────────────────┘
        │                           │
        │                           │
        ▼                           ▼
  生成 Rollout              训练 LoRA 模型
```

### 启动方式

#### 方式 1：使用 Docker Compose（推荐）

`docker-compose.areal.yml` 已经配置了 vLLM 和训练服务的编排：

```bash
# 自动启动 vLLM 和训练服务
./docker_train_areal.sh
```

#### 方式 2：手动启动 vLLM

```bash
# 启动 vLLM 服务
./scripts/run_vllm_base.sh

# 或使用训练模式（支持 LoRA）
./scripts/run_vllm_training.sh

# 然后启动训练
python -m areal.launcher.local train_areal.py --config configs/cua_grpo.yaml
```

#### 方式 3：使用现有 vLLM 服务

如果已有 vLLM 服务运行，只需设置环境变量：

```bash
export VLLM_API_BASE=http://your-vllm-server:8000/v1
python -m areal.launcher.local train_areal.py --config configs/cua_grpo.yaml
```

### 配置

在 `configs/cua_grpo.yaml` 中配置 vLLM：

```yaml
inference:
  backend: "vllm"
  vllm:
    api_base: "http://localhost:8000/v1"  # vLLM 服务地址
    max_tokens: 2048
    temperature: 0.7
    top_p: 0.9
    timeout: 120.0
    max_retries: 3
```

### GPU 资源分配建议

#### 单节点多 GPU

假设有 8 个 GPU：

- **vLLM 服务**：使用 4 个 GPU（tensor-parallel-size=4）
- **训练服务**：使用 4 个 GPU（FSDP 自动分配）

#### 多节点

- **节点 1**：运行 vLLM 服务（推理）
- **节点 2-N**：运行训练服务（FSDP 分布式训练）

### 注意事项

1. **LoRA 更新**：如果使用动态 LoRA，需要确保 vLLM 支持运行时 LoRA 更新
   ```bash
   # vLLM 启动时启用
   --enable-lora
   --max-lora-rank 32
   ```

2. **网络连接**：确保训练服务可以访问 vLLM 服务
   - 本地：`http://localhost:8000/v1`
   - 远程：`http://vllm-server-ip:8000/v1`

3. **健康检查**：vLLM 服务启动后，可以通过以下方式检查：
   ```bash
   curl http://localhost:8000/health
   ```

### 故障排查

1. **连接失败**：检查 vLLM 服务是否运行
   ```bash
   docker ps | grep vllm
   # 或
   ps aux | grep vllm
   ```

2. **超时**：增加 `timeout` 配置，或检查网络延迟

3. **内存不足**：减少 vLLM 的 `gpu-memory-utilization` 或 `tensor-parallel-size`

