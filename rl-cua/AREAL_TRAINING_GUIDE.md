# AReaL GRPO 训练启动指南

本指南将帮助你从零开始使用 AReaL 框架训练 CUA Agent。

## 📋 目录

1. [前置条件](#前置条件)
2. [环境准备](#环境准备)
3. [配置设置](#配置设置)
4. [启动训练](#启动训练)
5. [监控和日志](#监控和日志)
6. [断点续训](#断点续训)
7. [故障排查](#故障排查)

---

## 前置条件

### 硬件要求

- **GPU**: 至少 4 个 GPU（推荐 8 个或更多）
  - 训练：4+ GPU（FSDP 分布式训练）
  - 推理：4 GPU（vLLM 服务）
- **内存**: 至少 64GB 系统内存
- **存储**: 至少 500GB 可用空间（用于模型和检查点）

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **Docker**: 20.10+ (如果使用 Docker)
- **Python**: 3.10+ (如果本地运行)
- **CUDA**: 12.1+ (与 PyTorch 2.2.0 兼容)

---

## 环境准备

### Docker

使用 Docker 可以避免环境配置问题，推荐用于生产环境。

#### 1.1 安装 Docker 和 Docker Compose

```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version
```

#### 1.2 配置 NVIDIA Docker（GPU 支持）

```bash
# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证 GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
---

## 配置设置

### 步骤 1: 环境变量配置

复制环境变量模板并填写：

```bash
cp env.example .env
```

编辑 `.env` 文件，设置必要的 API 密钥：

```bash
# 必需的
GBOX_API_KEY=your_gbox_api_key_here

# 可选的（如果使用 HuggingFace 镜像）
HF_ENDPOINT=https://hf-mirror.com
HF_TOKEN=your_hf_token_if_needed

# 可选的（如果使用 Wandb）
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=cua-grpo

# vLLM 服务地址（如果使用外部服务）
VLLM_API_BASE=http://localhost:8000/v1
```

### 步骤 2: 训练配置

编辑 `configs/cua_grpo.yaml`，根据你的资源调整配置：

```yaml
# 模型配置
model:
  name: "Qwen/Qwen3-VL-32B-Instruct"  # 根据你的模型调整
  max_seq_length: 16384
  
  # LoRA 配置
  use_lora: true
  lora:
    r: 16  # 根据 GPU 内存调整（8/16/32）
    alpha: 32

# 训练配置
training:
  learning_rate: 1.0e-5
  batch_size: 4  # 根据 GPU 数量调整
  max_steps: 200
  
  # 输出目录
  output_dir: "outputs/grpo_cua"

# Rollout 配置
rollout:
  num_rollouts: 4  # 每个任务的 rollout 数量
  max_turns: 15
  concurrency: 4  # 并行 rollout 数量

# 推理配置
inference:
  backend: "vllm"
  vllm:
    api_base: "http://localhost:8000/v1"  # vLLM 服务地址
```

**重要配置说明**：

- **batch_size**: 每 GPU 的任务数，总 batch = batch_size × GPU 数量
- **num_rollouts**: GRPO 组大小，建议 4-8
- **concurrency**: 并行 rollout 数量，不要超过可用 GPU 数量
- **lora.r**: LoRA rank，越大效果越好但需要更多内存

### 步骤 3: GPU 资源规划

假设你有 8 个 GPU：

**方案 A: 训练和推理分离（推荐）**
- GPU 0-3: vLLM 推理服务（tensor-parallel-size=4）
- GPU 4-7: 训练服务（FSDP 使用 4 个 GPU）

**方案 B: 共享 GPU**
- GPU 0-7: 训练服务（FSDP 使用 8 个 GPU）
- vLLM 使用训练 GPU（需要配置 GPU 内存利用率）

---

## 启动训练

### Docker Compose

这是最简单的方式，自动管理 vLLM 和训练服务。

#### 1.1 启动完整训练流程

```bash
# 确保 .env 文件已配置
cat .env | grep GBOX_API_KEY

# 启动训练（自动启动 vLLM 和训练服务）
./docker_train_areal.sh
```

这个脚本会：
1. 构建 Docker 镜像
2. 启动 vLLM 服务并等待就绪
3. 启动训练服务
4. 训练完成后自动清理

#### 1.2 手动控制服务

```bash
# 启动 vLLM 服务
docker-compose -f docker-compose.areal.yml up -d vllm

# 等待 vLLM 就绪（检查健康状态）
docker-compose -f docker-compose.areal.yml exec vllm curl http://localhost:8000/health

# 启动训练服务
docker-compose -f docker-compose.areal.yml up trainer

# 查看日志
docker-compose -f docker-compose.areal.yml logs -f trainer

# 停止所有服务
docker-compose -f docker-compose.areal.yml down
```

#### 2.4 多节点训练（可选）

```bash
# 使用 Ray 进行多节点训练
python -m areal.launcher.ray train_areal.py --config configs/cua_grpo.yaml
```

### 方式 3: 使用现有 vLLM 服务

如果你已经有运行中的 vLLM 服务：

```bash
# 设置 vLLM 服务地址
export VLLM_API_BASE=http://your-vllm-server:8000/v1

# 启动训练
python -m areal.launcher.local train_areal.py --config configs/cua_grpo.yaml
```

记得在 `configs/cua_grpo.yaml` 中更新 `inference.vllm.api_base`。

---

## 监控和日志

### 训练日志

训练日志会保存在配置的输出目录：

```bash
# 查看训练日志
tail -f outputs/grpo_cua/logs/training.log

# 查看 Rollout 详细日志
tail -f outputs/grpo_cua/logs/rollouts.log
```

### 检查点

训练过程中会自动保存检查点：

```bash
# 查看检查点
ls -lh outputs/grpo_cua/checkpoints/

# 检查点结构
outputs/grpo_cua/
├── checkpoints/
│   ├── checkpoint-10/
│   ├── checkpoint-20/
│   └── ...
└── best_model/  # 最佳模型（如果配置了）
```

---

## 断点续训

### 从最新检查点恢复

```bash
# 自动查找最新检查点
python -m areal.launcher.local train_areal.py \
  --config configs/cua_grpo.yaml \
  --resume
```

### 从指定检查点恢复

```bash
# 指定检查点路径
python -m areal.launcher.local train_areal.py \
  --config configs/cua_grpo.yaml \
  --resume_from_checkpoint outputs/grpo_cua/checkpoints/checkpoint-50
```

### 在配置文件中设置

编辑 `configs/cua_grpo.yaml`：

```yaml
checkpoint:
  resume_from_checkpoint: "outputs/grpo_cua/checkpoints/checkpoint-50"
  ignore_data_skip: false
```

---

## 故障排查

### 问题 1: vLLM 服务无法启动

**症状**: vLLM 容器启动失败或无法访问

**解决方案**:

```bash
# 检查 GPU 是否可用
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 检查端口是否被占用
netstat -tuln | grep 8000

# 查看 vLLM 日志
docker logs vllm-cua-areal
```

### 问题 2: 训练时连接 vLLM 失败

**症状**: `ConnectionError` 或 `TimeoutError`

**解决方案**:

```bash
# 验证 vLLM 服务是否运行
curl http://localhost:8000/health

# 检查网络连接
ping localhost  # 或 vLLM 服务器 IP

# 增加超时时间（在 configs/cua_grpo.yaml 中）
inference:
  vllm:
    timeout: 300.0  # 增加到 5 分钟
```

### 问题 3: GPU 内存不足

**症状**: `CUDA out of memory`

**解决方案**:

1. **减少 batch_size**:
   ```yaml
   training:
     batch_size: 2  # 从 4 减少到 2
   ```

2. **减少 LoRA rank**:
   ```yaml
   model:
     lora:
       r: 8  # 从 16 减少到 8
   ```

3. **减少 vLLM GPU 内存使用**:
   ```yaml
   # 在 vLLM 启动参数中
   --gpu-memory-utilization 0.7  # 从 0.85 减少到 0.7
   ```

4. **减少并发数**:
   ```yaml
   rollout:
     concurrency: 2  # 从 4 减少到 2
   ```

### 问题 4: GBox API 错误

**症状**: `GBOX_API_KEY is not set` 或 API 调用失败

**解决方案**:

```bash
# 检查环境变量
echo $GBOX_API_KEY

# 在 .env 文件中设置
GBOX_API_KEY=your_actual_api_key

# 重新加载环境变量
source .env  # 或重启终端
```

### 问题 5: AReaL 导入错误

**症状**: `ImportError: cannot import name 'GRPOTrainer'`

**解决方案**:

```bash
# 检查 AReaL 是否安装
pip show areal

# 重新安装 AReaL
pip install --upgrade areal>=0.5.0

# 或从源码安装
pip install git+https://github.com/inclusionAI/AReaL.git
```

### 问题 6: 模型加载失败

**症状**: 模型下载失败或加载错误

**解决方案**:

```bash
# 使用 HuggingFace 镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct \
  --local-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct
```

---

## 快速开始示例

### 最小化测试运行

如果你想快速测试训练流程是否正常：

```bash
# 1. 修改配置文件，减少训练步数
# 编辑 configs/cua_grpo.yaml
training:
  max_steps: 2  # 只训练 2 步
  batch_size: 1
  eval_steps: 1
  save_steps: 1

rollout:
  num_rollouts: 1  # 每个任务只做 1 个 rollout
  concurrency: 1

# 2. 启动训练
./docker_train_areal.sh
```

### 完整训练示例

```bash
# 1. 配置环境
cp env.example .env
# 编辑 .env，设置 GBOX_API_KEY

# 2. 检查配置
cat configs/cua_grpo.yaml

# 3. 启动训练（Docker）
./docker_train_areal.sh

# 或本地环境
source venv/bin/activate
./scripts/run_vllm_base.sh  # 在另一个终端
python -m areal.launcher.local train_areal.py --config configs/cua_grpo.yaml
```

---

## 下一步

训练完成后，你可以：

1. **评估模型**: 使用评估数据集测试模型性能
2. **部署模型**: 将训练好的 LoRA 适配器部署到生产环境
3. **继续训练**: 从检查点继续训练，优化模型性能
4. **调整超参数**: 根据训练结果调整学习率、batch size 等

---

## 参考文档

- [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) - 完整迁移计划
- [VLLM_SETUP.md](./VLLM_SETUP.md) - vLLM 详细设置
- [DOCKERFILE_README.md](./DOCKERFILE_README.md) - Dockerfile 说明
- [MIGRATION_FIXES.md](./MIGRATION_FIXES.md) - 迁移修复说明

---

## 获取帮助

如果遇到问题：

1. 查看日志文件：`outputs/grpo_cua/logs/`
2. 检查配置文件：`configs/cua_grpo.yaml`
3. 验证环境变量：`.env`
4. 参考故障排查章节

祝训练顺利！🚀

