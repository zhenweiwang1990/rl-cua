# AReaL 多卡训练进度同步机制详解

## 一、多卡训练中的同步问题

在多卡训练中，需要解决几个关键的同步问题：

1. **梯度同步**：多个 GPU 上的梯度需要聚合
2. **训练进度同步**：所有 GPU 需要保持相同的 step/epoch
3. **模型参数同步**：所有 GPU 需要保持相同的模型参数
4. **异步训练中的同步**：Rollout workers 和 Training workers 之间的协调
5. **Checkpoint 同步**：保存和加载 checkpoint 时的同步

---

## 二、AReaL 的分布式架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                  AReaL 分布式训练架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Rollout Workers (推理 GPU)                  │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐            │   │
│  │  │GPU 0 │  │GPU 1 │  │GPU 2 │  │GPU 3 │            │   │
│  │  │vLLM  │  │vLLM  │  │vLLM  │  │vLLM  │            │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘            │   │
│  │     │         │         │         │                 │   │
│  │     └─────────┴─────────┴─────────┘                 │   │
│  │                    │                                 │   │
│  │                    ▼                                 │   │
│  │            Replay Buffer                             │   │
│  │            (共享存储)                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                    │                                         │
│                    ▼                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │      Training Workers (训练 GPU, FSDP)               │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐            │   │
│  │  │GPU 4 │  │GPU 5 │  │GPU 6 │  │GPU 7 │            │   │
│  │  │Train │  │Train │  │Train │  │Train │            │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘            │   │
│  │     │         │         │         │                 │   │
│  │     └─────────┴─────────┴─────────┘                 │   │
│  │              FSDP All-Reduce                        │   │
│  │              (梯度同步)                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                    │                                         │
│                    ▼                                         │
│         Parameter Store (分布式存储)                        │
│                    │                                         │
│                    └─────────────────────────────────────┘  │
│                            │                                 │
│                            ▼                                 │
│              Rollout Workers 加载新参数                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件

1. **Rollout Workers**：使用 vLLM 进行推理（可以多 GPU tensor parallel）
2. **Training Workers**：使用 FSDP 进行训练（数据并行）
3. **Replay Buffer**：共享的经验池（存储 rollouts）
4. **Parameter Store**：分布式参数存储（存储模型权重）

---

## 三、梯度同步（FSDP）

### 3.1 FSDP 的工作原理

FSDP (Fully Sharded Data Parallel) 是 AReaL 使用的分布式训练后端。

**FSDP 的同步机制**：

```python
# AReaL 内部处理（简化版）
from torch.distributed import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

# 1. 用 FSDP 包装模型
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
    mixed_precision=MixedPrecision(...),            # 混合精度
)

# 2. 前向传播（自动处理参数收集）
outputs = model(inputs)  # FSDP 自动收集所有 GPU 上的参数

# 3. 反向传播（自动处理梯度同步）
loss.backward()  # FSDP 自动进行 All-Reduce 同步梯度

# 4. 优化器步进（自动同步参数更新）
optimizer.step()  # FSDP 确保所有 GPU 的参数更新一致
```

### 3.2 梯度同步流程

```
训练步骤：

Step 1: Forward Pass
  GPU 4: [收集参数分片] → Forward → loss_4
  GPU 5: [收集参数分片] → Forward → loss_5
  GPU 6: [收集参数分片] → Forward → loss_6
  GPU 7: [收集参数分片] → Forward → loss_7

Step 2: Backward Pass
  GPU 4: loss_4.backward() → grad_4
  GPU 5: loss_5.backward() → grad_5
  GPU 6: loss_6.backward() → grad_6
  GPU 7: loss_7.backward() → grad_7
  
  FSDP All-Reduce:
  grad_avg = (grad_4 + grad_5 + grad_6 + grad_7) / 4
  所有 GPU 得到相同的 grad_avg

Step 3: Optimizer Step
  GPU 4: params_4 -= lr * grad_avg
  GPU 5: params_5 -= lr * grad_avg
  GPU 6: params_6 -= lr * grad_avg
  GPU 7: params_7 -= lr * grad_avg
  
  所有 GPU 的参数更新保持一致
```

### 3.3 AReaL 的 FSDP 配置

```python
# AReaL 配置 FSDP
config = {
    "training_backend": "fsdp",  # 或 "megatron"
    "fsdp": {
        "sharding_strategy": "FULL_SHARD",  # 完全分片
        "mixed_precision": "bf16",          # 混合精度
        "gradient_clipping": 1.0,           # 梯度裁剪
        "gradient_accumulation_steps": 1,   # 梯度累积
    }
}
```

**关键点**：
- FSDP 自动处理梯度同步（All-Reduce）
- 所有 GPU 的梯度会被平均
- 参数更新保持一致
- 不需要手动同步

---

## 四、训练进度同步

### 4.1 Step/Epoch 同步

在多 GPU 训练中，所有 GPU 必须保持相同的训练步数。

**AReaL 的处理方式**：

```python
# AReaL 内部（简化版）
import torch.distributed as dist

class GRPOTrainer:
    def __init__(self):
        # 只在 rank 0 上初始化进度
        if dist.get_rank() == 0:
            self.global_step = 0
        else:
            self.global_step = 0  # 其他 rank 从 0 开始
        
        # 在第一个 training step 同步
        if dist.is_initialized():
            dist.barrier()  # 所有 GPU 同步
    
    def train_step(self, batch):
        # 所有 GPU 执行相同的训练步骤
        loss = self.model(batch)
        loss.backward()
        
        # 梯度同步（FSDP 自动处理）
        # ...
        
        self.optimizer.step()
        
        # Step 同步（确保所有 GPU 步进相同）
        if dist.get_rank() == 0:
            self.global_step += 1
        
        # 广播 step 到所有 GPU
        if dist.is_initialized():
            step_tensor = torch.tensor([self.global_step], device=self.device)
            dist.broadcast(step_tensor, src=0)
            self.global_step = step_tensor.item()
        
        return loss
```

### 4.2 数据并行同步

**问题**：不同 GPU 可能处理不同的 batch，如何保证进度一致？

**AReaL 的解决方案**：

```python
# AReaL 使用 DistributedSampler
from torch.utils.data.distributed import DistributedSampler

# 每个 GPU 处理不同的数据
sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),  # 总 GPU 数
    rank=dist.get_rank(),                 # 当前 GPU rank
    shuffle=True,
)

dataloader = DataLoader(dataset, sampler=sampler, ...)

# 每个 GPU 处理 batch_size / num_gpus 的数据
# 但训练步数保持一致
for step, batch in enumerate(dataloader):
    # 所有 GPU 执行相同的 step
    loss = train_step(batch)
    
    # 每个 step，所有 GPU 同步
    if step % save_steps == 0:
        # 只在 rank 0 保存 checkpoint
        if dist.get_rank() == 0:
            save_checkpoint(step)
        dist.barrier()  # 等待 rank 0 保存完成
```

### 4.3 同步 Barrier

AReaL 在关键点使用 barrier 确保同步：

```python
# 关键同步点
dist.barrier()  # 所有 GPU 等待其他 GPU 到达这里

# 使用场景：
# 1. Checkpoint 保存前（确保所有 GPU 完成当前 step）
# 2. Checkpoint 加载后（确保所有 GPU 加载完成）
# 3. 评估开始前（确保所有 GPU 准备好）
# 4. 模型参数更新后（确保 Rollout workers 加载最新参数）
```

---

## 五、异步训练中的同步

### 5.1 Rollout Workers 和 Training Workers 的同步

这是 AReaL 的独特挑战：Rollout 和 Training 是异步的，如何保证同步？

**AReaL 的解决方案**：

#### 方案 1: 模型版本号

```python
# Parameter Store 管理模型版本
class ParameterStore:
    def __init__(self):
        self.current_version = 0
        self.lock = threading.Lock()
    
    def save_model(self, weights):
        """Training worker 保存新模型"""
        with self.lock:
            self.current_version += 1
            version = self.current_version
            
            # 保存到分布式存储
            save_to_storage(version, weights)
        
        return version
    
    def get_current_version(self):
        """获取当前模型版本"""
        with self.lock:
            return self.current_version
    
    def load_model(self, version):
        """Rollout worker 加载指定版本的模型"""
        return load_from_storage(version)
```

#### 方案 2: 可中断生成

```python
# Rollout Worker
class RolloutWorker:
    async def generate_rollout(self, task):
        # 开始生成时记录模型版本
        start_version = parameter_store.get_current_version()
        
        trajectory = []
        for turn in range(max_turns):
            # 检查是否有新版本
            current_version = parameter_store.get_current_version()
            
            if current_version > start_version:
                # 中断并加载新模型
                model = parameter_store.load_model(current_version)
                start_version = current_version
                logger.info(f"Switched to model version {current_version}")
            
            # 继续生成（使用新模型）
            action = model.generate(state)
            trajectory.append(action)
            state = await env.step(action)
        
        return trajectory, start_version  # 返回使用的模型版本
```

#### 方案 3: 数据陈旧度控制

```python
# Training Worker 训练时检查数据陈旧度
class TrainingWorker:
    def train_step(self):
        # 从 replay buffer 采样
        batch = replay_buffer.sample(
            batch_size=32,
            max_staleness=5,  # 最大允许陈旧度
            current_version=self.model_version,
        )
        
        # 只有符合陈旧度要求的数据才用于训练
        # 这保证了训练使用的数据不会太旧
```

### 5.2 进度同步总结

**Rollout Workers**：
- 使用模型版本号跟踪使用的模型
- 可以中断并切换到新模型
- 每个 rollout 标记使用的模型版本

**Training Workers**：
- 使用全局 step 跟踪训练进度
- FSDP 保证所有 GPU 的 step 同步
- 保存 checkpoint 时同步所有 GPU

**同步点**：
- Checkpoint 保存/加载
- 模型版本更新
- 数据采样（检查陈旧度）

---

## 六、Checkpoint 同步

### 6.1 保存 Checkpoint

在分布式训练中，checkpoint 保存需要特别处理。

**AReaL 的处理方式**：

```python
# AReaL 内部（简化版）
def save_checkpoint(self, step):
    # 1. 所有 GPU 同步（确保完成当前 step）
    if dist.is_initialized():
        dist.barrier()
    
    # 2. 只在 rank 0 保存（避免重复）
    if dist.get_rank() == 0:
        # 收集所有 GPU 的模型状态
        # FSDP 会自动处理参数收集
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "model_version": self.model_version,
        }
        
        # 保存到共享存储
        torch.save(state_dict, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # 3. 所有 GPU 等待保存完成
    if dist.is_initialized():
        dist.barrier()
```

### 6.2 加载 Checkpoint

```python
def load_checkpoint(self, checkpoint_path):
    # 1. 所有 GPU 同步（确保同时开始加载）
    if dist.is_initialized():
        dist.barrier()
    
    # 2. Rank 0 加载 checkpoint
    if dist.get_rank() == 0:
        state_dict = torch.load(checkpoint_path)
        
        # 加载模型状态（FSDP 会自动分发到各 GPU）
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.global_step = state_dict["global_step"]
        self.model_version = state_dict["model_version"]
    
    # 3. 广播状态到所有 GPU
    if dist.is_initialized():
        # 广播 global_step
        step_tensor = torch.tensor([self.global_step], device=self.device)
        dist.broadcast(step_tensor, src=0)
        self.global_step = step_tensor.item()
        
        # 广播 model_version
        version_tensor = torch.tensor([self.model_version], device=self.device)
        dist.broadcast(version_tensor, src=0)
        self.model_version = version_tensor.item()
        
        # FSDP 会自动同步模型参数
        dist.barrier()
    
    logger.info(f"Checkpoint loaded: step={self.global_step}, version={self.model_version}")
```

### 6.3 FSDP 的 Checkpoint 特殊处理

FSDP 需要特殊处理，因为参数是分片的：

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

def save_checkpoint(self, step):
    # FSDP 需要先收集完整状态
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
        # 现在可以获取完整的状态
        model_state_dict = self.model.state_dict()
        
        if dist.get_rank() == 0:
            # 保存完整状态
            torch.save({
                "model": model_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            }, checkpoint_path)

def load_checkpoint(self, checkpoint_path):
    # 加载完整状态
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # FSDP 会自动分发参数到各 GPU
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, load_policy):
        self.model.load_state_dict(checkpoint["model"])
```

---

## 七、实际配置示例

### 7.1 多卡训练配置

```yaml
# config.yaml
training:
  backend: "fsdp"  # 使用 FSDP
  
  # 分布式配置
  world_size: 8     # 总 GPU 数
  rank: 0           # 主 GPU（会自动设置）
  master_addr: "localhost"
  master_port: 29500
  
  # FSDP 配置
  fsdp:
    sharding_strategy: "FULL_SHARD"
    mixed_precision: "bf16"
    gradient_clipping: 1.0
    
  # 进度同步
  save_steps: 100
  eval_steps: 50
  
  # Checkpoint
  output_dir: "outputs/grpo_cua"
  resume_from_checkpoint: null
```

### 7.2 启动分布式训练

```bash
# 方式 1: 使用 torchrun（推荐）
torchrun --nproc_per_node=8 \
  --master_addr=localhost \
  --master_port=29500 \
  train_script.py \
  --config config.yaml

# 方式 2: 使用 AReaL launcher
python -m areal.launcher.local \
  train_script.py \
  --config config.yaml \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=8

# 方式 3: 使用 Ray（多节点）
python -m areal.launcher.ray \
  train_script.py \
  --config config.yaml \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=8
```

### 7.3 代码示例

```python
# train_distributed.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from areal.trainer import GRPOTrainer
from areal.config import GRPOConfig

def setup_distributed():
    """初始化分布式环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    
    # 设置当前 GPU
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def main():
    # 初始化分布式
    rank, world_size, local_rank = setup_distributed()
    
    # 加载配置
    config = GRPOConfig.from_yaml("config.yaml")
    
    # 创建 trainer（AReaL 会自动处理 FSDP）
    trainer = GRPOTrainer(config=config)
    
    # 训练（所有同步自动处理）
    trainer.train()
    
    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

---

## 八、关键要点总结

### 8.1 梯度同步

- ✅ **FSDP 自动处理**：All-Reduce 同步梯度
- ✅ **参数更新一致**：所有 GPU 使用相同的梯度更新参数
- ✅ **不需要手动同步**：FSDP 底层自动处理

### 8.2 训练进度同步

- ✅ **Step 同步**：所有 GPU 保持相同的 global_step
- ✅ **Barrier 同步**：在关键点使用 barrier 确保同步
- ✅ **分布式采样**：使用 DistributedSampler 保证数据分配

### 8.3 异步训练同步

- ✅ **模型版本号**：跟踪模型版本，支持可中断生成
- ✅ **数据陈旧度**：控制训练数据的时效性
- ✅ **参数存储**：分布式参数存储，支持版本管理

### 8.4 Checkpoint 同步

- ✅ **Rank 0 保存**：只在主 GPU 保存，避免重复
- ✅ **FSDP 特殊处理**：使用 FullStateDictConfig 收集完整状态
- ✅ **状态广播**：加载后广播状态到所有 GPU

---

## 九、与你当前实现的对比

### 9.1 你的当前实现

```python
# 你的代码是单 GPU 或简单数据并行
# 没有复杂的同步问题
for step in range(max_steps):
    rollouts = collect_rollouts()  # 单 GPU
    train_step(rollouts)          # 单 GPU
    save_checkpoint()              # 单 GPU
```

### 9.2 AReaL 的增强

**使用 AReaL 后**：

```python
# AReaL 自动处理所有同步
trainer = GRPOTrainer(config=config)  # 自动初始化 FSDP
trainer.train()  # 自动处理：
                 # - 梯度同步（FSDP）
                 # - 进度同步（DistributedSampler）
                 # - Checkpoint 同步（Rank 0 保存）
                 # - 异步协调（模型版本管理）
```

**优势**：
- ✅ 不需要手动处理同步
- ✅ 支持多 GPU、多节点训练
- ✅ 支持异步训练（Rollout 和 Training 并行）
- ✅ 更可靠的同步机制

---

## 十、常见问题

### Q1: 如果某个 GPU 慢了怎么办？

**A**: FSDP 使用 All-Reduce，所有 GPU 会等待最慢的 GPU。AReaL 的异步架构可以部分缓解这个问题（Rollout 和 Training 并行）。

### Q2: Checkpoint 保存在哪里？

**A**: 通常保存在共享存储（NFS、S3 等），所有 GPU 都可以访问。只有 Rank 0 执行保存操作。

### Q3: 如何从 checkpoint 恢复？

**A**: AReaL 会自动处理：
1. 所有 GPU 同步（barrier）
2. Rank 0 加载 checkpoint
3. FSDP 分发参数到各 GPU
4. 广播训练状态（step、version）

### Q4: Rollout workers 如何知道模型更新了？

**A**: 通过模型版本号：
1. Training worker 更新模型后，版本号 +1
2. Rollout worker 在生成过程中检查版本号
3. 如果有新版本，中断并加载新模型

---

**结论**：AReaL 提供了完善的分布式训练同步机制，自动处理梯度同步、进度同步、checkpoint 同步等所有问题。你只需要配置和使用，不需要手动处理同步细节。

