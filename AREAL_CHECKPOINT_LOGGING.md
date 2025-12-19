# AReaL 的 Checkpoint、日志和监控功能详解

## 一、功能概述

AReaL 提供了完整的训练管理功能，包括：

1. ✅ **自动保存 Checkpoint**：定期保存模型和训练状态
2. ✅ **断点续训**：从任意 checkpoint 恢复训练
3. ✅ **详细日志**：多层次的日志记录
4. ✅ **监控集成**：支持 Weights & Biases (wandb)、TensorBoard 等
5. ✅ **最佳模型跟踪**：自动保存性能最好的模型

---

## 二、Checkpoint 功能详解

### 2.1 AReaL 的 Checkpoint 机制

AReaL 的 checkpoint 系统非常完善，自动管理模型状态、优化器状态、训练进度等。

#### 保存的内容

```python
checkpoint-100/
├── model/
│   ├── adapter_model.safetensors  # LoRA 权重（如果使用 LoRA）
│   ├── adapter_config.json        # LoRA 配置
│   └── config.json                # 模型配置
├── tokenizer/
│   ├── tokenizer_config.json
│   └── ...
├── optimizer.pt                   # 优化器状态
├── scheduler.pt                   # 学习率调度器状态
├── training_state.json            # 训练状态（step, epoch, best_metric等）
├── training_metadata.json         # 训练元数据（metrics, config等）
└── model_version.txt              # 模型版本号（用于异步训练）
```

#### 自动保存策略

**AReaL 配置示例**：

```yaml
# config.yaml
training:
  save_strategy: "steps"  # 或 "epoch"
  save_steps: 100         # 每 N 步保存一次
  save_total_limit: 5     # 最多保留 N 个 checkpoint（自动清理旧的）
  
  # 最佳模型保存
  load_best_model_at_end: true
  metric_for_best_model: "eval/accuracy"
  greater_is_better: true
  
  # 输出目录
  output_dir: "outputs/grpo_cua"
  
checkpoint:
  # Checkpoint 配置
  resume_from_checkpoint: null  # 或路径，如 "outputs/grpo_cua/checkpoints/checkpoint-500"
  ignore_data_skip: false       # 是否忽略数据跳过（resume 时）
```

### 2.2 断点续训

#### 基本用法

```python
from areal.trainer import GRPOTrainer
from areal.config import GRPOConfig

# 方式 1: 通过配置文件指定
config = GRPOConfig.from_yaml("config.yaml")
# config.yaml 中设置: resume_from_checkpoint: "outputs/grpo_cua/checkpoints/checkpoint-500"

trainer = GRPOTrainer(config=config)
trainer.train()  # 自动从 checkpoint-500 恢复

# 方式 2: 通过命令行参数
python -m areal.launcher.local train_script.py \
  --config config.yaml \
  --resume_from_checkpoint outputs/grpo_cua/checkpoints/checkpoint-500

# 方式 3: 自动查找最新 checkpoint
python -m areal.launcher.local train_script.py \
  --config config.yaml \
  --resume  # 自动从最新的 checkpoint 恢复
```

#### 恢复的内容

AReaL 会完整恢复：

1. **模型参数**：LoRA 权重或完整模型权重
2. **优化器状态**：优化器的 momentum、Adam 的状态等
3. **学习率调度器状态**：当前的 learning rate
4. **训练进度**：global_step、epoch 等
5. **随机数生成器状态**：保证可复现性
6. **最佳模型标记**：如果该 checkpoint 是 best model

#### 代码示例

```python
# AReaL 内部处理（简化版）
class GRPOTrainer:
    def __init__(self, config, resume_from_checkpoint=None):
        self.config = config
        self.resume_from_checkpoint = resume_from_checkpoint or config.resume_from_checkpoint
        
        # 加载模型
        self.model = self._load_model()
        
        # 如果指定了 resume，加载 checkpoint
        if self.resume_from_checkpoint:
            self._load_checkpoint(self.resume_from_checkpoint)
        else:
            # 尝试自动查找最新 checkpoint
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                self._load_checkpoint(latest_checkpoint)
    
    def _load_checkpoint(self, checkpoint_path):
        """加载 checkpoint"""
        # 1. 加载模型权重
        if self.config.use_lora:
            # 只加载 LoRA 权重
            self.model.load_adapter(checkpoint_path / "model")
        else:
            # 加载完整模型
            self.model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
        
        # 2. 加载优化器状态
        if (checkpoint_path / "optimizer.pt").exists():
            self.optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))
        
        # 3. 加载调度器状态
        if (checkpoint_path / "scheduler.pt").exists():
            self.scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt"))
        
        # 4. 加载训练状态
        with open(checkpoint_path / "training_state.json") as f:
            state = json.load(f)
            self.global_step = state["global_step"]
            self.current_epoch = state["epoch"]
            self.best_metric = state.get("best_metric", None)
        
        logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
```

### 2.3 最佳模型跟踪

AReaL 会自动跟踪和保存最佳模型：

```python
# 配置
training:
  load_best_model_at_end: true
  metric_for_best_model: "eval/accuracy"  # 或 "eval/reward", "train/loss" 等
  greater_is_better: true  # accuracy 越大越好，loss 越小越好

# AReaL 会自动：
# 1. 在每个 eval step 比较当前 metric 和 best_metric
# 2. 如果更好，保存为 best_model/
# 3. 训练结束时，加载最佳模型
```

**保存结构**：

```
outputs/grpo_cua/
├── checkpoints/
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   ├── checkpoint-300/  # 这个是最佳模型
│   └── checkpoint-400/
├── best_model/  # 指向 checkpoint-300 的副本
│   └── ...
└── final/  # 最终模型（训练结束时的状态）
    └── ...
```

### 2.4 与你的当前实现对比

**你的当前实现**（已经很好）：

```python
# cua_agent/grpo_trainer.py
def _save_checkpoint(self, metrics, is_best=False):
    checkpoint_path = self.checkpoint_dir / f"checkpoint-{self.global_step}"
    
    # 保存模型
    self.model.save_pretrained(str(checkpoint_path))
    
    # 保存训练状态
    training_state = {
        "global_step": self.global_step,
        "best_model_path": best_path if is_best else None,
        ...
    }
    
    # 保存元数据
    metadata = {
        "accuracy": metrics.accuracy,
        "reward": metrics.avg_reward,
        ...
    }
```

**AReaL 的增强**：

```python
# AReaL 额外保存：
# 1. 优化器状态（你的实现没有）
# 2. 调度器状态（你的实现没有）
# 3. 随机数生成器状态（保证可复现）
# 4. 更完善的版本管理（用于异步训练）
# 5. 自动清理旧 checkpoint
```

**建议**：
- ✅ 你的 checkpoint 机制已经很好
- ⚠️ 可以考虑添加优化器状态保存（如果训练中断，学习率调度会丢失）
- ✅ AReaL 会自动处理这些细节

---

## 三、日志功能详解

### 3.1 AReaL 的多层次日志

AReaL 提供了非常完善的日志系统：

#### 日志级别

```python
# AReaL 的日志配置
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
  log_file: "outputs/grpo_cua/logs/training.log"
  
  # 不同组件的日志级别
  rollout_log_level: "DEBUG"  # Rollout 收集的详细日志
  training_log_level: "INFO"  # 训练的日志
  eval_log_level: "INFO"      # 评估的日志
```

#### 日志内容

**训练日志**：

```
2025-01-XX 10:00:00 - INFO - Step 100/1000 (10.0%) | elapsed 0:15:30 | ETA 2:19:30
2025-01-XX 10:00:01 - INFO - Collecting rollouts: 4 tasks × 4 rollouts
2025-01-XX 10:00:45 - INFO - Rollouts collected | time=44.2s | success_rate=0.75
2025-01-XX 10:00:46 - INFO - Computing GRPO advantages | groups_kept=4 | groups_filtered=0
2025-01-XX 10:01:00 - INFO - Training step | loss=0.234 | policy_loss=0.201 | kl_loss=0.033
2025-01-XX 10:01:01 - INFO - Eval step | accuracy=0.65 | reward=0.650 | std=0.120
2025-01-XX 10:01:02 - INFO - Checkpoint saved: checkpoint-100
```

**详细 Rollout 日志**（如果启用）：

```
2025-01-XX 10:00:10 - DEBUG - [ROLLOUT] Task train_01_open_settings | Rollout 1/4
2025-01-XX 10:00:11 - DEBUG - [ROLLOUT] Turn 1: action=click, target="Settings icon"
2025-01-XX 10:00:12 - DEBUG - [ROLLOUT] Turn 1: success=True
2025-01-XX 10:00:13 - DEBUG - [ROLLOUT] Turn 2: action=scroll, direction=down
...
2025-01-XX 10:00:45 - DEBUG - [ROLLOUT] Completed | turns=3 | reward=1.0 | success=True
```

### 3.2 与你的当前实现对比

**你的当前实现**（已经很好）：

```python
# 你的实现已经有：
# 1. 基础日志（logging）
logger.info("Step %d/%d", step, max_steps)

# 2. 详细 rollout 日志（可选）
if self.config.enable_detailed_logging:
    print(f"[ROLLOUT] Task {task.id} | Rollout {rollout_idx + 1}/{num_rollouts}")

# 3. Wandb 集成
if self.use_wandb:
    wandb.log({"train/loss": loss, "train/accuracy": accuracy})
```

**AReaL 的增强**：

```python
# AReaL 额外提供：
# 1. 结构化日志（JSON 格式，便于解析）
# 2. 日志文件自动轮转（避免文件过大）
# 3. 不同组件的独立日志级别
# 4. 更详细的异步训练日志（rollout worker、trainer worker 的状态）
# 5. 性能分析日志（GPU 利用率、内存使用等）
```

### 3.3 AReaL 的日志配置示例

```yaml
# config.yaml
logging:
  # 基础配置
  level: "INFO"
  log_to_file: true
  log_file: "outputs/grpo_cua/logs/training.log"
  
  # 详细日志
  enable_detailed_logging: true  # 启用详细 rollout 日志
  detailed_log_file: "outputs/grpo_cua/logs/rollouts.log"
  
  # 性能日志
  log_performance: true
  performance_log_file: "outputs/grpo_cua/logs/performance.log"
  
  # 组件日志级别
  rollout_log_level: "DEBUG"
  training_log_level: "INFO"
  eval_log_level: "INFO"
```

---

## 四、监控和可视化

### 4.1 Weights & Biases (wandb) 集成

AReaL 原生支持 wandb，自动记录所有训练指标。

#### 配置

```yaml
# config.yaml
monitoring:
  enable_wandb: true
  wandb_project: "cua-grpo"
  wandb_entity: "your-entity"  # 可选
  wandb_run_name: "grpo-qwen3vl-32b"  # 可选
  wandb_tags: ["grpo", "qwen3vl", "cua"]  # 可选
```

#### 自动记录的内容

AReaL 会自动记录：

1. **训练指标**：
   - `train/loss`, `train/policy_loss`, `train/kl_loss`
   - `train/accuracy`, `train/avg_reward`, `train/reward_std`
   - `train/learning_rate`, `train/epoch`, `train/step`

2. **评估指标**：
   - `eval/accuracy`, `eval/avg_reward`, `eval/reward_std`
   - `eval/best_accuracy`

3. **性能指标**：
   - `perf/rollout_time`, `perf/training_time`
   - `perf/throughput` (samples/sec)
   - `perf/gpu_utilization`, `perf/memory_usage`

4. **系统指标**：
   - `system/gpu_memory_used`, `system/gpu_memory_total`
   - `system/cpu_usage`, `system/disk_usage`

5. **配置信息**：
   - 完整的 config（hyperparameters）
   - 模型架构信息

#### 使用示例

```python
# AReaL 自动处理，你只需要配置
config = GRPOConfig.from_yaml("config.yaml")
trainer = GRPOTrainer(config=config)
trainer.train()  # wandb 自动初始化并记录

# 在 wandb 网页上可以看到：
# - 实时的训练曲线
# - 超参数对比
# - 系统资源使用情况
# - 最佳模型标记
```

### 4.2 TensorBoard 支持

AReaL 也支持 TensorBoard：

```yaml
monitoring:
  enable_tensorboard: true
  tensorboard_log_dir: "outputs/grpo_cua/tensorboard"
```

### 4.3 与你的当前实现对比

**你的当前实现**：

```python
# 你已经实现了 wandb 集成
if self.use_wandb:
    wandb.init(project=self.config.wandb_project, ...)
    wandb.log({
        "train/loss": loss,
        "train/accuracy": accuracy,
        ...
    })
```

**AReaL 的增强**：

```python
# AReaL 自动记录：
# 1. 更多指标（性能指标、系统指标）
# 2. 配置信息（自动记录完整 config）
# 3. 模型架构可视化
# 4. 超参数对比（如果运行多次实验）
# 5. 更好的图表组织（训练/评估分开）
```

---

## 五、完整示例

### 5.1 配置 AReaL 的 Checkpoint 和日志

```yaml
# config.yaml

# 训练配置
training:
  max_steps: 1000
  save_steps: 100
  eval_steps: 50
  save_total_limit: 5
  
  # 最佳模型
  load_best_model_at_end: true
  metric_for_best_model: "eval/accuracy"
  greater_is_better: true
  
  # 输出目录
  output_dir: "outputs/grpo_cua"

# Checkpoint 配置
checkpoint:
  resume_from_checkpoint: null  # 设为路径可恢复训练
  # resume_from_checkpoint: "outputs/grpo_cua/checkpoints/checkpoint-500"

# 日志配置
logging:
  level: "INFO"
  log_to_file: true
  log_file: "outputs/grpo_cua/logs/training.log"
  enable_detailed_logging: true
  detailed_log_file: "outputs/grpo_cua/logs/rollouts.log"

# 监控配置
monitoring:
  enable_wandb: true
  wandb_project: "cua-grpo"
  wandb_entity: "your-entity"
  enable_tensorboard: false
```

### 5.2 使用示例

```python
# train_with_areal.py
from areal.trainer import GRPOTrainer
from areal.config import GRPOConfig

# 加载配置
config = GRPOConfig.from_yaml("config.yaml")

# 创建 trainer（AReaL 会自动处理 checkpoint 和日志）
trainer = GRPOTrainer(config=config)

# 训练（所有功能自动启用）
trainer.train()

# 训练完成后：
# - 最佳模型保存在 outputs/grpo_cua/best_model/
# - 最终模型保存在 outputs/grpo_cua/final/
# - Checkpoints 保存在 outputs/grpo_cua/checkpoints/
# - 日志保存在 outputs/grpo_cua/logs/
# - Wandb 记录可在网页查看
```

### 5.3 恢复训练

```bash
# 方式 1: 自动从最新 checkpoint 恢复
python train_with_areal.py --resume

# 方式 2: 从指定 checkpoint 恢复
python train_with_areal.py \
  --resume_from_checkpoint outputs/grpo_cua/checkpoints/checkpoint-500

# 方式 3: 从最佳模型恢复
python train_with_areal.py \
  --resume_from_checkpoint outputs/grpo_cua/best_model
```

---

## 六、功能对比总结

| 功能 | 你的当前实现 | AReaL 提供 | 说明 |
|------|-------------|-----------|------|
| **Checkpoint 保存** | ✅ 已实现 | ✅ 原生支持 | AReaL 更完善（优化器状态等） |
| **断点续训** | ✅ 已实现 | ✅ 原生支持 | AReaL 支持更多恢复选项 |
| **最佳模型跟踪** | ✅ 已实现 | ✅ 原生支持 | 功能类似 |
| **自动清理旧 checkpoint** | ✅ 已实现 | ✅ 原生支持 | 功能类似 |
| **基础日志** | ✅ 已实现 | ✅ 原生支持 | AReaL 更结构化 |
| **详细日志** | ✅ 已实现 | ✅ 原生支持 | 功能类似 |
| **Wandb 集成** | ✅ 已实现 | ✅ 原生支持 | AReaL 记录更多指标 |
| **TensorBoard** | ❌ 未实现 | ✅ 原生支持 | AReaL 额外支持 |
| **性能日志** | ❌ 未实现 | ✅ 原生支持 | AReaL 额外支持 |
| **日志文件轮转** | ❌ 未实现 | ✅ 原生支持 | AReaL 额外支持 |

---

## 七、迁移建议

### 7.1 可以保留的功能

你的当前实现已经很好，可以保留：
- ✅ Checkpoint 保存逻辑（基本功能完整）
- ✅ 详细日志功能
- ✅ Wandb 集成

### 7.2 AReaL 带来的增强

使用 AReaL 可以获得：

1. **更完善的 checkpoint**：
   - 优化器状态保存
   - 调度器状态保存
   - 更可靠的恢复机制

2. **更丰富的监控**：
   - 性能指标自动记录
   - 系统资源监控
   - TensorBoard 支持

3. **更结构化的日志**：
   - JSON 格式日志（便于解析）
   - 日志文件自动轮转
   - 不同组件的独立日志级别

4. **更少的代码**：
   - 不需要自己实现 checkpoint 管理
   - 不需要自己实现日志配置
   - 专注于训练逻辑

### 7.3 迁移步骤

1. **保留现有代码作为备份**
2. **使用 AReaL 的 checkpoint 系统**：
   ```python
   # 之前：自己实现 _save_checkpoint, _load_checkpoint
   # 之后：使用 AReaL 的自动 checkpoint 管理
   trainer = GRPOTrainer(config=config)
   trainer.train()  # 自动处理所有 checkpoint
   ```

3. **使用 AReaL 的日志系统**：
   ```yaml
   # 配置文件中启用
   logging:
     enable_detailed_logging: true
   ```

4. **使用 AReaL 的监控系统**：
   ```yaml
   # 配置文件中启用
   monitoring:
     enable_wandb: true
   ```

---

## 八、关键要点

1. ✅ **AReaL 完全支持 checkpoint 和断点续训**
   - 自动保存模型、优化器、调度器状态
   - 支持从任意 checkpoint 恢复
   - 自动清理旧 checkpoint

2. ✅ **AReaL 提供完善的日志系统**
   - 多层次的日志记录
   - 详细的 rollout 日志
   - 结构化日志（JSON 格式）

3. ✅ **AReaL 原生支持监控工具**
   - Wandb 集成（自动记录所有指标）
   - TensorBoard 支持
   - 性能监控

4. ✅ **你的当前实现已经很好了**
   - 基本功能都有
   - AReaL 提供的是增强和完善

5. ✅ **迁移成本低**
   - AReaL 的接口与你的实现类似
   - 主要是配置文件的改变
   - 代码可以逐步迁移

---

**结论**：AReaL 完全支持你需要的所有功能，并且提供更完善的实现。你的当前代码已经很好，使用 AReaL 可以获得更多的功能和更好的可靠性。

