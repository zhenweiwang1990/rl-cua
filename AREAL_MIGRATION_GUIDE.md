# AReaL 迁移指南：Unsloth 集成与代码更改

## 一、AReaL 与 Unsloth 的关系

### 1.1 两者的定位

**Unsloth**：
- 定位：**模型加载和训练优化库**
- 功能：
  - 快速加载模型（`FastLanguageModel.from_pretrained()`）
  - LoRA 配置和管理
  - 梯度检查点优化
  - 内存优化（4-bit 量化、Flash Attention 等）
  - 训练加速（2-5x）

**AReaL**：
- 定位：**强化学习训练框架**
- 功能：
  - 异步训练架构（rollout 收集 + 模型训练解耦）
  - GRPO 等 RL 算法实现
  - 分布式训练支持（Megatron、FSDP）
  - vLLM/SGLang 推理后端集成
  - 训练流程管理

### 1.2 兼容性分析

**✅ 可以配合使用**：
- Unsloth 专注于**模型层**的优化（加载、内存、速度）
- AReaL 专注于**训练流程**的优化（异步、分布式、算法）
- 两者在不同的抽象层次上工作，理论上可以互补

**⚠️ 需要注意的问题**：
- AReaL 使用 PyTorch FSDP 或 Megatron 进行分布式训练
- Unsloth 的 `FastLanguageModel` 可能需要在 FSDP 包装之前使用
- 需要确保 Unsloth 加载的模型兼容 AReaL 的训练接口

### 1.3 推荐方案

**方案 A：继续使用 Unsloth（推荐）**
- 使用 Unsloth 加载模型和配置 LoRA
- 使用 AReaL 的训练框架进行 RL 训练
- 需要做适配层来桥接两者

**方案 B：使用标准 PEFT（更简单）**
- 直接使用 HuggingFace Transformers + PEFT
- 完全兼容 AReaL 的 FSDP 支持
- 放弃 Unsloth 的一些优化（但 AReaL 本身也有优化）

---

## 二、代码更改详细说明

### 2.1 当前代码结构分析

你的当前实现：

```python
# train_grpo_cua.py
def load_model_with_unsloth(...):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(...)
    model = FastLanguageModel.get_peft_model(model, ...)
    return model, tokenizer

# grpo_trainer.py
class CUAGRPOTrainer:
    def compute_loss_for_trajectory(...):
        outputs = self.model(input_ids=input_ids, labels=labels)
        # 手动计算 loss
```

### 2.2 迁移到 AReaL 需要做的更改

#### 更改 1：模型加载方式（可选 - 如果要继续用 Unsloth）

**当前代码**：
```python
# train_grpo_cua.py
def load_model_with_unsloth(model_name: str, max_seq_length: int, load_in_4bit: bool):
    from unsloth import FastLanguageModel
    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
        device_map="auto",
    )
```

**AReaL 方式（方案 A - 保留 Unsloth）**：
```python
# 在 AReaL 训练脚本中
from unsloth import FastLanguageModel
from areal.trainer import GRPOTrainer

# 加载模型（使用 Unsloth）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.model_name,
    max_seq_length=config.max_seq_length,
    load_in_4bit=config.load_in_4bit,
    dtype=None,
    # 注意：device_map 在 FSDP 模式下不能使用
)

# 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=config.lora.r,
    target_modules=config.lora.target_modules,
    lora_alpha=config.lora.alpha,
    lora_dropout=config.lora.dropout,
)

# 传递给 AReaL trainer（AReaL 会处理 FSDP 包装）
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    # ... 其他配置
)
```

**AReaL 方式（方案 B - 使用标准 PEFT）**：
```python
# 使用标准 HuggingFace 方式（AReaL 推荐）
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    # 不使用 device_map（FSDP 会处理）
)

lora_config = LoraConfig(
    r=config.lora.r,
    lora_alpha=config.lora.alpha,
    target_modules=config.lora.target_modules,
    lora_dropout=config.lora.dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# AReaL 会自动用 FSDP 包装
trainer = GRPOTrainer(model=model, tokenizer=tokenizer, ...)
```

#### 更改 2：训练循环替换

**当前代码**（自定义训练循环）：
```python
# grpo_trainer.py
class CUAGRPOTrainer:
    def train(self):
        for step in range(self.config.max_steps):
            # 1. 收集 rollouts（同步）
            groups = self._collect_rollouts(self.train_tasks)
            # 2. 计算 advantages
            groups, groups_kept, groups_filtered = self._compute_advantages(groups)
            # 3. 训练步骤
            loss = self._train_step(groups)
            # 4. 评估
            if step % self.config.eval_steps == 0:
                self._run_evaluation()
```

**AReaL 方式**（使用 AReaL 的训练接口）：
```python
# 使用 AReaL 的 GRPO trainer
from areal.trainer import GRPOTrainer
from areal.config import GRPOConfig
from areal.rollout import AsyncRolloutCollector

# 配置 AReaL
config = GRPOConfig(
    model_name=config.model_name,
    # ... 其他配置
    rollout=AsyncRolloutConfig(
        num_rollouts=4,
        # ... rollout 配置
    ),
    training=TrainingConfig(
        learning_rate=1e-5,
        batch_size=4,
        # ... 训练配置
    ),
)

# 创建 rollout collector（异步）
rollout_collector = AsyncRolloutCollector(
    model=model,  # 用于推理的模型（vLLM）
    env=your_gbox_env,  # 你的 GBox 环境
    config=config.rollout,
)

# 创建 trainer
trainer = GRPOTrainer(
    model=model,  # 训练用的模型
    tokenizer=tokenizer,
    config=config.training,
    rollout_collector=rollout_collector,
)

# 训练（AReaL 会自动处理异步调度）
trainer.train()
```

#### 更改 3：Rollout 收集（关键改动）

**当前代码**（同步收集）：
```python
# grpo_trainer.py
async def _collect_rollouts(self, tasks, is_eval=False):
    groups = []
    for task in tasks:
        # 逐个收集（同步）
        rollouts = []
        for i in range(self.config.rollout.num_rollouts):
            result = await self._run_single_rollout(task, i)
            rollouts.append(result)
        groups.append(TrajectoryGroup(task=task, samples=rollouts))
    return groups
```

**AReaL 方式**（异步收集）：
```python
# 使用 AReaL 的异步 rollout collector
# AReaL 会自动并行收集多个 rollouts
# 你只需要定义环境接口

from areal.env import BaseEnv

class GBoxEnv(BaseEnv):
    """适配 GBox 环境到 AReaL 接口"""
    
    async def reset(self, task):
        # 初始化任务
        self.agent = StandaloneGBoxCUAAgent(...)
        self.task = task
        return initial_state
    
    async def step(self, action):
        # 执行动作，返回 (observation, reward, done, info)
        result = await self.agent.execute_action(action)
        return result.screenshot, result.reward, result.done, result.metadata
    
    async def compute_reward(self, trajectory):
        # 计算奖励
        return calculate_reward(trajectory)

# AReaL 会自动异步收集 rollouts
# 你不需要手动循环
```

#### 更改 4：Loss 计算（可选）

**当前代码**（手动计算）：
```python
def compute_loss_for_trajectory(self, conversation, advantage):
    # 手动 tokenize
    input_ids, labels, loss_mask, advantage_mask = self.tokenize_conversation_with_mask(...)
    # 手动计算 loss
    outputs = self.model(input_ids=input_ids, labels=labels)
    # ... 复杂的 loss 计算
```

**AReaL 方式**（使用 AReaL 的 loss）：
```python
# AReaL 内置了 GRPO loss 计算
# 你只需要提供 trajectories 和 advantages
# AReaL 会自动处理 tokenization 和 loss 计算

# 但如果你想自定义，可以继承 GRPOTrainer
class CustomGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 自定义 loss 计算
        # 仍然可以使用你的 tokenize_conversation_with_mask 逻辑
        return super().compute_loss(model, inputs, return_outputs)
```

#### 更改 5：配置文件格式

**当前代码**（环境变量）：
```python
# grpo_config.py
@dataclass
class GRPOConfig:
    model_name: str = os.environ.get("MODEL_NAME", "unsloth/Qwen3-VL-32B-Instruct")
    learning_rate: float = get_env_float("LEARNING_RATE", "1e-5")
    # ...
```

**AReaL 方式**（YAML 配置文件）：
```yaml
# config.yaml（AReaL 标准格式）
model:
  name: "unsloth/Qwen3-VL-32B-Instruct"
  max_seq_length: 16384
  load_in_4bit: true

training:
  learning_rate: 1e-5
  batch_size: 4
  max_steps: 200
  # ...

rollout:
  num_rollouts: 4
  max_turns: 15
  # ...

lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**使用方式**：
```python
# AReaL 会自动加载 YAML 配置
python -m areal.launcher.local train_script.py --config config.yaml
```

---

## 三、具体迁移步骤

### Step 1：安装 AReaL

```bash
# 安装 AReaL
pip install areal

# 或从源码安装
git clone https://github.com/inclusionAI/AReaL
cd AReaL
pip install -e ".[dev]"
```

### Step 2：创建 AReaL 配置文件

```bash
# 创建 config.yaml
cp examples/math/gsm8k_grpo.yaml configs/cua_grpo.yaml
# 然后修改配置以匹配你的需求
```

### Step 3：适配环境接口

创建 `cua_agent/areal_env.py`：

```python
"""GBox 环境适配到 AReaL 接口"""
from areal.env import BaseEnv
from cua_agent.agent import StandaloneGBoxCUAAgent
from cua_agent.tasks import CUATask

class GBoxAReaLEnv(BaseEnv):
    def __init__(self, config):
        self.config = config
        self.agent = None
        self.task = None
    
    async def reset(self, task: CUATask):
        """初始化任务"""
        self.task = task
        self.agent = StandaloneGBoxCUAAgent(config)
        # 重置环境状态
        return {"task_description": task.description}
    
    async def step(self, action):
        """执行动作"""
        result = await self.agent.execute_action(action)
        return {
            "observation": result.screenshot,
            "reward": result.reward,
            "done": result.done,
            "info": result.metadata,
        }
    
    async def compute_reward(self, trajectory):
        """计算最终奖励"""
        return calculate_reward(trajectory, self.task)
```

### Step 4：修改训练脚本

创建 `train_grpo_areal.py`：

```python
"""使用 AReaL 的 GRPO 训练脚本"""
import argparse
from pathlib import Path

from areal.trainer import GRPOTrainer
from areal.config import GRPOConfig
from areal.rollout import AsyncRolloutCollector
from cua_agent.areal_env import GBoxAReaLEnv
from cua_agent.tasks import get_training_tasks, get_eval_tasks

# 方案 A：继续使用 Unsloth 加载模型
def load_model_with_unsloth(model_name, max_seq_length, load_in_4bit):
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
        # 注意：在 FSDP 模式下不使用 device_map
    )
    
    # 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
    )
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    # 加载 AReaL 配置
    config = GRPOConfig.from_yaml(args.config)
    
    # 加载模型（使用 Unsloth）
    model, tokenizer = load_model_with_unsloth(
        model_name=config.model.name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )
    
    # 创建环境
    env = GBoxAReaLEnv(config.rollout)
    
    # 创建 rollout collector
    # AReaL 会自动使用 vLLM 进行推理
    rollout_collector = AsyncRolloutCollector(
        env=env,
        config=config.rollout,
        inference_backend="vllm",  # 使用 vLLM
        vllm_config={
            "api_base": config.vllm.api_base,
            "model_name": config.model.name,
        },
    )
    
    # 创建 trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config.training,
        rollout_collector=rollout_collector,
    )
    
    # 加载任务
    train_tasks = get_training_tasks()
    eval_tasks = get_eval_tasks()
    
    # 训练
    trainer.train(train_tasks=train_tasks, eval_tasks=eval_tasks)

if __name__ == "__main__":
    main()
```

### Step 5：逐步迁移（推荐）

**阶段 1：保持现有代码，添加 AReaL 接口**
- 保留 `CUAGRPOTrainer` 作为备份
- 创建新的 `AReaLGRPOTrainer` 包装类
- 使用 AReaL 的异步 collector，但保留自己的训练逻辑

**阶段 2：部分迁移**
- 使用 AReaL 的 rollout collector
- 保留自己的 loss 计算
- 测试性能和正确性

**阶段 3：完全迁移**
- 使用 AReaL 的完整训练流程
- 移除自定义 trainer
- 优化配置和参数

---

## 四、Unsloth 的保留与替换

### 4.1 可以保留的 Unsloth 功能

✅ **模型加载优化**：
- Unsloth 的 `FastLanguageModel.from_pretrained()` 仍然可以使用
- 在传递给 AReaL trainer 之前使用 Unsloth 加载模型

✅ **LoRA 配置**：
- 使用 Unsloth 的 `get_peft_model()` 配置 LoRA
- 生成的 PEFT 模型与 AReaL 兼容

✅ **梯度检查点**：
- Unsloth 的梯度检查点优化仍然有效
- 或者使用 AReaL 的内置优化

### 4.2 需要替换的部分

❌ **训练循环**：
- 不再需要手动的训练循环
- AReaL 提供完整的训练管理

❌ **Rollout 收集逻辑**：
- 不再需要手动的异步循环
- AReaL 的 `AsyncRolloutCollector` 自动处理

❌ **Loss 计算**（可选）：
- 可以使用 AReaL 的内置 GRPO loss
- 或者保留自定义实现

### 4.3 推荐方案总结

**最佳实践**：
1. **继续使用 Unsloth 加载模型**（享受优化）
2. **使用 AReaL 的训练框架**（享受异步和分布式）
3. **保留自定义环境接口**（适配 GBox）

**代码示例**：
```python
# 1. 用 Unsloth 加载模型
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, ...)

# 2. 用 AReaL 训练
from areal.trainer import GRPOTrainer
trainer = GRPOTrainer(model=model, tokenizer=tokenizer, ...)
trainer.train()  # AReaL 处理所有训练逻辑
```

---

## 五、关键注意事项

### 5.1 FSDP 兼容性

⚠️ **重要**：如果使用 AReaL 的 FSDP 后端：

```python
# ❌ 错误：不能在 FSDP 模式下使用 device_map
model = FastLanguageModel.from_pretrained(
    ...,
    device_map="auto",  # 这会与 FSDP 冲突
)

# ✅ 正确：让 AReaL/FSDP 处理设备分配
model = FastLanguageModel.from_pretrained(
    ...,
    # 不使用 device_map
)
# AReaL 会用 FSDP 包装模型并处理设备分配
```

### 5.2 模型保存和加载

```python
# Unsloth 的保存方式仍然可用
model.save_pretrained("outputs/model")

# AReaL 的 checkpoint 会额外保存训练状态
# 使用 AReaL 的 resume 功能来恢复训练
```

### 5.3 动态 LoRA 切换

你的当前实现：
```python
# 手动同步 LoRA 到 vLLM
self._reload_vllm_lora_adapter(adapter_path)
```

AReaL 方式：
```python
# AReaL 可能会自动处理（取决于配置）
# 或者你仍然可以保留这个逻辑
```

---

## 六、迁移检查清单

- [ ] 安装 AReaL
- [ ] 创建 AReaL 配置文件（YAML）
- [ ] 适配环境接口（`GBoxAReaLEnv`）
- [ ] 修改模型加载代码（处理 FSDP 兼容性）
- [ ] 替换训练循环（使用 `GRPOTrainer`）
- [ ] 替换 rollout 收集（使用 `AsyncRolloutCollector`）
- [ ] 测试基本功能（单个任务）
- [ ] 测试异步性能（多个任务并行）
- [ ] 测试分布式训练（多 GPU）
- [ ] 验证模型性能（对比迁移前后）
- [ ] 优化配置参数
- [ ] 更新文档

---

## 七、参考资源

- [AReaL 文档](https://inclusionai.github.io/AReaL/)
- [AReaL 示例代码](https://github.com/inclusionAI/AReaL/tree/main/examples)
- [AReaL GRPO 示例](https://github.com/inclusionAI/AReaL/blob/main/examples/math/gsm8k_grpo.yaml)
- [Unsloth 文档](https://github.com/unslothai/unsloth)

---

**总结**：Unsloth 和 AReaL 可以配合使用。Unsloth 负责模型层优化，AReaL 负责训练流程优化。主要更改是使用 AReaL 的训练接口，而不是完全替换 Unsloth。

