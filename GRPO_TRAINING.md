# CUA Agent GRPO Training Guide

This guide explains how to train the CUA (Computer Use Agent) using GRPO (Group Relative Policy Optimization) with the AReaL-inspired framework.

## Overview

The GRPO training system enables training a vision-language model to control an Android device through multi-step interactions. Key features:

- **Multi-GPU vLLM Inference**: Uses vLLM for efficient parallel rollout collection
- **Dynamic LoRA Switching**: Updates LoRA adapters during training without restarting vLLM
- **Checkpoint & Resume**: Saves checkpoints with full training state for resumption
- **Detailed Logging**: Integrates with Weights & Biases for experiment tracking

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Sample batch of tasks                                    │
│  2. For each task, collect N rollouts:                       │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  vLLM Server (Multi-GPU)                            │ │
│     │  ├── Base Model: Qwen3-VL-32B                       │ │
│     │  └── LoRA Adapter: cua_agent_lora                   │ │
│     │                                                      │ │
│     │  GBox Android Box                                    │ │
│     │  ├── Take screenshot                                 │ │
│     │  ├── Execute action                                  │ │
│     │  └── Validate completion                             │ │
│     └─────────────────────────────────────────────────────┘ │
│  3. Compute GRPO advantages (group-relative)                 │
│  4. Train LoRA with token-level masking                     │
│  5. Update vLLM LoRA adapter (if enabled)                   │
│  6. Evaluate and checkpoint                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Python Environment** (Python 3.10+)
```bash
pip install -r requirements.txt

# Install Unsloth for efficient LoRA training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

2. **GBox API Key**
Get your API key from https://gbox.ai and set it in `.env`:
```bash
GBOX_API_KEY=your_gbox_api_key
```

3. **GPU Requirements**
- **Single GPU Setup** (24GB+ VRAM recommended):
  - vLLM and training share the same GPU
  - Requires careful memory management (see `SINGLE_GPU_SETUP.md`)
- **Multi-GPU Setup** (Recommended for production):
  - **8-GPU Setup (e.g., B200 x8)** - See `MULTI_GPU_SETUP.md` for details:
    - Option A: vLLM uses 7 GPUs (0-6), Training uses 1 GPUs (7) - **Recommended**
    - Option B: vLLM uses 2 GPUs (0-1), Training uses 6 GPUs (2-7)
    - Option C: vLLM uses 1 GPU (0), Training uses 7 GPUs (1-7)
  - **Important**: vLLM and training should use **separate GPUs** to avoid OOM errors

## Quick Start

### 1. Start vLLM Server for Rollouts

**For 8-GPU setup (e.g., B200 x8):**

```bash
# Option A: vLLM uses 4 GPUs (0-3), Training uses 4 GPUs (4-7)
# Recommended for large models (30B+)
GPU_DEVICES=0-3 \
TENSOR_PARALLEL_SIZE=4 \
GPU_MEMORY_UTILIZATION=0.85 \
./scripts/run_vllm_training.sh

# Option B: vLLM uses 2 GPUs (0-1), Training uses 6 GPUs (2-7)
# Good balance for medium models
GPU_DEVICES=0-1 \
TENSOR_PARALLEL_SIZE=2 \
GPU_MEMORY_UTILIZATION=0.85 \
./scripts/run_vllm_training.sh

# With initial LoRA adapter
LORA_PATH=outputs/grpo_cua/best_model \
GPU_DEVICES=0-3 \
TENSOR_PARALLEL_SIZE=4 \
./scripts/run_vllm_training.sh
```

### 2. Start Training (Docker - Recommended)

**Important**: Use different GPUs than vLLM to avoid OOM errors.

**For 8-GPU setup:**

```bash
# Option A: Training uses GPUs 4-7 (separate from vLLM's 0-3)
# Best for parallel execution
TRAIN_GPU_DEVICES=4-7 ./docker_train_grpo.sh

# Option B: Training uses GPUs 2-7 (if vLLM uses 0-1)
TRAIN_GPU_DEVICES=2-7 ./docker_train_grpo.sh

# Option C: Single GPU training (if using 4-bit quantization)
TRAIN_GPU_DEVICES=4 ./docker_train_grpo.sh

# Resume from checkpoint
TRAIN_GPU_DEVICES=4-7 ./docker_train_grpo.sh --resume

# Resume from best checkpoint
TRAIN_GPU_DEVICES=4-7 ./docker_train_grpo.sh --resume_best
```

### 2b. Start Training (Local - Alternative)

If running locally without Docker:

```bash
# Set CUDA_VISIBLE_DEVICES to use different GPUs than vLLM
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_grpo_cua.py

# With custom configuration
CUDA_VISIBLE_DEVICES=4 BATCH_SIZE=8 MAX_STEPS=500 python train_grpo_cua.py

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=4 python train_grpo_cua.py --resume
```

### Single GPU Configuration

If you only have **one GPU**, you need to carefully manage memory between vLLM and training:

**Step 1: Start vLLM with low memory utilization**

```bash
# Use only 40-50% of GPU memory for vLLM (leave room for training)
GPU_MEMORY_UTILIZATION=0.45 \
GPU_DEVICES=0 \
MAX_MODEL_LEN=8192 \
./scripts/run_vllm_training.sh
```

**Step 2: Start training (shares GPU 0 with vLLM)**

```bash
# Training will use the remaining GPU memory
# Make sure LOAD_IN_4BIT=true in .env (already default)
TRAIN_GPU_DEVICES=0 ./docker_train_grpo.sh
```

**Additional optimizations for single GPU:**

1. **Use a smaller model** (if available):
   ```bash
   MODEL_NAME=unsloth/Qwen3-VL-8B-Instruct  # Instead of 30B/32B
   ```

2. **Reduce vLLM memory** in `.env`:
   ```bash
   # Lower memory utilization
   GPU_MEMORY_UTILIZATION=0.4  # Use only 40% of GPU
   MAX_MODEL_LEN=8192          # Reduce max sequence length
   ```

3. **Reduce training batch size**:
   ```bash
   BATCH_SIZE=2                # Smaller batches
   NUM_ROLLOUTS=2              # Fewer rollouts per task
   ```

4. **Enable gradient checkpointing** (already enabled via Unsloth)

**Note**: With single GPU, training will be slower as vLLM and training compete for GPU resources. Consider:
- Running vLLM and training at different times (not recommended for GRPO)
- Using a smaller model (8B instead of 30B/32B)
- Upgrading to multiple GPUs for better performance

## Configuration

All configuration is done via environment variables. Copy `env.example` to `.env` and modify:

```bash
cp env.example .env
# Edit .env with your settings
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `unsloth/Qwen3-VL-32B-Instruct` | Base VLM model |
| `LEARNING_RATE` | `1e-5` | Learning rate |
| `BATCH_SIZE` | `4` | Tasks per batch |
| `NUM_ROLLOUTS` | `4` | Rollouts per task (GRPO group) |
| `MAX_STEPS` | `200` | Total training steps |
| `TARGET_ACCURACY` | `0.80` | Target success rate |
| `LORA_R` | `16` | LoRA rank |

See `env.example` for full configuration options.

## Training Tasks

The training uses 10 sample Android tasks:

### Training Tasks (8)
1. **Open Settings** - Launch Settings app
2. **Enable WiFi** - Turn on WiFi
3. **Maximum Brightness** - Set brightness to max
4. **Open Chrome** - Launch Chrome browser
5. **Enable Airplane Mode** - Turn on airplane mode
6. **Check Battery** - Navigate to battery info
7. **Go Home** - Return to home screen
8. **Scroll Settings** - Scroll to find "About phone"

### Evaluation Tasks (2)
1. **Display Timeout** - Change screen timeout
2. **Do Not Disturb** - Enable DND mode

### Adding Custom Tasks

Edit `cua_agent/tasks.py` to add new tasks:

```python
CUATask(
    id="custom_task_01",
    name="My Custom Task",
    description="Description of what to do",
    difficulty=TaskDifficulty.MEDIUM,
    category=TaskCategory.SETTINGS,
    max_steps=10,
    validation_type="state",
    validation_query="some_state",
    expected_result="expected_value",
)
```

## Reward Function

The default reward function is simple binary:
- **1.0**: Task completed successfully
- **0.1**: Task marked complete but failed
- **0.0**: Task not completed (timeout)

Modify `cua_agent/reward.py` for custom reward shaping.

## GRPO Algorithm

GRPO (Group Relative Policy Optimization) computes advantages relative to the group:

```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

This ensures:
- High variance groups have meaningful gradients
- Low variance groups are filtered out (`min_group_std=0.05`)
- Only assistant turns contribute to the loss

## Dynamic LoRA Switching

When `ENABLE_DYNAMIC_LORA=true`:
1. After every `LORA_UPDATE_STEPS` training steps
2. The current LoRA weights are saved to a temp directory
3. vLLM server is notified to reload the adapter
4. Subsequent rollouts use the updated policy

This enables on-policy training without restarting the server.

## Checkpoints

Checkpoints are saved to `OUTPUT_DIR/checkpoints/`:

```
outputs/grpo_cua/
├── checkpoints/
│   ├── checkpoint-10/
│   │   ├── adapter_model.safetensors
│   │   ├── adapter_config.json
│   │   ├── training_state.json
│   │   └── training_metadata.json
│   └── checkpoint-20/
├── best_model/
│   └── ... (copy of best checkpoint)
├── final/
│   └── ... (final model after training)
└── logs/
    └── ... (detailed logs if enabled)
```

### Resume Training

```bash
# Auto-resume from best-marked checkpoint (if any) or latest checkpoint
python train_grpo_cua.py

# Resume from specific checkpoint
python train_grpo_cua.py --resume_from_checkpoint outputs/grpo_cua/checkpoints/checkpoint-50

# Resume from best checkpoint
python train_grpo_cua.py --resume_best
```

Behind the scenes, the training script now uses an **auto-resume strategy similar to `rl-unsloth`**:

- If you run `python train_grpo_cua.py` **without** any resume flags:
  - It first looks for a checkpoint whose `training_state.json` contains a non-empty `best_model_path`
    (i.e., a checkpoint that was marked as best during training), and resumes from the **latest such checkpoint**.
  - If no checkpoint has a best marker, it falls back to the **latest checkpoint** in `checkpoints/`.
  - If no checkpoints exist, it clearly prints that it is **starting from scratch**.

When resuming, the console output will explicitly indicate whether it is:

- Resuming from a **specified** checkpoint
- Resuming from the **best** checkpoint (by accuracy)
- Auto-resuming from a checkpoint **with a best marker**
- Auto-resuming from the **latest** checkpoint

## Monitoring

### Weights & Biases

Set `ENABLE_WANDB=true` and your W&B credentials:

```bash
wandb login
WANDB_PROJECT=cua-grpo python train_grpo_cua.py
```

Tracked metrics:
- `train/loss`, `train/policy_loss`
- `train/avg_reward`, `train/accuracy`
- `train/rollout_time`, `train/training_time`
- `eval/accuracy`, `eval/avg_reward`

### Console Output

```
Step 10/200 | Loss: 0.2345 | Accuracy: 25.00% | Reward: 0.250 | Rollout: 45.3s | Train: 2.1s

📊 Eval Step 10: Accuracy=30.00%, Reward=0.300

🎯 New best accuracy: 30.00%
```

## Troubleshooting

### Out of Memory

1. Reduce `MAX_SEQ_LENGTH` (default: 16384)
2. Enable `LOAD_IN_4BIT=true`
3. Reduce `BATCH_SIZE`
4. Use fewer `NUM_ROLLOUTS`

### vLLM Connection Failed

1. Check vLLM server is running: `docker logs vllm-cua-training`
2. Verify API endpoint: `curl http://localhost:8000/v1/models`
3. Check `VLLM_API_BASE` in your `.env`

### Low Reward/Accuracy

1. Increase `NUM_ROLLOUTS` for better advantage estimation
2. Increase `MAX_TURNS` for complex tasks
3. Review task descriptions for clarity
4. Check GBox connection and box creation

## References

- [AReaL Documentation](https://inclusionai.github.io/AReaL/)
- [GRPO Algorithm (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [GBox Documentation](https://docs.gbox.ai)
- [vLLM Documentation](https://docs.vllm.ai)

