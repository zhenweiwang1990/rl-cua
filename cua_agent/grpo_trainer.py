"""GRPO Trainer for CUA Agent.

This module implements a custom GRPO trainer for training CUA agents
using reinforcement learning with vLLM-based rollout collection.

Based on AReaL framework patterns and rl-people-search implementation.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from cua_agent.grpo_config import GRPOConfig, get_env_int, get_env_float
from cua_agent.tasks import (
    CUATask,
    get_training_tasks,
    get_eval_tasks,
    create_task_system_prompt,
)
from cua_agent.reward import (
    CUARolloutResult,
    simple_reward_function,
    calculate_grpo_advantages,
    RewardTracker,
)
from gbox_cua.agent import StandaloneGBoxCUAAgent

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySample:
    """Single trajectory collected from a rollout."""
    task_id: str
    task: CUATask
    conversation: List[Dict]
    reward: float
    metadata: Dict[str, Any]
    rollout_idx: int
    group_id: int
    advantage: Optional[float] = None


@dataclass
class TrajectoryGroup:
    """Grouped trajectories for GRPO (one per task)."""
    task: CUATask
    group_id: int
    samples: List[TrajectorySample] = field(default_factory=list)


@dataclass
class TrainingMetrics:
    """Metrics for a training step."""
    loss: float
    policy_loss: float
    kl_loss: float
    avg_reward: float
    max_reward: float
    min_reward: float
    accuracy: float  # Success rate
    num_trainable_tokens: int = 0
    num_total_tokens: int = 0
    rollout_time: float = 0.0
    training_time: float = 0.0
    reward_std: float = 0.0
    groups_kept: int = 0
    groups_filtered: int = 0


class CUAGRPOTrainer:
    """Custom GRPO Trainer for CUA Agent with vLLM-based rollouts.
    
    This trainer:
    1. Collects rollouts using vLLM for inference on multiple GPUs
    2. Computes GRPO advantages within task groups
    3. Trains LoRA adapters with token-level masking
    4. Supports dynamic LoRA switching during training
    5. Saves checkpoints with resume capability
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: GRPOConfig,
        train_tasks: Optional[List[CUATask]] = None,
        eval_tasks: Optional[List[CUATask]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """Initialize CUA GRPO trainer.
        
        Args:
            model: The model to train (with LoRA applied)
            tokenizer: Tokenizer for the model
            config: GRPO configuration
            train_tasks: Training tasks (uses default if None)
            eval_tasks: Evaluation tasks (uses default if None)
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Load default tasks if not provided
        self.train_tasks = train_tasks or get_training_tasks()
        self.eval_tasks = eval_tasks or get_eval_tasks()
        
        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize standalone GBox CUA agent (from gbox-cua)
        gbox_api_key = config.rollout.gbox_api_key or os.environ.get("GBOX_API_KEY", "")
        if not gbox_api_key:
            logger.warning(
                "GBOX_API_KEY is not set. Rollouts will fail until a valid API key is provided."
            )

        # Use vLLM as backend; model_name should match the LoRA name that vLLM
        # exposes via `--enable-lora --lora-modules {lora_name}=/workspace/lora_adapter`.
        # This allows dynamic rollout: trainer syncs latest LoRA weights into the
        # shared adapter directory (LORA_PATH), and vLLM always serves `lora_name`.
        self.lora_path: Optional[Path] = None
        if self.config.vllm.enable_dynamic_lora and self.config.vllm.lora_path:
            # Interpret LORA_PATH as a path inside the training container
            self.lora_path = Path(self.config.vllm.lora_path)
            self.lora_path.mkdir(parents=True, exist_ok=True)
        elif self.config.vllm.enable_dynamic_lora:
            logger.warning(
                "ENABLE_DYNAMIC_LORA is True but LORA_PATH is not set. "
                "Dynamic LoRA rollout will be disabled."
            )

        self.standalone_agent = StandaloneGBoxCUAAgent(
            gbox_api_key=gbox_api_key,
            vlm_provider="vllm",
            vlm_api_base=config.vllm.api_base,
            vlm_api_key=config.vllm.api_key,
            model_name=config.vllm.lora_name or config.vllm.base_model,
            max_turns=config.rollout.max_turns,
            max_tokens=config.vllm.max_tokens,
            temperature=config.vllm.temperature,
            top_p=config.vllm.top_p,
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Reference model state for KL divergence
        self.ref_model_state = None
        if config.beta > 0:
            logger.info("Saving reference model state for KL divergence...")
            self.ref_model_state = {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters()
            }
        
        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.evals_without_improvement = 0
        self.training_start_time = None
        
        # Reward tracker
        self.train_reward_tracker = RewardTracker()
        self.eval_reward_tracker = RewardTracker()
        
        # Wandb
        self.use_wandb = False
        
        # Pad token
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self._load_checkpoint(Path(resume_from_checkpoint))
        
        # Log configuration
        self._log_config()
    
    def _log_config(self):
        """Log training configuration."""
        logger.info("=" * 60)
        logger.info("CUA GRPO Trainer initialized")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Train tasks: {len(self.train_tasks)}")
        logger.info(f"Eval tasks: {len(self.eval_tasks)}")
        logger.info(f"Rollouts per task: {self.config.rollout.num_rollouts}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"KL penalty (beta): {self.config.beta}")
        logger.info(f"Max steps: {self.config.max_steps}")
        logger.info(f"Target accuracy: {self.config.target_accuracy * 100:.1f}%")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info("=" * 60)

    def _reload_vllm_lora_adapter(self, adapter_path: Path) -> None:
        """Notify vLLM server to load/reload the latest LoRA adapter.

        Uses vLLM's runtime LoRA updating API:

            POST {VLLM_API_BASE}/v1/load_lora_adapter
            {
              "lora_name": "...",
              "lora_path": "/path/to/adapter"
            }

        Requirements on the vLLM side:
            - VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
            - The adapter_path must be visible inside the vLLM container.
        """
        if not self.config.vllm.enable_dynamic_lora:
            return

        api_base = (self.config.vllm.api_base or "").rstrip("/")
        if not api_base:
            logger.warning(
                "Dynamic LoRA is enabled but VLLM API base is empty. "
                "Skip notifying vLLM to reload LoRA."
            )
            return

        lora_name = self.config.vllm.lora_name or self.config.vllm.base_model
        if not lora_name:
            logger.warning(
                "Dynamic LoRA is enabled but lora_name/base_model is empty. "
                "Skip notifying vLLM to reload LoRA."
            )
            return

        import json as _json
        import urllib.request as _urllib_request

        url = f"{api_base}/v1/load_lora_adapter"
        payload = {
            "lora_name": lora_name,
            "lora_path": str(adapter_path),
        }
        data = _json.dumps(payload).encode("utf-8")

        req = _urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with _urllib_request.urlopen(req, timeout=10) as resp:
                status = resp.getcode()
                if 200 <= status < 300:
                    logger.info(
                        "Requested vLLM to load LoRA adapter '%s' from '%s' "
                        "(status=%s)",
                        lora_name,
                        adapter_path,
                        status,
                    )
                else:
                    logger.warning(
                        "vLLM load_lora_adapter returned non-2xx status: %s",
                        status,
                    )
        except Exception as e:
            logger.error(
                "Failed to notify vLLM to reload LoRA adapter '%s' from '%s': %s",
                lora_name,
                adapter_path,
                e,
            )
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load training state from checkpoint."""
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.best_accuracy = state.get("best_accuracy", 0.0)
            self.best_model_path = state.get("best_model_path")
            self.evals_without_improvement = state.get("evals_without_improvement", 0)
            
            logger.info(
                f"Loaded training state: step={self.global_step}, "
                f"best_acc={self.best_accuracy:.2%}"
            )
        
        # Load LoRA weights
        adapter_file = checkpoint_path / "adapter_model.safetensors"
        if adapter_file.exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            adapter_weights = load_file(str(adapter_file))
            set_peft_model_state_dict(self.model, adapter_weights)
            logger.info(f"Loaded LoRA weights from {checkpoint_path}")
    
    def _save_checkpoint(self, metrics: TrainingMetrics, is_best: bool = False):
        """Save model checkpoint and training state."""
        checkpoint_name = f"checkpoint-{self.global_step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "evals_without_improvement": self.evals_without_improvement,
            "metrics": {
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "avg_reward": metrics.avg_reward,
                "reward_std": metrics.reward_std,
            },
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save metadata
        metadata = {
            "step": self.global_step,
            "accuracy": metrics.accuracy,
            "avg_reward": metrics.avg_reward,
            "is_best": is_best,
        }
        with open(checkpoint_path / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # If dynamic LoRA rollout is enabled, sync latest adapter weights to the
        # shared LoRA directory (LORA_PATH). vLLM should mount this path as its
        # LoRA adapter directory (e.g., /workspace/lora_adapter).
        if self.config.vllm.enable_dynamic_lora and self.lora_path is not None:
            try:
                adapter_src = checkpoint_path / "adapter_model.safetensors"
                if not adapter_src.exists():
                    adapter_src = checkpoint_path / "adapter_model.bin"
                if adapter_src.exists():
                    self.lora_path.mkdir(parents=True, exist_ok=True)
                    import shutil as _shutil
                    adapter_dst = self.lora_path / adapter_src.name
                    _shutil.copy2(adapter_src, adapter_dst)
                    logger.info(
                        f"Synced LoRA adapter to shared path: {adapter_dst} "
                        f"(step={self.global_step})"
                    )
                    # Notify vLLM to reload / load the latest adapter.
                    self._reload_vllm_lora_adapter(adapter_dst)
                else:
                    logger.warning(
                        f"No adapter_model.[safetensors|bin] found in {checkpoint_path}; "
                        "dynamic LoRA sync skipped."
                    )
            except Exception as e:
                logger.error(f"Failed to sync LoRA adapter to {self.lora_path}: {e}")
        
        # If best model, also save to best_model directory
        if is_best:
            best_model_path = self.output_dir / "best_model"
            if best_model_path.exists():
                shutil.rmtree(best_model_path)
            shutil.copytree(checkpoint_path, best_model_path)
            self.best_model_path = best_model_path
            logger.info(f"Best model saved: {best_model_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                try:
                    step = int(path.name.split("-")[1])
                    checkpoints.append((step, path))
                except (IndexError, ValueError):
                    continue
        
        # Sort by step
        checkpoints.sort(key=lambda x: x[0])
        
        # Remove old ones
        while len(checkpoints) > self.config.save_total_limit:
            _, old_path = checkpoints.pop(0)
            shutil.rmtree(old_path)
            logger.info(f"Removed old checkpoint: {old_path}")
    
    async def _collect_rollouts(
        self,
        tasks: List[CUATask],
        is_eval: bool = False,
    ) -> List[TrajectoryGroup]:
        """Collect rollouts for a batch of tasks using gbox-cua's standalone agent.

        When detailed logging is enabled, this prints rich, step-by-step rollout logs
        similar to rl-qwen3:
          - Per-task / per-group headers
          - Per-rollout summary (reward, success, num_turns)
          - Per-action tool call details from `action_history`
        """
        groups: List[TrajectoryGroup] = []

        num_rollouts = 1 if is_eval else self.config.rollout.num_rollouts

        for group_id, task in enumerate(tasks):
            group = TrajectoryGroup(task=task, group_id=group_id)

            if self.config.enable_detailed_logging:
                print("\n" + "-" * 80, flush=True)
                phase = "EVAL" if is_eval else "TRAIN"
                print(
                    f"[{phase}] Task Group {group_id} | Task ID={task.id} | "
                    f"Rollouts={num_rollouts}",
                    flush=True,
                )
                print(f"Description: {task.description}", flush=True)

            for rollout_idx in range(num_rollouts):
                try:
                    # Run one task with standalone agent
                    result_dict, action_history = await self.standalone_agent.run_task(
                        task_description=task.description,
                        box_type="android",
                        verbose=self.config.verbose,
                    )

                    if self.config.enable_detailed_logging:
                        phase = "EVAL" if is_eval else "TRAIN"
                        print(
                            f"\n[{phase}] Rollout {rollout_idx + 1}/{num_rollouts} "
                            f"for Task {task.id}",
                            flush=True,
                        )

                    # Copy conversation from agent state
                    conversation = deepcopy(self.standalone_agent.conversation)

                    # Build rollout result for reward function
                    errors = [
                        step.get("error", "")
                        for step in action_history
                        if not step.get("success", True)
                    ]
                    rollout_result = CUARolloutResult(
                        task_id=task.id,
                        task_completed=result_dict.get("task_completed", False),
                        task_success=result_dict.get("task_success", False),
                        num_turns=result_dict.get("num_turns", len(action_history)),
                        max_turns=result_dict.get("max_turns", self.config.rollout.max_turns),
                        errors=errors,
                    )
                    reward = simple_reward_function(rollout_result, task)

                    if self.config.enable_detailed_logging:
                        print(
                            f"  ↳ Rollout summary: "
                            f"completed={rollout_result.task_completed}, "
                            f"success={rollout_result.task_success}, "
                            f"num_turns={rollout_result.num_turns}, "
                            f"reward={reward:.3f}",
                            flush=True,
                        )
                        if errors:
                            print(f"  ↳ Errors ({len(errors)}):", flush=True)
                            for e in errors:
                                print(f"      - {e}", flush=True)

                        # Detailed tool / action history
                        if action_history:
                            print("  ↳ Action history:", flush=True)
                            for step_idx, step in enumerate(action_history, start=1):
                                try:
                                    action_type = step.get("action") or step.get("type") or "unknown"
                                    success = step.get("success", True)
                                    error = step.get("error")
                                    tool_name = step.get("tool_name") or step.get("tool") or ""
                                    summary_parts = [
                                        f"step={step_idx}",
                                        f"action={action_type}",
                                        f"success={success}",
                                    ]
                                    if tool_name:
                                        summary_parts.append(f"tool={tool_name}")
                                    if error:
                                        summary_parts.append(f"error={error}")
                                    print("      " + " | ".join(summary_parts), flush=True)
                                except Exception:
                                    # Fallback: best-effort print of raw step dict
                                    print(f"      step={step_idx}: {step}", flush=True)

                    sample = TrajectorySample(
                        task_id=task.id,
                        task=task,
                        conversation=conversation,
                        reward=reward,
                        metadata={
                            "rollout_result": rollout_result.to_dict(),
                            "raw_result": result_dict,
                        },
                        rollout_idx=rollout_idx,
                        group_id=group_id,
                    )
                    group.samples.append(sample)
                except Exception as e:
                    logger.error(f"Rollout failed for task {task.id}: {e}", exc_info=True)
                    continue

            if group.samples:
                if self.config.enable_detailed_logging:
                    rewards = [s.reward for s in group.samples]
                    successes = [
                        s.metadata.get("rollout_result", {}).get("task_success", False)
                        for s in group.samples
                    ]
                    avg_reward = float(np.mean(rewards)) if rewards else 0.0
                    success_rate = float(np.mean(successes)) if successes else 0.0
                    print(
                        f"\n[GROUP STATS] Task {task.id} | "
                        f"Rollouts={len(group.samples)} | "
                        f"avg_reward={avg_reward:.3f} | "
                        f"success_rate={success_rate:.2%}",
                        flush=True,
                    )
                groups.append(group)

        return groups
    
    def _compute_advantages(self, groups: List[TrajectoryGroup]) -> Tuple[List[TrajectoryGroup], int, int]:
        """Compute GRPO advantages for each group.
        
        Returns:
            Tuple of (groups, groups_kept, groups_filtered)
        """
        groups_kept = 0
        groups_filtered = 0
        
        for group in groups:
            rewards = [s.reward for s in group.samples]
            advantages = calculate_grpo_advantages(rewards, self.config.min_group_std)
            
            # Check if group was filtered (all zero advantages)
            if all(a == 0 for a in advantages):
                groups_filtered += 1
            else:
                groups_kept += 1
            
            # Assign advantages
            for sample, advantage in zip(group.samples, advantages):
                sample.advantage = advantage
        
        return groups, groups_kept, groups_filtered
    
    def tokenize_conversation_with_mask(
        self,
        conversation: List[Dict],
        advantage: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize conversation and create loss mask.
        
        Only assistant turns are included in the loss calculation.
        
        Returns:
            Tuple of (input_ids, labels, loss_mask, advantage_mask)
        """
        all_tokens = []
        all_masks = []
        all_advantages = []
        
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            
            is_model_generated = (role == "assistant")
            
            # Serialize message in ChatML format
            if role == "system":
                text = f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text = f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                if tool_calls:
                    tool_calls_str = json.dumps(tool_calls)
                    text = f"<|im_start|>assistant\n{tool_calls_str}<|im_end|>\n"
                else:
                    text = f"<|im_start|>assistant\n{content or ''}<|im_end|>\n"
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                text = f"<|im_start|>tool\ntool_call_id: {tool_call_id}\n{content}<|im_end|>\n"
            else:
                text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if is_model_generated:
                mask = [1.0] * len(tokens)
                advantages = [advantage] * len(tokens)
            else:
                mask = [0.0] * len(tokens)
                advantages = [0.0] * len(tokens)
            
            all_tokens.extend(tokens)
            all_masks.extend(mask)
            all_advantages.extend(advantages)
        
        # Add EOS
        all_tokens.append(self.tokenizer.eos_token_id)
        all_masks.append(0.0)
        all_advantages.append(0.0)
        
        # Truncate if too long
        max_len = self.config.max_seq_length
        if len(all_tokens) > max_len:
            all_tokens = all_tokens[:max_len]
            all_masks = all_masks[:max_len]
            all_advantages = all_advantages[:max_len]
        
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        labels = input_ids.clone()
        loss_mask = torch.tensor(all_masks, dtype=torch.float)
        advantage_mask = torch.tensor(all_advantages, dtype=torch.float)
        
        return input_ids, labels, loss_mask, advantage_mask
    
    def compute_loss_for_trajectory(
        self,
        conversation: List[Dict],
        advantage: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a single trajectory."""
        input_ids, labels, loss_mask, advantage_mask = self.tokenize_conversation_with_mask(
            conversation, advantage
        )
        
        device = next(self.model.parameters()).device
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        loss_mask = loss_mask.to(device)
        advantage_mask = advantage_mask.to(device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = loss_mask[1:].contiguous()
        shift_advantage = advantage_mask[1:].contiguous()
        
        # Compute token-level losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Apply mask and advantages
        masked_losses = token_losses * shift_mask * shift_advantage
        
        num_trainable = shift_mask.sum().item()
        
        if num_trainable == 0:
            policy_loss = torch.tensor(0.0, device=device)
        else:
            policy_loss = masked_losses.sum() / shift_mask.sum()
        
        # KL divergence (simplified)
        kl_loss = torch.tensor(0.0, device=device)
        
        total_loss = policy_loss + self.config.beta * kl_loss
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "num_trainable_tokens": num_trainable,
            "num_total_tokens": shift_mask.numel(),
        }
        
        return total_loss, metrics
    
    def _run_evaluation(self) -> TrainingMetrics:
        """Run evaluation on eval set."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        self.eval_reward_tracker.reset()
        
        # Collect rollouts
        loop = asyncio.get_event_loop()
        groups = loop.run_until_complete(
            self._collect_rollouts(self.eval_tasks, is_eval=True)
        )
        
        rewards = []
        successes = []
        
        for group in groups:
            for sample in group.samples:
                rewards.append(sample.reward)
                success = sample.metadata.get("task_success", False)
                successes.append(success)
                self.eval_reward_tracker.add(sample.reward, success, sample.metadata.get("num_turns", 0))
        
        avg_reward = np.mean(rewards) if rewards else 0.0
        accuracy = np.mean(successes) if successes else 0.0
        
        self.model.train()
        
        return TrainingMetrics(
            loss=0.0,
            policy_loss=0.0,
            kl_loss=0.0,
            avg_reward=avg_reward,
            max_reward=max(rewards) if rewards else 0.0,
            min_reward=min(rewards) if rewards else 0.0,
            accuracy=accuracy,
            reward_std=np.std(rewards) if rewards else 0.0,
        )
    
    async def _update_vllm_lora(self):
        """Placeholder for dynamic LoRA switching on vLLM server.

        NOTE: Standalone gbox-cua agent talks directly to vLLM via HTTP.
        Dynamic LoRA reloading should be handled by the vLLM deployment
        (e.g., by mounting updated adapter path into the container).
        """
        if not self.config.vllm.enable_dynamic_lora:
            return
        logger.info(
            "Dynamic LoRA switching is enabled, but automatic reload is not "
            "implemented yet. Please ensure vLLM container watches the "
            "adapter path for updates."
        )
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        print("\n" + "=" * 80, flush=True)
        print(f"🚀 Starting CUA GRPO Training", flush=True)
        print("=" * 80, flush=True)
        
        self.training_start_time = time.time()
        self.model.train()
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"cua-grpo-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.to_dict(),
            )

        # Baseline evaluation before training (like rl-qwen3 flow)
        print("\n📊 Running baseline evaluation before training...", flush=True)
        baseline_metrics = self._run_evaluation()
        print(
            f"   Baseline accuracy = {baseline_metrics.accuracy:.2%}, "
            f"avg_reward = {baseline_metrics.avg_reward:.3f}",
            flush=True,
        )
        if self.use_wandb:
            wandb.log(
                {
                    "baseline/accuracy": baseline_metrics.accuracy,
                    "baseline/avg_reward": baseline_metrics.avg_reward,
                    "baseline/reward_std": baseline_metrics.reward_std,
                },
                step=self.global_step,
            )
        
        task_idx = 0
        loop = asyncio.get_event_loop()
        
        for step in range(self.global_step, self.config.max_steps):
            self.global_step = step + 1
            step_start = time.time()

            print("\n" + "=" * 80, flush=True)
            print(
                f"STEP {self.global_step}/{self.config.max_steps}",
                flush=True,
            )
            print("=" * 80, flush=True)
            
            # Get batch of tasks
            batch_tasks = []
            for _ in range(self.config.batch_size):
                batch_tasks.append(self.train_tasks[task_idx % len(self.train_tasks)])
                task_idx += 1
            
            # Collect rollouts
            rollout_start = time.time()
            print(
                f"🎯 COLLECTING ROLLOUTS "
                f"({len(batch_tasks)} tasks × {self.config.rollout.num_rollouts} rollouts)...",
                flush=True,
            )
            groups = loop.run_until_complete(self._collect_rollouts(batch_tasks))
            rollout_time = time.time() - rollout_start
            
            # Compute advantages
            print("📐 Computing GRPO advantages for trajectory groups...", flush=True)
            groups, groups_kept, groups_filtered = self._compute_advantages(groups)
            
            # Training step
            train_start = time.time()
            
            total_loss = 0.0
            total_policy_loss = 0.0
            num_samples = 0
            
            self.optimizer.zero_grad()
            
            for group in groups:
                for sample in group.samples:
                    if sample.advantage is None or sample.advantage == 0:
                        continue
                    
                    try:
                        loss, metrics = self.compute_loss_for_trajectory(
                            sample.conversation,
                            sample.advantage,
                        )
                        
                        if loss.item() != 0 and not torch.isnan(loss):
                            (loss / self.config.gradient_accumulation_steps).backward()
                            total_loss += loss.item()
                            total_policy_loss += metrics["policy_loss"]
                            num_samples += 1
                    except Exception as e:
                        logger.error(f"Error computing loss: {e}")
                        continue
            
            if num_samples > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            train_time = time.time() - train_start
            
            # Collect metrics
            all_rewards = [s.reward for g in groups for s in g.samples]
            all_successes = [s.metadata.get("task_success", False) for g in groups for s in g.samples]
            
            metrics = TrainingMetrics(
                loss=total_loss / max(num_samples, 1),
                policy_loss=total_policy_loss / max(num_samples, 1),
                kl_loss=0.0,
                avg_reward=np.mean(all_rewards) if all_rewards else 0.0,
                max_reward=max(all_rewards) if all_rewards else 0.0,
                min_reward=min(all_rewards) if all_rewards else 0.0,
                accuracy=np.mean(all_successes) if all_successes else 0.0,
                reward_std=np.std(all_rewards) if all_rewards else 0.0,
                rollout_time=rollout_time,
                training_time=train_time,
                groups_kept=groups_kept,
                groups_filtered=groups_filtered,
            )
            
            # Log progress
            print(
                f"✅ Step {self.global_step}/{self.config.max_steps} | "
                f"Loss={metrics.loss:.4f} | "
                f"Accuracy={metrics.accuracy:.2%} | "
                f"Reward={metrics.avg_reward:.3f} | "
                f"Rollout={rollout_time:.1f}s | "
                f"Train={train_time:.1f}s | "
                f"Groups kept={groups_kept} / {groups_kept + groups_filtered}",
                flush=True
            )
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "train/loss": metrics.loss,
                    "train/policy_loss": metrics.policy_loss,
                    "train/avg_reward": metrics.avg_reward,
                    "train/accuracy": metrics.accuracy,
                    "train/reward_std": metrics.reward_std,
                    "train/rollout_time": rollout_time,
                    "train/training_time": train_time,
                    "train/groups_kept": groups_kept,
                    "train/groups_filtered": groups_filtered,
                }, step=self.global_step)
            
            # Update vLLM LoRA periodically
            if self.global_step % self.config.vllm.lora_update_steps == 0:
                loop.run_until_complete(self._update_vllm_lora())
            
            # Evaluation
            if self.global_step % self.config.eval_steps == 0:
                eval_metrics = self._run_evaluation()
                
                print(
                    f"\n📊 Eval Step {self.global_step}: "
                    f"Accuracy={eval_metrics.accuracy:.2%}, "
                    f"Reward={eval_metrics.avg_reward:.3f}\n",
                    flush=True
                )
                
                if self.use_wandb:
                    wandb.log({
                        "eval/accuracy": eval_metrics.accuracy,
                        "eval/avg_reward": eval_metrics.avg_reward,
                        "eval/reward_std": eval_metrics.reward_std,
                    }, step=self.global_step)
                
                # Check for improvement
                is_best = eval_metrics.accuracy > self.best_accuracy
                if is_best:
                    self.best_accuracy = eval_metrics.accuracy
                    self.evals_without_improvement = 0
                    print(f"🎯 New best accuracy: {self.best_accuracy:.2%}", flush=True)
                else:
                    self.evals_without_improvement += 1
                
                # Save checkpoint
                self._save_checkpoint(eval_metrics, is_best=is_best)
                
                # Early stopping
                if self.evals_without_improvement >= self.config.patience:
                    print(f"Early stopping: no improvement for {self.config.patience} evaluations", flush=True)
                    break
                
                # Target reached
                if eval_metrics.accuracy >= self.config.target_accuracy:
                    print(f"🎉 Target accuracy {self.config.target_accuracy:.2%} reached!", flush=True)
                    self._save_checkpoint(eval_metrics, is_best=True)
                    break
            
            # Save checkpoint periodically
            elif self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(metrics)
        
        # Final save
        final_dir = self.output_dir / "final"
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        
        total_time = time.time() - self.training_start_time
        
        print("\n" + "=" * 60, flush=True)
        print("✅ Training complete!", flush=True)
        print("=" * 60, flush=True)
        print(f"Total time: {total_time / 60:.1f} minutes", flush=True)
        print(f"Best accuracy: {self.best_accuracy:.2%}", flush=True)
        print(f"Final model: {final_dir}", flush=True)
        if self.best_model_path:
            print(f"Best model: {self.best_model_path}", flush=True)
        print("=" * 60, flush=True)
        
        if self.use_wandb:
            wandb.finish()
    
    async def close(self):
        """Close trainer resources."""
        await self.standalone_agent.close()


__all__ = [
    "CUAGRPOTrainer",
    "TrajectorySample",
    "TrajectoryGroup",
    "TrainingMetrics",
]

