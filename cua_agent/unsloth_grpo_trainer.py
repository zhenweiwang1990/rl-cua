"""Unsloth-based GRPO Trainer for CUA Agent.

This module implements GRPO training using:
- RL: AReaL-style async rollout pattern
- Rollout: HuggingFace + Unsloth for inference
- LoRA: Vision + cross-modal targeting
- Reference: HuggingFace frozen model

This replaces the vLLM-based trainer with a local inference approach.
"""

import asyncio
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from cua_agent.unsloth_inference import (
    UnslothInferenceConfig,
    UnslothVLMInference,
    FrozenReferenceModel,
)
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

logger = logging.getLogger(__name__)


@dataclass
class UnslothGRPOConfig:
    """Configuration for Unsloth-based GRPO training."""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_seq_length: int = 8192
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    
    # LoRA settings (vision + cross-modal)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        # Language model
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # Vision (cross-modal)
        # Note: actual module names depend on model architecture
    ])
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    beta: float = 0.01  # KL penalty coefficient
    clip_epsilon: float = 0.2
    
    # Batch settings
    batch_size: int = 2  # Number of tasks per batch
    num_rollouts: int = 4  # Rollouts per task (GRPO group size)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Training schedule
    max_steps: int = 200
    warmup_steps: int = 10
    
    # Rollout settings
    max_turns: int = 15
    max_new_tokens: int = 1024
    temperature: float = 0.7
    
    # Evaluation and checkpointing
    eval_steps: int = 10
    save_steps: int = 10
    save_total_limit: int = 5
    
    # Early stopping
    target_accuracy: float = 0.80
    patience: int = 5
    
    # GRPO specific
    min_group_std: float = 0.05
    
    # Output
    output_dir: str = "outputs/unsloth_grpo"
    seed: int = 42
    
    # Environment
    gbox_api_key: Optional[str] = None
    
    # Logging
    verbose: bool = False
    enable_wandb: bool = False
    wandb_project: str = "cua-grpo"
    enable_detailed_logging: bool = True
    
    @classmethod
    def from_env(cls) -> "UnslothGRPOConfig":
        """Create config from environment variables."""
        def get_env_int(key, default):
            val = os.environ.get(key, str(default))
            return int(val.split('#')[0].strip())
        
        def get_env_float(key, default):
            val = os.environ.get(key, str(default))
            return float(val.split('#')[0].strip())
        
        def get_env_bool(key, default):
            val = os.environ.get(key, str(default)).lower().strip()
            return val in ("true", "1", "yes")
        
        return cls(
            model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"),
            max_seq_length=get_env_int("MAX_SEQ_LENGTH", 8192),
            load_in_4bit=get_env_bool("LOAD_IN_4BIT", "true"),
            use_lora=get_env_bool("USE_LORA", "true"),
            lora_r=get_env_int("LORA_R", 16),
            lora_alpha=get_env_int("LORA_ALPHA", 32),
            learning_rate=get_env_float("LEARNING_RATE", 1e-5),
            beta=get_env_float("BETA", 0.01),
            batch_size=get_env_int("BATCH_SIZE", 2),
            num_rollouts=get_env_int("NUM_ROLLOUTS", 4),
            max_steps=get_env_int("MAX_STEPS", 200),
            max_turns=get_env_int("MAX_TURNS", 15),
            temperature=get_env_float("TEMPERATURE", 0.7),
            eval_steps=get_env_int("EVAL_STEPS", 10),
            save_steps=get_env_int("SAVE_STEPS", 10),
            target_accuracy=get_env_float("TARGET_ACCURACY", 0.80),
            output_dir=os.environ.get("OUTPUT_DIR", "outputs/unsloth_grpo"),
            seed=get_env_int("SEED", 42),
            gbox_api_key=os.environ.get("GBOX_API_KEY"),
            verbose=get_env_bool("VERBOSE", "false"),
            enable_wandb=get_env_bool("ENABLE_WANDB", "false"),
            enable_detailed_logging=get_env_bool("ENABLE_DETAILED_LOGGING", "true"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "beta": self.beta,
            "batch_size": self.batch_size,
            "num_rollouts": self.num_rollouts,
            "max_steps": self.max_steps,
            "max_turns": self.max_turns,
            "target_accuracy": self.target_accuracy,
            "output_dir": self.output_dir,
        }


@dataclass
class TrajectorySample:
    """Single trajectory from a rollout."""
    task_id: str
    task: CUATask
    conversation: List[Dict]
    reward: float
    metadata: Dict[str, Any]
    rollout_idx: int
    group_id: int
    advantage: Optional[float] = None
    # Token-level data for training
    input_ids: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None


@dataclass
class TrajectoryGroup:
    """Grouped trajectories for GRPO."""
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
    accuracy: float
    num_trainable_tokens: int = 0
    num_total_tokens: int = 0
    rollout_time: float = 0.0
    training_time: float = 0.0
    reward_std: float = 0.0
    groups_kept: int = 0
    groups_filtered: int = 0


class UnslothGRPOTrainer:
    """GRPO Trainer using Unsloth + HuggingFace for local inference.
    
    Architecture:
    - Policy Model: Unsloth-loaded VLM with LoRA (trainable)
    - Reference Model: HuggingFace frozen model (for KL divergence)
    - Rollout: Local inference using policy model
    - RL Algorithm: GRPO with group-wise advantages
    """
    
    def __init__(
        self,
        config: Optional[UnslothGRPOConfig] = None,
        train_tasks: Optional[List[CUATask]] = None,
        eval_tasks: Optional[List[CUATask]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """Initialize the Unsloth GRPO trainer.
        
        Args:
            config: Training configuration
            train_tasks: Training tasks
            eval_tasks: Evaluation tasks
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.config = config or UnslothGRPOConfig.from_env()
        
        # Load tasks
        self.train_tasks = train_tasks or get_training_tasks()
        self.eval_tasks = eval_tasks or get_eval_tasks()
        
        # Setup directories
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize policy model with Unsloth
        logger.info("Loading policy model with Unsloth...")
        inference_config = UnslothInferenceConfig(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=self.config.dtype,
            use_lora=self.config.use_lora,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_target_modules=self.config.lora_target_modules,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        
        self.policy_inference = UnslothVLMInference(config=inference_config)
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            self.policy_inference.apply_lora()
        
        self.model = self.policy_inference.model
        self.tokenizer = self.policy_inference.tokenizer
        
        # Initialize frozen reference model for KL divergence
        self.ref_model = None
        if self.config.beta > 0:
            logger.info("Loading frozen reference model...")
            self.ref_model = FrozenReferenceModel(
                model_name=self.config.model_name,
                load_in_4bit=self.config.load_in_4bit,
                dtype=self.config.dtype,
            )
            self.ref_model.load()
        
        # Initialize GBox agent for rollouts
        gbox_api_key = self.config.gbox_api_key or os.environ.get("GBOX_API_KEY", "")
        if not gbox_api_key:
            logger.warning("GBOX_API_KEY not set, rollouts will fail")
        
        # Import GBox agent
        from gbox_cua.agent import StandaloneGBoxCUAAgent
        
        # Create a custom agent that uses our Unsloth model
        self.gbox_api_key = gbox_api_key
        
        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
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
        self.use_wandb = self.config.enable_wandb and WANDB_AVAILABLE
        
        # Color support
        self._enable_color = os.environ.get("NO_COLOR", "").strip() == ""
        
        # Resume from checkpoint
        if resume_from_checkpoint:
            self._load_checkpoint(Path(resume_from_checkpoint))
        
        self._log_config()
    
    def _color(self, text: str, color: str) -> str:
        """Apply ANSI colors."""
        if not self._enable_color:
            return text
        colors = {
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "cyan": "\033[96m",
            "magenta": "\033[95m",
            "blue": "\033[94m",
            "bold": "\033[1m",
        }
        reset = "\033[0m"
        prefix = colors.get(color)
        if not prefix:
            return text
        return f"{prefix}{text}{reset}"
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into H:MM:SS."""
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
    
    def _log_config(self):
        """Log training configuration."""
        logger.info("=" * 60)
        logger.info("Unsloth GRPO Trainer Initialized")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"LoRA: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        logger.info(f"Train tasks: {len(self.train_tasks)}")
        logger.info(f"Eval tasks: {len(self.eval_tasks)}")
        logger.info(f"Rollouts per task: {self.config.num_rollouts}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"KL penalty (beta): {self.config.beta}")
        logger.info(f"Max steps: {self.config.max_steps}")
        logger.info(f"Target accuracy: {self.config.target_accuracy * 100:.1f}%")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Reference model: {'Enabled (HF frozen)' if self.ref_model else 'Disabled'}")
        logger.info("=" * 60)
    
    async def _collect_rollouts(
        self,
        tasks: List[CUATask],
        is_eval: bool = False,
    ) -> List[TrajectoryGroup]:
        """Collect rollouts using local Unsloth model inference.
        
        Args:
            tasks: Tasks to collect rollouts for
            is_eval: Whether this is evaluation (single rollout)
            
        Returns:
            List of trajectory groups
        """
        from gbox_cua.gbox_client import GBoxClient
        from gbox_cua.prompts import create_system_prompt
        from gbox_cua.agent import GBoxAgentCore
        
        groups: List[TrajectoryGroup] = []
        num_rollouts = 1 if is_eval else self.config.num_rollouts
        
        # Set model to eval mode for inference
        self.model.eval()
        
        for group_id, task in enumerate(tasks):
            group = TrajectoryGroup(task=task, group_id=group_id)
            
            if self.config.enable_detailed_logging:
                phase = "EVAL" if is_eval else "TRAIN"
                print(f"\n[{phase}] Task Group {group_id} | Task: {task.id}")
                print(f"Description: {task.description[:100]}...")
            
            for rollout_idx in range(num_rollouts):
                try:
                    # Create GBox client
                    gbox = GBoxClient(
                        api_key=self.gbox_api_key,
                        box_type="android",
                        timeout="60s",
                        wait=True,
                    )
                    
                    # Create agent core
                    agent_core = GBoxAgentCore(
                        gbox_client=gbox,
                        max_turns=self.config.max_turns,
                    )
                    
                    conversation = []
                    action_history = []
                    task_success = False
                    task_completed = False
                    
                    try:
                        # Create box
                        await gbox.create_box("android")
                        await asyncio.sleep(1.0)
                        
                        # Build system prompt
                        system_prompt = create_system_prompt(
                            task_description=task.description,
                            max_turns=self.config.max_turns,
                        )
                        
                        conversation.append({
                            "role": "system",
                            "content": system_prompt,
                        })
                        
                        # Temperature variation for diversity
                        temp = self.config.temperature + (rollout_idx * 0.1)
                        
                        # Episode loop
                        for turn in range(1, self.config.max_turns + 1):
                            # Take screenshot
                            await asyncio.sleep(0.3)
                            screenshot_bytes, screenshot_uri = await gbox.take_screenshot()
                            
                            # Build user message
                            user_msg = f"Turn {turn}/{self.config.max_turns}. Analyze the screenshot and take the next action to complete the task."
                            conversation.append({
                                "role": "user",
                                "content": user_msg,
                            })
                            
                            # Generate response using local model
                            from PIL import Image
                            import io
                            image = Image.open(io.BytesIO(screenshot_bytes))
                            
                            response = self.policy_inference.generate(
                                messages=conversation,
                                image=image,
                                temperature=temp,
                                max_new_tokens=self.config.max_new_tokens,
                                return_logprobs=True,
                            )
                            
                            # Add assistant response
                            assistant_msg = {"role": "assistant", "content": response.content}
                            if response.tool_calls:
                                assistant_msg["tool_calls"] = response.tool_calls
                            conversation.append(assistant_msg)
                            
                            # Parse and execute tool call
                            tool_call = None
                            if response.tool_calls:
                                tool_call = response.tool_calls[0]
                            else:
                                # Try to parse from content
                                tool_call = GBoxAgentCore.parse_tool_call_from_response(
                                    response.content, turn
                                )
                            
                            if tool_call:
                                func_name = tool_call.get("function", {}).get("name", "")
                                func_args_str = tool_call.get("function", {}).get("arguments", "{}")
                                
                                try:
                                    func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                                except json.JSONDecodeError:
                                    func_args = {}
                                
                                # Execute action
                                action_result, done, is_success, _ = await agent_core.execute_tool_call(
                                    tool_call, screenshot_uri
                                )
                                
                                action_history.append({
                                    "turn": turn,
                                    "action": func_name,
                                    "success": action_result.get("status") == "success",
                                    "error": action_result.get("error"),
                                })
                                
                                # Add tool response
                                conversation.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.get("id", f"call_{turn}"),
                                    "content": json.dumps(action_result),
                                })
                                
                                if done:
                                    task_completed = True
                                    task_success = is_success
                                    break
                            else:
                                # No tool call, continue
                                conversation.append({
                                    "role": "tool",
                                    "tool_call_id": f"call_{turn}",
                                    "content": json.dumps({"status": "no_action"}),
                                })
                        
                    finally:
                        try:
                            await gbox.terminate_box()
                        except Exception:
                            pass
                    
                    # Build rollout result
                    errors = [h.get("error") for h in action_history if h.get("error")]
                    rollout_result = CUARolloutResult(
                        task_id=task.id,
                        task_completed=task_completed,
                        task_success=task_success,
                        num_turns=len(action_history),
                        max_turns=self.config.max_turns,
                        errors=errors,
                    )
                    
                    reward = simple_reward_function(rollout_result, task)
                    
                    if self.config.enable_detailed_logging:
                        phase = "EVAL" if is_eval else "TRAIN"
                        print(f"  [{phase}] Rollout {rollout_idx + 1}: "
                              f"success={task_success}, turns={len(action_history)}, reward={reward:.3f}")
                    
                    sample = TrajectorySample(
                        task_id=task.id,
                        task=task,
                        conversation=conversation,
                        reward=reward,
                        metadata={
                            "rollout_result": rollout_result.to_dict(),
                            "action_history": action_history,
                        },
                        rollout_idx=rollout_idx,
                        group_id=group_id,
                    )
                    group.samples.append(sample)
                    
                except Exception as e:
                    logger.error(f"Rollout failed for task {task.id}: {e}", exc_info=True)
                    continue
            
            if group.samples:
                groups.append(group)
        
        # Set model back to train mode
        self.model.train()
        
        return groups
    
    def _compute_advantages(
        self,
        groups: List[TrajectoryGroup],
    ) -> Tuple[List[TrajectoryGroup], int, int]:
        """Compute GRPO advantages for each group."""
        groups_kept = 0
        groups_filtered = 0
        
        for group in groups:
            rewards = [s.reward for s in group.samples]
            advantages = calculate_grpo_advantages(rewards, self.config.min_group_std)
            
            if all(a == 0 for a in advantages):
                groups_filtered += 1
            else:
                groups_kept += 1
            
            for sample, advantage in zip(group.samples, advantages):
                sample.advantage = advantage
        
        return groups, groups_kept, groups_filtered
    
    def tokenize_conversation_with_mask(
        self,
        conversation: List[Dict],
        advantage: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize conversation and create loss mask.
        
        Only assistant turns contribute to the loss.
        """
        all_tokens = []
        all_masks = []
        all_advantages = []
        
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            
            is_model_generated = (role == "assistant")
            
            # Serialize in ChatML format
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
        eos_id = self.tokenizer.eos_token_id or 0
        all_tokens.append(eos_id)
        all_masks.append(0.0)
        all_advantages.append(0.0)
        
        # Truncate
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
        """Compute GRPO loss for a single trajectory."""
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
        
        # Token-level cross-entropy loss
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
        
        # KL divergence (if reference model is available)
        kl_loss = torch.tensor(0.0, device=device)
        if self.ref_model is not None and self.config.beta > 0:
            with torch.no_grad():
                ref_logprobs = self.ref_model.compute_logprobs(input_ids)
            
            # Compute policy logprobs
            policy_log_probs = torch.log_softmax(shift_logits, dim=-1)
            policy_token_logprobs = torch.gather(
                policy_log_probs.view(-1, policy_log_probs.size(-1)),
                dim=-1,
                index=shift_labels.view(-1, 1),
            ).squeeze(-1)
            
            # KL = policy_logp - ref_logp
            ref_logprobs_shifted = ref_logprobs[:, :policy_token_logprobs.shape[0]].view(-1)
            kl = (policy_token_logprobs - ref_logprobs_shifted) * shift_mask
            kl_loss = kl.sum() / max(shift_mask.sum(), 1)
        
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
        
        loop = asyncio.get_event_loop()
        groups = loop.run_until_complete(
            self._collect_rollouts(self.eval_tasks, is_eval=True)
        )
        
        rewards = []
        successes = []
        
        for group in groups:
            for sample in group.samples:
                rewards.append(sample.reward)
                success = sample.metadata.get("rollout_result", {}).get("task_success", False)
                successes.append(success)
        
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
    
    def _save_checkpoint(self, metrics: TrainingMetrics, is_best: bool = False):
        """Save checkpoint."""
        checkpoint_name = f"checkpoint-{self.global_step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model (LoRA adapter)
        self.policy_inference.save_lora(str(checkpoint_path))
        
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
            },
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_model_path = self.output_dir / "best_model"
            if best_model_path.exists():
                shutil.rmtree(best_model_path)
            shutil.copytree(checkpoint_path, best_model_path)
            self.best_model_path = best_model_path
            logger.info(f"Best model saved: {best_model_path}")
        
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint."""
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.best_accuracy = state.get("best_accuracy", 0.0)
            self.best_model_path = state.get("best_model_path")
            self.evals_without_improvement = state.get("evals_without_improvement", 0)
            logger.info(f"Loaded training state: step={self.global_step}, best_acc={self.best_accuracy:.2%}")
        
        # Load LoRA weights
        adapter_file = checkpoint_path / "adapter_model.safetensors"
        if adapter_file.exists():
            self.policy_inference.load_lora(str(checkpoint_path))
            logger.info(f"Loaded LoRA weights from {checkpoint_path}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints."""
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                try:
                    step = int(path.name.split("-")[1])
                    checkpoints.append((step, path))
                except (IndexError, ValueError):
                    continue
        
        checkpoints.sort(key=lambda x: x[0])
        
        while len(checkpoints) > self.config.save_total_limit:
            _, old_path = checkpoints.pop(0)
            shutil.rmtree(old_path)
            logger.info(f"Removed old checkpoint: {old_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting Unsloth GRPO training...")
        print("\n" + "=" * 80)
        print(self._color("🚀 Starting Unsloth GRPO Training", "green"))
        print("=" * 80)
        
        self.training_start_time = time.time()
        self.model.train()
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=f"unsloth-grpo-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.to_dict(),
            )
        
        # Baseline evaluation
        print(self._color("\n📊 Running baseline evaluation...", "cyan"))
        baseline_metrics = self._run_evaluation()
        print(f"   Baseline accuracy: {baseline_metrics.accuracy:.2%}, reward: {baseline_metrics.avg_reward:.3f}")
        
        task_idx = 0
        loop = asyncio.get_event_loop()
        
        for step in range(self.global_step, self.config.max_steps):
            self.global_step = step + 1
            step_start = time.time()
            
            # Progress
            progress = self.global_step / self.config.max_steps
            elapsed = time.time() - self.training_start_time
            eta = elapsed / max(self.global_step, 1) * max(self.config.max_steps - self.global_step, 0)
            
            print(f"\n{'='*80}")
            print(self._color(
                f"STEP {self.global_step}/{self.config.max_steps} ({progress*100:.1f}%) | "
                f"elapsed {self._format_time(elapsed)} | ETA {self._format_time(eta)}",
                "magenta"
            ))
            
            # Get batch of tasks
            batch_tasks = []
            for _ in range(self.config.batch_size):
                batch_tasks.append(self.train_tasks[task_idx % len(self.train_tasks)])
                task_idx += 1
            
            # Collect rollouts
            rollout_start = time.time()
            print(self._color(
                f"🎯 Collecting rollouts ({len(batch_tasks)} tasks × {self.config.num_rollouts} rollouts)...",
                "cyan"
            ))
            groups = loop.run_until_complete(self._collect_rollouts(batch_tasks))
            rollout_time = time.time() - rollout_start
            
            # Compute advantages
            print(self._color("📐 Computing GRPO advantages...", "cyan"))
            groups, groups_kept, groups_filtered = self._compute_advantages(groups)
            
            # Training step
            train_start = time.time()
            
            total_loss = 0.0
            total_policy_loss = 0.0
            total_kl_loss = 0.0
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
                            total_kl_loss += metrics["kl_loss"]
                            num_samples += 1
                    except Exception as e:
                        logger.error(f"Error computing loss: {e}")
                        continue
            
            if num_samples > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )
                self.optimizer.step()
            
            train_time = time.time() - train_start
            
            # Collect metrics
            all_rewards = [s.reward for g in groups for s in g.samples]
            all_successes = [
                s.metadata.get("rollout_result", {}).get("task_success", False)
                for g in groups for s in g.samples
            ]
            
            metrics = TrainingMetrics(
                loss=total_loss / max(num_samples, 1),
                policy_loss=total_policy_loss / max(num_samples, 1),
                kl_loss=total_kl_loss / max(num_samples, 1),
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
            color = "green" if metrics.accuracy >= 0.5 else ("yellow" if metrics.accuracy >= 0.2 else "red")
            print(self._color(
                f"✅ Loss={metrics.loss:.4f} | Acc={metrics.accuracy:.2%} | "
                f"Reward={metrics.avg_reward:.3f} | KL={metrics.kl_loss:.4f} | "
                f"Rollout={rollout_time:.1f}s | Train={train_time:.1f}s",
                color
            ))
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    "train/loss": metrics.loss,
                    "train/policy_loss": metrics.policy_loss,
                    "train/kl_loss": metrics.kl_loss,
                    "train/avg_reward": metrics.avg_reward,
                    "train/accuracy": metrics.accuracy,
                }, step=self.global_step)
            
            # Evaluation
            if self.global_step % self.config.eval_steps == 0:
                eval_metrics = self._run_evaluation()
                
                print(self._color(
                    f"\n📊 Eval: Accuracy={eval_metrics.accuracy:.2%}, Reward={eval_metrics.avg_reward:.3f}",
                    "blue"
                ))
                
                # Check improvement
                is_best = eval_metrics.accuracy > self.best_accuracy
                if is_best:
                    self.best_accuracy = eval_metrics.accuracy
                    self.evals_without_improvement = 0
                    print(self._color(f"🎯 New best accuracy: {self.best_accuracy:.2%}", "green"))
                else:
                    self.evals_without_improvement += 1
                
                # Save checkpoint
                self._save_checkpoint(eval_metrics, is_best=is_best)
                
                # Early stopping
                if self.evals_without_improvement >= self.config.patience:
                    print(self._color(
                        f"Early stopping: no improvement for {self.config.patience} evaluations",
                        "yellow"
                    ))
                    break
                
                # Target reached
                if eval_metrics.accuracy >= self.config.target_accuracy:
                    print(self._color(
                        f"🎉 Target accuracy {self.config.target_accuracy:.2%} reached!",
                        "green"
                    ))
                    self._save_checkpoint(eval_metrics, is_best=True)
                    break
            
            elif self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(metrics)
        
        # Final save
        final_dir = self.output_dir / "final"
        self.policy_inference.save_lora(str(final_dir))
        
        total_time = time.time() - self.training_start_time
        
        print("\n" + "=" * 60)
        print(self._color("✅ Training complete!", "green"))
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best accuracy: {self.best_accuracy:.2%}")
        print(f"Final model: {final_dir}")
        print("=" * 60)
        
        if self.use_wandb:
            wandb.finish()


__all__ = [
    "UnslothGRPOConfig",
    "UnslothGRPOTrainer",
    "TrajectorySample",
    "TrajectoryGroup",
    "TrainingMetrics",
]

