"""AReaL CUA Workflow - Async Rollout for CUA Agent Training.

This module implements the CUA Agent workflow using AReaL's async infrastructure:
- RolloutWorkflow interface for seamless AReaL integration
- Async episode execution with GBox environments
- Multi-turn VLM interaction with tool calling
- GRPO group-wise parallel episodes

Key Features:
- Uses AReaL's async rollout for efficient parallel execution
- Integrates with GBox CUA for Android environment interaction
- Supports both HF and Unsloth model loaders
- Automatic checkpoint saving and recovery

Architecture:
    AReaL Trainer
        ↓
    CUAEnvRolloutWorkflow (RolloutWorkflow)
        ↓
    EpisodeRunner (per episode)
        ↓
    GBoxAgentCore (action execution)
        ↓
    GBox Android Environment
"""

import asyncio
import base64
import io
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.api.io_struct import ModelRequest
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64

from gbox_cua.gbox_client import GBoxClient
from gbox_cua.agent import GBoxAgentCore, RolloutLogger
from gbox_cua.prompts import create_system_prompt
from gbox_cua.tools import get_tools_schema

from cua_agent.tasks import CUATask, TaskCategory, TaskDifficulty
from cua_agent.reward import validate_task_completion

logger = logging.getLogger(__name__)


def create_empty_result() -> Dict[str, torch.Tensor]:
    """Create empty result tensors for failed episodes."""
    return {
        "input_ids": torch.tensor([], dtype=torch.long),
        "labels": torch.tensor([], dtype=torch.long),
        "logprobs": torch.tensor([], dtype=torch.float),
        "ref_logprobs": torch.tensor([], dtype=torch.float),
        "rewards": torch.tensor([], dtype=torch.float),
        "advantages": torch.tensor([], dtype=torch.float),
        "mask": torch.tensor([], dtype=torch.bool),
        "version": torch.tensor([0], dtype=torch.long),
    }


def build_episode_result(
    episode_output_tokens: List[int],
    episode_output_logprobs: List[float],
    episode_output_versions: List[int],
    episode_token_boundaries: List[Tuple[int, int]],
    episode_turn_rewards: List[float],
    final_reward: float,
) -> Dict[str, torch.Tensor]:
    """Build training tensors from episode data.
    
    Args:
        episode_output_tokens: All output tokens from the episode
        episode_output_logprobs: Log probs for each token
        episode_output_versions: Model version for each token
        episode_token_boundaries: (start, end) indices for each turn
        episode_turn_rewards: Process reward for each turn
        final_reward: Final episode reward (0 or 1)
        
    Returns:
        Dict of training tensors
    """
    if not episode_output_tokens:
        return create_empty_result()
    
    # Convert to tensors
    output_ids = torch.tensor(episode_output_tokens, dtype=torch.long)
    logprobs = torch.tensor(episode_output_logprobs, dtype=torch.float)
    
    # Create rewards tensor - assign final reward to all tokens
    # (GRPO uses sequence-level rewards, not per-token)
    rewards = torch.full((len(episode_output_tokens),), final_reward, dtype=torch.float)
    
    # Mask: all tokens are valid for training
    mask = torch.ones(len(episode_output_tokens), dtype=torch.bool)
    
    # Version
    version = torch.tensor(
        episode_output_versions if episode_output_versions else [0],
        dtype=torch.long
    )
    
    return {
        "input_ids": output_ids,
        "labels": output_ids.clone(),  # For causal LM
        "logprobs": logprobs,
        "ref_logprobs": torch.zeros_like(logprobs),  # Filled by trainer
        "rewards": rewards,
        "advantages": torch.zeros_like(rewards),  # Computed by trainer
        "mask": mask,
        "version": version,
    }


class CUAEnvRolloutWorkflow(RolloutWorkflow):
    """Multi-turn CUA Environment Rollout Workflow for AReaL training.

    This workflow handles the complete lifecycle of a CUA agent episode:
    1. Create GBox environment
    2. Take initial screenshot
    3. Build messages with system prompt, task, and screenshot
    4. Generate model response (tool_call) using AReaL InferenceEngine
    5. Execute action on GBox (via GBoxAgentCore)
    6. Take new screenshot, update messages
    7. Repeat until done or max_turns reached
    8. Compute final reward and return training tensors

    For GRPO training, multiple parallel episodes are run for each task
    to enable group-wise comparison.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        processor: Optional[AutoProcessor] = None,
        enable_thinking: bool = True,
        rollout_stat_scope: str = "rollout",
        dump_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CUA Environment Rollout Workflow.

        Args:
            reward_fn: Reward function for computing rewards
            gconfig: Generation hyperparameters
            tokenizer: Tokenizer or path to tokenizer
            processor: AutoProcessor for vision-language processing
            enable_thinking: Whether to enable thinking in chat template
            rollout_stat_scope: Scope name for stats tracking
            dump_dir: Directory for dumping rollout traces
            **kwargs: Additional configuration:
                - gbox_api_key: GBox API key
                - gbox_model: GBox model for coordinate generation
                - max_turns: Maximum turns per episode
                - context_window: Number of recent turns to keep
                - trace_dir: Directory for saving screenshots
        """
        # Extract custom parameters
        self.gbox_api_key = kwargs.pop("gbox_api_key", None) or os.getenv("GBOX_API_KEY", "")
        self.gbox_model = kwargs.pop("gbox_model", None) or os.getenv("GBOX_MODEL", "gbox-handy-1")
        self.max_turns = kwargs.pop("max_turns", None) or int(os.getenv("CUA_MAX_TURNS", "15"))
        self.context_window = kwargs.pop("context_window", None) or int(os.getenv("CUA_CONTEXT_WINDOW", "5"))
        self.trace_dir = kwargs.pop("trace_dir", None)
        
        # Internal counters
        self._group_counter = 0

        # Load tokenizer
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            self.tokenizer = load_hf_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Store processor
        self.processor = processor

        # Initialize gconfig with stop tokens
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir

        # Store reward function
        self.reward_fn = reward_fn

        # Get tools schema
        self.tools_schema = get_tools_schema()

        # Create directories
        if self.dump_dir and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        
        if self.trace_dir and not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir, exist_ok=True)

        if not self.gbox_api_key:
            logger.warning(
                "CUAEnvRolloutWorkflow initialized without GBOX_API_KEY; "
                "environment interaction will not work."
            )

        logger.info(f"[CUA Workflow] Initialized with max_turns={self.max_turns}, "
                   f"context_window={self.context_window}")

    def _build_task(self, data: Dict[str, Any]) -> CUATask:
        """Build CUATask from input data."""
        task_metadata = data.get("task_metadata", {})
        if isinstance(task_metadata, str):
            try:
                task_metadata = json.loads(task_metadata) if task_metadata else {}
            except json.JSONDecodeError:
                task_metadata = {}

        # Check if task object is directly provided
        if "task" in data and isinstance(data["task"], CUATask):
            return data["task"]

        task_id = (
            task_metadata.get("task_id") or
            data.get("task_id") or
            data.get("id") or
            f"task_{datetime.now().strftime('%H%M%S')}"
        )
        task_name = task_metadata.get("task_name") or task_metadata.get("name", task_id)
        task_description = (
            task_metadata.get("task_description") or
            task_metadata.get("description") or
            data.get("answer") or
            data.get("prompt") or
            data.get("messages", [{}])[0].get("content", "Complete the task")
        )

        difficulty_str = task_metadata.get("task_difficulty", task_metadata.get("difficulty", "medium"))
        try:
            difficulty = TaskDifficulty(difficulty_str)
        except ValueError:
            difficulty = TaskDifficulty.MEDIUM

        category_str = task_metadata.get("task_category", task_metadata.get("category", "system"))
        try:
            category = TaskCategory(category_str)
        except ValueError:
            category = TaskCategory.SYSTEM

        return CUATask(
            id=task_id,
            name=task_name,
            description=task_description,
            difficulty=difficulty,
            category=category,
            max_steps=task_metadata.get("max_steps", self.max_turns),
            validation_query=task_metadata.get("validation_query"),
            expected_result=task_metadata.get("expected_result"),
        )

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Run n_samples independent CUA episodes for the same task.

        This is the main entry point called by AReaL training loop.
        For GRPO, we run n_samples parallel episodes to enable group-wise comparison.
        
        Args:
            engine: AReaL inference engine
            data: Input data containing task information
            
        Returns:
            Concatenated training tensors from all episodes
        """
        if not self.gbox_api_key:
            logger.warning("No GBOX_API_KEY, returning empty result")
            return create_empty_result()

        # Get n_samples from gconfig (GRPO group size)
        n_samples = getattr(self.gconfig, 'n_samples', 1) or 1
        
        # Build task
        task = self._build_task(data)
        
        # Build training context
        try:
            version = engine.get_version()
        except Exception:
            version = 0
        
        task_metadata = data.get("task_metadata", data.get("metadata", {}))
        if isinstance(task_metadata, str):
            try:
                task_metadata = json.loads(task_metadata) if task_metadata else {}
            except json.JSONDecodeError:
                task_metadata = {}
        
        training_context = {
            "version": version,
            "epoch": task_metadata.get("epoch"),
            "global_step": task_metadata.get("global_step"),
            "group_id": task_metadata.get("group_id", self._group_counter),
        }
        
        self._group_counter += 1
        
        logger.info(f"[CUA Workflow] Running {n_samples} episodes for task: {task.id}")
        
        # Run episodes
        if n_samples == 1:
            return await self._run_single_episode(
                engine, task, sample_idx=0, training_context=training_context
            )
        
        # Run n_samples parallel episodes for GRPO
        episode_tasks = [
            self._run_single_episode(engine, task, sample_idx=i, training_context=training_context)
            for i in range(n_samples)
        ]
        
        episode_results = await asyncio.gather(*episode_tasks, return_exceptions=True)
        
        # Collect valid results
        valid_results = []
        for i, result in enumerate(episode_results):
            if isinstance(result, Exception):
                logger.warning(f"[CUA Workflow] Episode {i} failed: {result}")
                valid_results.append(create_empty_result())
            elif result is not None and "input_ids" in result and result["input_ids"].numel() > 0:
                valid_results.append(result)
            else:
                logger.warning(f"[CUA Workflow] Episode {i} returned empty result")
                valid_results.append(create_empty_result())
        
        if not valid_results:
            return create_empty_result()
        
        # Filter non-empty results for concatenation
        non_empty_results = [r for r in valid_results if r["input_ids"].numel() > 0]
        if not non_empty_results:
            return create_empty_result()
        
        return concat_padded_tensors(non_empty_results)

    async def _run_single_episode(
        self,
        engine: InferenceEngine,
        task: CUATask,
        sample_idx: int = 0,
        training_context: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run a single CUA episode.
        
        Args:
            engine: AReaL inference engine
            task: The CUA task to execute
            sample_idx: Index of this sample (for GRPO grouping)
            training_context: Training context info (version, epoch, etc.)
            
        Returns:
            Training tensors for this episode
        """
        training_context = training_context or {}
        
        # Build rollout ID
        rollout_id = self._build_rollout_id(task, sample_idx, training_context)
        
        # Initialize logger
        rollout_logger = RolloutLogger(
            rollout_id=rollout_id,
            task_id=task.id,
            task_description=task.description,
        )
        rollout_logger.set_model_info(
            model_name=str(getattr(self.tokenizer, 'name_or_path', 'unknown')),
            provider="areal",
        )
        
        # Episode-level data
        episode_output_tokens: List[int] = []
        episode_output_logprobs: List[float] = []
        episode_output_versions: List[int] = []
        episode_turn_rewards: List[float] = []
        episode_token_boundaries: List[Tuple[int, int]] = []
        turn_history: List[Dict] = []
        task_success = False
        version = training_context.get("version", 0)
        final_reward = 0.0
        
        # Create GBox client
        gbox = GBoxClient(
            api_key=self.gbox_api_key,
            model=self.gbox_model,
            box_type="android",
            timeout="60s",
            wait=True,
            expires_in="15m",
        )
        
        try:
            # Create box
            box_create_start = datetime.now()
            await gbox.create_box("android")
            box_create_time = (datetime.now() - box_create_start).total_seconds()
            
            rollout_logger.set_box_info(gbox.box_id or "", "android", box_create_time)
            logger.info(f"[Episode {sample_idx}] Box created: {gbox.box_id} ({box_create_time:.2f}s)")
            
            # Create agent core
            agent_core = GBoxAgentCore(
                gbox_client=gbox,
                max_turns=self.max_turns,
                context_window=self.context_window,
            )
            
            # Get engine version
            try:
                version = engine.get_version()
            except Exception:
                version = 0
            
            # Build system prompt
            system_prompt = create_system_prompt(
                task_description=task.description,
                max_turns=self.max_turns,
            )
            
            # Setup trace directory
            trace_path = None
            if self.trace_dir:
                trace_path = Path(self.trace_dir) / task.id / rollout_id
                trace_path.mkdir(parents=True, exist_ok=True)
            
            # Wait for box to be ready
            await asyncio.sleep(1.0)
            
            # Main interaction loop
            for turn in range(1, self.max_turns + 1):
                turn_data = await self._run_turn(
                    engine=engine,
                    gbox=gbox,
                    agent_core=agent_core,
                    task=task,
                    turn=turn,
                    turn_history=turn_history,
                    system_prompt=system_prompt,
                    rollout_logger=rollout_logger,
                    trace_path=trace_path,
                )
                
                if turn_data is None:
                    continue
                
                # Accumulate episode data
                turn_start_idx = len(episode_output_tokens)
                episode_output_tokens.extend(turn_data["output_tokens"])
                episode_output_logprobs.extend(turn_data["output_logprobs"])
                episode_output_versions.extend(turn_data["output_versions"])
                turn_end_idx = len(episode_output_tokens)
                
                episode_token_boundaries.append((turn_start_idx, turn_end_idx))
                episode_turn_rewards.append(turn_data["turn_reward"])
                
                # Update turn history
                turn_history.append(turn_data["history_entry"])
                
                # Check if done
                if turn_data["done"]:
                    task_success = turn_data["is_success"]
                    break
            
            # Compute final reward
            final_reward = await self._compute_final_reward(task, gbox, task_success)
            
            # Track stats
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=final_reward)
            
            # Set completion
            rollout_logger.set_final_result(
                success=task_success,
                message="Task completed" if task_success else "Task failed",
                total_turns=len(turn_history),
                max_turns=self.max_turns,
                reward=final_reward,
            )
            
            logger.info(f"[Episode {sample_idx}] Completed: success={task_success}, "
                       f"turns={len(turn_history)}, reward={final_reward}")
            
        except Exception as e:
            rollout_logger.set_completion(reward=0.0, success=False, error=str(e))
            logger.error(f"[Episode {sample_idx}] Failed: {e}", exc_info=True)
            if not episode_output_tokens:
                return create_empty_result()
            final_reward = 0.0
            
        finally:
            # Cleanup
            try:
                await gbox.terminate_box()
            except Exception as e:
                logger.warning(f"Failed to terminate box: {e}")
            
            # Print rollout summary
            if os.getenv("AREAL_VERBOSE_LOGGING", "").lower() in ("true", "1"):
                rollout_logger.print_log()
        
        # Build result
        if not episode_output_tokens:
            return create_empty_result()
        
        return build_episode_result(
            episode_output_tokens=episode_output_tokens,
            episode_output_logprobs=episode_output_logprobs,
            episode_output_versions=episode_output_versions,
            episode_token_boundaries=episode_token_boundaries,
            episode_turn_rewards=episode_turn_rewards,
            final_reward=final_reward,
        )

    def _build_rollout_id(
        self,
        task: CUATask,
        sample_idx: int,
        training_context: Dict,
    ) -> str:
        """Build comprehensive rollout ID with training context."""
        parts = [task.id]
        
        global_step = training_context.get("global_step")
        version = training_context.get("version", 0)
        if global_step is not None:
            parts.append(f"s{global_step:05d}")
        elif version > 0:
            parts.append(f"v{version:04d}")
        
        epoch = training_context.get("epoch")
        if epoch is not None:
            parts.append(f"e{epoch:03d}")
        
        group_id = training_context.get("group_id", 0)
        parts.append(f"g{group_id:02d}")
        parts.append(f"r{sample_idx:02d}")
        parts.append(datetime.now().strftime("%H%M%S_%f")[:-3])
        
        return "_".join(parts)

    async def _run_turn(
        self,
        engine: InferenceEngine,
        gbox: GBoxClient,
        agent_core: GBoxAgentCore,
        task: CUATask,
        turn: int,
        turn_history: List[Dict],
        system_prompt: str,
        rollout_logger: RolloutLogger,
        trace_path: Optional[Path],
    ) -> Optional[Dict]:
        """Run a single turn of the episode.
        
        Returns:
            Dictionary with turn data or None if turn failed
        """
        turn_log = rollout_logger.start_turn(turn, self.max_turns)
        
        # 1. Take screenshot
        await asyncio.sleep(0.3)
        screenshot_start = datetime.now()
        screenshot_bytes, screenshot_uri = await gbox.take_screenshot()
        screenshot_time = (datetime.now() - screenshot_start).total_seconds()
        
        turn_log.screenshot_time = screenshot_time
        turn_log.screenshot_bytes = len(screenshot_bytes)
        turn_log.screenshot_size_kb = len(screenshot_bytes) / 1024
        
        # Save screenshot
        if trace_path:
            try:
                screenshot_path = trace_path / f"turn_{turn}.png"
                screenshot_path.write_bytes(screenshot_bytes)
            except Exception as e:
                logger.warning(f"[Turn {turn}] Failed to save screenshot: {e}")
        
        screenshot_b64 = f"data:image/png;base64,{base64.b64encode(screenshot_bytes).decode()}"
        
        # Get PIL image
        try:
            pil_image = Image.open(io.BytesIO(screenshot_bytes))
            turn_log.screenshot_size = pil_image.size
        except Exception:
            pil_image = None
        
        # 2. Build messages for VLM
        messages = self._build_messages(
            system_prompt=system_prompt,
            turn_history=turn_history,
            current_screenshot_b64=screenshot_b64,
            turn=turn,
        )
        
        # 3. Process input for model
        if pil_image is None:
            pil_image = Image.open(io.BytesIO(screenshot_bytes))
        
        # Build messages for vLLM (without base64 images)
        messages_for_vllm = self._build_messages_for_vllm(
            system_prompt=system_prompt,
            turn_history=turn_history,
            turn=turn,
        )
        
        if self.processor is not None:
            try:
                processed = self.processor(
                    images=[pil_image],
                    text=messages_for_vllm,
                    padding=False,
                    return_tensors="pt",
                )
                input_ids = processed["input_ids"].tolist()[0]
            except TypeError:
                processed = self.processor.apply_chat_template(
                    messages_for_vllm,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                input_ids = processed["input_ids"].tolist()[0]
        else:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            input_ids = list(input_ids)
        
        turn_log.prompt_tokens = len(input_ids)
        
        # 4. Build ModelRequest for AReaL
        byte_images = image2base64([pil_image])
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            image_data=byte_images,
            vision_msg_vllm=[messages],
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        
        # 5. Generate response using AReaL engine
        vlm_start = datetime.now()
        resp = await engine.agenerate(req)
        vlm_time = (datetime.now() - vlm_start).total_seconds()
        
        turn_log.vlm_time = vlm_time
        
        # Decode response
        response_text = self.tokenizer.decode(resp.output_tokens)
        turn_log.raw_output = response_text
        turn_log.completion_tokens = resp.output_len
        turn_log.total_tokens = turn_log.prompt_tokens + turn_log.completion_tokens
        
        # Extract thinking and content
        thinking, content = GBoxAgentCore.extract_thinking_and_content(response_text)
        turn_log.thinking = thinking
        turn_log.content = content
        
        # 6. Parse tool call
        tool_call = GBoxAgentCore.parse_tool_call_from_response(response_text, turn)
        func_name = tool_call.get("function", {}).get("name", "")
        func_args_str = tool_call.get("function", {}).get("arguments", "{}")
        
        try:
            func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
        except json.JSONDecodeError:
            func_args = {}
        
        turn_log.tool_calls = [tool_call]
        turn_log.action_type = func_name
        turn_log.action_params = func_args
        
        # 7. Execute action
        action_start = datetime.now()
        try:
            action_result, done, is_success, action_timing = await agent_core.execute_tool_call(
                tool_call, screenshot_uri
            )
            turn_log.action_result = action_result
            turn_log.action_success = action_result.get("status") == "success" or is_success
            turn_log.action_timing = action_timing
        except Exception as e:
            action_result = {"status": "error", "message": str(e)}
            turn_log.action_error = str(e)
            done = False
            is_success = False
        
        action_time = (datetime.now() - action_start).total_seconds()
        turn_log.action_time = action_time
        
        if done:
            turn_log.task_completed = True
            turn_log.task_success = is_success
            turn_log.task_message = action_result.get("message", "")
        
        rollout_logger.end_turn()
        
        # Compute turn reward (placeholder for PRM)
        turn_reward = 0.0  # All credit assigned at episode end
        
        # Build history entry
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
            "tool_calls": [tool_call],
        }
        tool_response_msg = {
            "role": "tool",
            "tool_call_id": tool_call.get("id", f"call_{turn}"),
            "content": json.dumps(action_result),
        }
        
        return {
            "output_tokens": resp.output_tokens,
            "output_logprobs": resp.output_logprobs,
            "output_versions": resp.output_versions,
            "turn_reward": turn_reward,
            "done": done,
            "is_success": is_success,
            "history_entry": {
                "assistant": assistant_msg,
                "tool_response": tool_response_msg,
            },
        }

    def _build_messages(
        self,
        system_prompt: str,
        turn_history: List[Dict],
        current_screenshot_b64: str,
        turn: int,
    ) -> List[Dict]:
        """Build messages for model input with context window."""
        messages = []
        
        # 1. System prompt (always kept)
        messages.append({
            "role": "system",
            "content": system_prompt,
        })
        
        # 2. Recent history (last context_window turns)
        recent_history = turn_history[-self.context_window:]
        for hist_item in recent_history:
            if hist_item.get("assistant"):
                messages.append(hist_item["assistant"])
            if hist_item.get("tool_response"):
                messages.append(hist_item["tool_response"])
        
        # 3. Current turn user message with screenshot
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": current_screenshot_b64},
                },
                {
                    "type": "text",
                    "text": f"Turn {turn}/{self.max_turns}. Analyze the screenshot and take the next action to complete the task.",
                },
            ],
        })
        
        return messages

    def _build_messages_for_vllm(
        self,
        system_prompt: str,
        turn_history: List[Dict],
        turn: int,
    ) -> str:
        """Build text-only messages for vLLM (image handled separately)."""
        messages = []
        
        # System prompt
        messages.append({
            "role": "system",
            "content": system_prompt,
        })
        
        # Recent history
        recent_history = turn_history[-self.context_window:]
        for hist_item in recent_history:
            if hist_item.get("assistant"):
                assistant = hist_item["assistant"].copy()
                # Remove tool_calls for text serialization
                if "tool_calls" in assistant:
                    del assistant["tool_calls"]
                messages.append(assistant)
            if hist_item.get("tool_response"):
                messages.append(hist_item["tool_response"])
        
        # Current turn
        messages.append({
            "role": "user",
            "content": f"[Image] Turn {turn}/{self.max_turns}. Analyze the screenshot and take the next action to complete the task.",
        })
        
        # Apply chat template
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def _compute_final_reward(
        self,
        task: CUATask,
        gbox: GBoxClient,
        task_success: bool,
    ) -> float:
        """Compute final reward for the episode."""
        if task.validation_query and task.expected_result is not None:
            try:
                validated = await validate_task_completion(task, gbox)
                return 1.0 if validated else 0.0
            except Exception as e:
                logger.warning(f"Validation failed, using agent report: {e}")
        
        return 1.0 if task_success else 0.0


__all__ = [
    "CUAEnvRolloutWorkflow",
    "create_empty_result",
    "build_episode_result",
]

