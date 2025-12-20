"""AReaL Workflow for CUA Agent training.

This module implements CUAEnvRolloutWorkflow for multi-turn CUA agent training.
It follows the VisionRLVRWorkflow pattern from AReaL for handling vision inputs.

Key Design:
1. Uses gbox_cua.agent components for core agent logic (GBoxAgentCore, RolloutLogger)
2. Focuses only on AReaL-specific integration:
   - RolloutWorkflow interface
   - ModelRequest/ModelResponse handling
   - Training tensor construction
   - Reward computation

Architecture:
- CUAEnvRolloutWorkflow: Main workflow class (handles GRPO parallel episodes)
- EpisodeRunner: Runs single episodes (in episode_runner.py)
- workflow_utils: Message building and logging utilities

References:
- https://github.com/inclusionAI/AReaL/blob/main/areal/workflow/vision_rlvr.py
- https://github.com/inclusionAI/AReaL/blob/main/examples/search-agent/
"""

import asyncio
import atexit
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable

import torch
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors

from gbox_cua.tools import get_tools_schema

from cua_agent.tasks import CUATask, TaskCategory, TaskDifficulty
from cua_agent.episode_runner import EpisodeRunner
from cua_agent.workflow_utils import create_empty_result
from cua_agent.profiler import init_tracing, shutdown_tracing

logger = logging.getLogger(__name__)

# Initialize tracing at module load if ENABLE_TRACING is set
_tracing_initialized = init_tracing(
    service_name=os.getenv("OTEL_SERVICE_NAME", "cua-agent"),
    service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
    deployment_environment=os.getenv("DEPLOYMENT_ENVIRONMENT", "development"),
)

# Register shutdown handler
if _tracing_initialized:
    atexit.register(shutdown_tracing)


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
        processor: AutoProcessor | None = None,
        enable_thinking: bool = True,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
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

        if not self.gbox_api_key:
            logger.warning(
                "CUAEnvRolloutWorkflow initialized without GBOX_API_KEY; "
                "environment interaction will not work."
            )

        # Create episode runner
        self.episode_runner = EpisodeRunner(
            tokenizer=self.tokenizer,
            processor=self.processor,
            gconfig=self.gconfig,
            gbox_api_key=self.gbox_api_key,
            gbox_model=self.gbox_model,
            max_turns=self.max_turns,
            context_window=self.context_window,
            dump_dir=self.dump_dir,
            trace_dir=self.trace_dir,
            rollout_stat_scope=self.rollout_stat_scope,
        )

    def _build_task(self, data: dict[str, Any]) -> CUATask:
        """Build CUATask from input data."""
        task_metadata = data.get("task_metadata", {})
        if isinstance(task_metadata, str):
            try:
                task_metadata = json.loads(task_metadata) if task_metadata else {}
            except json.JSONDecodeError:
                task_metadata = {}

        task_id = (
            task_metadata.get("task_id") or
            data.get("task_id") or
            f"task_{datetime.now().strftime('%H%M%S')}"
        )
        task_name = task_metadata.get("task_name", task_id)
        task_description = (
            task_metadata.get("task_description") or
            data.get("answer") or
            data.get("messages", [{}])[0].get("content", "Complete the task")
        )

        difficulty_str = task_metadata.get("task_difficulty", "medium")
        try:
            difficulty = TaskDifficulty(difficulty_str)
        except ValueError:
            difficulty = TaskDifficulty.MEDIUM

        category_str = task_metadata.get("task_category", "system")
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
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
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

        # Get n_samples from gconfig
        n_samples = getattr(self.gconfig, 'n_samples', 1) or 1
        
        # Build task
        task = self._build_task(data)
        
        # Build training context
        try:
            version = engine.get_version()
        except Exception:
            version = 0
        
        task_metadata = data.get("task_metadata", {})
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
        
        # Run episodes
        if n_samples == 1:
            return await self.episode_runner.run(
                engine, task, sample_idx=0, training_context=training_context
            )
        
        # Run n_samples parallel episodes
        logger.info(f"[GRPO] Running {n_samples} parallel episodes for task {task.id}")
        
        episode_tasks = [
            self.episode_runner.run(engine, task, sample_idx=i, training_context=training_context)
            for i in range(n_samples)
        ]
        
        episode_results = await asyncio.gather(*episode_tasks, return_exceptions=True)
        
        # Collect results
        valid_results = []
        for i, result in enumerate(episode_results):
            if isinstance(result, Exception):
                logger.warning(f"[GRPO] Episode {i} for task {task.id} failed: {result}")
                valid_results.append(create_empty_result())
            elif result is not None and "input_ids" in result and result["input_ids"].numel() > 0:
                valid_results.append(result)
            else:
                logger.warning(f"[GRPO] Episode {i} for task {task.id} returned empty result")
                valid_results.append(create_empty_result())
        
        if not valid_results:
            return create_empty_result()
        
        non_empty_results = [r for r in valid_results if r["input_ids"].numel() > 0]
        if not non_empty_results:
            return create_empty_result()
        
        return concat_padded_tensors(non_empty_results)


__all__ = ["CUAEnvRolloutWorkflow"]
