"""Single episode runner for CUA Agent training.

This module handles the execution of a single CUA episode,
including environment interaction, action execution, and data collection.
"""

import asyncio
import base64
import io
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import colorama
import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.utils import stats_tracker
from areal.utils.image import image2base64
from areal.utils.perf_tracer import atrace_session_phase, trace_session

from gbox_cua.gbox_client import GBoxClient
from gbox_cua.agent import GBoxAgentCore, RolloutLogger
from gbox_cua.prompts import create_system_prompt

from cua_agent.tasks import CUATask
from cua_agent.reward import validate_task_completion
from cua_agent.workflow_utils import (
    build_messages,
    build_messages_for_vllm,
    create_empty_result,
    build_episode_result,
    print_rollout_summary,
)
from cua_agent.profiler import get_tracer, get_trace_url

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """Runs a single CUA episode with multi-turn environment interaction.
    
    This class encapsulates all the logic for running one complete episode,
    from environment setup to cleanup.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor | None,
        gconfig: GenerationHyperparameters,
        gbox_api_key: str,
        gbox_model: str,
        max_turns: int,
        context_window: int,
        dump_dir: str | None = None,
        trace_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
    ):
        """Initialize the episode runner.
        
        Args:
            tokenizer: Tokenizer for text processing
            processor: Optional processor for vision-language models
            gconfig: Generation hyperparameters
            gbox_api_key: API key for GBox service
            gbox_model: Model name for GBox coordinate generation
            max_turns: Maximum turns per episode
            context_window: Number of recent turns to keep in context
            dump_dir: Directory for trace dumps
            trace_dir: Directory for screenshot traces
            rollout_stat_scope: Scope name for stats tracking
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.gconfig = gconfig
        self.gbox_api_key = gbox_api_key
        self.gbox_model = gbox_model
        self.max_turns = max_turns
        self.context_window = context_window
        self.dump_dir = dump_dir
        self.trace_dir = trace_dir
        self.rollout_stat_scope = rollout_stat_scope

    async def run(
        self,
        engine: InferenceEngine,
        task: CUATask,
        sample_idx: int = 0,
        training_context: dict | None = None,
    ) -> dict[str, torch.Tensor]:
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
        
        # Get tracer for this episode
        tracer = get_tracer("episode_runner")
        
        # Start root span for the entire episode/rollout
        with tracer.start_as_current_span(
            f"rollout:{task.id}",
            attributes={
                "rollout.id": rollout_id,
                "task.id": task.id,
                "task.description": task.description[:200],
                "task.difficulty": task.difficulty.value if hasattr(task.difficulty, 'value') else str(task.difficulty),
                "sample_idx": sample_idx,
                "max_turns": self.max_turns,
                "training.version": training_context.get("version", 0),
                "training.epoch": training_context.get("epoch", 0),
                "training.global_step": training_context.get("global_step", 0),
            }
        ) as rollout_span:
            # Log trace URL for debugging
            trace_url = get_trace_url(rollout_span)
            if trace_url:
                logger.info(f"[Trace] Rollout {rollout_id}: {trace_url}")
            
            # Initialize logger
            rollout_logger = RolloutLogger(
                rollout_id=rollout_id,
                task_id=task.id,
                task_description=task.description,
            )
            rollout_logger.set_model_info(
                model_name=str(self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else "unknown"),
                provider="areal",
            )
            
            # Create GBox client
            gbox = GBoxClient(
                api_key=self.gbox_api_key,
                model=self.gbox_model,
                box_type="android",
                timeout="60s",
                wait=True,
                expires_in="15m",
            )
            
            # Create agent core
            agent_core = GBoxAgentCore(
                gbox_client=gbox,
                max_turns=self.max_turns,
                context_window=self.context_window,
            )
            
            # Episode-level data
            episode_output_tokens = []
            episode_output_logprobs = []
            episode_output_versions = []
            episode_turn_rewards = []
            episode_token_boundaries = []
            turn_history = []
            task_success = False
            version = 0
            final_reward = 0.0
            
            try:
                # Create box with tracing
                with tracer.start_as_current_span(
                    "gbox.create_box",
                    attributes={"box_type": "android"}
                ) as box_span:
                    box_create_start = datetime.now()
                    await gbox.create_box("android")
                    await asyncio.sleep(2.0)
                    box_create_time = (datetime.now() - box_create_start).total_seconds()
                    box_span.set_attribute("box.id", gbox.box_id or "")
                    box_span.set_attribute("duration_sec", box_create_time)
                
                rollout_logger.set_box_info(gbox.box_id or "", "android", box_create_time)
                rollout_span.set_attribute("box.id", gbox.box_id or "")
                
                # Get engine version
                version = engine.get_version()
                rollout_span.set_attribute("model.version", version)
                
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
                    await self._save_initial_screenshot(gbox, trace_path)
                
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
                        tracer=tracer,
                    )
                    
                    # Unpack turn data
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
                
                # Compute final reward with tracing
                with tracer.start_as_current_span("compute_reward") as reward_span:
                    final_reward = await self._compute_final_reward(task, gbox, task_success)
                    reward_span.set_attribute("reward", final_reward)
                    reward_span.set_attribute("task_success", task_success)
                
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
                
                # Set final attributes on rollout span
                rollout_span.set_attribute("result.success", task_success)
                rollout_span.set_attribute("result.reward", final_reward)
                rollout_span.set_attribute("result.turns_used", len(turn_history))
                rollout_span.set_attribute("result.total_tokens", sum(
                    len(t.get("output_tokens", [])) for t in [turn_history] if isinstance(t, dict)
                ) if turn_history else 0)
                
            except Exception as e:
                rollout_logger.set_completion(reward=0.0, success=False, error=str(e))
                logger.error(f"[EpisodeRunner] Task {task.id} failed: {e}", exc_info=True)
                rollout_span.set_attribute("error", str(e))
                rollout_span.set_attribute("result.success", False)
                if not episode_output_tokens:
                    print_rollout_summary(rollout_logger, task, self.dump_dir)
                    return create_empty_result()
                final_reward = 0.0
                
            finally:
                # Cleanup with tracing
                with tracer.start_as_current_span("gbox.terminate_box"):
                    try:
                        await gbox.terminate_box()
                    except Exception as e:
                        logger.warning(f"Failed to terminate box: {e}")
                
                print_rollout_summary(rollout_logger, task, self.dump_dir)
            
            # Dump traces
            if self.dump_dir and episode_output_tokens:
                await self._dump_trace(version, task.id, turn_history, final_reward)
            
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
        training_context: dict,
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

    async def _save_initial_screenshot(
        self,
        gbox: GBoxClient,
        trace_path: Path,
    ) -> None:
        """Save initial screenshot to trace directory."""
        try:
            initial_screenshot_bytes, _ = await gbox.take_screenshot()
            initial_screenshot_path = trace_path / "turn_0_initial.png"
            initial_screenshot_path.write_bytes(initial_screenshot_bytes)
            logger.info(f"Saved initial screenshot to {initial_screenshot_path}")
        except Exception as e:
            logger.warning(f"Failed to save initial screenshot: {e}")

    async def _run_turn(
        self,
        engine: InferenceEngine,
        gbox: GBoxClient,
        agent_core: GBoxAgentCore,
        task: CUATask,
        turn: int,
        turn_history: list[dict],
        system_prompt: str,
        rollout_logger: RolloutLogger,
        trace_path: Path | None,
        tracer=None,
    ) -> dict | None:
        """Run a single turn of the episode.
        
        Returns:
            Dictionary with turn data or None if turn failed
        """
        # Use provided tracer or get default
        if tracer is None:
            tracer = get_tracer("episode_runner")
        
        # Start turn span
        with tracer.start_as_current_span(
            f"turn:{turn}",
            attributes={
                "turn.number": turn,
                "turn.max_turns": self.max_turns,
            }
        ) as turn_span:
            turn_log = rollout_logger.start_turn(turn, self.max_turns)
            
            # 1. Take screenshot with tracing
            with tracer.start_as_current_span("screenshot.capture") as screenshot_span:
                await asyncio.sleep(0.3)
                screenshot_start = datetime.now()
                screenshot_bytes, screenshot_uri = await gbox.take_screenshot()
                screenshot_time = (datetime.now() - screenshot_start).total_seconds()
                
                turn_log.screenshot_time = screenshot_time
                turn_log.screenshot_bytes = len(screenshot_bytes)
                turn_log.screenshot_size_kb = len(screenshot_bytes) / 1024
                
                screenshot_span.set_attribute("duration_sec", screenshot_time)
                screenshot_span.set_attribute("size_bytes", len(screenshot_bytes))
                screenshot_span.set_attribute("size_kb", len(screenshot_bytes) / 1024)
            
            # Save screenshot
            if trace_path:
                try:
                    screenshot_path = trace_path / f"turn_{turn}.png"
                    screenshot_path.write_bytes(screenshot_bytes)
                except Exception as e:
                    logger.warning(f"[Turn {turn}] Failed to save screenshot: {e}")
            
            screenshot_b64 = f"data:image/png;base64,{base64.b64encode(screenshot_bytes).decode()}"
            
            # Get image
            try:
                pil_image = Image.open(io.BytesIO(screenshot_bytes))
                turn_log.screenshot_size = pil_image.size
                turn_span.set_attribute("screenshot.width", pil_image.size[0])
                turn_span.set_attribute("screenshot.height", pil_image.size[1])
            except Exception:
                pil_image = None
            
            # 2. Build messages with tracing
            with tracer.start_as_current_span("build_messages") as msg_span:
                messages = build_messages(
                    system_prompt=system_prompt,
                    turn_history=turn_history,
                    current_screenshot_b64=screenshot_b64,
                    turn=turn,
                    max_turns=self.max_turns,
                    context_window=self.context_window,
                )
                msg_span.set_attribute("message_count", len(messages))
                msg_span.set_attribute("history_turns", len(turn_history))
            
            # 3. Process input with tracing
            with tracer.start_as_current_span("process_input") as process_span:
                if pil_image is None:
                    pil_image = Image.open(io.BytesIO(screenshot_bytes))
                
                messages_for_vllm = build_messages_for_vllm(
                    system_prompt=system_prompt,
                    turn_history=turn_history,
                    turn=turn,
                    max_turns=self.max_turns,
                    context_window=self.context_window,
                )
                
                if self.processor is not None:
                    try:
                        processed = self.processor(
                            images=[pil_image],
                            text=messages_for_vllm,
                            padding=False,
                            return_tensors="pt",
                        )
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
                process_span.set_attribute("prompt_tokens", len(input_ids))
            
            # 4. Build ModelRequest
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
            
            # 5. Generate response with detailed tracing
            with tracer.start_as_current_span(
                "vlm.inference",
                attributes={"prompt_tokens": len(input_ids)}
            ) as vlm_span:
                vlm_start = datetime.now()
                async with atrace_session_phase("generate"):
                    resp = await engine.agenerate(req)
                vlm_time = (datetime.now() - vlm_start).total_seconds()
                
                turn_log.vlm_time = vlm_time
                vlm_span.set_attribute("duration_sec", vlm_time)
                vlm_span.set_attribute("output_tokens", resp.output_len)
                vlm_span.set_attribute("total_tokens", len(input_ids) + resp.output_len)
                vlm_span.set_attribute("tokens_per_sec", resp.output_len / vlm_time if vlm_time > 0 else 0)
            
            # Decode response
            response_text = self.tokenizer.decode(resp.output_tokens)
            turn_log.raw_output = response_text
            turn_log.completion_tokens = resp.output_len
            turn_log.total_tokens = turn_log.prompt_tokens + turn_log.completion_tokens
            
            turn_span.set_attribute("tokens.prompt", turn_log.prompt_tokens)
            turn_span.set_attribute("tokens.completion", turn_log.completion_tokens)
            turn_span.set_attribute("tokens.total", turn_log.total_tokens)
            
            # Extract thinking and content
            thinking, content = GBoxAgentCore.extract_thinking_and_content(response_text)
            turn_log.thinking = thinking
            turn_log.content = content
            
            # 6. Parse tool call
            with tracer.start_as_current_span("parse_tool_call") as parse_span:
                tool_call = GBoxAgentCore.parse_tool_call_from_response(response_text, turn)
                func_name = tool_call.get("function", {}).get("name", "")
                func_args_str = tool_call.get("function", {}).get("arguments", "{}")
                
                try:
                    func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                except json.JSONDecodeError:
                    func_args = {}
                
                parse_span.set_attribute("function_name", func_name)
                parse_span.set_attribute("has_arguments", bool(func_args))
            
            turn_log.tool_calls = [tool_call]
            turn_log.action_type = func_name
            turn_log.action_params = func_args
            turn_span.set_attribute("action.type", func_name)
            
            # 7. Execute action with detailed tracing
            with tracer.start_as_current_span(
                f"action.execute:{func_name}",
                attributes={
                    "action.type": func_name,
                    "action.params": json.dumps(func_args)[:500],
                }
            ) as action_span:
                action_start = datetime.now()
                await asyncio.sleep(0.5)
                try:
                    action_result, done, is_success = await agent_core.execute_tool_call(
                        tool_call, screenshot_uri
                    )
                    turn_log.action_result = action_result
                    turn_log.action_success = action_result.get("status") == "success" or is_success
                    action_span.set_attribute("action.success", turn_log.action_success)
                    action_span.set_attribute("action.status", action_result.get("status", "unknown"))
                except Exception as e:
                    action_result = {"status": "error", "message": str(e)}
                    turn_log.action_error = str(e)
                    done = False
                    is_success = False
                    action_span.set_attribute("action.success", False)
                    action_span.set_attribute("action.error", str(e))
                
                action_time = (datetime.now() - action_start).total_seconds()
                turn_log.action_time = action_time
                action_span.set_attribute("duration_sec", action_time)
            
            if done:
                turn_log.task_completed = True
                turn_log.task_success = is_success
                turn_log.task_message = action_result.get("message", "")
                turn_span.set_attribute("task.completed", True)
                turn_span.set_attribute("task.success", is_success)
            
            rollout_logger.end_turn()
            
            # Set turn duration
            turn_duration = turn_log.screenshot_time + vlm_time + action_time
            turn_span.set_attribute("duration_sec", turn_duration)
            
            # Compute turn reward
            turn_reward = self._compute_turn_reward(
                turn=turn,
                action_type=func_name,
                action_result=action_result,
                task=task,
                done=done,
                is_success=is_success,
            )
            
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

    def _compute_turn_reward(
        self,
        turn: int,
        action_type: str,
        action_result: dict,
        task: CUATask,
        done: bool,
        is_success: bool,
    ) -> float:
        """Compute process reward for a single turn.
        
        Placeholder for Process Reward Model (PRM) integration.
        Currently returns 0 for intermediate turns.
        
        Args:
            turn: Current turn number
            action_type: Type of action taken
            action_result: Result of action execution
            task: The task being executed
            done: Whether the episode is complete
            is_success: Whether the task was completed successfully
            
        Returns:
            Process reward for this turn
        """
        # All credit assigned at episode end via final_reward
        if not done:
            return 0.0
        return 0.0

    @trace_session("reward")
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

    async def _dump_trace(
        self,
        version: int,
        task_id: str,
        turn_history: list[dict],
        final_reward: float,
    ) -> None:
        """Dump rollout trace to file."""
        dump_path = f"{self.dump_dir}/{version}"
        await aiofiles.os.makedirs(dump_path, exist_ok=True)
        
        file_path = f"{dump_path}/{task_id}.txt"
        async with aiofiles.open(file_path, "a") as f:
            for i, turn in enumerate(turn_history):
                assistant = turn.get("assistant", {})
                response = assistant.get("content", "")
                info = "\n".join([
                    f"idx: {i + 1} / {len(turn_history)}, reward: {final_reward if i == len(turn_history) - 1 else 0.0}",
                    f"response: {colorama.Fore.YELLOW}{response[:500]}{colorama.Style.RESET_ALL}",
                ])
                await f.write(info + "\n\n")


__all__ = ["EpisodeRunner"]

