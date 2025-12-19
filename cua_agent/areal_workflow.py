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

References:
- https://github.com/inclusionAI/AReaL/blob/main/areal/workflow/vision_rlvr.py
- https://github.com/inclusionAI/AReaL/blob/main/examples/search-agent/
"""

import asyncio
import base64
import io
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

import aiofiles
import aiofiles.os
import colorama
import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

# Import reusable components from gbox_cua
from gbox_cua.gbox_client import GBoxClient
from gbox_cua.agent import (
    GBoxAgentCore,
    RolloutLogger,
    TurnLog,
)
from gbox_cua.prompts import create_system_prompt
from gbox_cua.tools import get_tools_schema

from cua_agent.tasks import CUATask, TaskCategory, TaskDifficulty
from cua_agent.reward import validate_task_completion

logger = logging.getLogger(__name__)


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

    Uses gbox_cua.agent components for:
    - GBoxAgentCore: Action execution, tool call parsing
    - RolloutLogger: Human-friendly logging
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
        """
        # Extract custom parameters
        self.gbox_api_key = kwargs.pop("gbox_api_key", None) or os.getenv("GBOX_API_KEY", "")
        self.gbox_model = kwargs.pop("gbox_model", None) or os.getenv("GBOX_MODEL", "gbox-handy-1")
        self.max_turns = kwargs.pop("max_turns", None) or int(os.getenv("CUA_MAX_TURNS", "15"))
        self.context_window = kwargs.pop("context_window", None) or int(os.getenv("CUA_CONTEXT_WINDOW", "5"))
        self.trace_dir = kwargs.pop("trace_dir", None)

        # Load tokenizer if path provided
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            self.tokenizer = load_hf_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Store processor for vision-language processing
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

        # Create dump directory if needed
        if self.dump_dir and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        if not self.gbox_api_key:
            logger.warning(
                "CUAEnvRolloutWorkflow initialized without GBOX_API_KEY; "
                "environment interaction will not work."
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

    def _build_messages(
        self,
        system_prompt: str,
        turn_history: list[dict],
        current_screenshot_b64: str,
        turn: int,
    ) -> list[dict]:
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
        turn_history: list[dict],
        turn: int,
    ) -> list[dict]:
        """Build messages for vLLM/processor input (Qwen-VL compatible)."""
        messages = []

        # 1. System prompt
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

        # 2. Recent history (text only for history)
        recent_history = turn_history[-self.context_window:]
        for hist_item in recent_history:
            if hist_item.get("assistant"):
                assistant = hist_item["assistant"]
                content = assistant.get("content", "")
                if content:
                    messages.append({
                        "role": "assistant",
                        "content": content,
                    })
            if hist_item.get("tool_response"):
                tool_resp = hist_item["tool_response"]
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {tool_resp.get('content', '')}",
                })

        # 3. Current turn user message with image placeholder
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},  # Qwen-VL format
                {
                    "type": "text",
                    "text": f"Turn {turn}/{self.max_turns}. Analyze the screenshot and take the next action to complete the task.",
                },
            ],
        })

        return messages

    @trace_session("reward")
    async def _compute_final_reward(
        self,
        task: CUATask,
        gbox: GBoxClient,
        task_success: bool,
    ) -> float:
        """Compute final reward for the episode."""
        # Try validation if defined
        if task.validation_query and task.expected_result is not None:
            try:
                validated = await validate_task_completion(task, gbox)
                return 1.0 if validated else 0.0
            except Exception as e:
                logger.warning(f"Validation failed, using agent report: {e}")

        # Use agent-reported success
        return 1.0 if task_success else 0.0

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Run a complete CUA episode with multi-turn environment interaction.

        This is the main entry point called by AReaL training loop.
        """
        if not self.gbox_api_key:
            logger.warning("No GBOX_API_KEY, returning empty result")
            return self._empty_result()

        # Build task
        task = self._build_task(data)
        rollout_id = f"{task.id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Initialize per-rollout logger (using gbox_cua.agent.RolloutLogger)
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

        # Create agent core for action execution
        agent_core = GBoxAgentCore(
            gbox_client=gbox,
            max_turns=self.max_turns,
            context_window=self.context_window,
        )

        # Track all responses and rewards for training
        all_results = []
        turn_history = []
        task_success = False
        version = 0

        try:
            # Create box
            box_create_start = datetime.now()
            await gbox.create_box("android")
            await asyncio.sleep(2.0)  # Wait for box to be ready
            box_create_time = (datetime.now() - box_create_start).total_seconds()
            rollout_logger.set_box_info(gbox.box_id or "", "android", box_create_time)

            # Get engine version for tracking
            version = engine.get_version()

            # Build system prompt
            system_prompt = create_system_prompt(
                task_description=task.description,
                max_turns=self.max_turns,
            )

            # Main interaction loop
            for turn in range(1, self.max_turns + 1):
                turn_log = rollout_logger.start_turn(turn, self.max_turns)

                # 1. Take screenshot
                await asyncio.sleep(0.3)
                screenshot_start = datetime.now()
                screenshot_bytes, screenshot_uri = await gbox.take_screenshot()
                turn_log.screenshot_time = (datetime.now() - screenshot_start).total_seconds()
                turn_log.screenshot_bytes = len(screenshot_bytes)
                turn_log.screenshot_size_kb = len(screenshot_bytes) / 1024

                screenshot_b64 = f"data:image/png;base64,{base64.b64encode(screenshot_bytes).decode()}"

                # Get image size
                try:
                    pil_image = Image.open(io.BytesIO(screenshot_bytes))
                    turn_log.screenshot_size = pil_image.size
                except Exception:
                    pil_image = None

                # 2. Build messages
                messages = self._build_messages(
                    system_prompt=system_prompt,
                    turn_history=turn_history,
                    current_screenshot_b64=screenshot_b64,
                    turn=turn,
                )

                # 3. Process with processor for vision-language input
                if pil_image is None:
                    pil_image = Image.open(io.BytesIO(screenshot_bytes))

                messages_for_vllm = self._build_messages_for_vllm(
                    system_prompt=system_prompt,
                    turn_history=turn_history,
                    turn=turn,
                )

                # Process input
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
                    multi_modal_data = {
                        "pixel_values": processed.get("pixel_values"),
                    }
                    if "image_grid_thw" in processed:
                        multi_modal_data["image_grid_thw"] = processed["image_grid_thw"]
                else:
                    input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    input_ids = list(input_ids)
                    multi_modal_data = None

                turn_log.prompt_tokens = len(input_ids)

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

                # 5. Generate response
                vlm_start = datetime.now()
                async with atrace_session_phase("generate"):
                    resp = await engine.agenerate(req)
                turn_log.vlm_time = (datetime.now() - vlm_start).total_seconds()

                # Decode response
                response_text = self.tokenizer.decode(resp.output_tokens)
                turn_log.raw_output = response_text
                turn_log.completion_tokens = resp.output_len
                turn_log.total_tokens = turn_log.prompt_tokens + turn_log.completion_tokens

                # Extract thinking and content
                thinking, content = GBoxAgentCore.extract_thinking_and_content(response_text)
                turn_log.thinking = thinking
                turn_log.content = content

                # 6. Parse tool call (using GBoxAgentCore)
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

                # 7. Execute action (using GBoxAgentCore)
                action_start = datetime.now()
                await asyncio.sleep(0.5)
                try:
                    action_result, done, is_success = await agent_core.execute_tool_call(
                        tool_call, screenshot_uri
                    )
                    turn_log.action_result = action_result
                    turn_log.action_success = action_result.get("status") == "success" or is_success
                except Exception as e:
                    action_result = {"status": "error", "message": str(e)}
                    turn_log.action_error = str(e)
                    done = False
                    is_success = False

                turn_log.action_time = (datetime.now() - action_start).total_seconds()
                
                # Update task completion in turn log
                if done:
                    turn_log.task_completed = True
                    turn_log.task_success = is_success
                    turn_log.task_message = action_result.get("message", "")

                rollout_logger.end_turn()

                # 8. Build result for this turn
                seq = resp.input_tokens + resp.output_tokens
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions

                result = {
                    "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
                    "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
                    "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
                    "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
                    "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                    "rewards": torch.tensor([0.0], dtype=torch.float32),
                }

                if multi_modal_data and multi_modal_data.get("pixel_values") is not None:
                    result["multi_modal_input"] = [multi_modal_data]

                all_results.append(result)

                # 9. Update turn history for context window
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
                turn_history.append({
                    "assistant": assistant_msg,
                    "tool_response": tool_response_msg,
                })

                # 10. Check if done
                if done:
                    task_success = is_success
                    break

            # Compute final reward
            final_reward = await self._compute_final_reward(task, gbox, task_success)

            # Update last result with reward
            if all_results:
                all_results[-1]["rewards"] = torch.tensor([final_reward], dtype=torch.float32)

            # Track stats
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=final_reward)

            # Set completion in logger
            rollout_logger.set_final_result(
                success=task_success,
                message="Task completed" if task_success else "Task failed",
                total_turns=len(turn_history),
                max_turns=self.max_turns,
                reward=final_reward,
            )

        except Exception as e:
            rollout_logger.set_completion(reward=0.0, success=False, error=str(e))
            logger.error(f"[CUAEnvRollout] Task {task.id} failed: {e}", exc_info=True)
            if not all_results:
                # Print rollout log directly to stdout for visibility
                self._print_rollout_summary(rollout_logger, task)
                return self._empty_result()

        finally:
            # Cleanup
            try:
                await gbox.terminate_box()
            except Exception as e:
                logger.warning(f"Failed to terminate box: {e}")

            # Print rollout log directly to stdout for visibility in Docker
            self._print_rollout_summary(rollout_logger, task)

        # Dump traces if configured
        if self.dump_dir and all_results:
            await self._dump_trace(
                version=version,
                task_id=task.id,
                turn_history=turn_history,
                final_reward=final_reward if 'final_reward' in dir() else 0.0,
            )

        # Concatenate all results
        if not all_results:
            return self._empty_result()

        return concat_padded_tensors(all_results)

    def _print_rollout_summary(self, rollout_logger: RolloutLogger, task: CUATask):
        """Print detailed rollout execution log to stdout.
        
        Uses print() directly to ensure visibility in Docker containers.
        Prints both a summary line and detailed per-turn execution records.
        """
        duration = rollout_logger.end_time - rollout_logger.start_time if rollout_logger.end_time else 0
        
        # Status indicators
        if rollout_logger.final_success:
            status = "✅ SUCCESS"
            status_color = "\033[92m"  # Green
        else:
            status = "❌ FAILED"
            status_color = "\033[91m"  # Red
        reset = "\033[0m"
        
        # Print summary line
        summary_parts = [
            f"[Rollout] {task.id}",
            f"{status_color}{status}{reset}",
            f"Turns: {rollout_logger.total_turns}/{rollout_logger.max_turns}",
            f"Reward: {rollout_logger.final_reward:.2f}",
            f"Time: {duration:.1f}s",
            f"Tokens: {rollout_logger.total_model_tokens}",
        ]
        print(" | ".join(summary_parts), flush=True)
        print("", flush=True)  # Empty line for readability
        
        # Try to get detailed log using format_log() method
        try:
            detailed_log = rollout_logger.format_log()
            if detailed_log:
                # Print detailed log with indentation
                for line in detailed_log.split('\n'):
                    if line.strip():  # Skip empty lines
                        print(f"  {line}", flush=True)
                print("", flush=True)  # Empty line after detailed log
        except (AttributeError, TypeError):
            # If format_log() doesn't exist or fails, try manual formatting
            try:
                # Try to access turns attribute directly
                if hasattr(rollout_logger, 'turns') and rollout_logger.turns:
                    for turn_idx, turn_log in enumerate(rollout_logger.turns, 1):
                        self._print_turn_details(turn_log, turn_idx)
                    print("", flush=True)
            except (AttributeError, TypeError):
                # Fallback: try to get formatted log from format method
                try:
                    if hasattr(rollout_logger, 'format'):
                        formatted = rollout_logger.format()
                        if formatted:
                            for line in formatted.split('\n'):
                                if line.strip():
                                    print(f"  {line}", flush=True)
                            print("", flush=True)
                except (AttributeError, TypeError):
                    pass  # If all methods fail, just print summary
        
        # Print errors if any
        if hasattr(rollout_logger, 'errors') and rollout_logger.errors:
            print(f"  Errors ({len(rollout_logger.errors)}):", flush=True)
            for error in rollout_logger.errors[-5:]:  # Last 5 errors
                error_msg = str(error)[:200]  # Truncate long errors
                print(f"    └─ {error_msg}", flush=True)
            print("", flush=True)
        
        # Optionally save full log to file if dump_dir is configured
        if self.dump_dir:
            try:
                log_path = os.path.join(self.dump_dir, "rollout_logs", f"{task.id}.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "w") as f:
                    try:
                        f.write(rollout_logger.format_log())
                    except (AttributeError, TypeError):
                        # Fallback: write what we can
                        f.write(f"Task: {task.id}\n")
                        f.write(f"Success: {rollout_logger.final_success}\n")
                        f.write(f"Reward: {rollout_logger.final_reward}\n")
                        f.write(f"Turns: {rollout_logger.total_turns}/{rollout_logger.max_turns}\n")
            except Exception:
                pass  # Silently ignore log file errors
    
    def _print_turn_details(self, turn_log, turn_idx: int):
        """Print detailed information for a single turn."""
        print(f"  Turn {turn_idx}:", flush=True)
        
        # Action information
        if hasattr(turn_log, 'action_type') and turn_log.action_type:
            action_params = ""
            if hasattr(turn_log, 'action_params') and turn_log.action_params:
                # Format parameters nicely
                params_str = json.dumps(turn_log.action_params, ensure_ascii=False)
                if len(params_str) > 100:
                    params_str = params_str[:97] + "..."
                action_params = f" ({params_str})"
            print(f"    Action: {turn_log.action_type}{action_params}", flush=True)
        
        # Action result
        if hasattr(turn_log, 'action_result') and turn_log.action_result:
            result_status = "unknown"
            result_msg = ""
            if isinstance(turn_log.action_result, dict):
                result_status = turn_log.action_result.get("status", "unknown")
                result_msg = turn_log.action_result.get("message", "")
            else:
                result_status = str(turn_log.action_result)[:50]
            
            success_marker = "✓" if hasattr(turn_log, 'action_success') and turn_log.action_success else "✗"
            print(f"    Result: {success_marker} {result_status}", flush=True)
            if result_msg and len(result_msg) <= 150:
                print(f"      {result_msg}", flush=True)
        
        # Thinking and content (if available)
        if hasattr(turn_log, 'thinking') and turn_log.thinking:
            thinking_preview = turn_log.thinking[:150] + "..." if len(turn_log.thinking) > 150 else turn_log.thinking
            print(f"    Thinking: {thinking_preview}", flush=True)
        
        # Token usage
        if hasattr(turn_log, 'total_tokens'):
            tokens_info = f"Tokens: {turn_log.total_tokens}"
            if hasattr(turn_log, 'prompt_tokens') and hasattr(turn_log, 'completion_tokens'):
                tokens_info += f" (prompt: {turn_log.prompt_tokens}, completion: {turn_log.completion_tokens})"
            print(f"    {tokens_info}", flush=True)
        
        # Timing information
        timing_parts = []
        if hasattr(turn_log, 'vlm_time') and turn_log.vlm_time:
            timing_parts.append(f"VLM: {turn_log.vlm_time:.2f}s")
        if hasattr(turn_log, 'action_time') and turn_log.action_time:
            timing_parts.append(f"Action: {turn_log.action_time:.2f}s")
        if hasattr(turn_log, 'screenshot_time') and turn_log.screenshot_time:
            timing_parts.append(f"Screenshot: {turn_log.screenshot_time:.2f}s")
        if timing_parts:
            print(f"    Timing: {' | '.join(timing_parts)}", flush=True)
        
        # Task completion (if this turn completed the task)
        if hasattr(turn_log, 'task_completed') and turn_log.task_completed:
            success_msg = "successfully" if (hasattr(turn_log, 'task_success') and turn_log.task_success) else "unsuccessfully"
            print(f"    Task completed {success_msg}", flush=True)
            if hasattr(turn_log, 'task_message') and turn_log.task_message:
                print(f"      {turn_log.task_message}", flush=True)
        
        # Errors
        if hasattr(turn_log, 'action_error') and turn_log.action_error:
            print(f"    Error: {turn_log.action_error[:150]}", flush=True)
        
        print("", flush=True)  # Empty line between turns

    def _empty_result(self) -> dict[str, torch.Tensor]:
        """Return empty result tensor dict."""
        return {
            "input_ids": torch.tensor([[]], dtype=torch.int32),
            "loss_mask": torch.tensor([[]], dtype=torch.int32),
            "logprobs": torch.tensor([[]], dtype=torch.float32),
            "versions": torch.tensor([[]], dtype=torch.int32),
            "attention_mask": torch.tensor([[]], dtype=torch.bool),
            "rewards": torch.tensor([0.0], dtype=torch.float32),
        }

    async def _dump_trace(
        self,
        version: int,
        task_id: str,
        turn_history: list[dict],
        final_reward: float,
    ):
        """Dump rollout trace to file."""
        dump_path = os.path.join(self.dump_dir, str(version))
        await aiofiles.os.makedirs(dump_path, exist_ok=True)

        file_path = os.path.join(dump_path, f"{task_id}.txt")
        async with aiofiles.open(file_path, "a") as f:
            for i, turn in enumerate(turn_history):
                assistant = turn.get("assistant", {})
                response = assistant.get("content", "")
                info = "\n".join([
                    f"idx: {i + 1} / {len(turn_history)}, reward: {final_reward if i == len(turn_history) - 1 else 0.0}",
                    f"response: {colorama.Fore.YELLOW}{response[:500]}{colorama.Style.RESET_ALL}",
                ])
                await f.write(info + "\n\n")


__all__ = ["CUAEnvRolloutWorkflow"]
