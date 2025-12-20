"""Utility functions for CUA Agent AReaL workflow.

This module contains helper functions for:
- Message building for VLM input
- Rollout logging and printing
- Result tensor construction
"""

import json
import os
from typing import Any

import torch

from cua_agent.tasks import CUATask


def build_messages(
    system_prompt: str,
    turn_history: list[dict],
    current_screenshot_b64: str,
    turn: int,
    max_turns: int,
    context_window: int,
) -> list[dict]:
    """Build messages for model input with context window.
    
    Args:
        system_prompt: System prompt for the agent
        turn_history: List of previous turn messages
        current_screenshot_b64: Base64 encoded current screenshot
        turn: Current turn number
        max_turns: Maximum turns allowed
        context_window: Number of recent turns to keep in context
        
    Returns:
        List of messages for VLM input
    """
    messages = []

    # 1. System prompt (always kept)
    messages.append({
        "role": "system",
        "content": system_prompt,
    })

    # 2. Recent history (last context_window turns)
    recent_history = turn_history[-context_window:]
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
                "text": f"Turn {turn}/{max_turns}. Analyze the screenshot and take the next action to complete the task.",
            },
        ],
    })

    return messages


def build_messages_for_vllm(
    system_prompt: str,
    turn_history: list[dict],
    turn: int,
    max_turns: int,
    context_window: int,
) -> list[dict]:
    """Build messages for vLLM/processor input (Qwen-VL compatible).
    
    Args:
        system_prompt: System prompt for the agent
        turn_history: List of previous turn messages
        turn: Current turn number
        max_turns: Maximum turns allowed
        context_window: Number of recent turns to keep in context
        
    Returns:
        List of messages for vLLM input
    """
    messages = []

    # 1. System prompt
    messages.append({
        "role": "system",
        "content": system_prompt,
    })

    # 2. Recent history (text only for history)
    recent_history = turn_history[-context_window:]
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
                "text": f"Turn {turn}/{max_turns}. Analyze the screenshot and take the next action to complete the task.",
            },
        ],
    })

    return messages


def create_empty_result() -> dict[str, torch.Tensor]:
    """Return empty result tensor dict."""
    return {
        "input_ids": torch.tensor([[]], dtype=torch.int32),
        "loss_mask": torch.tensor([[]], dtype=torch.int32),
        "logprobs": torch.tensor([[]], dtype=torch.float32),
        "versions": torch.tensor([[]], dtype=torch.int32),
        "attention_mask": torch.tensor([[]], dtype=torch.bool),
        "rewards": torch.tensor([0.0], dtype=torch.float32),
    }


def build_episode_result(
    episode_output_tokens: list[int],
    episode_output_logprobs: list[float],
    episode_output_versions: list[int],
    episode_token_boundaries: list[tuple[int, int]],
    episode_turn_rewards: list[float],
    final_reward: float,
) -> dict[str, torch.Tensor]:
    """Build episode-level result tensors for GRPO training.
    
    Args:
        episode_output_tokens: All output tokens across turns
        episode_output_logprobs: Corresponding log probabilities
        episode_output_versions: Version tracking for each token
        episode_token_boundaries: List of (start, end) indices for each turn
        episode_turn_rewards: Process reward for each turn
        final_reward: Final outcome reward
        
    Returns:
        Dictionary of tensors for training
    """
    episode_len = len(episode_output_tokens)
    
    # Build per-token rewards with process reward support
    token_rewards = torch.zeros(episode_len, dtype=torch.float32)
    
    for turn_idx, (start_idx, end_idx) in enumerate(episode_token_boundaries):
        turn_reward = episode_turn_rewards[turn_idx] if turn_idx < len(episode_turn_rewards) else 0.0
        
        # Assign turn's process reward to last token of turn
        if end_idx > start_idx:
            token_rewards[end_idx - 1] = turn_reward
    
    # Add final outcome reward to the last token
    if episode_len > 0:
        token_rewards[-1] += final_reward
    
    # Total episode reward for GRPO advantage computation
    total_episode_reward = sum(episode_turn_rewards) + final_reward
    
    return {
        "input_ids": torch.tensor(episode_output_tokens, dtype=torch.int32).unsqueeze(0),
        "loss_mask": torch.ones(episode_len, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(episode_output_logprobs, dtype=torch.float32).unsqueeze(0),
        "versions": torch.tensor(episode_output_versions, dtype=torch.int32).unsqueeze(0),
        "attention_mask": torch.ones(episode_len, dtype=torch.bool).unsqueeze(0),
        "token_rewards": token_rewards.unsqueeze(0),
        "rewards": torch.tensor([total_episode_reward], dtype=torch.float32),
    }


def print_rollout_summary(
    rollout_logger: Any,
    task: CUATask,
    dump_dir: str | None = None,
) -> None:
    """Print detailed rollout execution log to stdout.
    
    Args:
        rollout_logger: RolloutLogger instance with execution data
        task: The task that was executed
        dump_dir: Optional directory to save log files
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
    print("", flush=True)
    
    # Try to get detailed log
    try:
        detailed_log = rollout_logger.format_log()
        if detailed_log:
            for line in detailed_log.split('\n'):
                if line.strip():
                    print(f"  {line}", flush=True)
            print("", flush=True)
    except (AttributeError, TypeError):
        # Fallback to manual formatting
        try:
            if hasattr(rollout_logger, 'turns') and rollout_logger.turns:
                for turn_idx, turn_log in enumerate(rollout_logger.turns, 1):
                    _print_turn_details(turn_log, turn_idx)
                print("", flush=True)
        except (AttributeError, TypeError):
            pass
    
    # Print errors if any
    if hasattr(rollout_logger, 'errors') and rollout_logger.errors:
        print(f"  Errors ({len(rollout_logger.errors)}):", flush=True)
        for error in rollout_logger.errors[-5:]:
            error_msg = str(error)[:200]
            print(f"    └─ {error_msg}", flush=True)
        print("", flush=True)
    
    # Save full log to file if dump_dir is configured
    if dump_dir:
        try:
            log_path = os.path.join(dump_dir, "rollout_logs", f"{task.id}.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as f:
                try:
                    f.write(rollout_logger.format_log())
                except (AttributeError, TypeError):
                    f.write(f"Task: {task.id}\n")
                    f.write(f"Success: {rollout_logger.final_success}\n")
                    f.write(f"Reward: {rollout_logger.final_reward}\n")
                    f.write(f"Turns: {rollout_logger.total_turns}/{rollout_logger.max_turns}\n")
        except Exception:
            pass


def _print_turn_details(turn_log: Any, turn_idx: int) -> None:
    """Print detailed information for a single turn."""
    print(f"  Turn {turn_idx}:", flush=True)
    
    # Action information
    if hasattr(turn_log, 'action_type') and turn_log.action_type:
        action_params = ""
        if hasattr(turn_log, 'action_params') and turn_log.action_params:
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
    
    # Thinking preview
    if hasattr(turn_log, 'thinking') and turn_log.thinking:
        thinking_preview = turn_log.thinking[:150] + "..." if len(turn_log.thinking) > 150 else turn_log.thinking
        print(f"    Thinking: {thinking_preview}", flush=True)
    
    # Token usage
    if hasattr(turn_log, 'total_tokens'):
        tokens_info = f"Tokens: {turn_log.total_tokens}"
        if hasattr(turn_log, 'prompt_tokens') and hasattr(turn_log, 'completion_tokens'):
            tokens_info += f" (prompt: {turn_log.prompt_tokens}, completion: {turn_log.completion_tokens})"
        print(f"    {tokens_info}", flush=True)
    
    # Timing
    timing_parts = []
    if hasattr(turn_log, 'vlm_time') and turn_log.vlm_time:
        timing_parts.append(f"VLM: {turn_log.vlm_time:.2f}s")
    if hasattr(turn_log, 'action_time') and turn_log.action_time:
        timing_parts.append(f"Action: {turn_log.action_time:.2f}s")
    if hasattr(turn_log, 'screenshot_time') and turn_log.screenshot_time:
        timing_parts.append(f"Screenshot: {turn_log.screenshot_time:.2f}s")
    if timing_parts:
        print(f"    Timing: {' | '.join(timing_parts)}", flush=True)
    
    # Task completion
    if hasattr(turn_log, 'task_completed') and turn_log.task_completed:
        success_msg = "successfully" if (hasattr(turn_log, 'task_success') and turn_log.task_success) else "unsuccessfully"
        print(f"    Task completed {success_msg}", flush=True)
        if hasattr(turn_log, 'task_message') and turn_log.task_message:
            print(f"      {turn_log.task_message}", flush=True)
    
    # Errors
    if hasattr(turn_log, 'action_error') and turn_log.action_error:
        print(f"    Error: {turn_log.action_error[:150]}", flush=True)
    
    print("", flush=True)


__all__ = [
    "build_messages",
    "build_messages_for_vllm",
    "create_empty_result",
    "build_episode_result",
    "print_rollout_summary",
]

