"""Reward functions for CUA Agent GRPO training.

This module provides simple reward functions that can be extended
to more sophisticated reward shaping.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from cua_agent.tasks import CUATask

logger = logging.getLogger(__name__)


@dataclass
class CUARolloutResult:
    """Result of a CUA rollout."""
    
    task_id: str
    task_completed: bool
    task_success: bool
    num_turns: int
    max_turns: int
    errors: List[str]
    
    # Validation results
    validation_passed: bool = False
    validation_details: Optional[Dict[str, Any]] = None
    
    # Timing
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "num_turns": self.num_turns,
            "max_turns": self.max_turns,
            "errors": self.errors,
            "validation_passed": self.validation_passed,
            "validation_details": self.validation_details,
            "total_time": self.total_time,
        }


def simple_reward_function(
    result: CUARolloutResult,
    task: Optional[CUATask] = None,
) -> float:
    """Simple binary reward function.
    
    Returns 1.0 for successful completion, 0.0 otherwise.
    
    Args:
        result: Rollout result
        task: Original task (optional, for additional context)
        
    Returns:
        Reward value (0.0 or 1.0)
    """
    if result.task_success:
        return 1.0
    return 0.0


def completion_reward_function(
    result: CUARolloutResult,
    task: Optional[CUATask] = None,
) -> float:
    """Reward function with partial credit for completion attempt.
    
    - 1.0 for successful completion
    - 0.1 for attempted completion (marked as complete but failed)
    - 0.0 for timeout/incomplete
    
    Args:
        result: Rollout result
        task: Original task (optional)
        
    Returns:
        Reward value
    """
    if result.task_success:
        return 1.0
    elif result.task_completed:
        return 0.1  # Attempted but failed
    return 0.0


def efficiency_reward_function(
    result: CUARolloutResult,
    task: Optional[CUATask] = None,
    step_penalty: float = -0.02,
    error_penalty: float = -0.1,
    timeout_penalty: float = -0.5,
) -> float:
    """Reward function that considers efficiency.
    
    Rewards:
    - Base reward for success: 1.0
    - Step penalty: -0.02 per step
    - Error penalty: -0.1 per error
    - Timeout penalty: -0.5 if not completed
    
    Args:
        result: Rollout result
        task: Original task
        step_penalty: Penalty per step
        error_penalty: Penalty per error
        timeout_penalty: Penalty for timeout
        
    Returns:
        Reward value (can be negative)
    """
    reward = 0.0
    
    # Base reward for outcome
    if result.task_success:
        reward = 1.0
    elif result.task_completed:
        reward = 0.1
    else:
        reward = timeout_penalty
    
    # Step efficiency penalty
    reward += step_penalty * result.num_turns
    
    # Error penalty
    reward += error_penalty * len(result.errors)
    
    return reward


def shaped_reward_function(
    result: CUARolloutResult,
    task: CUATask,
    validation_weight: float = 0.5,
) -> float:
    """Shaped reward function with validation bonus.
    
    Combines completion reward with validation state check.
    
    Args:
        result: Rollout result
        task: Original task
        validation_weight: Weight for validation bonus
        
    Returns:
        Reward value
    """
    base_reward = completion_reward_function(result, task)
    
    # Add validation bonus
    if result.validation_passed:
        base_reward += validation_weight
    
    # Normalize to [0, 1]
    return min(1.0, base_reward)


class RewardTracker:
    """Track reward statistics during training."""
    
    def __init__(self):
        self.rewards: List[float] = []
        self.successes: List[bool] = []
        self.num_turns: List[int] = []
        
    def add(self, reward: float, success: bool, turns: int):
        """Add a rollout result."""
        self.rewards.append(reward)
        self.successes.append(success)
        self.num_turns.append(turns)
    
    def reset(self):
        """Reset tracker."""
        self.rewards.clear()
        self.successes.clear()
        self.num_turns.clear()
    
    @property
    def mean_reward(self) -> float:
        """Get mean reward."""
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
    
    @property
    def success_rate(self) -> float:
        """Get success rate."""
        return sum(self.successes) / len(self.successes) if self.successes else 0.0
    
    @property
    def mean_turns(self) -> float:
        """Get mean turns."""
        return sum(self.num_turns) / len(self.num_turns) if self.num_turns else 0.0
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "mean_reward": self.mean_reward,
            "success_rate": self.success_rate,
            "mean_turns": self.mean_turns,
            "total_rollouts": len(self.rewards),
        }

async def validate_task_completion(
    task: CUATask,
    gbox_client,
) -> bool:
    """Validate task completion using ADB/shell commands via GBox.

    We avoid using gbox-handy-1 for validation, and instead:
      1. Execute Android shell commands inside the box (via GBox Command API).
      2. Parse the output to determine whether the target system state matches
         the expected result (e.g., WiFi on/off, current app, screen timeout).
    """

    async def _run_shell(cmd: str) -> str:
        """Run a shell command inside the box via GBox SDK and return stdout."""
        sdk = getattr(gbox_client, "_sdk", None)
        box_id = getattr(gbox_client, "box_id", None)
        if sdk is None or not box_id:
            raise RuntimeError("gbox_client must provide `_sdk` and `box_id` for command execution")

        try:
            # Use SDK's get() method to get the box object, then use box.command()
            box = sdk.get(box_id)
            result = box.command(command=cmd)
            # result should have stdout, stderr, and exitCode attributes
            if hasattr(result, 'stdout'):
                return str(result.stdout) if result.stdout else ""
            # Fallback for dict-like responses
            if isinstance(result, dict):
                return str(result.get("stdout") or result.get("output") or "")
            return str(result)
        except Exception as e:
            logger.error(f"Failed to run shell command '{cmd}': {e}")
            return ""

    try:
        q = (task.validation_query or "").lower()
        expected = task.expected_result

        # 1) 当前前台 App (Settings / Chrome 等)
        if q == "current_app":
            out = await _run_shell(
                "dumpsys window | grep mCurrentFocus || "
                "dumpsys activity | grep mResumedActivity"
            )
            if not out:
                return False
            if isinstance(expected, str) and expected:
                return expected in out
            return "mCurrentFocus" in out or "mResumedActivity" in out

        # 2) WiFi 开关
        if q == "wifi_enabled":
            out = await _run_shell("settings get global wifi_on")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return bool(val) == bool(expected)

        # 3) 飞行模式
        if q == "airplane_mode":
            out = await _run_shell("settings get global airplane_mode_on")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return bool(val) == bool(expected)

        # 4) 亮度
        if q == "brightness_level":
            out = await _run_shell("settings get system screen_brightness")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return val == int(expected) if isinstance(expected, int) else False

        # 5) 当前 Activity（比如电池页）
        if q == "current_activity":
            out = await _run_shell(
                "dumpsys window | grep mCurrentFocus || "
                "dumpsys activity | grep mResumedActivity"
            )
            if not out:
                return False
            if isinstance(expected, str) and expected:
                return expected.lower() in out.lower()
            return False

        # 6) 是否在桌面
        if q == "is_home_screen":
            out = await _run_shell(
                "dumpsys window | grep mCurrentFocus || "
                "dumpsys activity | grep mResumedActivity"
            )
            if not out:
                return False
            lowered = out.lower()
            is_home = (
                "launcher" in lowered
                or "home" in lowered
                or "com.android.launcher" in lowered
                or "launcher3" in lowered
            )
            return is_home == bool(expected)

        # 7) 屏幕熄屏时间 (ms)
        if q == "screen_timeout":
            out = await _run_shell("settings get system screen_off_timeout")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return val == int(expected) if isinstance(expected, int) else False

        # 8) 勿扰模式 (DND / zen_mode)
        if q == "dnd_enabled":
            out = await _run_shell("settings get global zen_mode")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            is_enabled = val > 0
            return is_enabled == bool(expected)

        # 9) 蓝牙开关（配合 train_08_enable_bluetooth）
        if q == "bluetooth_enabled":
            out = await _run_shell(
                "settings get global bluetooth_on || "
                "settings get secure bluetooth_on"
            )
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return bool(val) == bool(expected)

        logger.warning(f"No shell-based validator implemented for query '{q}' (task {task.id})")
        return False

    except Exception as e:
        logger.error(f"Validation failed for task {task.id}: {e}", exc_info=True)
        return False
        
def calculate_grpo_advantages(
    rewards: List[float],
    min_std: float = 0.05,
) -> List[float]:
    """Calculate GRPO advantages for a group of rewards.
    
    GRPO uses group-relative normalization:
    advantage_i = (reward_i - mean(rewards)) / std(rewards)
    
    Args:
        rewards: List of rewards for a group
        min_std: Minimum standard deviation threshold
        
    Returns:
        List of advantages (or zeros if std < min_std)
    """
    import numpy as np
    
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Skip low-variance groups
    if std_reward < min_std:
        return [0.0] * len(rewards)
    
    # Normalize advantages
    advantages = [
        (r - mean_reward) / (std_reward + 1e-8)
        for r in rewards
    ]
    
    return advantages


__all__ = [
    "CUARolloutResult",
    "simple_reward_function",
    "completion_reward_function",
    "efficiency_reward_function",
    "shaped_reward_function",
    "RewardTracker",
    "validate_task_completion",
    "calculate_grpo_advantages",
    "cua_reward_fn",
]


def cua_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    """CUA 奖励函数 - 直接使用 workflow 传入的终局奖励。
    
    这个函数用于 AReaL 训练框架，直接从 kwargs 中获取终局奖励。
    
    Args:
        prompt: 输入提示
        completions: 模型输出
        prompt_ids: 提示的 token IDs
        completion_ids: 输出的 token IDs
        answer: 答案（未使用）
        **kwargs: 包含 reward, task_success, task_completed 等字段
        
    Returns:
        奖励值（float）
    """
    if isinstance(completions, list):
        completion = completions[0] if completions else ""
    else:
        completion = completions
    
    # 直接使用 workflow 传入的 reward
    reward = kwargs.get("reward")
    if reward is None:
        task_success = kwargs.get("task_success", False)
        task_completed = kwargs.get("task_completed", False)
        reward = 1.0 if (task_success or task_completed) else 0.0
    
    try:
        reward = float(reward)
    except Exception:
        reward = 0.0
    
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        task_id = kwargs.get("task_id", "N/A")
        num_turns = kwargs.get("num_turns", 0)
        max_turns = kwargs.get("max_turns", 15)
        task_success = kwargs.get("task_success", False)
        errors = kwargs.get("errors", [])
        
        logger.info("=" * 80)
        logger.info(f"[CUA Rollout Complete] Task ID: {task_id}")
        logger.info(f"  Turns: {num_turns}/{max_turns}")
        logger.info(f"  Task Success: {task_success}")
        logger.info(f"  Reward: {reward:.4f}")
        if errors:
            logger.info(f"  Errors ({len(errors)}): {errors[:3]}")
        logger.info("=" * 80)
    
    return reward

