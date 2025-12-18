"""CUA (Computer Use Agent) for GRPO training."""

from cua_agent.config import CUAConfig, GBoxConfig
from gbox_cua.gbox_client import GBoxClient
from cua_agent.actions import ActionType, Target, CUAAction, parse_action, action_to_dict
from cua_agent.tools import CUA_TOOLS_SCHEMA, get_tools_schema
from cua_agent.agent import CUAAgent

# GRPO Training modules
from cua_agent.tasks import (
    CUATask,
    TaskDifficulty,
    TaskCategory,
    TRAINING_TASKS,
    EVAL_TASKS,
    get_training_tasks,
    get_eval_tasks,
    get_task_by_id,
    create_task_prompt,
    create_task_system_prompt,
)
from cua_agent.grpo_config import (
    GRPOConfig,
    LoRAConfig,
    VLLMConfig,
    RolloutConfig,
)
from cua_agent.reward import (
    CUARolloutResult,
    simple_reward_function,
    completion_reward_function,
    efficiency_reward_function,
    RewardTracker,
    calculate_grpo_advantages,
)
from cua_agent.vllm_client import VLLMClient, VLLMResponse, VLLMRolloutCollector
from cua_agent.grpo_trainer import CUAGRPOTrainer, TrainingMetrics

__all__ = [
    # Core agent
    "CUAConfig",
    "GBoxConfig",
    "GBoxClient",
    "ActionType",
    "Target",
    "CUAAction",
    "parse_action",
    "action_to_dict",
    "CUA_TOOLS_SCHEMA",
    "get_tools_schema",
    "CUAAgent",
    # Tasks
    "CUATask",
    "TaskDifficulty",
    "TaskCategory",
    "TRAINING_TASKS",
    "EVAL_TASKS",
    "get_training_tasks",
    "get_eval_tasks",
    "get_task_by_id",
    "create_task_prompt",
    "create_task_system_prompt",
    # GRPO Config
    "GRPOConfig",
    "LoRAConfig",
    "VLLMConfig",
    "RolloutConfig",
    # Rewards
    "CUARolloutResult",
    "simple_reward_function",
    "completion_reward_function",
    "efficiency_reward_function",
    "RewardTracker",
    "calculate_grpo_advantages",
    # vLLM Client
    "VLLMClient",
    "VLLMResponse",
    "VLLMRolloutCollector",
    # Trainer
    "CUAGRPOTrainer",
    "TrainingMetrics",
]

