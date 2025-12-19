"""CUA (Computer Use Agent) for AReaL GRPO training."""

# Core modules
from cua_agent.config import CUAConfig, GBoxConfig
from cua_agent.agent import CUAAgent
from cua_agent.actions import ActionType, parse_action

# Reward
from cua_agent.reward import (
    CUARolloutResult,
    simple_reward_function,
    calculate_grpo_advantages,
)

# Tasks
from cua_agent.tasks import (
    CUATask,
    TaskDifficulty,
    TaskCategory,
    get_training_tasks,
    get_eval_tasks,
    get_areal_train_dataset,
    get_areal_eval_dataset,
)

# AReaL environment
from cua_agent.areal_env import (
    GBoxEnvConfig,
    GBoxActorEnv,
    create_env_factory,
)

__all__ = [
    # Config
    "CUAConfig",
    "GBoxConfig",
    
    # Agent
    "CUAAgent",
    
    # Actions
    "ActionType",
    "parse_action",
    
    # Reward
    "CUARolloutResult",
    "simple_reward_function",
    "calculate_grpo_advantages",
    
    # Tasks
    "CUATask",
    "TaskDifficulty",
    "TaskCategory",
    "get_training_tasks",
    "get_eval_tasks",
    "get_areal_train_dataset",
    "get_areal_eval_dataset",
    
    # AReaL Environment
    "GBoxEnvConfig",
    "GBoxActorEnv",
    "create_env_factory",
]
