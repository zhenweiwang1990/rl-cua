"""CUA (Computer Use Agent) for GRPO training.

Supports two training backends:
1. AReaL + vLLM (legacy)
2. Unsloth + HuggingFace (recommended)
"""

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

# AReaL environment (legacy)
from cua_agent.areal_env import (
    GBoxEnvConfig,
    GBoxActorEnv,
    create_env_factory,
)

# Unsloth-based training (recommended)
from cua_agent.unsloth_inference import (
    UnslothInferenceConfig,
    UnslothVLMInference,
    FrozenReferenceModel,
)
from cua_agent.unsloth_grpo_trainer import (
    UnslothGRPOConfig,
    UnslothGRPOTrainer,
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
    
    # AReaL Environment (legacy)
    "GBoxEnvConfig",
    "GBoxActorEnv",
    "create_env_factory",
    
    # Unsloth Training (recommended)
    "UnslothInferenceConfig",
    "UnslothVLMInference",
    "FrozenReferenceModel",
    "UnslothGRPOConfig",
    "UnslothGRPOTrainer",
]
