"""Configuration for CUA Agent."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class GBoxConfig:
    """Configuration for GBox API connection."""
    
    api_key: str = ""
    # Note: api_base_url is no longer used with official SDK
    # The SDK handles the API endpoint automatically
    
    # Box creation settings
    box_type: str = "android"  # "android" or "linux"
    timeout: str = "60s"
    wait: bool = True
    
    # Optional box configuration
    expires_in: str = "15m"
    labels: dict = field(default_factory=dict)
    envs: dict = field(default_factory=dict)
    
    # Model for coordinate generation
    model: str = "gbox-handy-1"


@dataclass
class CUAConfig:
    """Configuration for CUA Agent."""
    
    # Model settings
    model_name: str = "unsloth/Qwen3-VL-32B-Instruct"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Agent loop settings
    max_turns: int = 20
    
    # Action settings
    action_delay: float = 0.5  # Delay after each action
    screenshot_delay: float = 0.3  # Delay before screenshot
    
    # Reward weights (for future GRPO training)
    success_reward: float = 1.0
    step_penalty: float = -0.02
    error_penalty: float = -0.5
    timeout_penalty: float = -1.0
    
    # GBox configuration
    gbox: GBoxConfig = field(default_factory=GBoxConfig)
    
    # vLLM server settings (when using server mode)
    vllm_api_base: Optional[str] = None
    
    # Dynamic temperature (for GRPO diverse rollouts)
    enable_dynamic_temperature: bool = False
    base_temperature: float = 0.6
    temperature_increment: float = 0.1

