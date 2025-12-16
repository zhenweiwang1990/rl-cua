"""Configuration for CUA Agent."""

import os
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
    model_name: str = "unsloth/Qwen3-VL-30B-A3B-Instruct"
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
    
    # VLM provider settings
    # Options: "vllm" (local vLLM server) or "openrouter" (OpenRouter API)
    vlm_provider: str = "vllm"  # "vllm" or "openrouter"
    
    # vLLM server settings (when using vllm provider)
    vllm_api_base: Optional[str] = None
    
    # OpenRouter settings (when using openrouter provider)
    openrouter_api_key: Optional[str] = None
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    
    # Dynamic temperature (for GRPO diverse rollouts)
    enable_dynamic_temperature: bool = False
    base_temperature: float = 0.6
    temperature_increment: float = 0.1
    
    @classmethod
    def from_env(cls) -> "CUAConfig":
        """Create CUAConfig from environment variables.
        
        Environment variables:
            VLM_PROVIDER: "vllm" or "openrouter" (default: "vllm")
            VLLM_API_BASE: vLLM server URL (e.g., http://localhost:8000/v1)
            OPENROUTER_API_KEY: OpenRouter API key (required if VLM_PROVIDER=openrouter)
            OPENROUTER_API_BASE: OpenRouter API base URL (default: https://openrouter.ai/api/v1)
            MODEL_NAME: Model name (default: unsloth/Qwen3-VL-30B-A3B-Instruct)
            MAX_TOKENS: Maximum tokens (default: 2048)
            TEMPERATURE: Temperature (default: 0.7)
            TOP_P: Top-p sampling (default: 0.9)
            MAX_TURNS: Maximum turns (default: 20)
            GBOX_API_KEY: GBox API key
            BOX_TYPE: Box type (default: android)
        """
        # VLM provider settings
        vlm_provider = os.getenv("VLM_PROVIDER", "vllm").lower()
        if vlm_provider not in ["vllm", "openrouter"]:
            raise ValueError(f"Invalid VLM_PROVIDER: {vlm_provider}. Must be 'vllm' or 'openrouter'")
        
        vllm_api_base = os.getenv("VLLM_API_BASE")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        
        # Validate OpenRouter configuration
        if vlm_provider == "openrouter" and not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when VLM_PROVIDER=openrouter. "
                "Get your API key from https://openrouter.ai"
            )
        
        # Model settings
        model_name = os.getenv("MODEL_NAME", "unsloth/Qwen3-VL-30B-A3B-Instruct")
        max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        top_p = float(os.getenv("TOP_P", "0.9"))
        max_turns = int(os.getenv("MAX_TURNS", "20"))
        
        # GBox settings
        gbox_api_key = os.getenv("GBOX_API_KEY", "")
        box_type = os.getenv("BOX_TYPE", "android")
        
        gbox_config = GBoxConfig(
            api_key=gbox_api_key,
            box_type=box_type,
        )
        
        return cls(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_turns=max_turns,
            vlm_provider=vlm_provider,
            vllm_api_base=vllm_api_base,
            openrouter_api_key=openrouter_api_key,
            openrouter_api_base=openrouter_api_base,
            gbox=gbox_config,
        )

