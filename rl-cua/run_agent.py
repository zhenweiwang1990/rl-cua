#!/usr/bin/env python3
"""Run CUA Agent on a task.

Usage:
    python run_agent.py "Task description"
    python run_agent.py --task "Task description" --box-type android --verbose
"""

import argparse
import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

from cua_agent.config import CUAConfig, GBoxConfig
from cua_agent.agent import CUAAgent, calculate_reward


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_agent(
    task: str,
    box_type: str = "android",
    vllm_api_base: str | None = None,
    gbox_api_key: str | None = None,
    model_name: str = "unsloth/Qwen3-VL-30B-A3B-Instruct",
    max_turns: int = 20,
    verbose: bool = False,
    vlm_provider: str | None = None,
    openrouter_api_key: str | None = None,
) -> dict:
    """Run the CUA agent on a task.
    
    Args:
        task: Task description
        box_type: Type of GBox ("android" or "linux")
        vllm_api_base: vLLM server URL
        gbox_api_key: GBox API key
        model_name: Model name
        max_turns: Maximum turns
        verbose: Enable verbose output
        vlm_provider: VLM provider ("vllm" or "openrouter")
        openrouter_api_key: OpenRouter API key (required if vlm_provider=openrouter)
        
    Returns:
        Result dictionary
    """
    # Get API key
    if not gbox_api_key:
        gbox_api_key = os.environ.get("GBOX_API_KEY")
        if not gbox_api_key:
            raise ValueError(
                "GBOX_API_KEY not provided. Set it via --gbox-api-key or GBOX_API_KEY env var"
            )
    
    # Determine VLM provider from args or environment
    if not vlm_provider:
        vlm_provider = os.getenv("VLM_PROVIDER", "vllm").lower()
    
    # Get OpenRouter API key if needed
    if vlm_provider == "openrouter":
        if not openrouter_api_key:
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when using OpenRouter provider. "
                "Get your API key from https://openrouter.ai"
            )
    
    # Create config
    gbox_config = GBoxConfig(
        api_key=gbox_api_key,
        box_type=box_type,
    )
    
    # Get OpenRouter API base from environment if not provided
    openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    
    config = CUAConfig(
        model_name=model_name,
        max_turns=max_turns,
        vlm_provider=vlm_provider,
        vllm_api_base=vllm_api_base,
        openrouter_api_key=openrouter_api_key,
        openrouter_api_base=openrouter_api_base,
        gbox=gbox_config,
    )
    
    # Run agent
    async with CUAAgent(config) as agent:
        rubric, history = await agent.run_task(
            task_description=task,
            box_type=box_type,
            verbose=verbose,
        )
    
    # Calculate reward
    reward = calculate_reward(config, rubric)
    
    result = {
        "task": task,
        "rubric": rubric.to_dict(),
        "reward": reward,
        "num_steps": len(history),
    }
    
    return result


def main():
    """Main entry point."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Run CUA Agent on a task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_agent.py "Open the Settings app"
    python run_agent.py --task "Search for weather" --box-type android --verbose
    python run_agent.py --task "Navigate to gmail.com" --vllm-api-base http://localhost:8000/v1
        """,
    )
    
    parser.add_argument(
        "task",
        nargs="?",
        help="Task description",
    )
    parser.add_argument(
        "--task", "-t",
        dest="task_arg",
        help="Task description (alternative to positional)",
    )
    parser.add_argument(
        "--box-type", "-b",
        default="android",
        choices=["android", "linux"],
        help="Type of GBox environment (default: android)",
    )
    parser.add_argument(
        "--vllm-api-base",
        help="vLLM server URL (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--gbox-api-key",
        help="GBox API key (or set GBOX_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-VL-30B-A3B-Instruct",
        help="Model name (default: unsloth/Qwen3-VL-30B-A3B-Instruct)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns (default: 20)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--vlm-provider",
        choices=["vllm", "openrouter"],
        help="VLM provider: 'vllm' (local vLLM server) or 'openrouter' (OpenRouter API). "
             "Can also be set via VLM_PROVIDER env var (default: vllm)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        help="OpenRouter API key (required if --vlm-provider=openrouter). "
             "Can also be set via OPENROUTER_API_KEY env var",
    )
    
    args = parser.parse_args()
    
    # Get task from either positional or named argument
    task = args.task or args.task_arg
    if not task:
        parser.error("Task description is required")
    
    try:
        result = asyncio.run(run_agent(
            task=task,
            box_type=args.box_type,
            vllm_api_base=args.vllm_api_base,
            gbox_api_key=args.gbox_api_key,
            model_name=args.model,
            max_turns=args.max_turns,
            verbose=args.verbose,
            vlm_provider=args.vlm_provider,
            openrouter_api_key=args.openrouter_api_key,
        ))
        
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"Task: {result['task']}")
        print(f"Success: {result['rubric']['task_success']}")
        print(f"Completed: {result['rubric']['task_completed']}")
        print(f"Turns: {result['rubric']['num_turns']}/{result['rubric']['max_turns']}")
        print(f"Reward: {result['reward']:.3f}")
        print(f"Message: {result['rubric']['result_message']}")
        print("="*60)
        
        sys.exit(0 if result['rubric']['task_success'] else 1)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

