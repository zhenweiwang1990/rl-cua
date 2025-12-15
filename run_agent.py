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
    model_name: str = "unsloth/Qwen3-VL-32B-Instruct",
    max_turns: int = 20,
    verbose: bool = False,
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
    
    # Create config
    gbox_config = GBoxConfig(
        api_key=gbox_api_key,
        box_type=box_type,
    )
    
    config = CUAConfig(
        model_name=model_name,
        max_turns=max_turns,
        vllm_api_base=vllm_api_base,
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
        default="unsloth/Qwen3-VL-32B-Instruct",
        help="Model name (default: unsloth/Qwen3-VL-32B-Instruct)",
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
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

