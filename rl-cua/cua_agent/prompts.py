"""Prompt templates for CUA Agent.

This module re-exports shared prompt functions from gbox-cua package.
Install with: pip install git+https://github.com/babelcloud/gbox-cua.git
"""

from typing import Optional, List, Dict, Any
from gbox_cua.prompts import (
    create_system_prompt,
    create_user_message_with_screenshot,
)


def format_action_history(history: List[Dict[str, Any]]) -> str:
    """Format action history for context.
    
    Args:
        history: List of action dictionaries
        
    Returns:
        Formatted history string
    """
    if not history:
        return "No previous actions."
    
    lines = ["Action History:"]
    for i, action in enumerate(history, 1):
        action_type = action.get("action_type", "unknown")
        result = action.get("result", "")
        lines.append(f"  {i}. {action_type} - {result}")
    
    return "\n".join(lines)

__all__ = [
    "create_system_prompt",
    "create_user_message_with_screenshot",
    "format_action_history",
]

