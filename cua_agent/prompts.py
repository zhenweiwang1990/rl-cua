"""Prompt templates for CUA Agent."""

from typing import Optional, List, Dict, Any


SYSTEM_PROMPT_TEMPLATE = """You are a Computer Use Agent (CUA) that can interact with a device screen to complete tasks.

## Your Task
{task_description}

## Available Actions
You can perform the following actions by calling the `perform_action` tool:

1. **click** - Click on an element
   - option: "left" (default), "right", or "double"
   - target: Description of the element to click

2. **swipe** - Swipe from one point to another
   - start_target: Starting element
   - end_target: Ending element

3. **scroll** - Scroll in a direction
   - direction: "up", "down", "left", or "right"
   - distance: Scroll distance in pixels (default: 300)
   - target: Element to scroll on/near

4. **input** - Type text
   - text: Text to type
   - target: Element to type into

5. **key_press** - Press keys
   - keys: List of keys to press (e.g., ["Enter"], ["Ctrl", "A"])

6. **button_press** - Press device button
   - button: "back", "home", or "menu"

## Target Description
When specifying a target, describe the element with these fields:
- **element** (required): What the element is (e.g., "login button", "search field")
- **label**: Text shown on the element
- **color**: Element color
- **size**: Element size (small, medium, large)
- **location**: Where on screen (e.g., "top right", "center of the page")
- **shape**: Element shape (rectangle, circle, etc.)

## Task Completion
When you have completed the task or determine it cannot be completed, call the `report_task_complete` tool with:
- **success**: true if task completed, false otherwise
- **result_message**: Summary of what was accomplished

## Guidelines
1. Analyze the current screenshot carefully before each action
2. Take one action at a time and observe the result
3. Be precise in your target descriptions
4. If an action doesn't work as expected, try an alternative approach
5. Report completion when the task goal is achieved

## Constraints
- Maximum turns: {max_turns}
- Think step by step about what action to take
- Always provide reasoning before calling a tool
"""


def create_system_prompt(
    task_description: str,
    max_turns: int = 20,
) -> str:
    """Create the system prompt for the CUA agent.
    
    Args:
        task_description: Description of the task to complete
        max_turns: Maximum number of turns allowed
        
    Returns:
        Formatted system prompt
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        task_description=task_description,
        max_turns=max_turns,
    )


def create_user_message_with_screenshot(
    turn: int,
    screenshot_description: str = "Current screen state",
    previous_action: Optional[Dict[str, Any]] = None,
    action_result: Optional[str] = None,
) -> str:
    """Create a user message with screenshot context.
    
    Args:
        turn: Current turn number
        screenshot_description: Description of what the screenshot shows
        previous_action: The previous action taken
        action_result: Result of the previous action
        
    Returns:
        Formatted user message
    """
    parts = [f"Turn {turn}:"]
    
    if previous_action:
        action_type = previous_action.get("action_type", "unknown")
        parts.append(f"\nPrevious action: {action_type}")
        if action_result:
            parts.append(f"Action result: {action_result}")
    
    parts.append(f"\n{screenshot_description}")
    parts.append("\nAnalyze the screenshot and determine the next action to take.")
    
    return "\n".join(parts)


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

