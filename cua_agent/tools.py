"""Tool definitions for CUA Agent.

Defines the two tools available to the agent:
1. perform_action - Execute UI actions (click, swipe, scroll, etc.)
2. report_task_complete - Report that the task is finished
"""

from typing import List, Dict, Any


# Tool schema for perform_action
PERFORM_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "perform_action",
        "description": """Execute a UI action on the device screen. Use this tool to interact with UI elements.

Supported action types:
- click: Click on an element. Options: "left", "right", "double"
- swipe: Swipe from one element to another
- scroll: Scroll in a direction (up, down, left, right) with optional distance
- input: Type text into an element
- key_press: Press one or more keys
- button_press: Press device button (back, home, menu)

For actions with targets, describe the element using:
- element: Required. What the element is (e.g., "login button", "search field")
- label: Text shown on the element
- color: Element color
- size: Element size (small, medium, large)
- location: Where on screen (e.g., "top right", "center")
- shape: Element shape (rectangle, circle, etc.)""",
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["click", "swipe", "scroll", "input", "key_press", "button_press"],
                    "description": "Type of action to perform"
                },
                "option": {
                    "type": "string",
                    "enum": ["left", "right", "double"],
                    "description": "Click option (for click action)"
                },
                "target": {
                    "type": "object",
                    "description": "Target element for click, scroll, or input actions",
                    "properties": {
                        "element": {
                            "type": "string",
                            "description": "Description of the UI element"
                        },
                        "label": {
                            "type": "string",
                            "description": "Text label on the element"
                        },
                        "color": {
                            "type": "string",
                            "description": "Color of the element"
                        },
                        "size": {
                            "type": "string",
                            "description": "Size of the element (small, medium, large)"
                        },
                        "location": {
                            "type": "string",
                            "description": "Location on screen"
                        },
                        "shape": {
                            "type": "string",
                            "description": "Shape of the element"
                        }
                    },
                    "required": ["element"]
                },
                "start_target": {
                    "type": "object",
                    "description": "Start element for swipe action",
                    "properties": {
                        "element": {"type": "string"},
                        "label": {"type": "string"},
                        "color": {"type": "string"},
                        "size": {"type": "string"},
                        "location": {"type": "string"},
                        "shape": {"type": "string"}
                    },
                    "required": ["element"]
                },
                "end_target": {
                    "type": "object",
                    "description": "End element for swipe action",
                    "properties": {
                        "element": {"type": "string"},
                        "label": {"type": "string"},
                        "color": {"type": "string"},
                        "size": {"type": "string"},
                        "location": {"type": "string"},
                        "shape": {"type": "string"}
                    },
                    "required": ["element"]
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Scroll direction"
                },
                "distance": {
                    "type": "integer",
                    "description": "Scroll distance in pixels"
                },
                "text": {
                    "type": "string",
                    "description": "Text to input"
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to press (e.g., ['Enter'], ['Ctrl', 'A'])"
                },
                "button": {
                    "type": "string",
                    "enum": ["back", "home", "menu"],
                    "description": "Device button to press"
                }
            },
            "required": ["action_type"]
        }
    }
}


# Tool schema for report_task_complete
REPORT_TASK_COMPLETE_TOOL = {
    "type": "function",
    "function": {
        "name": "report_task_complete",
        "description": """Report that the task has been completed or cannot be completed.

Call this tool when:
1. You have successfully completed the task
2. You determine the task cannot be completed
3. You have reached a final state

Always provide a result_message explaining what was accomplished or why the task couldn't be completed.""",
        "parameters": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the task was completed successfully"
                },
                "result_message": {
                    "type": "string",
                    "description": "Summary of what was accomplished or why the task failed"
                }
            },
            "required": ["success", "result_message"]
        }
    }
}


# Combined tools schema
CUA_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    PERFORM_ACTION_TOOL,
    REPORT_TASK_COMPLETE_TOOL,
]


def get_tools_schema() -> List[Dict[str, Any]]:
    """Get the tools schema for the CUA agent."""
    return CUA_TOOLS_SCHEMA


def tool_call_to_action_dict(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a tool call to an action dictionary.
    
    Args:
        tool_name: Name of the tool called
        arguments: Tool arguments
        
    Returns:
        Action dictionary
    """
    if tool_name == "perform_action":
        return arguments
    elif tool_name == "report_task_complete":
        return {
            "action_type": "task_complete",
            "success": arguments.get("success", False),
            "result_message": arguments.get("result_message", ""),
        }
    else:
        return {
            "action_type": "task_complete",
            "success": False,
            "result_message": f"Unknown tool: {tool_name}",
        }

