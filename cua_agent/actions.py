"""Action types and parsing for CUA Agent."""

import json
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Types of UI actions the agent can perform."""
    
    CLICK = "click"
    SWIPE = "swipe"
    SCROLL = "scroll"
    INPUT = "input"
    KEY_PRESS = "key_press"
    BUTTON_PRESS = "button_press"
    SLEEP = "sleep"  # Wait for a duration
    TASK_COMPLETE = "task_complete"  # Report task completion


@dataclass
class Target:
    """Target element description for UI actions.
    
    The target describes a UI element using various attributes that help
    locate it on the screen. The gbox-handy-1 model uses this description
    to generate precise coordinates.
    """
    
    element: str  # Required: description of the element
    label: Optional[str] = None  # Text label on the element
    color: Optional[str] = None  # Color of the element
    size: Optional[str] = None  # Size description (small, medium, large)
    location: Optional[str] = None  # Location description
    shape: Optional[str] = None  # Shape description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {"element": self.element}
        if self.label:
            result["label"] = self.label
        if self.color:
            result["color"] = self.color
        if self.size:
            result["size"] = self.size
        if self.location:
            result["location"] = self.location
        if self.shape:
            result["shape"] = self.shape
        return result
    
    def to_description(self) -> str:
        """Convert to natural language description for coordinate generation."""
        parts = [self.element]
        if self.label:
            parts.append(f'labeled "{self.label}"')
        if self.color:
            parts.append(f"{self.color} colored")
        if self.size:
            parts.append(f"{self.size} sized")
        if self.location:
            parts.append(f"located at {self.location}")
        if self.shape:
            parts.append(f"with {self.shape} shape")
        return " ".join(parts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Target":
        """Create Target from dictionary."""
        return cls(
            element=data.get("element", ""),
            label=data.get("label"),
            color=data.get("color"),
            size=data.get("size"),
            location=data.get("location"),
            shape=data.get("shape"),
        )


@dataclass
class CUAAction:
    """A UI action with all necessary parameters."""
    
    action_type: ActionType
    
    # For click
    option: Optional[str] = None  # "left", "right", "double"
    target: Optional[Target] = None
    
    # For swipe
    start_target: Optional[Target] = None
    end_target: Optional[Target] = None
    
    # For scroll
    direction: Optional[str] = None  # "up", "down", "left", "right"
    distance: Optional[int] = None
    
    # For input
    text: Optional[str] = None
    
    # For key_press
    keys: List[str] = field(default_factory=list)
    
    # For button_press
    button: Optional[str] = None  # "back", "home", "menu"
    
    # For sleep
    duration: Optional[float] = None  # Sleep duration in seconds
    
    # For task_complete
    success: bool = False
    result_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {"action_type": self.action_type.value}
        
        if self.action_type == ActionType.CLICK:
            if self.option:
                result["option"] = self.option
            if self.target:
                result["target"] = self.target.to_dict()
        
        elif self.action_type == ActionType.SWIPE:
            if self.start_target:
                result["start_target"] = self.start_target.to_dict()
            if self.end_target:
                result["end_target"] = self.end_target.to_dict()
        
        elif self.action_type == ActionType.SCROLL:
            if self.target:
                result["target"] = self.target.to_dict()
            if self.direction:
                result["direction"] = self.direction
            if self.distance:
                result["distance"] = self.distance
        
        elif self.action_type == ActionType.INPUT:
            if self.text:
                result["text"] = self.text
            if self.target:
                result["target"] = self.target.to_dict()
        
        elif self.action_type == ActionType.KEY_PRESS:
            result["keys"] = self.keys
        
        elif self.action_type == ActionType.BUTTON_PRESS:
            if self.button:
                result["button"] = self.button
        
        elif self.action_type == ActionType.SLEEP:
            if self.duration:
                result["duration"] = self.duration
        
        elif self.action_type == ActionType.TASK_COMPLETE:
            result["success"] = self.success
            if self.result_message:
                result["result_message"] = self.result_message
        
        return result


def parse_target(data: Union[Dict, str, None]) -> Optional[Target]:
    """Parse target from various formats."""
    if data is None:
        return None
    if isinstance(data, str):
        return Target(element=data)
    if isinstance(data, dict):
        return Target.from_dict(data)
    return None


def parse_action(data: Dict[str, Any]) -> CUAAction:
    """Parse action from dictionary.
    
    Args:
        data: Action dictionary
        
    Returns:
        Parsed CUAAction
    """
    action_type_str = data.get("action_type", "").lower()
    
    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        logger.warning(f"Unknown action type: {action_type_str}")
        # Default to task_complete with failure
        return CUAAction(
            action_type=ActionType.TASK_COMPLETE,
            success=False,
            result_message=f"Unknown action type: {action_type_str}",
        )
    
    if action_type == ActionType.CLICK:
        return CUAAction(
            action_type=action_type,
            option=data.get("option", "left"),
            target=parse_target(data.get("target")),
        )
    
    elif action_type == ActionType.SWIPE:
        return CUAAction(
            action_type=action_type,
            start_target=parse_target(data.get("start_target")),
            end_target=parse_target(data.get("end_target")),
        )
    
    elif action_type == ActionType.SCROLL:
        return CUAAction(
            action_type=action_type,
            target=parse_target(data.get("target")),
            direction=data.get("direction", "down"),
            distance=data.get("distance", 300),
        )
    
    elif action_type == ActionType.INPUT:
        return CUAAction(
            action_type=action_type,
            text=data.get("text", ""),
            target=parse_target(data.get("target")),
        )
    
    elif action_type == ActionType.KEY_PRESS:
        keys = data.get("keys", [])
        if isinstance(keys, str):
            keys = [keys]
        return CUAAction(
            action_type=action_type,
            keys=keys,
        )
    
    elif action_type == ActionType.BUTTON_PRESS:
        return CUAAction(
            action_type=action_type,
            button=data.get("button", "home"),
        )
    
    elif action_type == ActionType.SLEEP:
        return CUAAction(
            action_type=action_type,
            duration=data.get("duration", 1.0),
        )
    
    elif action_type == ActionType.TASK_COMPLETE:
        return CUAAction(
            action_type=action_type,
            success=data.get("success", False),
            result_message=data.get("result_message"),
        )
    
    return CUAAction(action_type=action_type)


def action_to_dict(action: CUAAction) -> Dict[str, Any]:
    """Convert action to dictionary."""
    return action.to_dict()


def extract_action_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract action JSON from model response.
    
    Tries multiple patterns to find the action data.
    
    Args:
        response: Model response text
        
    Returns:
        Parsed action dictionary or None
    """
    # Try <tool_call> tags first
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(tool_call_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match.strip())
            # Check if it's a tool call with name/arguments
            if "name" in data and "arguments" in data:
                args = data["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                return args
            # Or direct action data
            if "action_type" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON object with action_type
    json_pattern = r'\{[^{}]*"action_type"[^{}]*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find any JSON object
    json_pattern = r'\{[\s\S]*?\}'
    matches = re.findall(json_pattern, response)
    
    for match in matches:
        try:
            data = json.loads(match)
            if "action_type" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    return None

