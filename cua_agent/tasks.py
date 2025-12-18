"""Sample tasks for CUA Agent GRPO training.

This module defines 10 sample Android tasks for training and evaluation.
Each task can be validated via gbox system state APIs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class TaskDifficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskCategory(str, Enum):
    """Task categories."""
    SYSTEM = "system"
    NAVIGATION = "navigation"
    SETTINGS = "settings"
    APP = "app"
    INPUT = "input"


@dataclass
class CUATask:
    """A task for CUA agent to complete on Android."""
    
    id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    category: TaskCategory
    max_steps: int = 10
    
    # Validation config
    validation_type: str = "state"  # "state", "screenshot", "api"
    validation_query: Optional[str] = None  # Query to check against gbox state
    expected_result: Optional[Any] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "category": self.category.value,
            "max_steps": self.max_steps,
            "validation_type": self.validation_type,
            "validation_query": self.validation_query,
            "expected_result": self.expected_result,
            "tags": self.tags,
            "prerequisites": self.prerequisites,
        }


# =============================================================================
# TRAINING TASKS (8 tasks)
# =============================================================================

TRAINING_TASKS = [
    # Task 1: Open Settings app
    CUATask(
        id="train_01_open_settings",
        name="Open Settings",
        description="Open the Settings app from the home screen.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=5,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.settings",
        tags=["app", "settings", "launch"],
    ),
    
    # Task 2: Enable WiFi
    CUATask(
        id="train_02_enable_wifi",
        name="Enable WiFi",
        description="Go to Settings and turn on WiFi if it's off.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="wifi_enabled",
        expected_result=True,
        tags=["settings", "wifi", "toggle"],
    ),
    
    # Task 3: Set screen brightness to maximum
    CUATask(
        id="train_03_max_brightness",
        name="Maximum Brightness",
        description="Open Settings and set the screen brightness to maximum.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="brightness_level",
        expected_result=255,
        tags=["settings", "display", "brightness"],
    ),
    
    # Task 4: Open Chrome browser
    CUATask(
        id="train_04_open_chrome",
        name="Open Chrome",
        description="Open the Chrome browser app.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=5,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.chrome",
        tags=["app", "browser", "chrome"],
    ),
    
    # Task 5: Enable Airplane Mode
    CUATask(
        id="train_05_airplane_mode",
        name="Enable Airplane Mode",
        description="Go to Settings and turn on Airplane Mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="airplane_mode",
        expected_result=True,
        tags=["settings", "network", "airplane"],
    ),
    
    # Task 6: Check battery level
    CUATask(
        id="train_06_check_battery",
        name="Check Battery Level",
        description="Open Settings and navigate to the Battery section to check the current battery level.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="current_activity",
        expected_result="battery",
        tags=["settings", "battery", "info"],
    ),
    
    # Task 7: Go to home screen
    CUATask(
        id="train_07_go_home",
        name="Go to Home Screen",
        description="Press the home button to return to the home screen.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.NAVIGATION,
        max_steps=3,
        validation_type="state",
        validation_query="is_home_screen",
        expected_result=True,
        tags=["navigation", "home"],
    ),
    # Task 8: Enable Bluetooth
    CUATask(
        id="train_08_enable_bluetooth",
        name="Enable Bluetooth",
        description="Go to Settings and turn on Bluetooth if it's off.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="bluetooth_enabled",
        expected_result=True,
        tags=["settings", "bluetooth", "toggle"],
    ),
]


# =============================================================================
# EVALUATION TASKS (2 tasks)
# =============================================================================

EVAL_TASKS = [
    # Eval Task 1: Change display timeout
    CUATask(
        id="eval_01_display_timeout",
        name="Change Display Timeout",
        description="Open Settings, go to Display, and change the screen timeout to 5 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=300000,  # 5 minutes in milliseconds
        tags=["settings", "display", "timeout"],
    ),
    
    # Eval Task 2: Enable Do Not Disturb
    CUATask(
        id="eval_02_dnd_mode",
        name="Enable Do Not Disturb",
        description="Enable Do Not Disturb mode from quick settings or Settings app.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="dnd_enabled",
        expected_result=True,
        tags=["settings", "notifications", "dnd"],
    ),
]


# =============================================================================
# TASK UTILITIES
# =============================================================================

def get_all_tasks() -> List[CUATask]:
    """Get all tasks (training + evaluation)."""
    return TRAINING_TASKS + EVAL_TASKS


def get_training_tasks() -> List[CUATask]:
    """Get training tasks only."""
    return TRAINING_TASKS


def get_eval_tasks() -> List[CUATask]:
    """Get evaluation tasks only."""
    return EVAL_TASKS


def get_task_by_id(task_id: str) -> Optional[CUATask]:
    """Get a task by its ID."""
    for task in get_all_tasks():
        if task.id == task_id:
            return task
    return None


def get_tasks_by_category(category: TaskCategory) -> List[CUATask]:
    """Get tasks by category."""
    return [t for t in get_all_tasks() if t.category == category]


def get_tasks_by_difficulty(difficulty: TaskDifficulty) -> List[CUATask]:
    """Get tasks by difficulty."""
    return [t for t in get_all_tasks() if t.difficulty == difficulty]


# =============================================================================
# TASK PROMPT GENERATION
# =============================================================================

def create_task_prompt(task: CUATask) -> str:
    """Create a prompt for the task."""
    return f"""Task: {task.name}

{task.description}

You have a maximum of {task.max_steps} steps to complete this task.
Report task_complete when done, indicating success or failure."""


def create_task_system_prompt(task: CUATask) -> str:
    """Create system prompt for the task."""
    return f"""You are a helpful AI assistant that controls an Android device to complete tasks.

Your goal: {task.name}
Description: {task.description}

You can perform the following actions:
- click: Click on a UI element
- swipe: Swipe from one point to another  
- scroll: Scroll in a direction
- input: Type text into a field
- button_press: Press a system button (back, home, menu)
- task_complete: Report when the task is done

Guidelines:
1. Analyze the screenshot to understand the current screen state
2. Plan your next action based on the goal
3. Be efficient - use the minimum number of steps
4. Report task_complete with success=true when done, or success=false if stuck

Maximum steps allowed: {task.max_steps}"""


__all__ = [
    "CUATask",
    "TaskDifficulty", 
    "TaskCategory",
    "TRAINING_TASKS",
    "EVAL_TASKS",
    "get_all_tasks",
    "get_training_tasks",
    "get_eval_tasks",
    "get_task_by_id",
    "get_tasks_by_category",
    "get_tasks_by_difficulty",
    "create_task_prompt",
    "create_task_system_prompt",
]

