"""CUA Agent - Main agent loop for Computer Use Agent."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from cua_agent.config import CUAConfig, GBoxConfig
from cua_agent.gbox_client import GBoxClient
from cua_agent.vlm_inference import VLMInference
from cua_agent.actions import (
    ActionType,
    CUAAction,
    Target,
    parse_action,
    extract_action_from_response,
)
from cua_agent.tools import get_tools_schema, tool_call_to_action_dict
from cua_agent.prompts import create_system_prompt, create_user_message_with_screenshot

logger = logging.getLogger(__name__)


@dataclass
class CUARubric:
    """Rubric for evaluating CUA agent performance."""
    
    # Task outcome
    task_completed: bool = False
    task_success: bool = False
    result_message: str = ""
    
    # Turn tracking
    num_turns: int = 0
    max_turns: int = 20
    
    # Action metrics
    num_clicks: int = 0
    num_swipes: int = 0
    num_scrolls: int = 0
    num_inputs: int = 0
    num_key_presses: int = 0
    num_button_presses: int = 0
    
    # Errors
    num_action_errors: int = 0
    num_coordinate_failures: int = 0
    num_parse_errors: int = 0
    
    # Timing
    total_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rubric to dictionary."""
        return {
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "result_message": self.result_message,
            "num_turns": self.num_turns,
            "max_turns": self.max_turns,
            "num_clicks": self.num_clicks,
            "num_swipes": self.num_swipes,
            "num_scrolls": self.num_scrolls,
            "num_inputs": self.num_inputs,
            "num_key_presses": self.num_key_presses,
            "num_button_presses": self.num_button_presses,
            "num_action_errors": self.num_action_errors,
            "num_coordinate_failures": self.num_coordinate_failures,
            "num_parse_errors": self.num_parse_errors,
            "total_time_seconds": self.total_time_seconds,
        }


@dataclass
class ActionStep:
    """Record of a single action step."""
    
    turn: int
    action: CUAAction
    coordinates: Optional[Dict[str, int]] = None
    success: bool = True
    error_message: Optional[str] = None
    screenshot_before: Optional[str] = None  # Path or reference
    screenshot_after: Optional[str] = None


class CUAAgent:
    """Computer Use Agent that interacts with device screens to complete tasks.
    
    The agent:
    1. Creates a GBox environment (Android/Linux)
    2. Takes screenshots and sends them to the VLM
    3. Receives action instructions from the VLM
    4. Converts actions to coordinates using gbox-handy-1
    5. Executes actions on the device
    6. Repeats until task completion or max turns
    """
    
    def __init__(
        self,
        config: CUAConfig,
        rollout_index: int = 0,
        enable_logging: bool = True,
    ):
        """Initialize CUA agent.
        
        Args:
            config: Agent configuration
            rollout_index: Index for diverse rollouts (GRPO training)
            enable_logging: Whether to log detailed steps
        """
        self.config = config
        self.rollout_index = rollout_index
        self.enable_logging = enable_logging
        
        # Initialize clients
        self.gbox_client = GBoxClient(config.gbox)
        self.vlm = VLMInference(
            model_name=config.model_name,
            api_base=config.vllm_api_base,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        
        # Tools schema
        self.tools = get_tools_schema()
        
        # State
        self.action_history: List[ActionStep] = []
        self.conversation: List[Dict[str, Any]] = []
    
    async def run_task(
        self,
        task_description: str,
        box_type: str = "android",
        verbose: bool = False,
    ) -> Tuple[CUARubric, List[ActionStep]]:
        """Run the agent on a task.
        
        Args:
            task_description: Description of the task to complete
            box_type: Type of box ("android" or "linux")
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of (rubric, action_history)
        """
        import time
        start_time = time.time()
        
        rubric = CUARubric(max_turns=self.config.max_turns)
        self.action_history = []
        self.conversation = []
        
        try:
            # Create GBox environment
            if verbose:
                print(f"\n{'='*80}")
                print(f"Creating {box_type} box...")
            
            await self.gbox_client.create_box(box_type)
            
            if verbose:
                print(f"Box created: {self.gbox_client.box_id}")
                print(f"Task: {task_description}")
                print(f"{'='*80}\n")
            
            # Wait for box to be ready
            await asyncio.sleep(2.0)
            
            # Create system prompt
            system_prompt = create_system_prompt(
                task_description=task_description,
                max_turns=self.config.max_turns,
            )
            self.conversation.append({
                "role": "system",
                "content": system_prompt,
            })
            
            # Agent loop
            for turn in range(self.config.max_turns):
                rubric.num_turns = turn + 1
                
                if verbose:
                    print(f"\n{'─'*60}")
                    print(f"Turn {turn + 1}/{self.config.max_turns}")
                    print(f"{'─'*60}")
                
                # Take screenshot
                await asyncio.sleep(self.config.screenshot_delay)
                screenshot_bytes, screenshot_uri = await self.gbox_client.take_screenshot()
                
                if verbose:
                    print(f"📸 Screenshot captured ({len(screenshot_bytes)} bytes)")
                
                # Create user message
                previous_action = None
                action_result = None
                if self.action_history:
                    last_step = self.action_history[-1]
                    previous_action = last_step.action.to_dict()
                    if last_step.success:
                        action_result = "Action executed successfully"
                    else:
                        action_result = f"Action failed: {last_step.error_message}"
                
                user_message = create_user_message_with_screenshot(
                    turn=turn + 1,
                    screenshot_description="[Screenshot attached - analyze and determine next action]",
                    previous_action=previous_action,
                    action_result=action_result,
                )
                
                self.conversation.append({
                    "role": "user",
                    "content": user_message,
                })
                
                # Calculate temperature for diverse rollouts
                if self.config.enable_dynamic_temperature:
                    temperature = (
                        self.config.base_temperature +
                        self.rollout_index * self.config.temperature_increment
                    )
                else:
                    temperature = self.config.temperature
                
                # Generate response
                raw_response, parsed_response = await self.vlm.generate(
                    messages=self.conversation,
                    tools=self.tools,
                    image_data=screenshot_bytes,
                    temperature=temperature,
                )
                
                if verbose:
                    print(f"\n🤖 Model response:")
                    print(f"   {raw_response[:200]}...")
                
                # Add assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": parsed_response.get("content"),
                }
                if parsed_response.get("tool_calls"):
                    assistant_msg["tool_calls"] = parsed_response["tool_calls"]
                self.conversation.append(assistant_msg)
                
                # Parse action
                action = None
                action_dict = None
                
                if parsed_response.get("tool_calls"):
                    tool_call = parsed_response["tool_calls"][0]
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    action_dict = tool_call_to_action_dict(tool_name, tool_args)
                    action = parse_action(action_dict)
                    
                    if verbose:
                        print(f"\n🔧 Tool call: {tool_name}")
                        print(f"   Arguments: {json.dumps(tool_args, indent=2)[:200]}")
                else:
                    # Try to extract action from text
                    action_dict = extract_action_from_response(raw_response)
                    if action_dict:
                        action = parse_action(action_dict)
                    else:
                        rubric.num_parse_errors += 1
                        if verbose:
                            print(f"\n⚠️ Could not parse action from response")
                        continue
                
                if action is None:
                    rubric.num_parse_errors += 1
                    continue
                
                # Check for task completion
                if action.action_type == ActionType.TASK_COMPLETE:
                    rubric.task_completed = True
                    rubric.task_success = action.success
                    rubric.result_message = action.result_message or ""
                    
                    if verbose:
                        status = "✅ SUCCESS" if action.success else "❌ FAILED"
                        print(f"\n{status}: {rubric.result_message}")
                    
                    # Add tool response
                    self.conversation.append({
                        "role": "tool",
                        "tool_call_id": "call_0",
                        "content": json.dumps({
                            "status": "task_complete",
                            "success": action.success,
                        }),
                    })
                    break
                
                # Execute action
                step = ActionStep(turn=turn + 1, action=action)
                
                try:
                    await self._execute_action(
                        action, screenshot_uri, step, rubric, verbose
                    )
                except Exception as e:
                    step.success = False
                    step.error_message = str(e)
                    rubric.num_action_errors += 1
                    logger.error(f"Action execution error: {e}")
                    if verbose:
                        print(f"\n❌ Action error: {e}")
                
                self.action_history.append(step)
                
                # Add tool response
                tool_response = {
                    "status": "success" if step.success else "error",
                }
                if step.error_message:
                    tool_response["error"] = step.error_message
                if step.coordinates:
                    tool_response["coordinates"] = step.coordinates
                
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": "call_0",
                    "content": json.dumps(tool_response),
                })
                
                # Wait after action
                await asyncio.sleep(self.config.action_delay)
            
            # Check if ran out of turns
            if not rubric.task_completed and rubric.num_turns >= self.config.max_turns:
                rubric.result_message = "Ran out of turns"
                if verbose:
                    print(f"\n⏱️ Ran out of turns")
        
        finally:
            # Clean up
            rubric.total_time_seconds = time.time() - start_time
            
            try:
                await self.gbox_client.terminate_box()
                if verbose:
                    print(f"\n🗑️ Box terminated")
            except Exception as e:
                logger.error(f"Failed to terminate box: {e}")
        
        if verbose:
            self._print_summary(rubric)
        
        return rubric, self.action_history
    
    async def _execute_action(
        self,
        action: CUAAction,
        screenshot_uri: str,
        step: ActionStep,
        rubric: CUARubric,
        verbose: bool,
    ):
        """Execute a UI action.
        
        Args:
            action: The action to execute
            screenshot_uri: Screenshot for coordinate generation
            step: Action step to update
            rubric: Rubric to update
            verbose: Whether to log
        """
        if action.action_type == ActionType.CLICK:
            rubric.num_clicks += 1
            
            if action.target:
                # Generate coordinates
                result = await self.gbox_client.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=action.target.to_description(),
                )
                
                coords = result.get("response", {}).get("coordinates", {})
                x = coords.get("x", 0)
                y = coords.get("y", 0)
                step.coordinates = {"x": x, "y": y}
                
                if verbose:
                    print(f"   📍 Coordinates: ({x}, {y})")
                
                # Execute click
                is_double = action.option == "double"
                await self.gbox_client.click(
                    x=x, y=y,
                    button=action.option if action.option in ["left", "right"] else "left",
                    double_click=is_double,
                )
        
        elif action.action_type == ActionType.SWIPE:
            rubric.num_swipes += 1
            
            start_desc = action.start_target.to_description() if action.start_target else "screen center"
            end_desc = action.end_target.to_description() if action.end_target else "screen center"
            
            # Generate start coordinates
            start_result = await self.gbox_client.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="click",
                target=start_desc,
            )
            start_coords = start_result.get("response", {}).get("coordinates", {})
            
            # Generate end coordinates
            end_result = await self.gbox_client.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="click",
                target=end_desc,
            )
            end_coords = end_result.get("response", {}).get("coordinates", {})
            
            step.coordinates = {
                "start_x": start_coords.get("x", 0),
                "start_y": start_coords.get("y", 0),
                "end_x": end_coords.get("x", 0),
                "end_y": end_coords.get("y", 0),
            }
            
            if verbose:
                print(f"   📍 Swipe: ({step.coordinates['start_x']}, {step.coordinates['start_y']}) → ({step.coordinates['end_x']}, {step.coordinates['end_y']})")
            
            await self.gbox_client.swipe(
                start_x=step.coordinates["start_x"],
                start_y=step.coordinates["start_y"],
                end_x=step.coordinates["end_x"],
                end_y=step.coordinates["end_y"],
            )
        
        elif action.action_type == ActionType.SCROLL:
            rubric.num_scrolls += 1
            
            target_desc = action.target.to_description() if action.target else "screen center"
            
            # Generate coordinates
            result = await self.gbox_client.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="scroll",
                target=target_desc,
                direction=action.direction,
            )
            
            coords = result.get("response", {}).get("coordinates", {})
            x = coords.get("x", 0)
            y = coords.get("y", 0)
            step.coordinates = {"x": x, "y": y}
            
            if verbose:
                print(f"   📍 Scroll at ({x}, {y}), direction: {action.direction}")
            
            await self.gbox_client.scroll(
                x=x, y=y,
                direction=action.direction or "down",
                distance=action.distance or 300,
            )
        
        elif action.action_type == ActionType.INPUT:
            rubric.num_inputs += 1
            
            x, y = None, None
            if action.target:
                # Generate coordinates for input field
                result = await self.gbox_client.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=action.target.to_description(),
                )
                coords = result.get("response", {}).get("coordinates", {})
                x = coords.get("x")
                y = coords.get("y")
                step.coordinates = {"x": x, "y": y}
                
                if verbose:
                    print(f"   📍 Input at ({x}, {y}): \"{action.text}\"")
            else:
                if verbose:
                    print(f"   ⌨️ Input: \"{action.text}\"")
            
            await self.gbox_client.type_text(
                text=action.text or "",
                x=x, y=y,
            )
        
        elif action.action_type == ActionType.KEY_PRESS:
            rubric.num_key_presses += 1
            
            if verbose:
                print(f"   ⌨️ Key press: {action.keys}")
            
            await self.gbox_client.press_key(keys=action.keys)
        
        elif action.action_type == ActionType.BUTTON_PRESS:
            rubric.num_button_presses += 1
            
            if verbose:
                print(f"   🔘 Button press: {action.button}")
            
            await self.gbox_client.press_button(button=action.button or "home")
    
    def _print_summary(self, rubric: CUARubric):
        """Print task summary."""
        print(f"\n{'='*60}")
        print("TASK SUMMARY")
        print(f"{'='*60}")
        print(f"Completed: {rubric.task_completed}")
        print(f"Success: {rubric.task_success}")
        print(f"Message: {rubric.result_message}")
        print(f"Turns: {rubric.num_turns}/{rubric.max_turns}")
        print(f"Time: {rubric.total_time_seconds:.1f}s")
        print(f"\nActions:")
        print(f"  Clicks: {rubric.num_clicks}")
        print(f"  Swipes: {rubric.num_swipes}")
        print(f"  Scrolls: {rubric.num_scrolls}")
        print(f"  Inputs: {rubric.num_inputs}")
        print(f"  Key presses: {rubric.num_key_presses}")
        print(f"  Button presses: {rubric.num_button_presses}")
        print(f"\nErrors:")
        print(f"  Action errors: {rubric.num_action_errors}")
        print(f"  Coordinate failures: {rubric.num_coordinate_failures}")
        print(f"  Parse errors: {rubric.num_parse_errors}")
        print(f"{'='*60}\n")
    
    async def close(self):
        """Close agent resources."""
        await self.vlm.close()
        await self.gbox_client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def calculate_reward(config: CUAConfig, rubric: CUARubric) -> float:
    """Calculate reward for GRPO training.
    
    Args:
        config: Agent configuration
        rubric: Evaluation rubric
        
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Base reward for task completion
    if rubric.task_completed:
        if rubric.task_success:
            reward += config.success_reward
        else:
            reward += 0.1  # Small reward for attempting completion
    else:
        reward += config.timeout_penalty
    
    # Step penalty (encourage efficiency)
    reward += config.step_penalty * rubric.num_turns
    
    # Error penalties
    total_errors = (
        rubric.num_action_errors +
        rubric.num_coordinate_failures +
        rubric.num_parse_errors
    )
    reward += config.error_penalty * total_errors
    
    return reward

