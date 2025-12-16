"""CUA Agent - Main agent loop for Computer Use Agent."""

import asyncio
import json
import logging
import time
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
        
        # Determine API base and key based on provider
        if config.vlm_provider == "openrouter":
            api_base = config.openrouter_api_base
            api_key = config.openrouter_api_key
        else:  # vllm
            api_base = config.vllm_api_base
            api_key = None  # vLLM doesn't require API key
        
        self.vlm = VLMInference(
            model_name=config.model_name,
            provider=config.vlm_provider,
            api_base=api_base,
            api_key=api_key,
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
                turn_start = time.time()
                rubric.num_turns = turn + 1
                
                # Initialize timing variables for this turn
                screenshot_time = 0.0
                vlm_time = 0.0
                parse_time = 0.0
                action_time = 0.0
                total_tokens = 0
                screenshot_size_kb = 0.0
                
                logger.info(f"{'='*80}")
                logger.info(f"[Turn {turn + 1}/{self.config.max_turns}] Starting turn")
                logger.info(f"{'='*80}")
                
                if verbose:
                    print(f"\n{'─'*60}")
                    print(f"Turn {turn + 1}/{self.config.max_turns}")
                    print(f"{'─'*60}")
                
                # Take screenshot
                screenshot_start = time.time()
                await asyncio.sleep(self.config.screenshot_delay)
                screenshot_bytes, screenshot_uri = await self.gbox_client.take_screenshot()
                screenshot_time = time.time() - screenshot_start
                
                screenshot_size_kb = len(screenshot_bytes) / 1024
                logger.info(
                    f"[Turn {turn + 1}] Screenshot captured: {screenshot_size_kb:.2f} KB, "
                    f"took {screenshot_time:.3f}s"
                )
                if verbose:
                    print(f"📸 Screenshot captured ({screenshot_size_kb:.2f} KB, {screenshot_time:.3f}s)")
                
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
                # Log input information
                num_messages = len(self.conversation)
                messages_summary = []
                for msg in self.conversation[-3:]:  # Last 3 messages for summary
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        content_preview = content[:100] + "..." if len(content) > 100 else content
                    else:
                        content_preview = str(content)[:100]
                    messages_summary.append(f"{role}: {content_preview}")
                
                logger.info(
                    f"[Turn {turn + 1}] VLM Input: "
                    f"{num_messages} messages, "
                    f"screenshot: {screenshot_size_kb:.2f} KB, "
                    f"temperature: {temperature:.2f}"
                )
                if verbose:
                    print(f"\n📥 Input to VLM:")
                    print(f"   Messages: {num_messages}")
                    print(f"   Screenshot: {screenshot_size_kb:.2f} KB")
                    print(f"   Temperature: {temperature:.2f}")
                    if messages_summary:
                        print(f"   Recent messages: {', '.join(messages_summary)}")
                
                # Call VLM
                vlm_start = time.time()
                raw_response, parsed_response, usage_info = await self.vlm.generate(
                    messages=self.conversation,
                    tools=self.tools,
                    image_data=screenshot_bytes,
                    temperature=temperature,
                )
                vlm_time = time.time() - vlm_start
                
                # Log VLM output and usage
                prompt_tokens = usage_info.get("prompt_tokens", 0)
                completion_tokens = usage_info.get("completion_tokens", 0)
                total_tokens = usage_info.get("total_tokens", 0)
                model_used = usage_info.get("model", "unknown")
                finish_reason = usage_info.get("finish_reason", "unknown")
                
                logger.info(
                    f"[Turn {turn + 1}] VLM Output: "
                    f"took {vlm_time:.3f}s, "
                    f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens}, "
                    f"model: {model_used}, "
                    f"finish_reason: {finish_reason}"
                )
                # Log complete VLM response
                logger.info(f"[Turn {turn + 1}] VLM Response (full):\n{raw_response}")
                
                # Log parsed response structure
                if parsed_response.get("tool_calls"):
                    logger.info(
                        f"[Turn {turn + 1}] VLM Response contains {len(parsed_response['tool_calls'])} tool call(s)"
                    )
                else:
                    logger.info(f"[Turn {turn + 1}] VLM Response contains no tool calls (text-only response)")
                
                if verbose:
                    print(f"\n🤖 Model response ({vlm_time:.3f}s):")
                    print(f"   Tokens: {prompt_tokens} (prompt) + {completion_tokens} (completion) = {total_tokens} (total)")
                    print(f"   Model: {model_used}")
                    print(f"   Finish reason: {finish_reason}")
                    print(f"\n   Full Response:")
                    print(f"   {raw_response}")
                    if parsed_response.get("tool_calls"):
                        print(f"\n   🔧 Tool Calls ({len(parsed_response['tool_calls'])}):")
                        for i, tc in enumerate(parsed_response["tool_calls"]):
                            print(f"\n   [{i+1}] Complete Tool Call:")
                            print(f"   {json.dumps(tc, indent=4, ensure_ascii=False)}")
                    if parsed_response.get("content"):
                        print(f"\n   Content: {parsed_response['content']}")
                
                # Add assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": parsed_response.get("content"),
                }
                if parsed_response.get("tool_calls"):
                    assistant_msg["tool_calls"] = parsed_response["tool_calls"]
                self.conversation.append(assistant_msg)
                
                # Parse action
                parse_start = time.time()
                action = None
                action_dict = None
                
                if parsed_response.get("tool_calls"):
                    # Log all tool calls completely
                    logger.info(
                        f"[Turn {turn + 1}] Complete Tool Calls ({len(parsed_response['tool_calls'])}):\n"
                        f"{json.dumps(parsed_response['tool_calls'], indent=2, ensure_ascii=False)}"
                    )
                    
                    if verbose:
                        print(f"\n🔧 Complete Tool Calls ({len(parsed_response['tool_calls'])}):")
                        for i, tc in enumerate(parsed_response["tool_calls"]):
                            print(f"\n   [{i+1}] {json.dumps(tc, indent=4, ensure_ascii=False)}")
                    
                    # Process first tool call for action execution
                    tool_call = parsed_response["tool_calls"][0]
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    action_dict = tool_call_to_action_dict(tool_name, tool_args)
                    action = parse_action(action_dict)
                    
                    logger.info(
                        f"[Turn {turn + 1}] Parsed tool call: {tool_name}, "
                        f"action_type: {action.action_type if action else 'None'}"
                    )
                    if verbose:
                        print(f"\n   Parsed as: {action.action_type if action else 'None'}")
                        if action and action.target:
                            print(f"   Target: {action.target.to_description()}")
                else:
                    # Try to extract action from text
                    action_dict = extract_action_from_response(raw_response)
                    if action_dict:
                        action = parse_action(action_dict)
                        logger.info(
                            f"[Turn {turn + 1}] Extracted action from text: {action.action_type if action else 'None'}"
                        )
                    else:
                        rubric.num_parse_errors += 1
                        logger.warning(f"[Turn {turn + 1}] Failed to parse action from response")
                        if verbose:
                            print(f"\n⚠️ Could not parse action from response")
                        continue
                
                parse_time = time.time() - parse_start
                logger.info(f"[Turn {turn + 1}] Action parsing took {parse_time:.3f}s")
                
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
                
                action_start = time.time()
                try:
                    logger.info(
                        f"[Turn {turn + 1}] Executing action: {action.action_type}, "
                        f"target: {action.target.to_description() if action.target else 'None'}"
                    )
                    await self._execute_action(
                        action, screenshot_uri, step, rubric, verbose
                    )
                    action_time = time.time() - action_start
                    logger.info(
                        f"[Turn {turn + 1}] Action executed successfully in {action_time:.3f}s, "
                        f"coordinates: {step.coordinates}"
                    )
                except Exception as e:
                    action_time = time.time() - action_start
                    step.success = False
                    step.error_message = str(e)
                    rubric.num_action_errors += 1
                    logger.error(
                        f"[Turn {turn + 1}] Action execution failed after {action_time:.3f}s: {e}"
                    )
                    if verbose:
                        print(f"\n❌ Action error ({action_time:.3f}s): {e}")
                
                # Wait after action
                await asyncio.sleep(self.config.action_delay)
                
                # Log turn summary
                turn_time = time.time() - turn_start
                logger.info(
                    f"[Turn {turn + 1}] Turn completed in {turn_time:.3f}s: "
                    f"screenshot: {screenshot_time:.3f}s, "
                    f"vlm: {vlm_time:.3f}s ({total_tokens} tokens), "
                    f"parse: {parse_time:.3f}s, "
                    f"action: {action_time:.3f}s, "
                    f"success: {step.success}"
                )
                if verbose:
                    print(f"\n⏱️ Turn {turn + 1} total time: {turn_time:.3f}s")
                    print(f"   Breakdown: screenshot={screenshot_time:.3f}s, "
                          f"vlm={vlm_time:.3f}s ({total_tokens} tokens), "
                          f"parse={parse_time:.3f}s, action={action_time:.3f}s")
                
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
                
                # Log turn summary
                turn_time = time.time() - turn_start
                logger.info(
                    f"[Turn {turn + 1}] Turn completed in {turn_time:.3f}s: "
                    f"screenshot: {screenshot_time:.3f}s, "
                    f"vlm: {vlm_time:.3f}s ({total_tokens} tokens), "
                    f"parse: {parse_time:.3f}s, "
                    f"action: {action_time:.3f}s, "
                    f"success: {step.success}"
                )
                if verbose:
                    print(f"\n⏱️ Turn {turn + 1} total time: {turn_time:.3f}s")
                    print(f"   Breakdown: screenshot={screenshot_time:.3f}s, "
                          f"vlm={vlm_time:.3f}s ({total_tokens} tokens), "
                          f"parse={parse_time:.3f}s, action={action_time:.3f}s")
            
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
            
            # Use drag type to get both start and end coordinates in one call
            # According to GBox docs: https://docs.gbox.ai/api-reference/model/generate-coordinates-for-a-model
            drag_result = await self.gbox_client.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="drag",
                target=start_desc,
                end_target=end_desc,
            )
            
            # Parse drag response - may return start/end coordinates or single coordinates
            response_data = drag_result.get("response", {})
            coordinates = response_data.get("coordinates", {})
            
            # Check if response has separate start/end coordinates
            if "start" in coordinates and "end" in coordinates:
                start_coords = coordinates.get("start", {})
                end_coords = coordinates.get("end", {})
                step.coordinates = {
                    "start_x": start_coords.get("x", 0),
                    "start_y": start_coords.get("y", 0),
                    "end_x": end_coords.get("x", 0),
                    "end_y": end_coords.get("y", 0),
                }
            else:
                # Fallback: if drag doesn't return both, use separate calls
                start_result = await self.gbox_client.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=start_desc,
                )
                start_coords = start_result.get("response", {}).get("coordinates", {})
                
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
            
            # For scroll, use "location" field - default to "screen center" if no target
            target_desc = action.target.to_description() if action.target else "center of the screen"
            
            # Generate coordinates using scroll type (which uses "location" field)
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

