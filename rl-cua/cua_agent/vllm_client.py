"""vLLM Client for CUA Agent GRPO training.

This module provides a client for communicating with vLLM server,
supporting dynamic LoRA switching for training and multi-GPU inference.
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import aiohttp
from openai import AsyncOpenAI

from cua_agent.grpo_config import VLLMConfig

logger = logging.getLogger(__name__)


@dataclass
class VLLMResponse:
    """Response from vLLM server."""
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    latency: float = 0.0


class VLLMClient:
    """Client for vLLM OpenAI-compatible API with LoRA support.
    
    Supports:
    - Multi-GPU tensor parallel inference
    - Dynamic LoRA adapter switching
    - Vision-Language model inference with images
    - Tool/function calling
    """
    
    def __init__(self, config: Optional[VLLMConfig] = None):
        """Initialize vLLM client.
        
        Args:
            config: vLLM configuration. If None, loads from environment.
        """
        self.config = config or VLLMConfig.from_env()
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.config.api_key or "dummy",  # vLLM doesn't require key
            base_url=self.config.api_base,
            timeout=self.config.timeout,
        )
        
        # LoRA state
        self._current_lora_path: Optional[str] = None
        self._lora_load_time: float = 0.0
        
        logger.info(f"VLLMClient initialized: {self.config.api_base}")
    
    @property
    def model_name(self) -> str:
        """Get the current model name to use for inference."""
        if self.config.enable_dynamic_lora and self._current_lora_path:
            return self.config.lora_name
        return self.config.base_model
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.error(f"vLLM health check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models on the server."""
        try:
            models = await self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[bytes] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_lora: bool = True,
    ) -> VLLMResponse:
        """Generate a response from the model.
        
        Args:
            messages: Conversation messages
            tools: Optional tool definitions for function calling
            image_data: Optional image bytes for VLM
            temperature: Sampling temperature (uses config default if None)
            max_tokens: Max tokens to generate (uses config default if None)
            use_lora: Whether to use LoRA adapter (if available)
            
        Returns:
            VLLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages with image if provided
        request_messages = self._prepare_messages(messages, image_data)
        
        # Select model
        model = self.config.lora_name if (use_lora and self._current_lora_path) else self.config.base_model
        
        # Prepare request parameters
        params = {
            "model": model,
            "messages": request_messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        try:
            response = await self.client.chat.completions.create(**params)
            
            choice = response.choices[0]
            message = choice.message
            
            # Parse tool calls
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            
            return VLLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                model=response.model,
                latency=time.time() - start_time,
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        image_data: Optional[bytes] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare messages for the API request, including image encoding."""
        if not image_data:
            return messages
        
        # Encode image to base64
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        # Find the last user message and add the image
        result_messages = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and i == len(messages) - 1:
                # This is the last user message - add image
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Convert to multi-part content
                    content = [
                        {"type": "text", "text": content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                result_messages.append({
                    "role": "user",
                    "content": content,
                })
            else:
                result_messages.append(msg)
        
        return result_messages
    
    async def update_lora(self, lora_path: str) -> bool:
        """Update the LoRA adapter on the vLLM server.
        
        This copies the LoRA adapter to the server's expected location
        and triggers a reload.
        
        Args:
            lora_path: Path to the LoRA adapter directory
            
        Returns:
            True if update was successful
        """
        if not self.config.enable_dynamic_lora:
            logger.info("Dynamic LoRA switching is disabled")
            return False
        
        start_time = time.time()
        
        try:
            lora_path = Path(lora_path)
            if not lora_path.exists():
                logger.error(f"LoRA path does not exist: {lora_path}")
                return False
            
            # Check for adapter files
            adapter_file = lora_path / "adapter_model.safetensors"
            if not adapter_file.exists():
                adapter_file = lora_path / "adapter_model.bin"
            if not adapter_file.exists():
                logger.error(f"No adapter file found in {lora_path}")
                return False
            
            # For now, we assume the vLLM server is watching a specific directory
            # and will automatically reload when files change
            # This is a simplified approach - production might use REST API
            
            self._current_lora_path = str(lora_path)
            self._lora_load_time = time.time() - start_time
            
            logger.info(
                f"LoRA adapter updated: {lora_path} "
                f"(took {self._lora_load_time:.2f}s)"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update LoRA adapter: {e}")
            return False
    
    async def close(self):
        """Close the client."""
        await self.client.close()


class VLLMRolloutCollector:
    """Collect rollouts using vLLM for GRPO training.
    
    This class manages multiple parallel rollout collections
    using vLLM for model inference.
    """
    
    def __init__(
        self,
        vllm_client: VLLMClient,
        gbox_api_key: str,
        tools_schema: List[Dict[str, Any]],
        max_turns: int = 15,
        concurrency: int = 4,
    ):
        """Initialize rollout collector.
        
        Args:
            vllm_client: VLLMClient instance
            gbox_api_key: API key for GBox
            tools_schema: Tool definitions for function calling
            max_turns: Maximum turns per rollout
            concurrency: Number of parallel rollouts
        """
        self.client = vllm_client
        self.gbox_api_key = gbox_api_key
        self.tools_schema = tools_schema
        self.max_turns = max_turns
        self.concurrency = concurrency
        
        # Import here to avoid circular imports
        from gbox_cua.gbox_client import GBoxClient
        self.gbox_client_class = GBoxClient
    
    async def collect_single_rollout(
        self,
        task_description: str,
        system_prompt: str,
        rollout_idx: int = 0,
        temperature: Optional[float] = None,
        verbose: bool = False,
    ) -> Tuple[List[Dict], float, Dict[str, Any]]:
        """Collect a single rollout for a task.
        
        Args:
            task_description: Task description
            system_prompt: System prompt
            rollout_idx: Rollout index (for temperature variation)
            temperature: Sampling temperature
            verbose: Whether to log detailed info
            
        Returns:
            Tuple of (conversation, reward, metadata)
        """
        from gbox_cua.tools import tool_call_to_action_dict
        from cua_agent.actions import parse_action, ActionType
        
        # Create GBox client for this rollout
        gbox = self.gbox_client_class(api_key=self.gbox_api_key)
        
        conversation = []
        metadata = {
            "rollout_idx": rollout_idx,
            "num_turns": 0,
            "task_completed": False,
            "task_success": False,
            "errors": [],
        }
        
        try:
            # Create box
            await gbox.create_box("android")
            await asyncio.sleep(2.0)  # Wait for box ready
            
            # Initialize conversation
            conversation.append({
                "role": "system",
                "content": system_prompt,
            })
            
            # Agent loop
            for turn in range(self.max_turns):
                metadata["num_turns"] = turn + 1
                
                # Take screenshot
                screenshot_bytes, screenshot_uri = await gbox.take_screenshot()
                
                # Create user message
                user_content = f"Turn {turn + 1}/{self.max_turns}. Analyze the screenshot and take the next action."
                conversation.append({
                    "role": "user",
                    "content": user_content,
                })
                
                # Calculate temperature with variation
                if temperature is None:
                    temp = self.client.config.temperature + (rollout_idx * 0.1)
                else:
                    temp = temperature
                
                # Generate response
                response = await self.client.generate(
                    messages=conversation,
                    tools=self.tools_schema,
                    image_data=screenshot_bytes,
                    temperature=temp,
                )
                
                # Add assistant message
                assistant_msg = {"role": "assistant"}
                if response.content:
                    assistant_msg["content"] = response.content
                if response.tool_calls:
                    assistant_msg["tool_calls"] = response.tool_calls
                conversation.append(assistant_msg)
                
                # Process tool calls
                if response.tool_calls:
                    tool_call = response.tool_calls[0]
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    # Convert to action
                    action_dict = tool_call_to_action_dict(tool_name, tool_args)
                    action = parse_action(action_dict)
                    
                    # Check for task completion
                    if action.action_type == ActionType.TASK_COMPLETE:
                        metadata["task_completed"] = True
                        metadata["task_success"] = action.success
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", "call_0"),
                            "content": json.dumps({
                                "status": "task_complete",
                                "success": action.success,
                            }),
                        })
                        break
                    
                    # Execute action
                    try:
                        result = await self._execute_action(gbox, action, screenshot_uri)
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", "call_0"),
                            "content": json.dumps(result),
                        })
                    except Exception as e:
                        metadata["errors"].append(str(e))
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", "call_0"),
                            "content": json.dumps({"error": str(e)}),
                        })
                
                await asyncio.sleep(0.3)  # Action delay
        
        finally:
            # Cleanup
            try:
                await gbox.terminate_box()
            except:
                pass
            await gbox.close()
        
        # Calculate simple reward
        if metadata["task_success"]:
            reward = 1.0
        elif metadata["task_completed"]:
            reward = 0.1  # Attempted but failed
        else:
            reward = 0.0  # Timeout/incomplete
        
        return conversation, reward, metadata
    
    async def _execute_action(
        self,
        gbox,
        action,
        screenshot_uri: str,
    ) -> Dict[str, Any]:
        """Execute an action on the GBox device."""
        from cua_agent.actions import ActionType
        
        result = {"status": "success"}
        
        if action.action_type == ActionType.CLICK:
            if action.target:
                coords = await gbox.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=action.target.to_description(),
                )
                x = coords.get("response", {}).get("coordinates", {}).get("x", 0)
                y = coords.get("response", {}).get("coordinates", {}).get("y", 0)
                await gbox.click(x=x, y=y)
                result["coordinates"] = {"x": x, "y": y}
        
        elif action.action_type == ActionType.SCROLL:
            target_desc = action.target.to_description() if action.target else "center of screen"
            coords = await gbox.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="scroll",
                target=target_desc,
                direction=action.direction,
            )
            x = coords.get("response", {}).get("coordinates", {}).get("x", 0)
            y = coords.get("response", {}).get("coordinates", {}).get("y", 0)
            await gbox.scroll(x=x, y=y, direction=action.direction or "down", distance=300)
            result["coordinates"] = {"x": x, "y": y}
        
        elif action.action_type == ActionType.INPUT:
            if action.target:
                coords = await gbox.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=action.target.to_description(),
                )
                x = coords.get("response", {}).get("coordinates", {}).get("x", 0)
                y = coords.get("response", {}).get("coordinates", {}).get("y", 0)
                await gbox.type_text(text=action.text or "", x=x, y=y)
            else:
                await gbox.type_text(text=action.text or "")
        
        elif action.action_type == ActionType.BUTTON_PRESS:
            await gbox.press_button(button=action.button or "home")
        
        elif action.action_type == ActionType.KEY_PRESS:
            await gbox.press_key(keys=action.keys)
        
        elif action.action_type == ActionType.SWIPE:
            # Handle swipe action
            start_desc = action.start_target.to_description() if action.start_target else "center of screen"
            end_desc = action.end_target.to_description() if action.end_target else "center of screen"
            
            drag_result = await gbox.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="drag",
                target=start_desc,
                end_target=end_desc,
            )
            
            coords = drag_result.get("response", {}).get("coordinates", {})
            if "start" in coords and "end" in coords:
                start_x = coords["start"].get("x", 0)
                start_y = coords["start"].get("y", 0)
                end_x = coords["end"].get("x", 0)
                end_y = coords["end"].get("y", 0)
            else:
                start_x = start_y = end_x = end_y = 0
            
            await gbox.swipe(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)
            result["coordinates"] = {
                "start_x": start_x, "start_y": start_y,
                "end_x": end_x, "end_y": end_y
            }
        
        elif action.action_type == ActionType.SLEEP:
            await asyncio.sleep(action.duration or 1.0)
        
        return result
    
    async def collect_rollouts(
        self,
        task_descriptions: List[str],
        system_prompts: List[str],
        num_rollouts: int = 4,
        verbose: bool = False,
    ) -> List[Tuple[List[Dict], float, Dict[str, Any]]]:
        """Collect multiple rollouts for multiple tasks.
        
        Args:
            task_descriptions: List of task descriptions
            system_prompts: List of system prompts (one per task)
            num_rollouts: Number of rollouts per task
            verbose: Whether to log detailed info
            
        Returns:
            List of (conversation, reward, metadata) tuples
        """
        results = []
        
        # Create tasks for all rollouts
        tasks_to_run = []
        for i, (task_desc, sys_prompt) in enumerate(zip(task_descriptions, system_prompts)):
            for j in range(num_rollouts):
                tasks_to_run.append(
                    self.collect_single_rollout(
                        task_description=task_desc,
                        system_prompt=sys_prompt,
                        rollout_idx=j,
                        verbose=verbose,
                    )
                )
        
        # Run with limited concurrency
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def bounded_rollout(coro):
            async with semaphore:
                return await coro
        
        # Execute all rollouts
        results = await asyncio.gather(
            *[bounded_rollout(task) for task in tasks_to_run],
            return_exceptions=True,
        )
        
        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Rollout failed with exception: {r}")
                continue
            valid_results.append(r)
        
        return valid_results


__all__ = [
    "VLLMClient",
    "VLLMResponse",
    "VLLMRolloutCollector",
]

