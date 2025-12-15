"""VLM (Vision-Language Model) inference module using vLLM.

This module provides inference capabilities for the Qwen3-VL model
using vLLM for efficient GPU inference.
"""

import base64
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class VLMInference:
    """Vision-Language Model inference using vLLM.
    
    Supports two modes:
    1. Server mode: Connect to a running vLLM server via OpenAI-compatible API
    2. Local mode: Load model directly (requires more GPU memory)
    """
    
    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-VL-32B-Instruct",
        api_base: Optional[str] = None,
        api_key: str = "EMPTY",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Initialize VLM inference.
        
        Args:
            model_name: Model name/path
            api_base: vLLM server base URL (e.g., "http://localhost:8000/v1")
            api_key: API key (default "EMPTY" for local vLLM)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
        """
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self._llm = None
        self._tokenizer = None
        self._client = None
        
        if api_base:
            # Server mode
            self._client = httpx.AsyncClient(
                base_url=api_base,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=300.0,
            )
    
    async def _ensure_local_model(self):
        """Lazily load local model if not using server mode."""
        if self._llm is None and self.api_base is None:
            try:
                from vllm import LLM, SamplingParams
                from transformers import AutoProcessor
                
                logger.info(f"Loading model: {self.model_name}")
                self._llm = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    max_model_len=32768,
                    limit_mm_per_prompt={"image": 1},
                )
                self._tokenizer = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
                logger.info("Model loaded successfully")
            except ImportError as e:
                raise ImportError(
                    "vLLM not installed. Install with: pip install vllm"
                ) from e
    
    def _encode_image(self, image_data: bytes) -> str:
        """Encode image bytes to base64 data URI.
        
        Args:
            image_data: Image bytes
            
        Returns:
            Base64 data URI
        """
        base64_data = base64.b64encode(image_data).decode("utf-8")
        
        # Detect image format
        if image_data[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = "image/png"
        elif image_data[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        else:
            mime_type = "image/png"  # Default
        
        return f"data:{mime_type};base64,{base64_data}"
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[bytes] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a response from the VLM.
        
        Args:
            messages: Conversation messages
            tools: Tool definitions
            image_data: Optional image bytes to include
            temperature: Override temperature
            
        Returns:
            Tuple of (raw_response, parsed_response)
        """
        if self.api_base:
            return await self._generate_server(
                messages, tools, image_data, temperature
            )
        else:
            return await self._generate_local(
                messages, tools, image_data, temperature
            )
    
    async def _generate_server(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[bytes] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate using vLLM server API.
        
        Args:
            messages: Conversation messages
            tools: Tool definitions
            image_data: Optional image bytes
            temperature: Override temperature
            
        Returns:
            Tuple of (raw_response, parsed_response)
        """
        if not self._client:
            raise RuntimeError("Client not initialized")
        
        # Prepare messages with image if provided
        api_messages = []
        for msg in messages:
            if msg["role"] == "user" and image_data:
                # Add image to user message
                content = msg.get("content", "")
                image_uri = self._encode_image(image_data)
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_uri}
                        },
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                })
                # Only add image to first user message
                image_data = None
            else:
                api_messages.append(msg)
        
        payload = {
            "model": self.model_name,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": self.top_p,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        
        result = response.json()
        choice = result["choices"][0]
        message = choice["message"]
        
        raw_content = message.get("content", "") or ""
        
        # Parse response
        parsed = {
            "content": raw_content,
            "tool_calls": message.get("tool_calls"),
        }
        
        return raw_content, parsed
    
    async def _generate_local(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[bytes] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate using local vLLM model.
        
        Args:
            messages: Conversation messages
            tools: Tool definitions
            image_data: Optional image bytes
            temperature: Override temperature
            
        Returns:
            Tuple of (raw_response, parsed_response)
        """
        await self._ensure_local_model()
        
        from vllm import SamplingParams
        
        # Process image if provided
        images = []
        if image_data:
            img = Image.open(BytesIO(image_data))
            images.append(img)
        
        # Apply chat template
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        sampling_params = SamplingParams(
            temperature=temperature or self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        
        # Generate
        if images:
            outputs = self._llm.generate(
                [{
                    "prompt": text,
                    "multi_modal_data": {"image": images[0]},
                }],
                sampling_params=sampling_params,
            )
        else:
            outputs = self._llm.generate([text], sampling_params=sampling_params)
        
        raw_content = outputs[0].outputs[0].text
        
        # Parse tool calls
        parsed = self._parse_tool_calls(raw_content)
        
        return raw_content, parsed
    
    def _parse_tool_calls(self, response: str) -> Dict[str, Any]:
        """Parse tool calls from model response.
        
        Args:
            response: Model response text
            
        Returns:
            Parsed response with tool_calls if found
        """
        result = {"content": response, "tool_calls": None}
        
        # Try <tool_call> tags
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        if matches:
            tool_calls = []
            for i, match in enumerate(matches):
                try:
                    tool_data = json.loads(match.strip())
                    tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tool_data.get("name", ""),
                            "arguments": json.dumps(tool_data.get("arguments", {})),
                        }
                    })
                except json.JSONDecodeError:
                    continue
            
            if tool_calls:
                result["tool_calls"] = tool_calls
                return result
        
        # Try direct JSON with name and arguments
        try:
            json_match = re.search(
                r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',
                response,
                re.DOTALL
            )
            if json_match:
                parsed = json.loads(json_match.group())
                if "name" in parsed and "arguments" in parsed:
                    args = parsed["arguments"]
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    result["tool_calls"] = [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": args,
                        }
                    }]
                    return result
        except (json.JSONDecodeError, KeyError):
            pass
        
        return result
    
    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

