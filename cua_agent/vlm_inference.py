"""VLM (Vision-Language Model) inference module.

This module provides inference capabilities for the Qwen3-VL model
using vLLM or OpenRouter API.
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
    """Vision-Language Model inference.
    
    Supports multiple providers:
    1. vLLM server mode: Connect to a running vLLM server via OpenAI-compatible API
    2. vLLM local mode: Load model directly (requires more GPU memory)
    3. OpenRouter: Use OpenRouter API for inference
    """
    
    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-VL-30B-A3B-Instruct",
        provider: str = "vllm",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Initialize VLM inference.
        
        Args:
            model_name: Model name/path
            provider: Provider to use ("vllm" or "openrouter")
            api_base: API base URL (for vLLM server or OpenRouter)
            api_key: API key (required for OpenRouter, "EMPTY" for local vLLM)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self._llm = None
        self._tokenizer = None
        self._client = None
        
        # Validate provider
        if self.provider not in ["vllm", "openrouter"]:
            raise ValueError(f"Invalid provider: {provider}. Must be 'vllm' or 'openrouter'")
        
        # Setup API client based on provider
        if self.provider == "openrouter":
            if not api_base:
                api_base = "https://openrouter.ai/api/v1"
            if not api_key:
                raise ValueError("api_key is required when using OpenRouter provider")
            self.api_base = api_base
            self._client = httpx.AsyncClient(
                base_url=api_base,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/rl-cua",  # Optional: for OpenRouter
                    "X-Title": "CUA Agent",  # Optional: for OpenRouter
                },
                timeout=300.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            logger.info(f"Initialized OpenRouter client: {api_base}")
        elif self.provider == "vllm" and api_base:
            # vLLM server mode
            self._client = httpx.AsyncClient(
                base_url=api_base,
                headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                timeout=300.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            logger.info(f"Initialized vLLM server client: {api_base}")
    
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
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Generate a response from the VLM.
        
        Args:
            messages: Conversation messages
            tools: Tool definitions
            image_data: Optional image bytes to include
            temperature: Override temperature
            
        Returns:
            Tuple of (raw_response, parsed_response, usage_info)
            usage_info contains: prompt_tokens, completion_tokens, total_tokens, model, etc.
        """
        if self.provider == "openrouter" or (self.provider == "vllm" and self.api_base):
            return await self._generate_server(
                messages, tools, image_data, temperature
            )
        else:
            # vLLM local mode
            return await self._generate_local(
                messages, tools, image_data, temperature
            )
    
    async def _generate_server(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[bytes] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Generate using server API (vLLM or OpenRouter).
        
        Args:
            messages: Conversation messages
            tools: Tool definitions
            image_data: Optional image bytes
            temperature: Override temperature
            
        Returns:
            Tuple of (raw_response, parsed_response, usage_info)
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
        
        # Map model name for OpenRouter if needed
        model_name = self.model_name
        if self.provider == "openrouter":
            # OpenRouter model name mapping
            # Map HuggingFace/ModelScope model names to OpenRouter model names
            # Note: OpenRouter model IDs are case-sensitive and use lowercase
            model_mapping = {
                # Qwen3-VL-30B-A3B (recommended - available on OpenRouter)
                "unsloth/Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "qwen/Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                # Qwen3-VL-32B (may be unavailable on OpenRouter)
                "unsloth/Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                "Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                "qwen/Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                "Qwen/Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                # Qwen3-VL-8B (alternative smaller model)
                "unsloth/Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                "Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                "qwen/Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                "Qwen/Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                # Qwen2.5-VL models
                "Qwen/Qwen2.5-VL-32B-Instruct": "qwen/qwen2.5-vl-32b-instruct",
                "qwen/Qwen2.5-VL-32B-Instruct": "qwen/qwen2.5-vl-32b-instruct",
            }
            model_name = model_mapping.get(model_name, model_name)
            # If model name still contains "/", assume it's already in OpenRouter format
            if "/" in model_name and model_name not in model_mapping.values():
                # Convert to lowercase for OpenRouter format (e.g., qwen/qwen3-vl-32b-instruct)
                parts = model_name.split("/")
                if len(parts) == 2:
                    model_name = f"{parts[0].lower()}/{parts[1].lower()}"
        
        payload = {
            "model": model_name,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": self.top_p,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.ReadError as e:
            provider_name = "OpenRouter" if self.provider == "openrouter" else "vLLM"
            logger.error(f"Failed to read from {provider_name} server at {self.api_base}: {e}")
            raise RuntimeError(
                f"Cannot connect to {provider_name} server at {self.api_base}. "
                f"Please ensure the server is running and accessible."
            ) from e
        except httpx.ConnectError as e:
            provider_name = "OpenRouter" if self.provider == "openrouter" else "vLLM"
            logger.error(f"Failed to connect to {provider_name} server at {self.api_base}: {e}")
            raise RuntimeError(
                f"Cannot connect to {provider_name} server at {self.api_base}. "
                f"Please check if the server is running and accessible."
            ) from e
        except httpx.HTTPStatusError as e:
            provider_name = "OpenRouter" if self.provider == "openrouter" else "vLLM"
            error_detail = ""
            try:
                error_body = e.response.json()
                error_detail = f": {error_body}"
                
                # Provide helpful error message for OpenRouter 404
                if self.provider == "openrouter" and e.response.status_code == 404:
                    error_msg = error_body.get("error", {}).get("message", "")
                    if "No endpoints found" in error_msg:
                        logger.error(
                            f"Model '{model_name}' is not available on OpenRouter. "
                            f"This may mean:\n"
                            f"  1. The model is temporarily unavailable\n"
                            f"  2. The model name is incorrect\n"
                            f"  3. You need to check available models at https://openrouter.ai/models\n"
                            f"  Try using an alternative model like 'qwen/qwen2.5-vl-32b-instruct' or 'qwen/qwen3-vl-8b-instruct'"
                        )
            except:
                error_detail = f": {e.response.text}"
            logger.error(f"{provider_name} API error: {e.response.status_code}{error_detail}")
            raise RuntimeError(
                f"{provider_name} API error: {e.response.status_code}{error_detail}"
            ) from e
        
        result = response.json()
        choice = result["choices"][0]
        message = choice["message"]
        
        raw_content = message.get("content", "") or ""
        
        # Parse response
        parsed = {
            "content": raw_content,
            "tool_calls": message.get("tool_calls"),
        }
        
        # Extract usage information
        usage_info = {
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": result.get("usage", {}).get("total_tokens", 0),
            "model": result.get("model", self.model_name),
            "finish_reason": choice.get("finish_reason", "unknown"),
        }
        
        # OpenRouter specific fields
        if self.provider == "openrouter":
            usage_info["prompt_tokens_details"] = result.get("usage", {}).get("prompt_tokens_details", {})
            usage_info["completion_tokens_details"] = result.get("usage", {}).get("completion_tokens_details", {})
        
        return raw_content, parsed, usage_info
    
    async def _generate_local(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[bytes] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Generate using local vLLM model.
        
        Args:
            messages: Conversation messages
            tools: Tool definitions
            image_data: Optional image bytes
            temperature: Override temperature
            
        Returns:
            Tuple of (raw_response, parsed_response, usage_info)
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
        
        # Extract usage information from vLLM output
        usage_info = {
            "prompt_tokens": len(outputs[0].prompt_token_ids) if hasattr(outputs[0], "prompt_token_ids") else 0,
            "completion_tokens": len(outputs[0].outputs[0].token_ids) if hasattr(outputs[0].outputs[0], "token_ids") else 0,
            "total_tokens": 0,
            "model": self.model_name,
            "finish_reason": outputs[0].outputs[0].finish_reason if hasattr(outputs[0].outputs[0], "finish_reason") else "unknown",
        }
        usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"]
        
        return raw_content, parsed, usage_info
    
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
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models from OpenRouter (OpenRouter only).
        
        Returns:
            List of available model dictionaries
        """
        if self.provider != "openrouter" or not self._client:
            raise ValueError("list_available_models is only available for OpenRouter provider")
        
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to list models from OpenRouter: {e}")
            raise
    
    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

