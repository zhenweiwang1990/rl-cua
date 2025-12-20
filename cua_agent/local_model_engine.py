"""Local PyTorch/Unsloth Model Engine for AReaL.

This module provides a local inference engine that uses PyTorch or Unsloth models
directly instead of requiring a separate vLLM server.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import torch
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest

from cua_agent.model_loader import ModelLoader, create_model_loader

logger = logging.getLogger(__name__)


class LocalModelEngine(InferenceEngine):
    """Local inference engine using PyTorch/Unsloth models.
    
    This engine loads the model locally and performs inference directly,
    without requiring a separate vLLM server.
    """
    
    def __init__(
        self,
        model_loader: ModelLoader,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize local model engine.
        
        Args:
            model_loader: Model loader instance (HF or Unsloth)
            device: Device to run inference on (default: cuda if available)
            dtype: Data type for inference (default: bfloat16)
        """
        self.model_loader = model_loader
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype or torch.bfloat16
        
        # Load model
        self.model, self.tokenizer, self.processor = model_loader.load()
        
        # Move model to device and set to eval mode
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        
        logger.info(f"LocalModelEngine initialized on {self.device}")
    
    async def agenerate(self, request: ModelRequest) -> Any:
        """Generate response asynchronously.
        
        Args:
            request: Model request with input_ids, image_data, etc.
            
        Returns:
            Response object with output_tokens
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, request)
    
    def _generate_sync(self, request: ModelRequest) -> Any:
        """Synchronous generation."""
        # Prepare input
        input_ids = torch.tensor([request.input_ids], device=self.device)
        
        # Handle images if present
        if request.image_data and len(request.image_data) > 0:
            # For vision-language models, we need to use the processor
            if self.processor is not None:
                # Reconstruct messages from request
                messages = request.vision_msg_vllm[0] if request.vision_msg_vllm else []
                
                # Process with processor
                from PIL import Image
                import base64
                import io
                
                images = []
                for img_data in request.image_data:
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    images.append(img)
                
                # Use processor to prepare inputs
                inputs = self.processor(
                    images=images,
                    text=messages,
                    padding=True,
                    return_tensors="pt",
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=request.gconfig.max_new_tokens,
                        temperature=request.gconfig.temperature,
                        do_sample=not request.gconfig.greedy,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Extract new tokens
                input_length = inputs["input_ids"].shape[1]
                output_tokens = outputs[0][input_length:].cpu().tolist()
            else:
                # Fallback: use tokenizer only
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=request.gconfig.max_new_tokens,
                        temperature=request.gconfig.temperature,
                        do_sample=not request.gconfig.greedy,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                output_tokens = outputs[0][input_ids.shape[1]:].cpu().tolist()
        else:
            # Text-only generation
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=request.gconfig.max_new_tokens,
                    temperature=request.gconfig.temperature,
                    do_sample=not request.gconfig.greedy,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            output_tokens = outputs[0][input_ids.shape[1]:].cpu().tolist()
        
        # Create response object
        class Response:
            def __init__(self, tokens):
                self.output_tokens = tokens
        
        return Response(output_tokens)
    
    def submit(self, *args, **kwargs):
        """Submit request (for compatibility with RemotevLLMEngine interface)."""
        raise NotImplementedError("LocalModelEngine does not support async submission")
    
    def wait(self, *args, **kwargs):
        """Wait for completion (for compatibility with RemotevLLMEngine interface)."""
        raise NotImplementedError("LocalModelEngine does not support async submission")
    
    def pause(self):
        """Pause engine (no-op for local engine)."""
        pass
    
    def resume(self):
        """Resume engine (no-op for local engine)."""
        pass
    
    def initialize(self, *args, **kwargs):
        """Initialize engine (already done in __init__)."""
        pass
    
    def set_version(self, version: int):
        """Set model version (for LoRA updates)."""
        # Reload LoRA if needed
        if hasattr(self.model_loader, "load_lora") and version > 0:
            # In a real implementation, you'd load the LoRA adapter for this version
            pass
    
    def destroy(self):
        """Clean up resources."""
        if hasattr(self.model, "cpu"):
            self.model = self.model.cpu()
        del self.model
        torch.cuda.empty_cache()

