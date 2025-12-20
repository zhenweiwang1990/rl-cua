"""Unsloth-based VLM Inference for CUA Agent GRPO training.

This module provides HuggingFace + Unsloth based inference,
replacing vLLM for local rollout collection.

Architecture:
- Uses Unsloth for efficient VLM loading with LoRA
- Supports vision + cross-modal LoRA targeting
- Runs inference locally on GPU
"""

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class UnslothInferenceConfig:
    """Configuration for Unsloth-based inference."""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_seq_length: int = 8192
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    
    # Vision + Cross-modal LoRA targets
    # These target both language and vision components
    lora_target_modules: List[str] = field(default_factory=lambda: [
        # Language model projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # Vision encoder projections (for cross-modal)
        "visual.attn.q_proj", "visual.attn.k_proj", "visual.attn.v_proj",
        "visual.attn.o_proj",
        # Cross-attention / merger layers
        "merger.mlp",
    ])
    
    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Device settings
    device: str = "cuda"
    
    @classmethod
    def from_env(cls) -> "UnslothInferenceConfig":
        """Create config from environment variables."""
        import os
        
        def get_env_int(key, default):
            val = os.environ.get(key, str(default))
            return int(val.split('#')[0].strip())
        
        def get_env_float(key, default):
            val = os.environ.get(key, str(default))
            return float(val.split('#')[0].strip())
        
        def get_env_bool(key, default):
            val = os.environ.get(key, str(default)).lower().strip()
            return val in ("true", "1", "yes")
        
        return cls(
            model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"),
            max_seq_length=get_env_int("MAX_SEQ_LENGTH", 8192),
            load_in_4bit=get_env_bool("LOAD_IN_4BIT", "true"),
            use_lora=get_env_bool("USE_LORA", "true"),
            lora_r=get_env_int("LORA_R", 16),
            lora_alpha=get_env_int("LORA_ALPHA", 32),
            max_new_tokens=get_env_int("MAX_NEW_TOKENS", 1024),
            temperature=get_env_float("TEMPERATURE", 0.7),
            top_p=get_env_float("TOP_P", 0.9),
        )


@dataclass
class InferenceResponse:
    """Response from Unsloth inference."""
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    output_ids: Optional[List[int]] = None
    output_logprobs: Optional[List[float]] = None
    finish_reason: str = ""
    latency: float = 0.0
    
    def get_text(self) -> str:
        """Get response text content."""
        return self.content or ""


class UnslothVLMInference:
    """Unsloth-based Vision Language Model inference.
    
    This class handles:
    - Model loading with Unsloth optimizations
    - LoRA adapter application (vision + cross-modal)
    - VLM inference with images
    - Tool call parsing
    """
    
    def __init__(
        self,
        config: Optional[UnslothInferenceConfig] = None,
        model=None,
        tokenizer=None,
        processor=None,
    ):
        """Initialize Unsloth VLM inference.
        
        Args:
            config: Inference configuration
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            processor: Pre-loaded processor (optional)
        """
        self.config = config or UnslothInferenceConfig()
        self._model = model
        self._tokenizer = tokenizer
        self._processor = processor
        self._lora_applied = False
        
        # Lazy loading
        if self._model is None and self._tokenizer is None:
            self._load_model()
    
    def _load_model(self):
        """Load model with Unsloth optimizations."""
        try:
            from unsloth import FastVisionModel
            logger.info(f"Loading model with Unsloth: {self.config.model_name}")
            
            # Determine dtype
            if self.config.dtype == "bfloat16":
                dtype = torch.bfloat16
            elif self.config.dtype == "float16":
                dtype = torch.float16
            else:
                dtype = None
            
            # Load with Unsloth
            self._model, self._tokenizer = FastVisionModel.from_pretrained(
                self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=dtype,
                load_in_4bit=self.config.load_in_4bit,
            )
            
            # Try to load processor
            try:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"Could not load processor: {e}")
                self._processor = None
            
            logger.info("Model loaded successfully with Unsloth")
            
        except ImportError:
            logger.warning("Unsloth not available, falling back to HuggingFace")
            self._load_model_hf()
    
    def _load_model_hf(self):
        """Fallback: Load model with HuggingFace transformers."""
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        
        logger.info(f"Loading model with HuggingFace: {self.config.model_name}")
        
        # Quantization config
        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        # Determine dtype
        if self.config.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.config.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # Load processor
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
        except Exception:
            self._processor = None
        
        logger.info("Model loaded successfully with HuggingFace")
    
    def apply_lora(
        self,
        target_modules: Optional[List[str]] = None,
        r: Optional[int] = None,
        alpha: Optional[int] = None,
        dropout: Optional[float] = None,
    ):
        """Apply LoRA adapters for training.
        
        Targets both vision and language components for cross-modal learning.
        
        Args:
            target_modules: Modules to apply LoRA to
            r: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout
        """
        if self._lora_applied:
            logger.warning("LoRA already applied, skipping")
            return self._model
        
        target_modules = target_modules or self.config.lora_target_modules
        r = r or self.config.lora_r
        alpha = alpha or self.config.lora_alpha
        dropout = dropout if dropout is not None else self.config.lora_dropout
        
        try:
            from unsloth import FastVisionModel
            
            # Filter target modules to only include valid ones
            # Unsloth may not support all module names
            valid_targets = self._get_valid_lora_targets(target_modules)
            
            self._model = FastVisionModel.get_peft_model(
                self._model,
                r=r,
                target_modules=valid_targets,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            
            self._lora_applied = True
            logger.info(f"LoRA applied with Unsloth: r={r}, alpha={alpha}, targets={valid_targets}")
            
        except ImportError:
            # Fallback to PEFT
            from peft import LoraConfig, get_peft_model, TaskType
            
            valid_targets = self._get_valid_lora_targets(target_modules)
            
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=valid_targets,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self._model = get_peft_model(self._model, lora_config)
            self._lora_applied = True
            logger.info(f"LoRA applied with PEFT: r={r}, alpha={alpha}, targets={valid_targets}")
        
        return self._model
    
    def _get_valid_lora_targets(self, target_modules: List[str]) -> List[str]:
        """Filter target modules to only include valid ones in the model."""
        valid_targets = []
        all_module_names = set()
        
        for name, _ in self._model.named_modules():
            all_module_names.add(name)
            # Also add the last part of the name for matching
            if '.' in name:
                all_module_names.add(name.split('.')[-1])
        
        for target in target_modules:
            # Check if this target exists in the model
            if target in all_module_names:
                valid_targets.append(target)
            else:
                # Check if it's a partial match (e.g., "q_proj" matches "model.layers.0.self_attn.q_proj")
                for module_name in all_module_names:
                    if module_name.endswith(target) or target in module_name:
                        if target not in valid_targets:
                            valid_targets.append(target)
                        break
        
        # If no valid targets found, use common defaults
        if not valid_targets:
            logger.warning(f"No valid LoRA targets found, using defaults")
            valid_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        return valid_targets
    
    @property
    def model(self):
        """Get the loaded model."""
        return self._model
    
    @property
    def tokenizer(self):
        """Get the loaded tokenizer."""
        return self._tokenizer
    
    @property
    def processor(self):
        """Get the loaded processor."""
        return self._processor
    
    def set_train_mode(self, train: bool = True):
        """Set model to training or evaluation mode."""
        if train:
            self._model.train()
        else:
            self._model.eval()
    
    @torch.inference_mode()
    def generate(
        self,
        messages: List[Dict[str, Any]],
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        return_logprobs: bool = False,
    ) -> InferenceResponse:
        """Generate response from the model.
        
        Args:
            messages: Conversation messages in OpenAI format
            image: PIL Image (optional)
            image_bytes: Raw image bytes (optional)
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            return_logprobs: Whether to return log probabilities
            
        Returns:
            InferenceResponse with generated content
        """
        start_time = time.time()
        
        # Process image
        if image_bytes is not None and image is None:
            image = Image.open(io.BytesIO(image_bytes))
        
        # Set generation params
        temperature = temperature or self.config.temperature
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Build input
        if self._processor is not None:
            inputs = self._build_inputs_with_processor(messages, image)
        else:
            inputs = self._build_inputs_with_tokenizer(messages)
        
        # Move to device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample and temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        
        if return_logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True
        
        outputs = self._model.generate(**inputs, **gen_kwargs)
        
        # Process outputs
        if return_logprobs:
            output_ids = outputs.sequences[0].tolist()
            # Compute log probabilities
            logprobs = self._compute_logprobs(outputs)
        else:
            output_ids = outputs[0].tolist()
            logprobs = None
        
        # Decode
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[input_len:]
        response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Parse tool calls
        tool_calls = self._parse_tool_calls(response_text)
        
        latency = time.time() - start_time
        
        return InferenceResponse(
            content=response_text,
            tool_calls=tool_calls,
            output_ids=new_tokens,
            output_logprobs=logprobs,
            finish_reason="stop",
            latency=latency,
        )
    
    def _build_inputs_with_processor(
        self,
        messages: List[Dict[str, Any]],
        image: Optional[Image.Image],
    ) -> Dict[str, torch.Tensor]:
        """Build inputs using processor (for VLMs)."""
        # Format messages for chat template
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_messages.append({"role": "system", "content": content})
            elif role == "user":
                if image is not None and len(formatted_messages) == 1:
                    # Add image placeholder
                    formatted_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": content},
                        ]
                    })
                else:
                    formatted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                formatted_messages.append({"role": "tool", "content": content})
        
        # Apply chat template
        text = self._processor.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Process
        if image is not None:
            inputs = self._processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self._processor(
                text=text,
                return_tensors="pt",
                padding=True,
            )
        
        return inputs
    
    def _build_inputs_with_tokenizer(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Build inputs using tokenizer only (fallback)."""
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        )
        
        return inputs
    
    def _compute_logprobs(self, outputs) -> List[float]:
        """Compute log probabilities from generation outputs."""
        logprobs = []
        if hasattr(outputs, 'scores') and outputs.scores:
            for i, score in enumerate(outputs.scores):
                # Get the selected token
                token_id = outputs.sequences[0, -len(outputs.scores) + i].item()
                # Compute log prob
                log_probs = torch.log_softmax(score[0], dim=-1)
                logprobs.append(log_probs[token_id].item())
        return logprobs
    
    def _parse_tool_calls(self, response_text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from response text.
        
        Supports multiple formats:
        1. JSON tool_calls array
        2. Function call format
        3. XML-style tool calls
        """
        tool_calls = []
        
        # Try JSON format
        try:
            # Look for tool_calls in JSON
            json_match = re.search(r'"tool_calls"\s*:\s*(\[.*?\])', response_text, re.DOTALL)
            if json_match:
                calls = json.loads(json_match.group(1))
                return calls
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Try function call format
        func_pattern = r'<function=(\w+)>\s*(\{.*?\})\s*</function>'
        matches = re.findall(func_pattern, response_text, re.DOTALL)
        for func_name, func_args in matches:
            try:
                args = json.loads(func_args)
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args),
                    }
                })
            except json.JSONDecodeError:
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": func_args,
                    }
                })
        
        if tool_calls:
            return tool_calls
        
        # Try JSON block format
        json_block_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_block_pattern, response_text, re.DOTALL)
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if "name" in data or "function" in data:
                    func_name = data.get("name") or data.get("function", {}).get("name")
                    func_args = data.get("arguments") or data.get("parameters") or data.get("function", {}).get("arguments", {})
                    if func_name:
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(func_args) if isinstance(func_args, dict) else func_args,
                            }
                        })
            except json.JSONDecodeError:
                pass
        
        return tool_calls if tool_calls else None
    
    def save_lora(self, save_path: str):
        """Save LoRA adapter weights."""
        if not self._lora_applied:
            logger.warning("No LoRA applied, nothing to save")
            return
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self._model.save_pretrained(str(save_path))
        self._tokenizer.save_pretrained(str(save_path))
        
        logger.info(f"LoRA adapter saved to {save_path}")
    
    def load_lora(self, adapter_path: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            logger.error(f"Adapter path does not exist: {adapter_path}")
            return False
        
        try:
            self._model = PeftModel.from_pretrained(
                self._model,
                str(adapter_path),
            )
            self._lora_applied = True
            logger.info(f"LoRA adapter loaded from {adapter_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            return False


class FrozenReferenceModel:
    """Frozen reference model for KL divergence computation.
    
    This model is loaded once and never updated during training.
    Used to compute KL penalty in GRPO.
    """
    
    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        """Initialize frozen reference model.
        
        Args:
            model_name: HuggingFace model name
            load_in_4bit: Whether to use 4-bit quantization
            dtype: Model dtype
            device: Device to load model on
        """
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.dtype = dtype
        self.device = device
        
        self._model = None
        self._tokenizer = None
    
    def load(self):
        """Load the reference model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        logger.info(f"Loading frozen reference model: {self.model_name}")
        
        # Quantization config
        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        # Dtype
        if self.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False
        
        self._model.eval()
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        logger.info("Frozen reference model loaded and frozen")
    
    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer
    
    @torch.inference_mode()
    def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log probabilities for input tokens.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Log probabilities [batch, seq_len]
        """
        if self._model is None:
            self.load()
        
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Compute log probs
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        return token_log_probs


__all__ = [
    "UnslothInferenceConfig",
    "InferenceResponse",
    "UnslothVLMInference",
    "FrozenReferenceModel",
]

