"""Unified Model Loader for CUA Agent GRPO Training.

Provides a unified interface for loading VLM models:
1. HuggingFace native loader (default, for validation)
2. Unsloth loader (for acceleration)

This allows switching between loaders without changing RL logic.

Architecture:
    ModelLoader (Abstract)
        ├── HFModelLoader      - HuggingFace Transformers
        └── UnslothModelLoader - Unsloth optimized

Usage:
    # HF mode (validation)
    loader = create_model_loader("hf", model_name="Qwen/Qwen3-VL-32B-Instruct")
    model, tokenizer, processor = loader.load()
    
    # Unsloth mode (acceleration)
    loader = create_model_loader("unsloth", model_name="Qwen/Qwen3-VL-32B-Instruct")
    model, tokenizer, processor = loader.load()
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA configuration for Vision + Cross-modal targeting."""
    
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    
    # Target modules for Qwen3-VL
    # Language + Vision + Cross-modal
    target_modules: List[str] = field(default_factory=lambda: [
        # Language model projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Vision-specific targets (for cross-modal learning)
    vision_target_modules: List[str] = field(default_factory=lambda: [
        "visual.attn.q_proj",
        "visual.attn.k_proj", 
        "visual.attn.v_proj",
        "visual.attn.o_proj",
        "merger.mlp",
    ])
    
    def get_all_targets(self) -> List[str]:
        """Get all target modules (language + vision)."""
        return list(set(self.target_modules + self.vision_target_modules))


@dataclass
class ModelLoaderConfig:
    """Configuration for model loading."""
    
    model_name: str = "Qwen/Qwen3-VL-32B-Instruct"
    max_seq_length: int = 8192
    dtype: str = "bfloat16"
    
    # Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Device
    device_map: str = "auto"
    
    # Trust remote code
    trust_remote_code: bool = True
    
    @classmethod
    def from_env(cls) -> "ModelLoaderConfig":
        """Create config from environment variables."""
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, str(default)).lower()
            return val in ("true", "1", "yes")
        
        def get_int(key: str, default: int) -> int:
            return int(os.environ.get(key, str(default)))
        
        return cls(
            model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-32B-Instruct"),
            max_seq_length=get_int("MAX_SEQ_LENGTH", 8192),
            dtype=os.environ.get("DTYPE", "bfloat16"),
            load_in_4bit=get_bool("LOAD_IN_4BIT", True),
            load_in_8bit=get_bool("LOAD_IN_8BIT", False),
            lora=LoRAConfig(
                enabled=get_bool("USE_LORA", True),
                r=get_int("LORA_R", 16),
                alpha=get_int("LORA_ALPHA", 32),
            ),
        )


class ModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    def __init__(self, config: ModelLoaderConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._lora_applied = False
    
    @abstractmethod
    def load(self) -> Tuple[Any, Any, Any]:
        """Load model, tokenizer, and processor.
        
        Returns:
            Tuple of (model, tokenizer, processor)
        """
        pass
    
    @abstractmethod
    def apply_lora(self) -> Any:
        """Apply LoRA adapters to the model.
        
        Returns:
            Model with LoRA adapters
        """
        pass
    
    @abstractmethod
    def save_lora(self, save_path: str):
        """Save LoRA adapter weights."""
        pass
    
    @abstractmethod
    def load_lora(self, adapter_path: str):
        """Load LoRA adapter weights."""
        pass
    
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
    
    def get_dtype(self) -> torch.dtype:
        """Get torch dtype from config."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.dtype, torch.bfloat16)


class HFModelLoader(ModelLoader):
    """HuggingFace Transformers model loader.
    
    Uses native HuggingFace for validation and debugging.
    """
    
    def load(self) -> Tuple[Any, Any, Any]:
        """Load model using HuggingFace Transformers."""
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        
        logger.info(f"[HF Loader] Loading model: {self.config.model_name}")
        
        # Quantization config
        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.get_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=self.get_dtype(),
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Ensure pad token
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
        # Load processor
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
            )
        except Exception as e:
            logger.warning(f"Could not load processor: {e}")
            self._processor = None
        
        logger.info(f"[HF Loader] Model loaded successfully")
        logger.info(f"[HF Loader] Model dtype: {self._model.dtype}")
        logger.info(f"[HF Loader] Model device: {next(self._model.parameters()).device}")
        
        return self._model, self._tokenizer, self._processor
    
    def apply_lora(self) -> Any:
        """Apply LoRA using PEFT library."""
        if not self.config.lora.enabled:
            logger.info("[HF Loader] LoRA disabled, skipping")
            return self._model
        
        if self._lora_applied:
            logger.warning("[HF Loader] LoRA already applied")
            return self._model
        
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Filter valid target modules
        valid_targets = self._get_valid_lora_targets()
        
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=valid_targets,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self._model = get_peft_model(self._model, lora_config)
        self._lora_applied = True
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(f"[HF Loader] LoRA applied")
        logger.info(f"[HF Loader] Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"[HF Loader] Target modules: {valid_targets}")
        
        return self._model
    
    def _get_valid_lora_targets(self) -> List[str]:
        """Filter target modules to only include valid ones."""
        all_targets = self.config.lora.get_all_targets()
        valid_targets = []
        
        all_module_names = set()
        for name, _ in self._model.named_modules():
            all_module_names.add(name)
            if '.' in name:
                all_module_names.add(name.split('.')[-1])
        
        for target in all_targets:
            if target in all_module_names:
                valid_targets.append(target)
            else:
                for module_name in all_module_names:
                    if module_name.endswith(target) or target in module_name:
                        if target not in valid_targets:
                            valid_targets.append(target)
                        break
        
        if not valid_targets:
            logger.warning("[HF Loader] No valid LoRA targets, using defaults")
            valid_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        return valid_targets
    
    def save_lora(self, save_path: str):
        """Save LoRA adapter weights."""
        if not self._lora_applied:
            logger.warning("[HF Loader] No LoRA applied, nothing to save")
            return
        
        from pathlib import Path
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self._model.save_pretrained(str(save_dir))
        self._tokenizer.save_pretrained(str(save_dir))
        
        logger.info(f"[HF Loader] LoRA adapter saved to {save_path}")
    
    def load_lora(self, adapter_path: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        from pathlib import Path
        
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            logger.error(f"[HF Loader] Adapter path does not exist: {adapter_path}")
            return False
        
        try:
            self._model = PeftModel.from_pretrained(self._model, str(adapter_dir))
            self._lora_applied = True
            logger.info(f"[HF Loader] LoRA adapter loaded from {adapter_path}")
            return True
        except Exception as e:
            logger.error(f"[HF Loader] Failed to load LoRA: {e}")
            return False


class UnslothModelLoader(ModelLoader):
    """Unsloth optimized model loader.
    
    Uses Unsloth for 2x faster training with lower memory.
    """
    
    def load(self) -> Tuple[Any, Any, Any]:
        """Load model using Unsloth."""
        try:
            from unsloth import FastVisionModel
        except ImportError:
            logger.error("[Unsloth Loader] Unsloth not installed, falling back to HF")
            hf_loader = HFModelLoader(self.config)
            return hf_loader.load()
        
        logger.info(f"[Unsloth Loader] Loading model: {self.config.model_name}")
        
        # Load with Unsloth
        self._model, self._tokenizer = FastVisionModel.from_pretrained(
            self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.get_dtype(),
            load_in_4bit=self.config.load_in_4bit,
        )
        
        # Ensure pad token
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
        # Load processor
        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
            )
        except Exception as e:
            logger.warning(f"[Unsloth Loader] Could not load processor: {e}")
            self._processor = None
        
        logger.info(f"[Unsloth Loader] Model loaded successfully with Unsloth")
        
        return self._model, self._tokenizer, self._processor
    
    def apply_lora(self) -> Any:
        """Apply LoRA using Unsloth's optimized PEFT."""
        if not self.config.lora.enabled:
            logger.info("[Unsloth Loader] LoRA disabled, skipping")
            return self._model
        
        if self._lora_applied:
            logger.warning("[Unsloth Loader] LoRA already applied")
            return self._model
        
        try:
            from unsloth import FastVisionModel
            
            # Filter valid targets
            valid_targets = self._get_valid_lora_targets()
            
            self._model = FastVisionModel.get_peft_model(
                self._model,
                r=self.config.lora.r,
                target_modules=valid_targets,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            
            self._lora_applied = True
            
            # Log trainable parameters
            trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"[Unsloth Loader] LoRA applied with Unsloth optimization")
            logger.info(f"[Unsloth Loader] Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            
            return self._model
            
        except ImportError:
            logger.warning("[Unsloth Loader] Unsloth not available for LoRA, using PEFT")
            return self._apply_lora_peft()
    
    def _apply_lora_peft(self) -> Any:
        """Fallback: Apply LoRA using PEFT."""
        from peft import LoraConfig, get_peft_model, TaskType
        
        valid_targets = self._get_valid_lora_targets()
        
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=valid_targets,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self._model = get_peft_model(self._model, lora_config)
        self._lora_applied = True
        
        logger.info(f"[Unsloth Loader] LoRA applied with PEFT fallback")
        return self._model
    
    def _get_valid_lora_targets(self) -> List[str]:
        """Filter target modules to only include valid ones."""
        all_targets = self.config.lora.get_all_targets()
        valid_targets = []
        
        all_module_names = set()
        for name, _ in self._model.named_modules():
            all_module_names.add(name)
            if '.' in name:
                all_module_names.add(name.split('.')[-1])
        
        for target in all_targets:
            if target in all_module_names:
                valid_targets.append(target)
            else:
                for module_name in all_module_names:
                    if module_name.endswith(target) or target in module_name:
                        if target not in valid_targets:
                            valid_targets.append(target)
                        break
        
        if not valid_targets:
            valid_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        return valid_targets
    
    def save_lora(self, save_path: str):
        """Save LoRA adapter weights."""
        if not self._lora_applied:
            logger.warning("[Unsloth Loader] No LoRA applied, nothing to save")
            return
        
        from pathlib import Path
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self._model.save_pretrained(str(save_dir))
        self._tokenizer.save_pretrained(str(save_dir))
        
        logger.info(f"[Unsloth Loader] LoRA adapter saved to {save_path}")
    
    def load_lora(self, adapter_path: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        from pathlib import Path
        
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            logger.error(f"[Unsloth Loader] Adapter path does not exist: {adapter_path}")
            return False
        
        try:
            self._model = PeftModel.from_pretrained(self._model, str(adapter_dir))
            self._lora_applied = True
            logger.info(f"[Unsloth Loader] LoRA adapter loaded from {adapter_path}")
            return True
        except Exception as e:
            logger.error(f"[Unsloth Loader] Failed to load LoRA: {e}")
            return False


def create_model_loader(
    loader_type: str = "hf",
    config: Optional[ModelLoaderConfig] = None,
    **kwargs,
) -> ModelLoader:
    """Factory function to create model loader.
    
    Args:
        loader_type: "hf" for HuggingFace, "unsloth" for Unsloth
        config: ModelLoaderConfig (optional)
        **kwargs: Override config values
        
    Returns:
        ModelLoader instance
    """
    if config is None:
        config = ModelLoaderConfig.from_env()
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    loader_type = loader_type.lower()
    
    if loader_type == "unsloth":
        logger.info("Creating Unsloth model loader")
        return UnslothModelLoader(config)
    else:
        logger.info("Creating HuggingFace model loader")
        return HFModelLoader(config)


class FrozenReferenceModel:
    """Frozen reference model for KL divergence computation.
    
    This model is loaded once and never updated during training.
    Used to compute KL penalty in GRPO.
    """
    
    def __init__(
        self,
        config: ModelLoaderConfig,
        loader_type: str = "hf",
    ):
        """Initialize frozen reference model."""
        self.config = config
        self.loader_type = loader_type
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load the reference model and freeze it."""
        if self._loaded:
            return
        
        logger.info("[Reference Model] Loading frozen reference model...")
        
        # Create loader
        loader = create_model_loader(self.loader_type, self.config)
        
        # Load model (without LoRA - we want the base model)
        self._model, self._tokenizer, _ = loader.load()
        
        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False
        
        self._model.eval()
        self._loaded = True
        
        logger.info("[Reference Model] Frozen reference model loaded")
    
    @property
    def model(self):
        if not self._loaded:
            self.load()
        return self._model
    
    @property
    def tokenizer(self):
        if not self._loaded:
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
            Log probabilities [batch, seq_len - 1]
        """
        if not self._loaded:
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
    "ModelLoaderConfig",
    "LoRAConfig",
    "ModelLoader",
    "HFModelLoader",
    "UnslothModelLoader",
    "FrozenReferenceModel",
    "create_model_loader",
]

