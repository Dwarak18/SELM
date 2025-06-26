"""Language model management using Hugging Face Transformers."""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import logging
from typing import List, Dict, Optional, Tuple
import os
from src.config import settings

logger = logging.getLogger(__name__)


class LanguageModel:
    """
    Manages the language model with LoRA adaptation capabilities.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.device = self._get_device()
        self.generation_config = None
        
    def _get_device(self) -> str:
        """Determine the best device for model inference."""
        if not settings.model.gpu_enabled:
            return "cpu"
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration for memory efficiency."""
        if not settings.model.gpu_enabled or self.device == "cpu":
            return None
        
        if settings.model.precision == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif settings.model.precision == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None
    
    def load_model(self) -> None:
        """Load the base language model and tokenizer."""
        try:
            logger.info(f"Loading model: {settings.model.name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model.name,
                cache_dir=settings.model.cache_dir,
                trust_remote_code=True
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization
            quantization_config = self._create_quantization_config()
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.model.name,
                cache_dir=settings.model.cache_dir,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if settings.model.precision == "fp16" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set up generation config
            self.generation_config = GenerationConfig(
                max_length=settings.model.max_length,
                temperature=settings.model.temperature,
                top_p=settings.model.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_lora(self) -> None:
        """Set up LoRA configuration for the model."""
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=settings.training.lora_rank,
                lora_alpha=settings.training.lora_alpha,
                lora_dropout=settings.training.lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            self.peft_model = get_peft_model(self.model, lora_config)
            logger.info("LoRA configuration applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            raise
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Update generation config with any provided kwargs
            gen_config = GenerationConfig(**{
                **self.generation_config.to_dict(),
                "max_new_tokens": max_new_tokens,
                **kwargs
            })
            
            # Generate response
            with torch.no_grad():
                model_to_use = self.peft_model if self.peft_model else self.model
                outputs = model_to_use.generate(
                    inputs,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a chat response for conversation format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Generated response
        """
        try:
            # Format messages into a prompt
            prompt = self._format_chat_prompt(messages)
            return self.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Failed to generate chat response: {e}")
            raise
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt string."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def save_lora_adapter(self, path: str) -> None:
        """Save the current LoRA adapter."""
        if self.peft_model:
            self.peft_model.save_pretrained(path)
            logger.info(f"LoRA adapter saved to {path}")
        else:
            logger.warning("No LoRA adapter to save")
    
    def load_lora_adapter(self, path: str) -> None:
        """Load a LoRA adapter from disk."""
        try:
            if os.path.exists(path):
                self.peft_model = PeftModel.from_pretrained(self.model, path)
                logger.info(f"LoRA adapter loaded from {path}")
            else:
                logger.warning(f"LoRA adapter path does not exist: {path}")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        info = {
            "model_name": settings.model.name,
            "device": self.device,
            "precision": settings.model.precision,
            "has_lora": self.peft_model is not None,
            "trainable_params": 0,
            "total_params": 0
        }
        
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            info["total_params"] = total_params
            
            if self.peft_model:
                trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
                info["trainable_params"] = trainable_params
                info["trainable_percent"] = (trainable_params / total_params) * 100
        
        return info


# Global model instance
language_model = LanguageModel()
