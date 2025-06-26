"""LoRA/QLoRA training module for adaptive model fine-tuning."""

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging
import os
import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from src.config import settings
from src.models.language_model import language_model
from src.memory.vector_store import vector_memory
from src.feedback.queue import feedback_queue, FeedbackItem, TrainingRequest

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    Handles LoRA fine-tuning of the language model based on feedback.
    """
    
    def __init__(self):
        self.trainer = None
        self.training_dataset = None
        self.output_dir = os.path.join("./models", "lora_checkpoints")
        self.current_training_request = None
        
    def prepare_training_data(self, feedback_items: List[FeedbackItem]) -> Dataset:
        """
        Prepare training data from feedback items.
        
        Args:
            feedback_items: List of feedback items for training
            
        Returns:
            Hugging Face Dataset object
        """
        try:
            training_examples = []
            
            for item in feedback_items:
                # Format training example
                if item.feedback_type == "correction":
                    # Create training pair: input -> corrected output
                    input_text = f"Correct the following text: {item.original_text}"
                    output_text = item.corrected_text
                    
                    # Combine input and output for causal LM training
                    full_text = f"{input_text}\nCorrected: {output_text}"
                    
                elif item.feedback_type == "preference":
                    # Handle preference feedback
                    input_text = f"Improve the following response: {item.original_text}"
                    output_text = item.corrected_text
                    full_text = f"{input_text}\nImproved: {output_text}"
                    
                else:
                    # Generic feedback handling
                    full_text = f"Input: {item.original_text}\nOutput: {item.corrected_text}"
                
                training_examples.append({
                    "text": full_text,
                    "feedback_id": item.id,
                    "feedback_type": item.feedback_type
                })
            
            # Create dataset
            dataset = Dataset.from_list(training_examples)
            
            # Tokenize dataset
            def tokenize_function(examples):
                tokenized = language_model.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=settings.model.max_length,
                    return_tensors="pt"
                )
                # For causal LM, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].clone()
                return tokenized
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            logger.info(f"Prepared training dataset with {len(tokenized_dataset)} examples")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def setup_lora_training(self, training_dataset: Dataset) -> None:
        """
        Set up LoRA configuration and trainer.
        
        Args:
            training_dataset: Prepared training dataset
        """
        try:
            # Ensure model has LoRA setup
            if not language_model.peft_model:
                language_model.setup_lora()
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=1,  # Few epochs for fine-tuning
                per_device_train_batch_size=settings.training.batch_size,
                gradient_accumulation_steps=2,
                warmup_steps=10,
                max_steps=settings.training.max_steps,
                learning_rate=settings.training.learning_rate,
                fp16=settings.model.precision == "fp16",
                logging_steps=10,
                save_steps=50,
                save_strategy="steps",
                evaluation_strategy="no",
                load_best_model_at_end=False,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                report_to=None,  # Disable wandb/tensorboard
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=language_model.tokenizer,
                mlm=False,  # Causal LM, not masked LM
                pad_to_multiple_of=8 if settings.model.precision == "fp16" else None
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=language_model.peft_model,
                args=training_args,
                train_dataset=training_dataset,
                data_collator=data_collator,
                tokenizer=language_model.tokenizer,
            )
            
            logger.info("LoRA training setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA training: {e}")
            raise
    
    def train_on_feedback(self, training_request: TrainingRequest) -> Dict[str, Any]:
        """
        Train the model on feedback items.
        
        Args:
            training_request: Training request with feedback items
            
        Returns:
            Training results dictionary
        """
        try:
            self.current_training_request = training_request
            
            # Update training request status
            feedback_queue.update_training_status(
                training_request.id,
                "processing",
                {"start_time": datetime.now().isoformat()}
            )
            
            # Get feedback items
            feedback_items = []
            for feedback_id in training_request.feedback_items:
                item = feedback_queue.get_feedback(feedback_id)
                if item:
                    feedback_items.append(item)
            
            if not feedback_items:
                raise ValueError("No valid feedback items found")
            
            logger.info(f"Training on {len(feedback_items)} feedback items")
            
            # Prepare training data
            training_dataset = self.prepare_training_data(feedback_items)
            
            # Setup training
            self.setup_lora_training(training_dataset)
            
            # Start training
            logger.info("Starting LoRA training...")
            train_result = self.trainer.train()
            
            # Save the adapter
            adapter_path = os.path.join(self.output_dir, f"adapter_{training_request.id}")
            language_model.peft_model.save_pretrained(adapter_path)
            
            # Store corrections in vector memory
            for item in feedback_items:
                vector_memory.add_correction(
                    original_text=item.original_text,
                    corrected_text=item.corrected_text,
                    context=item.context,
                    correction_type=item.feedback_type,
                    metadata={
                        "training_request_id": training_request.id,
                        "feedback_id": item.id,
                        "source": "training"
                    }
                )
                
                # Mark feedback as processed
                feedback_queue.mark_feedback_processed(item.id)
            
            # Prepare results
            training_results = {
                "training_loss": train_result.training_loss,
                "adapter_path": adapter_path,
                "feedback_count": len(feedback_items),
                "training_steps": train_result.global_step,
                "end_time": datetime.now().isoformat()
            }
            
            # Update training request status
            feedback_queue.update_training_status(
                training_request.id,
                "completed",
                training_results
            )
            
            logger.info(f"Training completed successfully. Adapter saved to: {adapter_path}")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Update training request status
            if self.current_training_request:
                feedback_queue.update_training_status(
                    self.current_training_request.id,
                    "failed",
                    {"error": str(e), "end_time": datetime.now().isoformat()}
                )
            
            raise
    
    def evaluate_model_performance(self, test_examples: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Evaluate model performance on test examples.
        
        Args:
            test_examples: List of test examples with 'input' and 'expected' keys
            
        Returns:
            Performance metrics
        """
        try:
            if not test_examples:
                return {}
            
            correct_predictions = 0
            total_predictions = len(test_examples)
            
            for example in test_examples:
                input_text = example["input"]
                expected_output = example["expected"]
                
                # Generate prediction
                predicted_output = language_model.generate_response(
                    input_text,
                    max_new_tokens=100
                )
                
                # Simple exact match evaluation (can be improved)
                if predicted_output.strip().lower() == expected_output.strip().lower():
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "correct_predictions": correct_predictions,
                "total_predictions": total_predictions
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate model performance: {e}")
            return {}
    
    def create_training_schedule(self, min_feedback_count: int = 10) -> None:
        """
        Create training requests from accumulated feedback.
        
        Args:
            min_feedback_count: Minimum number of feedback items to trigger training
        """
        try:
            # Get pending feedback
            pending_feedback = feedback_queue.get_pending_feedback(limit=100)
            
            if len(pending_feedback) >= min_feedback_count:
                # Group feedback by type
                feedback_groups = {}
                for item in pending_feedback:
                    feedback_type = item.feedback_type
                    if feedback_type not in feedback_groups:
                        feedback_groups[feedback_type] = []
                    feedback_groups[feedback_type].append(item.id)
                
                # Create training requests for each group
                for feedback_type, feedback_ids in feedback_groups.items():
                    if len(feedback_ids) >= 5:  # Minimum batch size
                        training_request_id = feedback_queue.create_training_request(
                            feedback_ids=feedback_ids,
                            training_type="lora",
                            priority=1,
                            metadata={"feedback_type": feedback_type}
                        )
                        logger.info(f"Created training request: {training_request_id} for {feedback_type}")
            
        except Exception as e:
            logger.error(f"Failed to create training schedule: {e}")
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get history of training sessions.
        
        Returns:
            List of training session information
        """
        try:
            history = []
            
            # List all adapter directories
            if os.path.exists(self.output_dir):
                for item in os.listdir(self.output_dir):
                    adapter_path = os.path.join(self.output_dir, item)
                    if os.path.isdir(adapter_path) and item.startswith("adapter_"):
                        request_id = item.replace("adapter_", "")
                        
                        # Get training request details
                        training_request = feedback_queue.get_training_request(request_id)
                        if training_request:
                            history.append({
                                "request_id": request_id,
                                "adapter_path": adapter_path,
                                "status": training_request.status,
                                "feedback_count": len(training_request.feedback_items),
                                "timestamp": training_request.timestamp,
                                "metadata": training_request.metadata
                            })
            
            # Sort by timestamp
            history.sort(key=lambda x: x["timestamp"], reverse=True)
            return history
            
        except Exception as e:
            logger.error(f"Failed to get training history: {e}")
            return []


# Global LoRA trainer instance
lora_trainer = LoRATrainer()
