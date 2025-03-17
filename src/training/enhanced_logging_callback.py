"""
Enhanced logging callback for Gemma training.

This module provides a TrainerCallback implementation that integrates with
the centralized logging system to track training progress, model outputs,
and metrics during Gemma training.
"""

import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import TrainerCallback

# Import the centralized logging system
from src.logging import (
    get_training_logger,
    get_model_logger,
    get_reward_logger,
    get_metrics_logger,
    log_model_output,
    log_reward,
    log_training_progress,
    log_memory_usage,
    log_reward_metrics,
    log_generation_metrics
)

# Helper functions for extracting reasoning and answers
def extract_reasoning_and_answer(text: str) -> Tuple[str, str]:
    """Extract reasoning and answer from model output.
    
    Args:
        text: Model output text containing XML tags
        
    Returns:
        Tuple of (reasoning, answer)
    """
    import re
    
    reasoning = ""
    answer = ""
    
    # Extract reasoning
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Extract answer
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return reasoning, answer


class EnhancedLoggingCallback(TrainerCallback):
    """Enhanced callback that integrates with the centralized logging system.
    
    This callback replaces the original OutputLoggingCallback in the training
    script, providing improved logging capabilities through the centralized
    logging system.
    """
    
    def __init__(self):
        """Initialize the callback."""
        # Initialize counters
        self.log_counter = 0
        self.total_answers = 0
        self.correct_answers = 0
        
        # Get loggers
        self.logger = get_training_logger()
        self.model_logger = get_model_logger()
        self.metrics_logger = get_metrics_logger()
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called at the end of initialization.
        
        Args:
            args: Training arguments
            state: Training state
            control: Trainer control object
            **kwargs: Additional arguments
        """
        self.logger.info("Training initialization completed")
        
        # Log initial configuration
        self.logger.info(f"Training arguments: {args}")
        self.logger.info(f"Total steps: {args.max_steps}")
        self.logger.info(f"Batch size: {args.per_device_train_batch_size}")
        self.logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        
        # Log initial memory usage
        log_memory_usage()
        
        return control
        
    def on_step_end(self, args, state, control, logs=None, model=None, tokenizer=None, **kwargs):
        """Called at the end of a training step.
        
        Args:
            args: Training arguments
            state: Training state
            control: Trainer control object
            logs: Log dictionary
            model: The model being trained
            tokenizer: The tokenizer being used
            **kwargs: Additional arguments
        """
        self.log_counter += 1
        
        # Update correct answers counter if available in logs
        if logs and 'rewards/correctness_reward_func' in logs:
            # Each step processes batch_size * num_generations examples
            batch_size = args.per_device_train_batch_size
            num_generations = args.num_generations
            examples_this_step = batch_size * num_generations
            
            self.total_answers += examples_this_step
            
            # Check for correctness based on non-zero reward
            # The reward value indicates how many were correct in this batch
            from src.training.reward_config import CORRECTNESS_REWARD
            correct_value = logs['rewards/correctness_reward_func']
            
            # If the reward is for the whole batch, divide by CORRECTNESS_REWARD to get count
            if correct_value > 0:
                correct_count = int(round(correct_value / CORRECTNESS_REWARD))
                self.correct_answers += correct_count
        
        # Calculate progress and accuracy
        progress_pct = (state.global_step / args.max_steps) * 100
        accuracy = 0.0
        if self.total_answers > 0:
            accuracy = (self.correct_answers / self.total_answers) * 100
        
        # Use the logging system to log progress
        log_training_progress(
            step=state.global_step,
            metrics={
                "progress_percent": progress_pct,
                "correct_answers": self.correct_answers,
                "total_answers": self.total_answers,
                "accuracy": accuracy,
                "learning_rate": state.learning_rate if hasattr(state, "learning_rate") else None,
                "loss": logs.get("loss", None) if logs else None
            }
        )
        
        # Get model outputs if available from the trainer
        trainer = kwargs.get('trainer', None)
        if hasattr(trainer, 'current_batch_info') and trainer.current_batch_info:
            try:
                batch_info = trainer.current_batch_info
                
                # Extract questions, true answers, model generations
                questions = batch_info.get('questions', [])
                true_answers = batch_info.get('true_answers', [])
                model_outputs = batch_info.get('model_outputs', [])
                
                # Process each example in the batch and log using new system
                for i in range(len(questions)):
                    if i < len(questions) and i < len(true_answers) and i < len(model_outputs):
                        question = questions[i]
                        true_answer = true_answers[i]
                        model_output = model_outputs[i]
                        
                        # Extract reasoning and answer from model output
                        llm_reasoning, llm_answer = extract_reasoning_and_answer(model_output)
                        
                        # Log through the logging system
                        log_model_output(
                            question=question,
                            true_answer=true_answer,
                            model_output=model_output,
                            reasoning=llm_reasoning,
                            answer=llm_answer
                        )
                        
                # Log generation metrics every 5 steps
                if self.log_counter % 5 == 0 and model_outputs:
                    log_generation_metrics(
                        step=state.global_step,
                        generations=model_outputs
                    )
            except Exception as e:
                self.logger.warning(f"Error capturing training data: {str(e)}")
        
        # Log rewards if available
        if logs:
            # Extract all reward-related metrics
            reward_metrics = {
                k.replace('rewards/', ''): v 
                for k, v in logs.items() 
                if k.startswith('rewards/')
            }
            
            if reward_metrics:
                # Convert scalar values to lists for the log_reward_metrics function
                rewards_dict = {
                    k: [v] for k, v in reward_metrics.items()
                }
                
                # Log using the metrics system
                log_reward_metrics(state.global_step, rewards_dict)
                
                # Log details to debug log
                self.logger.debug(f"Step {state.global_step} rewards: {reward_metrics}")
            
            # Custom detailed logging at lower frequency
            if self.log_counter % 20 == 0:
                # Check model gradients and parameters
                if model is not None:
                    param_with_grad = 0
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            param_with_grad += 1
                    
                    self.logger.info(f"Step {state.global_step}: Parameters with gradients: {param_with_grad}")
                    
                    # Update memory tracking
                    log_memory_usage()
        
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training.
        
        Args:
            args: Training arguments
            state: Training state
            control: Trainer control object
            **kwargs: Additional arguments
        """
        # Calculate final accuracy
        accuracy = 0.0
        if self.total_answers > 0:
            accuracy = (self.correct_answers / self.total_answers) * 100
            
        # Log final progress
        log_training_progress(
            step=state.global_step,
            metrics={
                "final_accuracy": accuracy,
                "correct_answers": self.correct_answers,
                "total_answers": self.total_answers,
                "training_complete": True
            }
        )
        
        # Log final memory usage
        log_memory_usage()
        
        self.logger.info(f"Training completed. Final accuracy: {accuracy:.2f}%")
        
        return control 