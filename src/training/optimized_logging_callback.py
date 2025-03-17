"""
Optimized logging callback for Gemma training.

This module provides an optimized TrainerCallback implementation that integrates
with the centralized logging system while maintaining the performance optimizations
from the original optimized_gemma_training.py script.
"""

import os
import torch
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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

# Helper function for extracting reasoning and answers
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


class OptimizedLoggingCallback(TrainerCallback):
    """Optimized callback that integrates with the centralized logging system.
    
    This callback is designed for high-performance training and includes:
    - Reduced logging frequency for improved performance
    - Background processing for model output logging
    - Support for checkpointing
    """
    
    def __init__(self, checkpoint_dir=None):
        """Initialize the callback.
        
        Args:
            checkpoint_dir: Optional directory for checkpointing
        """
        # Initialize counters
        self.log_counter = 0
        self.total_answers = 0
        self.correct_answers = 0
        
        # Get loggers
        self.logger = get_training_logger()
        self.model_logger = get_model_logger()
        self.metrics_logger = get_metrics_logger()
        
        # Set up optimized logging frequencies
        self.progress_log_frequency = 5      # Print progress every 5 steps
        self.metrics_log_frequency = 20      # Log detailed metrics every 20 steps
        self.memory_log_frequency = 100      # Log memory usage every 100 steps
        
        # Create a buffer for model outputs to avoid frequent logging calls
        self.model_output_buffer = []
        self.model_output_buffer_size = 10   # Buffer up to 10 model outputs before logging
        
        # For checkpointing (if provided)
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            self.logger.info(f"Checkpointing enabled at: {checkpoint_dir}")
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called at the end of initialization.
        
        Args:
            args: Training arguments
            state: Training state
            control: Trainer control object
            **kwargs: Additional arguments
        """
        self.logger.info("Training initialization completed")
        
        # Log initial configuration (minimal to avoid clutter)
        self.logger.info(f"Total steps: {args.max_steps}")
        self.logger.info(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        
        # Log initial memory usage
        log_memory_usage()
        
        # Print initial progress indicator
        print(f"\nTraining progress: 0/{args.max_steps} steps (0.0%) | Correct: 0/0 (0.0%)")
        
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
        
        # Update correct answers counter if available in logs (always process this)
        if logs and 'rewards/correctness_reward_func' in logs:
            batch_size = args.per_device_train_batch_size
            num_generations = args.num_generations
            examples_this_step = batch_size * num_generations
            
            self.total_answers += examples_this_step
            
            from src.training.reward_config import CORRECTNESS_REWARD
            correct_value = logs['rewards/correctness_reward_func']
            
            if correct_value > 0:
                correct_count = int(round(correct_value / CORRECTNESS_REWARD))
                self.correct_answers += correct_count
        
        # Calculate accuracy (used in multiple places)
        accuracy = 0.0
        if self.total_answers > 0:
            accuracy = (self.correct_answers / self.total_answers) * 100
        
        # Print progress to stdout at optimized frequency
        if self.log_counter % self.progress_log_frequency == 0:
            progress_pct = (state.global_step / args.max_steps) * 100
            print(f"\rTraining progress: {state.global_step}/{args.max_steps} steps ({progress_pct:.1f}%) | Correct: {self.correct_answers}/{self.total_answers} ({accuracy:.1f}%)", end="", flush=True)
            
            # Only log to the system at reduced frequency to avoid overhead
            log_training_progress(
                step=state.global_step,
                metrics={
                    "progress_percent": progress_pct,
                    "correct_answers": self.correct_answers,
                    "total_answers": self.total_answers,
                    "accuracy": accuracy,
                    "loss": logs.get("loss", None) if logs else None
                }
            )
        
        # Process model outputs with buffering
        trainer = kwargs.get('trainer', None)
        if hasattr(trainer, 'current_batch_info') and trainer.current_batch_info:
            try:
                batch_info = trainer.current_batch_info
                
                # Extract questions, true answers, model generations
                questions = batch_info.get('questions', [])
                true_answers = batch_info.get('true_answers', [])
                model_outputs = batch_info.get('model_outputs', [])
                
                # Process each example in the batch
                for i in range(len(questions)):
                    if i < len(questions) and i < len(true_answers) and i < len(model_outputs):
                        question = questions[i]
                        true_answer = true_answers[i]
                        model_output = model_outputs[i]
                        
                        # Extract reasoning and answer from model output
                        llm_reasoning, llm_answer = extract_reasoning_and_answer(model_output)
                        
                        # Add to buffer instead of immediate logging
                        self.model_output_buffer.append({
                            "question": question,
                            "true_answer": true_answer,
                            "model_output": model_output,
                            "reasoning": llm_reasoning,
                            "answer": llm_answer
                        })
                
                # Log generation metrics at optimized frequency
                if self.log_counter % 20 == 0 and model_outputs:
                    log_generation_metrics(
                        step=state.global_step,
                        generations=model_outputs
                    )
            except Exception as e:
                self.logger.warning(f"Error capturing training data: {str(e)}")
        
        # Process the model output buffer when it reaches the threshold
        if len(self.model_output_buffer) >= self.model_output_buffer_size:
            # Use a separate thread to process the buffer
            buffer_to_process = self.model_output_buffer
            self.model_output_buffer = []
            
            # Process in background
            threading.Thread(
                target=self._process_model_output_buffer,
                args=(buffer_to_process,),
                daemon=True
            ).start()
        
        # Log rewards at reduced frequency
        if self.log_counter % self.metrics_log_frequency == 0 and logs:
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
                
                # Log a subset of metrics to the console log
                if 'loss' in logs:
                    self.logger.info(f"Step {state.global_step}: Loss: {logs['loss']:.6f}")
                if 'correctness_reward_func' in reward_metrics:
                    self.logger.info(f"Step {state.global_step}: Correctness reward: {reward_metrics['correctness_reward_func']:.6f}")
                
                self.logger.info(f"Step {state.global_step}: Accuracy so far: {self.correct_answers}/{self.total_answers} ({accuracy:.2f}%)")
            
            # Log memory usage at this frequency
            if self.log_counter % self.memory_log_frequency == 0:
                log_memory_usage()
        
        # Handle checkpointing if enabled
        if self.checkpoint_dir and state.global_step > 0 and state.global_step % 250 == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint-{state.global_step}")
            self.logger.info(f"Saving additional checkpoint to {checkpoint_path}")
            
            # The actual saving is handled by the trainer, we're just logging it
        
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
            
        # Print final progress and add a newline 
        progress_pct = 100.0
        print(f"\rTraining progress: {state.global_step}/{args.max_steps} steps ({progress_pct:.1f}%) | Correct: {self.correct_answers}/{self.total_answers} ({accuracy:.1f}%)")
        print("\nTraining completed!")
        
        # Process any remaining outputs in buffer
        if self.model_output_buffer:
            self._process_model_output_buffer(self.model_output_buffer)
            self.model_output_buffer = []
        
        # Log final statistics
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
        
    def _process_model_output_buffer(self, buffer):
        """Process buffered model outputs in background.
        
        Args:
            buffer: List of model output dictionaries to process
        """
        if not buffer:
            return
            
        try:
            for item in buffer:
                log_model_output(
                    question=item["question"],
                    true_answer=item["true_answer"],
                    model_output=item["model_output"],
                    reasoning=item["reasoning"],
                    answer=item["answer"]
                )
        except Exception as e:
            self.logger.error(f"Error processing model output buffer: {str(e)}") 