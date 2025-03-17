"""
Centralized logging system for Gemma training scripts.

This module provides a unified logging interface for all components
of the Gemma training pipeline, with configurable handlers and formatters.
"""

import logging
import os
import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Module version
__version__ = "0.1.0"

# Initialize the module logger
logger = logging.getLogger(__name__)

def initialize(config_path: Optional[str] = None) -> None:
    """Initialize the logging system from configuration.
    
    Args:
        config_path: Path to a YAML configuration file. If None, 
                     default configuration will be used.
    """
    from .config import load_config, apply_config
    config = load_config(config_path)
    apply_config(config)
    logger.info(f"Logging system initialized with config from {config_path or 'default'}")
    
    # Initialize CSV logger if enabled in config
    if config.get('csv_logging', {}).get('enabled', False):
        # Get base configuration
        csv_base_path = config.get('csv_logging', {}).get('path', './logs/generations_and_rewards.csv')
        csv_max_queue_size = config.get('csv_logging', {}).get('max_queue_size', 1000)
        csv_flush_interval = config.get('csv_logging', {}).get('flush_interval_seconds', 10)
        
        # Create a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Split the path into directory and filename
        csv_dir = os.path.dirname(csv_base_path)
        csv_filename = os.path.basename(csv_base_path)
        
        # Ensure the logs directory exists
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
            logger.info(f"Ensuring logs directory exists: {csv_dir}")
        
        # Insert timestamp before the extension
        name_parts = csv_filename.split('.')
        if len(name_parts) > 1:
            # If there's an extension
            timestamped_filename = f"{name_parts[0]}_{timestamp}.{'.'.join(name_parts[1:])}"
        else:
            # No extension
            timestamped_filename = f"{csv_filename}_{timestamp}"
            
        # Create the full path with timestamp
        csv_path = os.path.join(csv_dir, timestamped_filename)
        
        # Define field names - make this explicit to ensure consistent ordering
        fieldnames = [
            'timestamp', 'global_step', 'question_id', 'question', 
            'ground_truth_answer', 'model_output', 'reasoning', 
            'model_answer', 'total_reward'
        ]
        
        # Add individual reward function columns
        reward_function_names = config.get('metrics', {}).get('reward_metrics', {}).get('functions', [])
        for reward_name in reward_function_names:
            fieldnames.append(f"{reward_name}")
        
        from .csv_logger import initialize_csv_logger
        initialize_csv_logger(
            csv_path=csv_path,
            fieldnames=fieldnames,
            max_queue_size=csv_max_queue_size,
            flush_interval=csv_flush_interval
        )
        logger.info(f"CSV logging initialized with timestamped file: {csv_path}")

# Logger getters
def get_training_logger() -> logging.Logger:
    """Get logger for training progress."""
    return logging.getLogger("logging.training")

def get_model_logger() -> logging.Logger:
    """Get logger for model outputs."""
    return logging.getLogger("logging.model")

def get_reward_logger() -> logging.Logger:
    """Get logger for reward function logging."""
    return logging.getLogger("logging.reward")

def get_system_logger() -> logging.Logger:
    """Get logger for system metrics."""
    return logging.getLogger("logging.system")

def get_metrics_logger() -> logging.Logger:
    """Get logger for training metrics."""
    return logging.getLogger("logging.metrics")

# Convenience functions
def log_model_output(question: str, true_answer: Optional[str] = None, 
                     model_output: str = "", reasoning: Optional[str] = None,
                     answer: Optional[str] = None, step: Optional[int] = None,
                     question_id: Optional[str] = None, 
                     log_to_csv: bool = True) -> None:
    """Log a model generation.
    
    Args:
        question: The input question or prompt
        true_answer: The expected answer
        model_output: The complete model output
        reasoning: Optional extracted reasoning section
        answer: Optional extracted answer
        step: Optional training step
        question_id: Optional unique identifier for the question
        log_to_csv: Whether to also log to CSV (if enabled)
    """
    logger = get_model_logger()
    
    # Create data dictionary
    data = {
        "question": question,
        "true_answer": true_answer if true_answer is not None else "",
        "model_output": model_output,
        "reasoning": reasoning if reasoning is not None else "",
        "answer": answer if answer is not None else ""
    }
    
    # Add step if provided
    if step is not None:
        data["step"] = step
        
    # Add question_id if provided
    if question_id is not None:
        data["question_id"] = question_id
        
    # Log to model logger
    logger.info(data)
    
    # Log to CSV if enabled
    if log_to_csv:
        try:
            from .csv_logger import log_generation_with_rewards, get_csv_logger
            
            if get_csv_logger() is not None:
                # Rename fields to match CSV schema
                csv_data = {
                    'question': question,
                    'ground_truth_answer': true_answer if true_answer is not None else "",
                    'model_output': model_output,
                    'reasoning': reasoning if reasoning is not None else "",
                    'model_answer': answer if answer is not None else ""
                }
                
                # Add step and question_id if provided
                if step is not None:
                    csv_data['global_step'] = step
                    
                if question_id is not None:
                    csv_data['question_id'] = question_id
                
                # Log to CSV
                log_generation_with_rewards(csv_data)
        except (ImportError, RuntimeError):
            # CSV logger not available or not initialized - just continue
            pass

def log_reward(reward_name: str, values: List[float], 
               samples: Optional[List[str]] = None) -> None:
    """Log reward function values.
    
    Args:
        reward_name: Name of the reward function
        values: List of reward values
        samples: Optional list of sample identifiers
    """
    logger = get_reward_logger()
    if samples is None:
        samples = [f"sample_{i}" for i in range(len(values))]
    
    for sample, value in zip(samples, values):
        logger.info({
            "reward_name": reward_name,
            "sample": sample,
            "value": value
        })

def log_generation_with_rewards(question: str, 
                               true_answer: str, 
                               model_output: str, 
                               rewards: Dict[str, float],
                               reasoning: Optional[str] = None,
                               answer: Optional[str] = None,
                               step: Optional[int] = None,
                               question_id: Optional[str] = None) -> None:
    """Log a model generation with associated reward values to the CSV file.
    
    Args:
        question: The input question or prompt
        true_answer: The expected answer
        model_output: The complete model output 
        rewards: Dictionary mapping reward function names to values
        reasoning: Optional extracted reasoning (if None, will be extracted)
        answer: Optional extracted answer (if None, will be extracted)
        step: Optional training step
        question_id: Optional unique identifier for the question
    """
    # First, extract reasoning and answer if not provided
    if reasoning is None or answer is None:
        import re
        
        if reasoning is None:
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", model_output, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
        if answer is None:
            answer_match = re.search(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else ""
    
    # Log to standard model logger
    log_model_output(
        question=question,
        true_answer=true_answer,
        model_output=model_output,
        reasoning=reasoning,
        answer=answer,
        step=step,
        question_id=question_id,
        log_to_csv=False  # Don't log to CSV here - we'll do it separately
    )
    
    # Calculate total reward
    total_reward = sum(rewards.values())
    
    # Log to CSV
    try:
        from .csv_logger import log_generation_with_rewards as csv_log
        
        # Create data for CSV
        csv_data = {
            'question': question,
            'ground_truth_answer': true_answer,
            'model_output': model_output,
            'reasoning': reasoning,
            'model_answer': answer,
            'total_reward': total_reward
        }
        
        # Add individual rewards
        for reward_name, reward_value in rewards.items():
            csv_data[reward_name] = reward_value
            
        # Add step and question_id if provided
        if step is not None:
            csv_data['global_step'] = step
            
        if question_id is not None:
            csv_data['question_id'] = question_id
            
        # Log to CSV
        csv_log(csv_data)
    except (ImportError, RuntimeError) as e:
        # CSV logger not available or not initialized - just log a warning
        logger.warning(f"Could not log to CSV: {str(e)}")

def log_training_progress(step: int, metrics: Dict[str, float]) -> None:
    """Log training progress metrics.
    
    Args:
        step: Current training step
        metrics: Dictionary of metrics to log
    """
    logger = get_training_logger()
    logger.info({
        "step": step,
        **metrics
    })

def log_memory_usage() -> None:
    """Log current memory usage."""
    import psutil
    
    logger = get_system_logger()
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    logger.info({
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024)
    })

def log_reward_metrics(step: int, rewards_dict: Dict[str, List[float]], 
                       history_window: int = 100) -> None:
    """Log aggregated reward metrics with trends.
    
    Args:
        step: Current training step
        rewards_dict: Dictionary mapping reward names to lists of values
        history_window: Window size for moving averages
    """
    import numpy as np
    logger = get_metrics_logger()
    
    # Also update metrics tracker
    from .metrics import global_metrics
    
    for reward_name, values in rewards_dict.items():
        if not values:
            continue
            
        # Calculate statistics
        values_array = np.array(values)
        metrics = {
            "step": step,
            "reward_name": reward_name,
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "std": float(np.std(values_array))
        }
        
        # Record in global metrics tracker
        global_metrics.add_metric(f"rewards/{reward_name}/mean", metrics["mean"], step)
        global_metrics.add_metric(f"rewards/{reward_name}/median", metrics["median"], step)
        
        logger.info(metrics)

def log_generation_metrics(step: int, generations: List[str], 
                          references: Optional[List[str]] = None) -> None:
    """Log metrics about model generations.
    
    Args:
        step: Current training step
        generations: List of model-generated texts
        references: Optional list of reference texts
    """
    logger = get_metrics_logger()
    
    # Use metrics module for more advanced metrics
    from .metrics import (
        calculate_token_entropy,
        calculate_format_adherence_rate,
        calculate_reasoning_answer_ratio,
        global_metrics
    )
    
    # Basic length statistics
    lengths = [len(gen.split()) for gen in generations]
    
    metrics = {
        "step": step,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "num_samples": len(generations)
    }
    
    # Advanced metrics
    if generations:
        # Calculate diversity
        metrics["token_entropy"] = calculate_token_entropy(generations)
        
        # Calculate format adherence
        metrics["format_adherence_rate"] = calculate_format_adherence_rate(generations)
        
        # Calculate reasoning/answer ratio
        reasoning_metrics = calculate_reasoning_answer_ratio(generations)
        metrics.update(reasoning_metrics)
    
    # Store in global metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and key != "step":
            global_metrics.add_metric(f"generation/{key}", value, step)
    
    logger.info(metrics)

def run_validation(step: int, model: Any, tokenizer: Any, 
                  num_samples: Optional[int] = None) -> None:
    """Run validation on held-out dataset and log results.
    
    Args:
        step: Current training step
        model: The model to evaluate
        tokenizer: The tokenizer to use
        num_samples: Number of validation samples to use (None = use config default)
    """
    from .validation import run_validation as run_validation_impl
    
    results = run_validation_impl(step, model, tokenizer, num_samples)
    
    # Log results to the metrics logger
    if results is not None:
        logger = get_metrics_logger()
        logger.info({
            "step": step,
            "validation": results
        })
        
        # Store in global metrics
        from .metrics import global_metrics
        if "accuracy" in results:
            global_metrics.add_metric("validation/accuracy", results["accuracy"], step)
        
        if "avg_generation_time" in results:
            global_metrics.add_metric("validation/avg_generation_time", 
                                    results["avg_generation_time"], step) 