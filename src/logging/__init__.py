"""
Centralized logging system for Gemma training scripts.

This module provides a unified logging interface for all components
of the Gemma training pipeline, with configurable handlers and formatters.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any

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
def log_model_output(question: str, true_answer: str, 
                     model_output: str, reasoning: Optional[str] = None) -> None:
    """Log a model generation.
    
    Args:
        question: The input question or prompt
        true_answer: The expected answer
        model_output: The complete model output
        reasoning: Optional extracted reasoning section
    """
    logger = get_model_logger()
    logger.info({
        "question": question,
        "true_answer": true_answer,
        "model_output": model_output,
        "reasoning": reasoning
    })

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