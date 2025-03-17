"""
Wrapped reward functions that integrate with the centralized logging system.

This module contains wrapper functions around the original reward functions
that add logging functionality while preserving the original behavior.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from functools import wraps
import threading

# Import the centralized logging system
from src.logging import (
    get_reward_logger,
    log_reward,
    log_reward_metrics
)

# Import original reward functions
from src.training.reward_config import (
    CORRECTNESS_REWARD,
    INTEGER_REWARD,
    STRICT_FORMAT_REWARD, 
    SOFT_FORMAT_REWARD,
    XML_COUNT_REWARD
)

# Import original reward functions 
# Note: Update this import path as needed based on actual location
from src.training.rewards import (
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    anti_repetition_reward_func,
    topic_relevance_reward_func
)

# Helper functions from the main rewards module
# Note: Update this import path as needed based on actual location
from src.training.rewards import extract_xml_answer

# Thread-local storage for collecting rewards during each batch
_reward_collection = threading.local()

def get_current_rewards_collection():
    """Get the thread-local rewards collection for the current batch."""
    if not hasattr(_reward_collection, 'rewards'):
        _reward_collection.rewards = {}
    return _reward_collection.rewards

def reset_rewards_collection():
    """Reset the thread-local rewards collection."""
    if hasattr(_reward_collection, 'rewards'):
        _reward_collection.rewards = {}

def get_collected_rewards():
    """Get all collected rewards and clear the collection."""
    rewards = getattr(_reward_collection, 'rewards', {})
    reset_rewards_collection()
    return rewards

def collect_reward(reward_name, index, value):
    """Collect a reward value for a particular sample index."""
    rewards = get_current_rewards_collection()
    
    # Create an entry for this sample index if it doesn't exist
    if index not in rewards:
        rewards[index] = {}
        
    # Add the reward value
    rewards[index][reward_name] = value


def wrapped_correctness_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """Wrapper around correctness_reward to log detailed metrics.
    
    Args:
        prompts: Prompts provided to model
        completions: Model completions
        answer: Expected answers
        **kwargs: Additional arguments
        
    Returns:
        List of reward values as returned by the original function
    """
    # Handle different completion formats
    if completions and isinstance(completions[0], str):
        responses = completions
    else:
        try:
            responses = [completion[0]['content'] for completion in completions]
        except (TypeError, IndexError, KeyError):
            responses = completions
    
    # Call original function
    rewards = correctness_reward_func(prompts, completions, answer, **kwargs)
    
    # Log individual rewards and details
    for i, reward in enumerate(rewards):
        if i < len(completions) and i < len(answer):
            # Extract data for logging
            response = responses[i] if i < len(responses) else "N/A"
            expected = answer[i] if i < len(answer) else "N/A"
            extracted = extract_xml_answer(response)
            
            # Log via the reward system - using correct signature
            log_reward(
                reward_name="correctness",
                values=[reward],
                samples=[f"sample_{i}_correct:{reward>0}_expected:{expected}_extracted:{extracted}"]
            )
            
            # Collect reward for CSV logging
            collect_reward("correctness_reward_func", i, reward)
    
    # Log aggregated metrics
    log_reward_metrics(step=kwargs.get('step', 0), 
                       rewards_dict={"correctness_reward_func": rewards})
    
    return rewards


def wrapped_int_reward(completions, **kwargs) -> List[float]:
    """Wrapper around int_reward to log detailed metrics."""
    # Handle different completion formats
    if completions and isinstance(completions[0], str):
        responses = completions
    else:
        try:
            responses = [completion[0]['content'] for completion in completions]
        except (TypeError, IndexError, KeyError):
            responses = completions
    
    rewards = int_reward_func(completions, **kwargs)
    
    # Extract responses for logging
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Log individual rewards
    for i, (reward, extracted) in enumerate(zip(rewards, extracted_responses)):
        log_reward(
            reward_name="int_reward",
            values=[reward],
            samples=[f"sample_{i}_is_integer:{extracted.isdigit()}"]
        )
        
        # Collect reward for CSV logging
        collect_reward("int_reward_func", i, reward)
    
    # Log aggregated metrics
    log_reward_metrics(step=kwargs.get('step', 0), 
                       rewards_dict={"int_reward_func": rewards})
    
    return rewards


def wrapped_strict_format_reward(completions, **kwargs) -> List[float]:
    """Wrapper around strict_format_reward to log detailed metrics."""
    # Handle different completion formats
    if completions and isinstance(completions[0], str):
        responses = completions
    else:
        try:
            responses = [completion[0]['content'] for completion in completions]
        except (TypeError, IndexError, KeyError):
            responses = completions
    
    rewards = strict_format_reward_func(completions, **kwargs)
    
    # Log individual rewards
    for i, (reward, response) in enumerate(zip(rewards, responses)):
        log_reward(
            reward_name="strict_format",
            values=[reward],
            samples=[f"sample_{i}_format_correct:{reward>0}_len:{len(response)}"]
        )
        
        # Collect reward for CSV logging
        collect_reward("strict_format_reward_func", i, reward)
    
    # Log aggregated metrics
    log_reward_metrics(step=kwargs.get('step', 0), 
                       rewards_dict={"strict_format_reward_func": rewards})
    
    return rewards


def wrapped_soft_format_reward(completions, **kwargs) -> List[float]:
    """Wrapper around soft_format_reward to log detailed metrics."""
    # Handle different completion formats
    if completions and isinstance(completions[0], str):
        responses = completions
    else:
        try:
            responses = [completion[0]['content'] for completion in completions]
        except (TypeError, IndexError, KeyError):
            responses = completions
    
    rewards = soft_format_reward_func(completions, **kwargs)
    
    # Log individual rewards
    for i, (reward, response) in enumerate(zip(rewards, responses)):
        log_reward(
            reward_name="soft_format",
            values=[reward],
            samples=[f"sample_{i}_format_correct:{reward>0}_len:{len(response)}"]
        )
        
        # Collect reward for CSV logging
        collect_reward("soft_format_reward_func", i, reward)
    
    # Log aggregated metrics
    log_reward_metrics(step=kwargs.get('step', 0), 
                       rewards_dict={"soft_format_reward_func": rewards})
    
    return rewards


def wrapped_xmlcount_reward(completions, **kwargs) -> List[float]:
    """Wrapper around xmlcount_reward to log detailed metrics."""
    rewards = xmlcount_reward_func(completions, **kwargs)
    
    # Extract responses for logging
    # Check if completions is a list of strings or a list of dictionaries
    if completions and isinstance(completions[0], str):
        responses = completions
    else:
        # Try to handle different completion formats
        try:
            responses = [completion[0]['content'] for completion in completions]
        except (TypeError, IndexError, KeyError):
            responses = completions
    
    # Log individual rewards
    for i, (reward, response) in enumerate(zip(rewards, responses)):
        # Count XML tags (simplified version)
        reasoning_open = response.count("<reasoning>")
        reasoning_close = response.count("</reasoning>")
        answer_open = response.count("<answer>")
        answer_close = response.count("</answer>")
        balanced = (reasoning_open == reasoning_close == answer_open == answer_close == 1)
        
        log_reward(
            reward_name="xmlcount",
            values=[reward],
            samples=[f"sample_{i}_balanced:{balanced}_tags:{reasoning_open},{reasoning_close},{answer_open},{answer_close}"]
        )
        
        # Collect reward for CSV logging
        collect_reward("xmlcount_reward_func", i, reward)
    
    # Log aggregated metrics
    log_reward_metrics(step=kwargs.get('step', 0), 
                       rewards_dict={"xmlcount_reward_func": rewards})
    
    return rewards


def wrapped_anti_repetition_reward(completions, **kwargs) -> List[float]:
    """Wrapper around anti_repetition_reward to log detailed metrics."""
    # Handle different completion formats
    if completions and isinstance(completions[0], str):
        responses = completions
    else:
        try:
            responses = [completion[0]['content'] for completion in completions]
        except (TypeError, IndexError, KeyError):
            responses = completions
    
    rewards = anti_repetition_reward_func(completions, **kwargs)
    
    # Log individual rewards
    for i, (reward, response) in enumerate(zip(rewards, responses)):
        log_reward(
            reward_name="anti_repetition",
            values=[reward],
            samples=[f"sample_{i}_len:{len(response)}_has_penalty:{reward<0}"]
        )
        
        # Collect reward for CSV logging
        collect_reward("anti_repetition_reward_func", i, reward)
    
    # Log aggregated metrics
    log_reward_metrics(step=kwargs.get('step', 0), 
                       rewards_dict={"anti_repetition_reward_func": rewards})
    
    return rewards


def wrapped_topic_relevance_reward(prompts, completions, **kwargs) -> List[float]:
    """Wrapper around topic_relevance_reward to log detailed metrics."""
    # Handle different completion formats
    if completions and isinstance(completions[0], str):
        responses = completions
    else:
        try:
            responses = [completion[0]['content'] for completion in completions]
        except (TypeError, IndexError, KeyError):
            responses = completions
    
    rewards = topic_relevance_reward_func(prompts, completions, **kwargs)
    
    # Log individual rewards
    for i, (reward, response) in enumerate(zip(rewards, responses)):
        log_reward(
            reward_name="topic_relevance",
            values=[reward],
            samples=[f"sample_{i}_relevant:{reward>0.5}"]
        )
        
        # Collect reward for CSV logging
        collect_reward("topic_relevance_reward_func", i, reward)
    
    # Log aggregated metrics
    log_reward_metrics(step=kwargs.get('step', 0), 
                       rewards_dict={"topic_relevance_reward_func": rewards})
    
    return rewards


# Map of original reward functions to wrapped versions
WRAPPED_REWARD_FUNCTIONS = {
    correctness_reward_func: wrapped_correctness_reward,
    int_reward_func: wrapped_int_reward,
    strict_format_reward_func: wrapped_strict_format_reward,
    soft_format_reward_func: wrapped_soft_format_reward,
    xmlcount_reward_func: wrapped_xmlcount_reward,
    anti_repetition_reward_func: wrapped_anti_repetition_reward,
    topic_relevance_reward_func: wrapped_topic_relevance_reward
}

# Create list of wrapped reward functions in the same order as the original REWARD_FUNCTIONS
def get_wrapped_reward_functions(original_reward_functions):
    """Get wrapped versions of reward functions in the same order as the originals."""
    # Reset rewards collection
    reset_rewards_collection()
    return [WRAPPED_REWARD_FUNCTIONS.get(func, func) for func in original_reward_functions] 