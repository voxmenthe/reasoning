"""
Example script demonstrating integration of the centralized logging system
with Gemma training scripts.

This script shows how to:
1. Initialize the logging system
2. Use the various loggers
3. Track metrics and model outputs
4. Integrate with reward functions

Run this script directly to see the logging system in action.
"""

import os
import torch
import numpy as np
from transformers import TrainerCallback
import threading
import time

# Import our centralized logging system
from src.logging import (
    initialize as init_logging,
    get_training_logger,
    get_model_logger,
    get_reward_logger,
    get_metrics_logger,
    get_system_logger,
    log_model_output,
    log_reward,
    log_training_progress,
    log_memory_usage,
    log_reward_metrics,
    log_generation_metrics
)

# Import our custom callback implementations
from src.training.enhanced_logging_callback import EnhancedLoggingCallback
from src.training.optimized_logging_callback import OptimizedLoggingCallback

# Import our wrapped reward functions
from src.training.wrapped_rewards import (
    wrapped_correctness_reward,
    wrapped_int_reward,
    wrapped_soft_format_reward,
    wrapped_strict_format_reward,
    wrapped_xmlcount_reward,
    wrapped_anti_repetition_reward,
    wrapped_topic_relevance_reward,
    get_wrapped_reward_functions
)

# Example implementation of a basic GRPOTrainer mock for testing
class MockTrainer:
    """Mock trainer for demonstration purposes."""
    
    def __init__(self, callbacks=None):
        """Initialize the mock trainer."""
        self.callbacks = callbacks or []
        self.state = MockTrainerState()
        self.args = MockTrainerArgs()
        self.model = MockModel()
        self.current_batch_info = None
    
    def run_training(self, num_steps=100):
        """Simulate running training for some steps."""
        logger = get_training_logger()
        logger.info("Starting mock training run")
        
        # Call on_init_end for all callbacks
        for callback in self.callbacks:
            callback.on_init_end(self.args, self.state, {})
        
        # Simulate training loop
        for step in range(1, num_steps + 1):
            self.state.global_step = step
            
            # Generate mock batch data
            self._generate_mock_batch()
            
            # Generate mock logs
            logs = self._generate_mock_logs(step)
            
            # Call on_step_end for all callbacks
            for callback in self.callbacks:
                callback.on_step_end(
                    self.args, self.state, {},
                    logs=logs, model=self.model, trainer=self
                )
            
            # Simulate training computation
            time.sleep(0.01)
        
        # Call on_train_end for all callbacks
        for callback in self.callbacks:
            callback.on_train_end(self.args, self.state, {})
            
        logger.info("Completed mock training run")
    
    def _generate_mock_batch(self):
        """Generate mock batch data for the current step."""
        # Create mock questions, answers, and model outputs
        questions = [
            f"What is {np.random.randint(1, 100)} + {np.random.randint(1, 100)}?",
            f"Calculate {np.random.randint(1, 50)} * {np.random.randint(1, 20)}."
        ]
        
        # Mock true answers
        true_answers = [
            str(int(questions[0].split()[2]) + int(questions[0].split()[4][:-1])),
            str(int(questions[1].split()[1]) * int(questions[1].split()[3][:-1]))
        ]
        
        # Mock model outputs with XML tags
        model_outputs = []
        for i, (q, a) in enumerate(zip(questions, true_answers)):
            # Occasionally generate incorrect answer for realism
            correct = np.random.random() > 0.3
            model_answer = a if correct else str(int(a) + np.random.randint(1, 10))
            
            # Generate mock reasoning
            reasoning = f"To solve {q}\n"
            if "+" in q:
                nums = q.split('+')
                n1 = nums[0].split()[-1]
                n2 = nums[1].split()[0]
                reasoning += f"I'll add {n1} and {n2}.\n{n1} + {n2} = {model_answer}"
            else:
                nums = q.split('*')
                n1 = nums[0].split()[-1]
                n2 = nums[1].split()[0]
                reasoning += f"I'll multiply {n1} and {n2}.\n{n1} * {n2} = {model_answer}"
            
            # Create model output with XML tags
            output = f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{model_answer}\n</answer>"
            model_outputs.append(output)
        
        # Store in the trainer's batch info
        self.current_batch_info = {
            'questions': questions,
            'true_answers': true_answers, 
            'model_outputs': model_outputs
        }
    
    def _generate_mock_logs(self, step):
        """Generate mock logs for the current step."""
        # Create mock log data including rewards
        logs = {
            'loss': 2.5 * np.exp(-step/50) + 0.1 * np.random.random(),
            'rewards/correctness_reward_func': np.random.random() * 2.5,
            'rewards/anti_repetition_reward_func': 0.5 - 0.1 * np.random.random(),
            'rewards/soft_format_reward_func': 0.48 + 0.04 * np.random.random(),
            'rewards/int_reward_func': 0.5,
            'reward': 3.0 * np.random.random()
        }
        return logs


class MockTrainerState:
    """Mock trainer state for demonstration purposes."""
    
    def __init__(self):
        """Initialize the mock state."""
        self.global_step = 0
        self.learning_rate = 5e-6


class MockTrainerArgs:
    """Mock trainer arguments for demonstration purposes."""
    
    def __init__(self):
        """Initialize the mock arguments."""
        self.max_steps = 100
        self.per_device_train_batch_size = 2
        self.gradient_accumulation_steps = 2
        self.num_generations = 1


class MockModel:
    """Mock model for demonstration purposes."""
    
    def __init__(self):
        """Initialize the mock model."""
        self.parameters = [
            ('layer1.weight', torch.rand(10, 10, requires_grad=True)),
            ('layer1.bias', torch.rand(10, requires_grad=True)),
            ('layer2.weight', torch.rand(5, 10, requires_grad=True))
        ]
    
    def named_parameters(self):
        """Return named parameters of the mock model."""
        for name, param in self.parameters:
            yield name, param


def get_mock_reward_functions():
    """Get mock reward functions for demonstration."""
    # These are simplified mock implementations that return random rewards
    def mock_correctness_reward(prompts, completions, answer, **kwargs):
        return [2.0 if np.random.random() > 0.3 else 0.0 for _ in completions]
    
    def mock_int_reward(completions, **kwargs):
        return [0.5 for _ in completions]
    
    def mock_format_reward(completions, **kwargs):
        return [0.5 if np.random.random() > 0.1 else 0.0 for _ in completions]
    
    # Return the list of mock reward functions
    return [
        mock_correctness_reward,
        mock_int_reward,
        mock_format_reward
    ]


def run_standard_example():
    """Run a standard example with the EnhancedLoggingCallback."""
    # Initialize the logging system
    init_logging("src/logging/config/gemma_logging_config.yaml")
    logger = get_training_logger()
    
    logger.info("Running standard integration example")
    
    # Create a mock trainer with the enhanced logging callback
    trainer = MockTrainer(callbacks=[EnhancedLoggingCallback()])
    
    # Run the mock training
    trainer.run_training(num_steps=20)
    
    logger.info("Standard integration example completed")


def run_optimized_example():
    """Run an optimized example with the OptimizedLoggingCallback."""
    # Initialize the logging system (same config, different callback)
    init_logging("src/logging/config/gemma_logging_config.yaml")
    logger = get_training_logger()
    
    logger.info("Running optimized integration example")
    
    # Create a mock trainer with the optimized logging callback
    callback = OptimizedLoggingCallback(checkpoint_dir="logs/checkpoints")
    trainer = MockTrainer(callbacks=[callback])
    
    # Run the mock training
    trainer.run_training(num_steps=30)
    
    logger.info("Optimized integration example completed")


def run_reward_wrapping_example():
    """Run an example showing reward function wrapping."""
    # Initialize the logging system
    init_logging("src/logging/config/gemma_logging_config.yaml")
    logger = get_reward_logger()
    
    logger.info("Running reward function wrapping example")
    
    # Get mock reward functions
    original_rewards = get_mock_reward_functions()
    
    # Just for the example, manually wrap one function
    def wrapped_mock_reward(prompts, completions, answer, **kwargs):
        # Call the original
        rewards = original_rewards[0](prompts, completions, answer, **kwargs)
        
        # Log rewards through the logging system
        log_reward_metrics(
            step=kwargs.get('step', 0),
            rewards_dict={"mock_correctness_reward": rewards}
        )
        
        return rewards
    
    # Run a few simulated batches to show the wrapping
    mock_prompts = [["What is 2+2?"]]
    mock_completions = [[{"content": "<reasoning>\n2+2=4\n</reasoning>\n<answer>\n4\n</answer>"}]]
    mock_answers = ["4"]
    
    for step in range(5):
        # Run the wrapped reward function
        rewards = wrapped_mock_reward(
            mock_prompts, mock_completions, mock_answers, step=step
        )
        logger.info(f"Step {step} rewards: {rewards}")
        
        # Simulate some time passing
        time.sleep(0.1)
    
    logger.info("Reward function wrapping example completed")


if __name__ == "__main__":
    # Create necessary log directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("logs/checkpoints", exist_ok=True)
    
    # Run the examples
    run_standard_example()
    print("\n" + "-" * 50 + "\n")
    run_optimized_example()
    print("\n" + "-" * 50 + "\n")
    run_reward_wrapping_example()
    
    print("\nAll examples completed. Check the logs directory for output files.") 