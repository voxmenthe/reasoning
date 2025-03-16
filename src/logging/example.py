"""
Example script demonstrating how to use the logging system.

This script shows how to initialize and use the logging system
with various features like model output logging, metrics tracking,
and validation.
"""

import logging
import time
import random
import os
from typing import Dict, List, Any, Optional

# Import the logging system
from src.logging import (
    initialize,
    get_training_logger,
    get_model_logger,
    get_reward_logger,
    get_metrics_logger,
    log_model_output,
    log_reward,
    log_training_progress,
    log_memory_usage,
    log_reward_metrics,
    log_generation_metrics,
    run_validation
)

# Create a mock model and tokenizer for demonstration
class MockModel:
    """Mock model for demonstration."""
    def generate(self, input_text: str) -> str:
        """Generate a mock response."""
        return f"<reasoning>I'm thinking about {input_text}...</reasoning>\n<answer>42</answer>"

class MockTokenizer:
    """Mock tokenizer for demonstration."""
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [1] * len(text.split())
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return "Decoded text"

def main() -> None:
    """Run the example script."""
    # Initialize the logging system
    print("Initializing logging system...")
    initialize()  # Use default configuration
    
    # Get loggers
    training_logger = get_training_logger()
    model_logger = get_model_logger()
    reward_logger = get_reward_logger()
    metrics_logger = get_metrics_logger()
    
    # Log a basic message
    training_logger.info("Starting training example...")
    
    # Create mock model and tokenizer
    model = MockModel()
    tokenizer = MockTokenizer()
    
    # Simulate training steps
    for step in range(1, 51):
        # Log training progress
        training_logger.info(f"Step {step}/50")
        
        # Simulate a batch of examples
        batch_size = 16
        questions = [f"What is {i} + {step}?" for i in range(batch_size)]
        true_answers = [str(i + step) for i in range(batch_size)]
        
        # Simulate model outputs
        model_outputs = []
        reasoning_parts = []
        
        for question in questions:
            output = model.generate(question)
            model_outputs.append(output)
            
            # Extract reasoning part for metrics
            import re
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', output)
            if reasoning_match:
                reasoning_parts.append(reasoning_match.group(1))
            else:
                reasoning_parts.append("")
        
        # Log one example from the batch
        if step % 10 == 0:
            log_model_output(
                question=questions[0],
                true_answer=true_answers[0],
                model_output=model_outputs[0],
                reasoning=reasoning_parts[0]
            )
        
        # Simulate reward calculation
        rewards = {
            "correctness": [random.random() for _ in range(batch_size)],
            "coherence": [random.random() for _ in range(batch_size)],
            "reasoning_quality": [random.random() for _ in range(batch_size)]
        }
        
        # Log rewards for one example
        for reward_name, values in rewards.items():
            log_reward(reward_name, values[:1], questions[:1])
        
        # Log aggregated reward metrics
        log_reward_metrics(step, rewards)
        
        # Log generation metrics every 5 steps
        if step % 5 == 0:
            log_generation_metrics(step, model_outputs)
        
        # Log training metrics
        metrics = {
            "loss": 1.0 - (step / 50),
            "learning_rate": 0.001 * (1.0 - step / 100),
            "examples_seen": step * batch_size
        }
        log_training_progress(step, metrics)
        
        # Log memory usage every 10 steps
        if step % 10 == 0:
            log_memory_usage()
        
        # Run validation every 25 steps
        if step % 25 == 0:
            run_validation(step, model, tokenizer)
        
        # Simulate training time
        time.sleep(0.1)
    
    # Log training completed
    training_logger.info("Training example completed!")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Run the example
    main()
    
    print("\nLogging example completed!")
    print("Check the following files for logs:")
    print("  - logs/training.log")
    print("  - logs/model_outputs.csv")
    print("TensorBoard logs are available in logs/tensorboard/")
    print("Run 'tensorboard --logdir=logs/tensorboard' to view them.") 