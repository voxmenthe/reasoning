# Logging System Integration Plan

## Analysis of Current Logging Patterns

After examining the existing Gemma training scripts, we've identified the following logging patterns:

### 1. In `new_gemma_training.py`

- **Standard Python Logging**: 
  - Uses `logging.basicConfig` with both file and console handlers
  - Log level set to INFO
  - Custom formatter for timestamp, logger name, level and message
  - Logs to "training_process.log"

- **Direct Print Statements**:
  - Device information and initialization messages
  - Training configuration details
  - Progress indicators

- **CSV Logging in Callback**:
  - Custom `OutputLoggingCallback` class implements CSV logging
  - Stores data in memory and writes periodically (every 50 steps)
  - Synchronous writing which can block training
  - Logs model outputs, questions, true answers, and extracted reasoning/answers

- **Metrics Logging**:
  - Loss values logged every 5 steps
  - Reward function values (correctness, anti-repetition)
  - Accuracy calculation
  - Parameter gradients logging (every 20 steps)

### 2. In `optimized_gemma_training.py`

- **Enhanced CSV Logging**:
  - Uses background thread via `CSVSaverThread` class
  - Non-blocking CSV writing
  - Similar data structure (questions, answers, reasoning)

- **Reduced Logging Frequency**:
  - Progress printed every 5 steps instead of every step
  - Metrics logged every 20 steps instead of every 5
  - CSV saved with same frequency (50 steps)

- **Overall Optimizations**:
  - Thread-based approach to avoid training interruptions
  - Careful management of logging frequency
  - Streamlined metrics reporting

### 3. Reward System

- **Multiple Reward Functions**:
  - Correctness reward (matching expected answer)
  - Format adherence rewards (XML structure)
  - Anti-repetition rewards
  - Topic relevance
  - Integer validation

- **Reward Configuration**:
  - Defined in `reward_config.py`
  - Contains weights, scales, and thresholds
  - Multiple penalties and specialized settings

## Integration Approach

We'll implement the integration in phases, focusing on one script at a time to minimize disruption:

### Phase 1: Custom Configuration for Gemma Training

1. **Create Gemma-specific logging configuration**:

```yaml
# src/logging/config/gemma_logging_config.yaml
version: 1
disable_existing_loggers: false

root:
  level: INFO
  handlers: [console]

loggers:
  logging.training:
    level: INFO
    handlers: [console, file, tensorboard]
    propagate: false
  
  logging.model:
    level: INFO
    handlers: [console, file, csv, tensorboard]
    propagate: false
    
  logging.reward:
    level: INFO
    handlers: [file, tensorboard]
    propagate: false
    
  logging.system:
    level: INFO
    handlers: [file]
    propagate: false
    
  logging.metrics:
    level: INFO
    handlers: [file, tensorboard]
    propagate: false

handlers:
  console:
    class: logging.StreamHandler
    formatter: standard
    level: INFO
    
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: standard
    filename: logs/training.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    
  csv:
    class: logging.custom.CSVHandler
    filename: logs/model_outputs.csv
    columns: [timestamp, question, answer, llm_reasoning, llm_answer]
    buffer_size: 100
    flush_interval: 5.0
    
  tensorboard:
    class: logging.custom.TensorBoardHandler
    log_dir: logs/tensorboard
    flush_secs: 30
    max_queue: 100

formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
    
  json:
    format: json
    fields: [timestamp, level, name, message]

# Model evaluation metrics configuration
metrics:
  # How frequently to log metrics (in training steps)
  logging_frequency: 5
  
  # How frequently to run validation (in training steps)
  validation_frequency: 250
  
  # Sample a subset of batches for more expensive metrics
  sampling_ratio: 0.1
  
  # Enable/disable specific metric categories
  enable:
    reward_metrics: true
    generation_metrics: true
    validation_metrics: true
    resource_metrics: true
```

### Phase 2: Integration with `new_gemma_training.py`

1. **Add imports at the top of the file**:
```python
# Import centralized logging system
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
```

2. **Replace logging setup**:
```python
# Initialize logging system with Gemma-specific configuration
init_logging("src/logging/config/gemma_logging_config.yaml")
logger = get_training_logger()
```

3. **Update print statements**:
```python
# Before:
print(f"MPS (Apple Silicon GPU) available: {is_mps_available}")

# After:
logger.info(f"MPS (Apple Silicon GPU) available: {is_mps_available}")
```

4. **Replace OutputLoggingCallback with new logging-integrated version**:
```python
class EnhancedLoggingCallback(TrainerCallback):
    def __init__(self):
        self.log_counter = 0
        self.total_answers = 0
        self.correct_answers = 0
    
    def on_init_end(self, args, state, control, **kwargs):
        logger.info("Training initialization completed")
        return control
        
    def on_step_end(self, args, state, control, logs=None, model=None, tokenizer=None, **kwargs):
        self.log_counter += 1
        
        # Update correct answers counter if available in logs
        if logs and 'rewards/correctness_reward_func' in logs:
            batch_size = args.per_device_train_batch_size
            num_generations = args.num_generations
            examples_this_step = batch_size * num_generations
            
            self.total_answers += examples_this_step
            
            from reward_config import CORRECTNESS_REWARD
            correct_value = logs['rewards/correctness_reward_func']
            
            if correct_value > 0:
                correct_count = int(round(correct_value / CORRECTNESS_REWARD))
                self.correct_answers += correct_count
        
        # Log training progress
        progress_pct = (state.global_step / args.max_steps) * 100
        accuracy = 0.0
        if self.total_answers > 0:
            accuracy = (self.correct_answers / self.total_answers) * 100
        
        # Use the new logging system to log progress
        log_training_progress(
            step=state.global_step,
            metrics={
                "progress_percent": progress_pct,
                "correct_answers": self.correct_answers,
                "total_answers": self.total_answers,
                "accuracy": accuracy
            }
        )
        
        # Get model outputs if available
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
                        
                        # Log using the new system
                        log_model_output(
                            question=question,
                            true_answer=true_answer,
                            model_output=model_output,
                            reasoning=llm_reasoning
                        )
            except Exception as e:
                logger.warning(f"Error capturing training data: {str(e)}")
        
        # Log rewards if available
        if logs:
            # Extract all reward-related metrics
            reward_metrics = {
                k.replace('rewards/', ''): v 
                for k, v in logs.items() 
                if k.startswith('rewards/')
            }
            
            # Convert scalar values to lists for the log_reward_metrics function
            rewards_dict = {
                k: [v] for k, v in reward_metrics.items()
            }
            
            # Log using the new system
            if rewards_dict:
                log_reward_metrics(state.global_step, rewards_dict)
        
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # Log final statistics
        accuracy = 0.0
        if self.total_answers > 0:
            accuracy = (self.correct_answers / self.total_answers) * 100
            
        log_training_progress(
            step=state.global_step,
            metrics={
                "final_accuracy": accuracy,
                "correct_answers": self.correct_answers,
                "total_answers": self.total_answers,
                "training_complete": True
            }
        )
        
        logger.info("Training completed")
        return control
```

5. **Modify training setup to use new callback**:
```python
# Initialize GRPO trainer with the new callback
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_functions=REWARD_FUNCTIONS,
    callbacks=[EnhancedLoggingCallback()]
)
```

6. **Add resource monitoring**:
```python
# Add memory usage tracking
if is_mps_available:
    initial_memory = torch.mps.current_allocated_memory() / (1024 * 1024)
    log_memory_usage()  # Use the new system for consistent reporting
```

### Phase 3: Integration with `optimized_gemma_training.py`

Follow a similar approach to the integration with `new_gemma_training.py`, but adapt to the optimized structure:

1. **Leverage existing optimizations**:
   - The background CSV thread is already implemented in the new logging system
   - Adopt the reduced logging frequency from the optimized script

2. **Create an enhanced callback class**:
```python
class OptimizedLoggingCallback(TrainerCallback):
    def __init__(self, checkpoint_dir=None):
        self.log_counter = 0
        self.total_answers = 0
        self.correct_answers = 0
        self.checkpoint_dir = checkpoint_dir
        
    # Similar implementation to the EnhancedLoggingCallback above,
    # but with optimized frequency settings
```

### Phase 4: Test and Validate Integration

1. **Create a small-scale test dataset**:
   - Use a subset of the original training data
   - Keep original structure but reduce size

2. **Run short training sessions**:
   - Validate that logs are correctly captured
   - Check TensorBoard output for metrics visualization
   - Verify CSV files contain expected data
   - Test with different logging configurations

3. **Performance benchmarking**:
   - Compare training throughput before and after integration
   - Measure memory usage impact
   - Check CPU utilization for background logging threads
   - Test with different batch sizes and logging frequencies

### Phase 5: Documentation Update

1. **Add integration documentation**:
   - Update README with integration details
   - Create examples specific to Gemma training
   - Document all available metrics and their meanings
   - Provide configuration examples for different scenarios

2. **Create quick reference guides**:
   - Command line overview of available commands
   - Configuration reference for all logging options
   - Metrics reference with visualization tips

## Handling Special Cases

### Reward Function Integration

For reward functions, we need special handling to ensure all metrics are properly captured:

1. **Create reward-specific wrappers**:
```python
def wrapped_correctness_reward(prompts, completions, answer, **kwargs):
    """Wrapper around correctness_reward to log detailed metrics"""
    rewards = correctness_reward_func(prompts, completions, answer, **kwargs)
    
    # Log individual rewards
    reward_logger = get_reward_logger()
    for i, reward in enumerate(rewards):
        if i < len(completions) and i < len(answer):
            response = completions[i][0]['content'] if i < len(completions) else "N/A"
            expected = answer[i] if i < len(answer) else "N/A"
            extracted = extract_xml_answer(response)
            
            # Log via the reward system
            log_reward(
                reward_name="correctness",
                values=[reward],
                samples=[f"sample_{i}"]
            )
    
    return rewards
```

2. **Update reward configuration**:
```python
# Modified REWARD_FUNCTIONS with wrapped versions
REWARD_FUNCTIONS = [
    wrapped_xmlcount_reward_func,
    wrapped_soft_format_reward_func,
    wrapped_strict_format_reward_func,
    wrapped_int_reward_func,
    wrapped_correctness_reward_func,
    wrapped_anti_repetition_reward_func,
    wrapped_topic_relevance_reward_func,
]
```

### Generation Metrics

Add generation metrics collection to capture model output quality:

```python
# Inside the callback's on_step_end method:

# Collect all model outputs for the batch
if hasattr(trainer, 'current_batch_info') and trainer.current_batch_info:
    model_outputs = trainer.current_batch_info.get('model_outputs', [])
    
    # Log generation metrics every 5 steps
    if self.log_counter % 5 == 0 and model_outputs:
        log_generation_metrics(
            step=state.global_step,
            generations=model_outputs
        )
```

## Timeline and Dependencies

1. **Week 1**: Integration with `new_gemma_training.py`
   - Day 1: Setup configuration and import structure
   - Day 2-3: Replace basic logging and print statements
   - Day 4-5: Implement enhanced callback class
   - Day 5: Initial testing

2. **Week 2**: Integration with `optimized_gemma_training.py`
   - Day 1-2: Adapt integration approach for optimized script
   - Day 3-4: Implement optimized callback with new logging
   - Day 5: Testing optimized version

3. **Week 3**: Full testing and refinement
   - Day 1-2: Comprehensive testing with various configurations
   - Day 3: Performance benchmarking
   - Day 4-5: Refinements based on test results

4. **Week 4**: Documentation and final integration
   - Day 1-2: Update documentation with Gemma-specific examples
   - Day 3: Create quick reference guides
   - Day 4-5: Final tests and submission 