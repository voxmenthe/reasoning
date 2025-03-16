# Logging Consolidation Plan

## Current State Analysis

The current logging implementation across the Gemma training scripts suffers from several issues:

1. **Inconsistent Logging Mechanisms**:
   - Standard Python `logging` module used in some places
   - Direct `print()` statements scattered throughout code
   - Custom CSV logging in callback classes
   - Specialized whitelist logging in `rewards.py`

2. **Scattered Configuration**:
   - Hard-coded file paths
   - Different logging frequencies
   - Inconsistent log formats
   - No centralized control for enabling/disabling specific log types

3. **Performance Issues**:
   - Some logging operations occur in the main training loop, potentially impacting performance
   - CSV writing happens in the same thread as training
   - No batching of log writes

## Goals

1. Create a unified, modular logging system
2. Make logging configurable through a single configuration file
3. Separate logging concerns from business logic
4. Improve performance by moving logging operations to background threads
5. Enable/disable specific log types without code changes
6. Standardize log formats and levels
7. Add meaningful metrics for model learning evaluation

## Implementation Strategy

### 1. Create a Core Logging Module Structure

```
/logging/
  __init__.py           # Exposes public API
  config.py             # Configuration handling
  handlers/
    __init__.py
    file_handler.py     # File logging
    console_handler.py  # Console output
    csv_handler.py      # CSV data logging
    tensorboard.py      # TensorBoard integration
  formatters/
    __init__.py
    standard.py         # Standard log format
    json.py             # JSON formatter
  loggers/
    __init__.py
    training.py         # Training progress logs
    model.py            # Model outputs/predictions
    reward.py           # Reward function logs
    system.py           # System metrics (memory, etc.)
    metrics.py          # Learning metrics tracking
  utils.py              # Utility functions
  metrics.py            # Model evaluation metrics
  validation.py         # Periodic validation runner
```

### 2. Configuration System

Create a YAML-based configuration system:

```yaml
# logging_config.yaml
version: 1
disable_existing_loggers: false

root:
  level: INFO
  handlers: [console, file]

loggers:
  training:
    level: INFO
    handlers: [console, file, tensorboard]
    propagate: false
  
  model_outputs:
    level: INFO
    handlers: [csv, file]
    propagate: false
    
  rewards:
    level: INFO
    handlers: [file, tensorboard]
    propagate: false
    
  whitelist:
    level: INFO
    handlers: [file]
    propagate: false
    
  metrics:
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
    columns: [timestamp, question, true_answer, model_answer, model_reasoning]
    buffer_size: 50
    
  tensorboard:
    class: logging.custom.TensorBoardHandler
    log_dir: logs/tensorboard

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
  logging_frequency: 50
  
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
    
  # Validation dataset settings
  validation:
    dataset_path: "data/validation_set.json"
    max_samples: 100
    timeout_per_sample: 5  # seconds
```

### 3. Core API Design

```python
# Main API in logging/__init__.py

# Initialize system from config
def initialize(config_path=None):
    """Initialize the logging system from configuration."""
    
# Get loggers
def get_training_logger():
    """Get logger for training progress."""
    
def get_model_logger():
    """Get logger for model outputs."""
    
def get_reward_logger():
    """Get logger for reward function logging."""
    
def get_system_logger():
    """Get logger for system metrics."""
    
def get_metrics_logger():
    """Get logger for training metrics."""

# Convenience functions
def log_model_output(question, true_answer, model_output, reasoning=None):
    """Log a model generation."""
    
def log_reward(reward_name, values, samples=None):
    """Log reward function values."""
    
def log_training_progress(step, metrics):
    """Log training progress metrics."""
    
def log_memory_usage():
    """Log current memory usage."""
    
def log_reward_metrics(step, rewards_dict, history_window=100):
    """Log aggregated reward metrics with trends."""
    
def log_generation_metrics(step, generations, references=None):
    """Log metrics about model generations (diversity, length, etc.)."""
    
def run_validation(step, model, tokenizer, num_samples=None):
    """Run validation on held-out dataset and log results."""
```

### 4. Custom Handlers Implementation

#### CSV Handler
Implement an asynchronous CSV handler that:
- Buffers writes in memory
- Writes to disk in a background thread
- Has configurable columns and formatting
- Supports automatic header writing

#### TensorBoard Handler
Create a handler that:
- Forwards metrics to TensorBoard
- Handles scalar, histogram, and text data
- Batches updates for efficiency

### 5. Integration Plan

1. **Phase 1: Core Module Implementation**
   - Create the basic module structure
   - Implement configuration loading
   - Develop custom handlers
   - Write tests for the logging system

2. **Phase 2: Replace Existing Logging**
   - Update `new_gemma_training.py` first
   - Replace direct print statements with appropriate logger calls
   - Move CSV logging to the new system
   - Test with a small training run

3. **Phase 3: Extend to Other Files**
   - Update `optimized_gemma_training.py`
   - Modify `rewards.py` to use the centralized logging
   - Update any inference scripts

### 6. Usage Examples

#### Initialization
```python
import logging_system

# Initialize with default config
logging_system.initialize()

# Or with custom config
logging_system.initialize("path/to/custom_config.yaml")
```

#### Training Progress Logging
```python
training_logger = logging_system.get_training_logger()

# Log step progress
training_logger.info(f"Step {step}/{total_steps} completed")

# Log metrics
logging_system.log_training_progress(
    step=current_step,
    metrics={
        "loss": current_loss,
        "accuracy": current_accuracy,
        "learning_rate": current_lr
    }
)
```

#### Model Output Logging
```python
# Log complete model outputs
logging_system.log_model_output(
    question="What is 2+2?",
    true_answer="4",
    model_output="<reasoning>\n2+2=4\n</reasoning>\n<answer>\n4\n</answer>",
    reasoning="2+2=4"
    copy_of_true_answer_for_comparison="4",
)
```

#### Reward Logging
```python
# Log rewards
logging_system.log_reward(
    reward_name="correctness_reward",
    values=[1.0, 0.0, 1.0],
    samples=["sample1", "sample2", "sample3"]
)
```

#### Metrics Logging
```python
# Log reward metrics with trend analysis
logging_system.log_reward_metrics(
    step=current_step,
    rewards_dict={
        "correctness_reward": batch_correctness_rewards,
        "anti_repetition_reward": batch_anti_repetition_rewards,
        "topic_relevance_reward": batch_topic_relevance_rewards
    }
)

# Log generation diversity metrics
logging_system.log_generation_metrics(
    step=current_step,
    generations=batch_generations
)

# Run validation on held-out data (automatically logs results)
if current_step % validation_frequency == 0:
    logging_system.run_validation(
        step=current_step,
        model=model,
        tokenizer=tokenizer
    )
```

### 7. Performance Considerations

- Use buffered logging for high-frequency events
- Move file I/O to background threads
- Implement rate limiting for certain log types
- Add log rotation for all file-based handlers
- Use sampling for very frequent events
- Defer expensive metric calculations to a background process
- Perform validation on a separate thread to avoid blocking training

### 8. Testing Strategy

- Unit tests for each handler and formatter
- Integration tests for the full system
- Performance benchmarks to ensure logging doesn't slow training
- Configuration validation tests
- Validation of metric calculation accuracy

## Model Learning Evaluation Metrics

Since this project uses GRPO (Generative Reinforcement Learning with Preferential Optimization), traditional loss metrics are less informative for tracking model learning. The following metrics will provide better insight while minimizing performance overhead:

### 1. Reward-Based Metrics

- **Per-reward function statistics**: Track mean, median, min, max for each reward function
- **Reward trends**: Track moving averages (window of 100 steps) to visualize learning progress
- **Reward distributions**: Periodically log histograms of reward values to detect mode collapse
- **Reward correlations**: Track correlations between different reward functions to detect conflicts

### 2. Generation Quality Metrics

- **Token entropy**: Measure diversity in model outputs to detect mode collapse
- **KL divergence from initial model**: Track how far the model has drifted from initial policy
- **Generation length distribution**: Track if model is generating longer/shorter outputs over time
- **XML format adherence rate**: Percentage of outputs properly formatted with reasoning+answer tags
- **Reasoning-to-answer ratio**: Track the ratio of tokens spent on reasoning vs. the answer

### 3. Task-Specific Performance Metrics

- **Answer correctness rate**: Percentage of outputs with correct numerical answers
- **Reasoning completeness**: Detect if all steps are present in reasoning section
- **Hallucination tracking**: Rate of unsupported statements in reasoning (requires validation set)

### 4. Validation Metrics (on held-out data)

- **Zero-shot accuracy**: Track model performance on unseen problems
- **Generalization gap**: Difference between training and validation metrics
- **Error pattern analysis**: Log common error types to guide reward function improvements

### 5. Resource Efficiency Metrics

- **Training throughput**: Examples processed per second
- **Memory utilization**: Peak and average memory usage
- **Computation-to-performance ratio**: Resource usage relative to improvement

### Implementation Strategy for Metrics

To ensure these metrics don't add excessive overhead:

1. **Frequency-based logging**: Log basic metrics every 50 steps, full metrics every 250 steps
2. **Sampling-based approach**: Calculate expensive metrics on a subset of examples
3. **Background processing**: Move metric calculation off the critical path
4. **Incremental computation**: Update metrics incrementally rather than from scratch
5. **Early stopping detection**: Track plateau in reward metrics for automatic checkpointing

### Visualization Dashboards

Create TensorBoard-compatible dashboards for:

1. **Reward trends**: Line charts of reward metrics over time
2. **Performance metrics**: Accuracy rates on validation set
3. **Generation quality**: Distribution of output characteristics
4. **Resource usage**: Memory and computation efficiency

These dashboards will automatically update as logs are generated, providing real-time insight into training progress without manual intervention.

## Timeline and Milestones

1. **Week 1**: Core module implementation and configuration system
2. **Week 2**: Custom handlers and formatters
3. **Week 3**: Integration with `new_gemma_training.py`
4. **Week 4**: Complete integration with all scripts and testing
5. **Week 5**: Implement metrics tracking and validation pipeline

## Success Metrics

1. All logging consolidated through the new system
2. Configuration changes possible without code modifications
3. No performance regression in training loop (less than 5% overhead)
4. Improved readability and organization of logs
5. Reduced code duplication related to logging
6. Meaningful learning metrics that correlate with final model performance
7. Ability to identify training issues early through metric analysis 