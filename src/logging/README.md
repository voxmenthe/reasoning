# Centralized Logging System for Gemma Training

A comprehensive, modular logging system for ML training that provides:

- Unified configuration via YAML files
- Consistent log formats across components
- Background thread processing for non-blocking CSV logging
- TensorBoard integration for metrics visualization
- Detailed reward function tracking
- Model output capture and analysis
- Resource usage monitoring

## Key Components

- **Configuration System**: YAML-based config for all logging settings
- **Core Logger Types**:
  - `training`: Training progress and high-level metrics
  - `model`: Model outputs and generations
  - `reward`: Reward function values and metrics
  - `system`: System and resource usage information
  - `metrics`: Detailed performance metrics
- **Custom Handlers**:
  - `CSVHandler`: Background thread-based CSV logging
  - `TensorBoardHandler`: Metrics visualization
- **Metrics Collection**:
  - Reward metrics tracking
  - Generation quality metrics
  - Resource usage metrics

## Using the Logging System

### 1. Initialize the System

```python
from src.logging import initialize as init_logging

# Initialize with a configuration file
init_logging("src/logging/config/gemma_logging_config.yaml")
```

### 2. Get Specific Loggers

```python
from src.logging import (
    get_training_logger,
    get_model_logger,
    get_reward_logger,
    get_metrics_logger
)

# Get specific loggers for different components
training_logger = get_training_logger()
model_logger = get_model_logger()
```

### 3. Log Messages and Data

```python
# Standard logging
training_logger.info("Starting training run")
training_logger.warning("Learning rate may be too high")

# Log model outputs
from src.logging import log_model_output

log_model_output(
    question="What is 2+2?",
    true_answer="4",
    model_output="<reasoning>\n2+2=4\n</reasoning>\n<answer>\n4\n</answer>",
    reasoning="2+2=4",
    answer="4"
)

# Log metrics
from src.logging import log_training_progress

log_training_progress(
    step=100,
    metrics={
        "loss": 0.5,
        "accuracy": 85.2,
        "learning_rate": 5e-6
    }
)
```

### 4. Track Resources

```python
from src.logging import log_memory_usage

# Log memory usage
log_memory_usage()
```

### 5. Log Reward Function Metrics

```python
from src.logging import log_reward_metrics

rewards_dict = {
    "correctness_reward_func": [2.0, 0.0, 2.0],
    "format_reward_func": [0.5, 0.5, 0.5]
}

log_reward_metrics(step=100, rewards_dict=rewards_dict)
```

## Integration with Gemma Training Scripts

The logging system has been integrated with both the standard and optimized Gemma training scripts:

### Standard Script (`new_gemma_training.py`)

- Uses the `EnhancedLoggingCallback` for tracking training progress
- Replaces print statements with structured logging
- Captures model outputs in standard format

### Optimized Script (`optimized_gemma_training.py`)

- Uses the `OptimizedLoggingCallback` with reduced logging frequency
- Implements background threading for non-blocking logging
- Provides memory-efficient CSV and metrics tracking

### Reward Function Integration

The logging system wraps reward functions to automatically capture detailed metrics:

```python
# Get wrapped versions of original reward functions
from src.training.wrapped_rewards import get_wrapped_reward_functions

wrapped_reward_functions = get_wrapped_reward_functions(REWARD_FUNCTIONS)

# Use in trainer
trainer = GRPOTrainer(
    # ...
    reward_funcs=wrapped_reward_functions,
    # ...
)
```

## Configuration

### Sample Configuration File

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
  
  # ... other loggers ...

handlers:
  console:
    class: logging.StreamHandler
    formatter: standard
    level: INFO
    
  # ... other handlers ...

# Model evaluation metrics configuration
metrics:
  logging_frequency: 5
  validation_frequency: 250
  # ... additional settings ...
```

## Analyzing Logs

### Available Output Formats

1. **Console Output**: Real-time progress and critical messages
2. **Log Files**: Detailed logs in `logs/gemma_training.log`
3. **CSV Data**: Model outputs in `logs/gemma_outputs.csv`
4. **TensorBoard**: Visualize metrics in `logs/gemma_tensorboard/`

### TensorBoard Visualization

Launch TensorBoard to visualize metrics:

```bash
tensorboard --logdir=logs/gemma_tensorboard
```

This will show:
- Training loss curves
- Reward function values
- Model accuracy
- Resource usage
- Generation quality metrics

## Performance Considerations

The logging system is designed for minimal training impact:

- Background thread processing for CSV logging
- Configurable logging frequency
- Buffered metrics collection
- Memory-efficient handlers

For maximum performance:
- Reduce `logging_frequency` in the config
- Set appropriate `buffer_size` values
- Disable detailed metrics during production runs 