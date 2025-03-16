# Gemma3 Reasoning Project

This project focuses on fine-tuning and using Gemma 3 models for reasoning tasks.

## Project Structure

```
├── src/
│   ├── inference/       # Inference related code
│   ├── training/        # Training related code
│   ├── notebooks/       # Jupyter notebooks
│   ├── datasets/        # Dataset handling code
│   ├── logging/         # Logging utilities
│   └── *.py             # Core modules
├── tests/               # Test directory
└── docs/                # Documentation
```

## Setup

### Prerequisites

- Python 3.10 or later
- Poetry for dependency management

### Installation

1. Run the setup script:

```bash
./project_setup.sh
```

This will:
- Create and activate the virtual environment at ~/venvs/gemma3 (if it doesn't exist)
- Install Poetry and project dependencies
- Configure the Jupyter kernel

### Manual Setup

If you prefer to set up manually:

1. Create and activate the virtual environment:
```bash
python -m venv ~/venvs/gemma3
source ~/venvs/gemma3/bin/activate
```

2. Install Poetry:
```bash
pip install --upgrade pip
pip install poetry
```

3. Configure Poetry to use the existing venv:
```bash
poetry config virtualenvs.create false
```

4. Install project dependencies:
```bash
poetry lock
poetry install
```

5. Install the Jupyter kernel:
```bash
python -m ipykernel install --user --name=gemma3 --display-name "Gemma3"
```

## Usage

TODO: Add usage instructions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Gemma Training Logging System

This module provides a centralized logging system for Gemma training scripts. It offers a unified interface for logging different types of information, with configurable handlers and formatters.

## Features

- **Unified API**: Consistent interface for all logging needs
- **Configurable**: YAML-based configuration system
- **Asynchronous Logging**: Background thread processing for high-frequency events
- **Multiple Output Formats**: Console, File, CSV, and TensorBoard
- **Model Metrics**: Specialized logging for GRPO/RL training metrics
- **Validation**: Periodic evaluation on held-out data

## Usage

### Basic Initialization

```python
from src.logging import initialize, get_training_logger

# Initialize with default configuration
initialize()

# Get a logger
logger = get_training_logger()
logger.info("Training started")
```

### Logging Model Outputs

```python
from src.logging import log_model_output

log_model_output(
    question="What is 2+2?",
    true_answer="4",
    model_output="<reasoning>2+2=4</reasoning>\n<answer>4</answer>",
    reasoning="2+2=4"
)
```

### Logging Rewards

```python
from src.logging import log_reward

log_reward(
    reward_name="correctness_reward",
    values=[1.0, 0.0, 1.0],
    samples=["sample1", "sample2", "sample3"]
)
```

### Logging Training Progress

```python
from src.logging import log_training_progress

log_training_progress(
    step=current_step,
    metrics={
        "loss": current_loss,
        "accuracy": current_accuracy,
        "learning_rate": current_lr
    }
)
```

### Logging Advanced Metrics

```python
from src.logging import log_reward_metrics, log_generation_metrics

# Log reward metrics with trend analysis
log_reward_metrics(
    step=current_step,
    rewards_dict={
        "correctness_reward": batch_correctness_rewards,
        "anti_repetition_reward": batch_anti_repetition_rewards,
        "topic_relevance_reward": batch_topic_relevance_rewards
    }
)

# Log generation diversity metrics
log_generation_metrics(
    step=current_step,
    generations=batch_generations
)
```

### Running Validation

```python
from src.logging import run_validation

run_validation(
    step=current_step,
    model=model,
    tokenizer=tokenizer
)
```

## Configuration

The logging system can be configured using a YAML file. Here's a basic example:

```yaml
version: 1
disable_existing_loggers: false

root:
  level: INFO
  handlers: [console]

loggers:
  logging.training:
    level: INFO
    handlers: [console, file]

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

formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
```

To use a custom configuration:

```python
from src.logging import initialize

initialize("path/to/custom_config.yaml")
```

## Full Example

See `example.py` for a complete demonstration of the logging system in action.

## Custom Handlers

The system includes several custom handlers:

- **CSVHandler**: Writes model outputs to CSV files asynchronously
- **TensorBoardHandler**: Logs metrics to TensorBoard for visualization

These are automatically configured when specified in the YAML configuration file.

## Metrics and Validation

The system provides specialized support for:

- **Reward Tracking**: Statistics for reward functions over time
- **Generation Metrics**: Diversity, format adherence, etc.
- **Validation**: Periodic evaluation on held-out data

These metrics are designed specifically for GRPO/RL-style training where traditional loss metrics may not be informative.

## Requirements

- Python 3.6+
- PyYAML
- TensorBoard (optional, for visualization)
- NumPy (for metrics calculation) 