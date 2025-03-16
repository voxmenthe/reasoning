#!/usr/bin/env python
# coding: utf-8

# # Enhanced Gemma 3 1b Thinking on MAC 🍎
# 
# This script is an enhanced version of the original Gemma3 1b Thinking notebook,
# optimized for Apple Silicon with improvements from the Fine_Tuning_Gemma_3 script.
#
# Improvements include:
# - Better LoRA configuration with targeted modules
# - Gradient checkpointing for memory efficiency
# - MPS-compatible optimizer settings
# - Enhanced tokenizer with chat template
# - Memory tracking for Apple Silicon

import os
import torch
import warnings
import logging
from transformers import Gemma3ForCausalLM, AutoTokenizer, GenerationConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
import re
import csv
from pathlib import Path

# Import reward functions configuration
from reward_config import REWARD_FUNCTIONS

# Custom forward wrapper to ensure inputs require gradients
class ForwardWrapper:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, *args, **kwargs):
        # Ensure inputs have requires_grad=True only for floating point tensors
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point and not v.requires_grad:
                kwargs[k] = v.detach().clone().requires_grad_(True)
                
        # For positional args
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point and not arg.requires_grad:
                new_args.append(arg.detach().clone().requires_grad_(True))
            else:
                new_args.append(arg)
        
        return self.model(*new_args, **kwargs)



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import dataset functions from separate module
from src.datasets.reasoning_dataset import (
    get_gsm8k_questions, 
    process_chat_data
)

# Set your Hugging Face token
from creds import all_creds
os.environ["HUGGING_FACE_HUB_TOKEN"] = all_creds['HUGGINGFACE_ACCESS_TOKEN_Gemma']

# Check for MPS (Apple Silicon) availability
is_mps_available = torch.backends.mps.is_available()
print(f"MPS (Apple Silicon GPU) available: {is_mps_available}")

# Set device and data type
if is_mps_available:
    device = torch.device("mps")
    print("Using MPS device for training")
    
    # Track initial memory usage on MPS
    initial_memory = torch.mps.current_allocated_memory() / (1024 * 1024)
    print(f"Initial MPS memory usage: {initial_memory:.2f} MB")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for training")

# Set dtype - bfloat16 is supported on Apple Silicon M2+ chips
torch_dtype = torch.bfloat16

# Model and tokenizer initialization
# --------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-4b-it" # "google/gemma-3-1b-it",

# Load base model
print("Loading base model...")
# First load the model on CPU without any dtype or device specification
model = Gemma3ForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    attn_implementation="eager"  # Explicitly set eager attention during loading
)

# Then set dtype and move to appropriate device
print(f"Converting model to {torch_dtype} and moving to device...")
if is_mps_available:
    # First convert to appropriate dtype while still on CPU
    model = model.to(torch_dtype)
    # Then move to MPS device
    model = model.to("mps")
else:
    model = model.to(device=device, dtype=torch_dtype)

# Enable gradient checkpointing for memory efficiency
print("Enabling gradient checkpointing...")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})

# Load and configure tokenizer
print("Loading tokenizer...")
processor = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos=True)
processor.padding_side = 'right'

# Add chat template from the instruction-tuned model (improvement from Fine_Tuning script)
processor.chat_template = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos=True).chat_template

# Configure LoRA - improved settings from Fine_Tuning script
print("Configuring LoRA adapters...")
peft_config = LoraConfig(
    lora_alpha=16,  # Increased from 4 in original script
    lora_dropout=0.05,
    r=16,  # Increased from 4 in original script
    bias="none",
    task_type="CAUSAL_LM",
    # Targeted modules instead of "all-linear" for better efficiency
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

# Apply LoRA adapters
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

# Load and process dataset
# --------------------------------------------------------------------------
print("Loading GSM8K dataset...")
dataset = get_gsm8k_questions()
dataset_size = len(dataset)
print(f"Dataset size: {dataset_size} examples")

# Training configuration
# --------------------------------------------------------------------------
# Set dynamic batch size based on device
if is_mps_available:
    per_device_batch = 2  # Increased from 1 to 2 to be divisible by num_generations
    grad_accum = 4  # Reduced from 8 since we increased batch size
    print("Using MPS-optimized batch size")
else:
    per_device_batch = 4  # Original value
    grad_accum = 4  # Original value

# Define training parameters
num_epochs = 3  # Define number of epochs upfront
effective_batch_size = per_device_batch * grad_accum
steps_per_epoch = dataset_size // effective_batch_size
total_steps = steps_per_epoch * num_epochs

print(f"\nTraining configuration:")
print(f"- Number of epochs: {num_epochs}")
print(f"- Effective batch size: {effective_batch_size}")
print(f"- Steps per epoch: {steps_per_epoch}")
print(f"- Total steps: {total_steps}")

# Configure sequence lengths
max_prompt_length = 1024
max_seq_length = 2048

# Configure training arguments with MPS-compatible optimizer
print("Configuring training arguments...")
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    
    # Use adamw_torch which is MPS-compatible (not 8bit version)
    optim="adamw_torch",
    
    # Increase logging frequency
    logging_steps=1,
    logging_first_step=True,
    log_level="info",
    
    per_device_train_batch_size=per_device_batch,
    gradient_accumulation_steps=grad_accum,
    num_generations=2,  # This needs to evenly divide into the batch size
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    num_train_epochs=num_epochs,
    max_steps=total_steps,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    cache_implementation="hybrid",
    
    # Ensure correct gradient computation
    gradient_checkpointing=True,  # Enable gradient checkpointing in trainer arguments
    
    # Output directory
    output_dir=f"./gemma3_{MODEL_NAME}_thinking_mac"
)

# Make sure we're not using the cache during training to ensure gradient flow
# (Set this on the model instead of in config)
model.config.use_cache = False

print(f"- Total steps for {training_args.num_train_epochs} epochs: {training_args.max_steps}")

# Initialize and start training
# --------------------------------------------------------------------------
print("Initializing GRPO trainer...")
logger.info("Initializing GRPO trainer with the following configuration:")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Learning rate: {training_args.learning_rate}")
logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
logger.info(f"Number of epochs: {training_args.num_train_epochs}")
logger.info(f"Max steps: {training_args.max_steps}")

# Ensure LoRA weights are correctly set to require gradients
for name, param in model.named_parameters():
    if 'lora' in name:  # Check if parameter is part of LoRA
        param.requires_grad = True
        
# Confirm trainable parameters have requires_grad=True
trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
logger.info(f"Trainable parameters: {len(trainable_params)}")
logger.info(f"Sample trainable parameters: {trainable_params[:5]}")

# Custom callback to log model outputs during training
class OutputLoggingCallback(TrainerCallback):
    def __init__(self):
        self.log_counter = 0
        self.total_answers = 0
        self.correct_answers = 0
        
        # Add CSV tracking
        self.training_data = []
        self.csv_save_frequency = 50  # Save every 50 steps
        self.csv_path = "training_outputs.csv"
        
        # Ensure CSV directory exists
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['question', 'answer', 'llm_reasoning', 'llm_answer'])
    
    def on_init_end(self, args, state, control, **kwargs):
        logger.info("Training initialization completed")
        print(f"\nTraining progress: 0/{args.max_steps} steps (0.0%) | Correct: 0/0 (0.0%)")
        return control
        
    def on_step_end(self, args, state, control, logs=None, model=None, tokenizer=None, **kwargs):
        self.log_counter += 1
        
        # Update correct answers counter if available in logs
        if logs and 'rewards/correctness_reward_func' in logs:
            # Each step processes batch_size * num_generations examples
            batch_size = args.per_device_train_batch_size
            num_generations = args.num_generations
            examples_this_step = batch_size * num_generations
            
            self.total_answers += examples_this_step
            
            # Check for correctness based on non-zero reward
            # The reward value indicates how many were correct in this batch
            from reward_config import CORRECTNESS_REWARD
            correct_value = logs['rewards/correctness_reward_func']
            
            # If the reward is for the whole batch, divide by CORRECTNESS_REWARD to get count
            if correct_value > 0:
                correct_count = int(round(correct_value / CORRECTNESS_REWARD))
                self.correct_answers += correct_count
        
        # Get model outputs if available
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
                        
                        # Store data for CSV
                        self.training_data.append({
                            'question': question,
                            'answer': true_answer,
                            'llm_reasoning': llm_reasoning,
                            'llm_answer': llm_answer
                        })
            except Exception as e:
                logger.warning(f"Error capturing training data: {str(e)}")
        
        # Calculate accuracy percentage
        accuracy = 0.0
        if self.total_answers > 0:
            accuracy = (self.correct_answers / self.total_answers) * 100
            
        # Print lightweight progress to stdout every step
        progress_pct = (state.global_step / args.max_steps) * 100
        print(f"\rTraining progress: {state.global_step}/{args.max_steps} steps ({progress_pct:.1f}%) | Correct: {self.correct_answers}/{self.total_answers} ({accuracy:.1f}%)", end="", flush=True)
        
        # Save CSV periodically
        if self.log_counter % self.csv_save_frequency == 0 and self.training_data:
            self.save_csv()
            logger.info(f"Training data saved to CSV at step {state.global_step}")
        
        if self.log_counter % 5 == 0 and logs:  # Log every 5 steps
            if 'loss' in logs:
                logger.info(f"Step {state.global_step}: Loss: {logs['loss']:.10f}")
            if 'rewards/correctness_reward_func' in logs:
                logger.info(f"Step {state.global_step}: Correctness reward: {logs['rewards/correctness_reward_func']:.6f}")
            if 'rewards/anti_repetition_reward_func' in logs:
                logger.info(f"Step {state.global_step}: Anti-repetition reward: {logs['rewards/anti_repetition_reward_func']:.6f}")
            logger.info(f"Step {state.global_step}: Total reward: {logs.get('reward', 'N/A')}")
            logger.info(f"Step {state.global_step}: Accuracy so far: {self.correct_answers}/{self.total_answers} ({accuracy:.2f}%)")
            
            # Log all available metrics for debugging
            logger.info(f"Step {state.global_step}: Available metrics: {', '.join(logs.keys())}")
            
            # Log trainable parameter gradients to verify they're being updated
            if self.log_counter % 20 == 0:  # Less frequent check
                param_with_grad = 0
                for name, param in trainer.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param_with_grad += 1
                logger.info(f"Step {state.global_step}: Parameters with gradients: {param_with_grad}")
                
                # Log sample parameter values to verify updates
                if param_with_grad > 0:
                    for name, param in list(trainer.model.named_parameters())[:3]:
                        if param.requires_grad:
                            logger.info(f"Parameter {name}: {param.data.flatten()[:3].tolist()}")
        
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # Print final progress and add a newline
        accuracy = 0.0
        if self.total_answers > 0:
            accuracy = (self.correct_answers / self.total_answers) * 100
            
        print(f"\rTraining progress: {state.global_step}/{args.max_steps} steps (100.0%) | Correct: {self.correct_answers}/{self.total_answers} ({accuracy:.1f}%)")
        print("\nTraining completed!")
        
        # Save final CSV
        self.save_csv()
        logger.info(f"Final training data saved to {self.csv_path}")
        
        logger.info(f"Final accuracy: {self.correct_answers}/{self.total_answers} ({accuracy:.2f}%)")
        logger.info("Training completed")
        return control
    
    def save_csv(self):
        """Save training data to CSV file"""
        if not self.training_data:
            logger.warning("No training data to save")
            return
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.training_data)
            
            # If file exists, append without headers
            if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
                df.to_csv(self.csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.csv_path, index=False)
                
            # Clear the stored data after saving
            self.training_data = []
            
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")

# Function to extract reasoning and answer from model output
def extract_reasoning_and_answer(text):
    """
    Extract reasoning and answer from model output using the XML format tags
    Returns tuple of (reasoning, answer)
    """
    reasoning = ""
    answer = ""
    
    # Extract reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return reasoning, answer

# Apply the forward wrapper to ensure inputs have requires_grad=True
original_forward = model.forward
model.forward = ForwardWrapper(model.forward)

# Modify GRPOTrainer to capture current batch information
class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_batch_info = None
        
    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        return result
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Store current batch information for the callback
        if hasattr(inputs, 'questions'):
            self.current_batch_info = {
                'questions': inputs.questions,
                'true_answers': inputs.answers if hasattr(inputs, 'answers') else None,
                'model_outputs': None  # Will be populated after generation
            }
            
        # Call the parent method
        result = super().compute_loss(model, inputs, return_outputs)
        
        # Update with model outputs if available
        if hasattr(self, 'last_outputs') and self.current_batch_info:
            self.current_batch_info['model_outputs'] = self.last_outputs
            
        return result

trainer = CustomGRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=REWARD_FUNCTIONS,  # Use reward functions from config
    args=training_args,
    train_dataset=dataset,
    callbacks=[OutputLoggingCallback()]
)

print("Starting training...")
logger.info("Starting training process")
trainer_output = trainer.train()
logger.info(f"Training completed in {trainer_output.metrics['train_runtime']:.2f} seconds")

# Save the model - both full model and PEFT adapters
# --------------------------------------------------------------------------
output_dir = f"./gemma3_{MODEL_NAME}_thinking_mac_final"
adapter_output_dir = f"./gemma3_{MODEL_NAME}_peft_adapters"

logger.info(f"Saving full model to {output_dir}")
trainer.save_model(output_dir)

# Save PEFT adapters separately for easier loading
logger.info(f"Saving PEFT adapters to {adapter_output_dir}")
trainer.model.save_pretrained(adapter_output_dir)

# Log training metrics
logger.info("Training metrics:")
for key, value in trainer_output.metrics.items():
    logger.info(f"{key}: {value}")

# Track final memory usage on MPS
if is_mps_available:
    try:
        final_memory = torch.mps.current_allocated_memory() / (1024 * 1024)
        print(f"Final MPS memory usage: {final_memory:.2f} MB")
    except Exception as e:
        print("Unable to track MPS memory usage")

# Test the trained model
# --------------------------------------------------------------------------
print("Testing the trained model...")

# Create a test pipeline that works with PEFT models
def generate_text(model, tokenizer, prompt, max_new_tokens=1024):
    # Prepare the prompt
    input_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

question = "The school principal decided that she wanted every class to have an equal number of boys and girls in each first-grade classroom. There are 4 classrooms. There are 56 boys and 44 girls. How many total students are in each classroom?"

output = generate_text(trainer.model, processor, question)

print("\nTest output:")
print(output)

print("\nTraining complete!")
