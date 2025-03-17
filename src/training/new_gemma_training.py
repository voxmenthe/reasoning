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
import logging
from transformers import Gemma3ForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

# Import our centralized logging system
from src.logging import (
    initialize as init_logging,
    get_training_logger,
    get_model_logger,
    get_reward_logger,
    get_metrics_logger,
    log_model_output,
    log_reward,
    log_training_progress,
    log_memory_usage,
    log_reward_metrics,
    log_generation_metrics
)

# Import our enhanced callback and wrapped rewards
from src.training.enhanced_logging_callback import EnhancedLoggingCallback
from src.training.wrapped_rewards import get_wrapped_reward_functions

# Initialize the logging system with Gemma-specific configuration
init_logging("src/logging/config/gemma_logging_config.yaml")
logger = get_training_logger()

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
logger.info(f"MPS (Apple Silicon GPU) available: {is_mps_available}")

# Set device and data type
if is_mps_available:
    device = torch.device("mps")
    logger.info("Using MPS device for training")
    
    # Track initial memory usage on MPS
    initial_memory = torch.mps.current_allocated_memory() / (1024 * 1024)
    logger.info(f"Initial MPS memory usage: {initial_memory:.2f} MB")
    log_memory_usage()  # Call without arguments
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device for training")

# Set dtype - bfloat16 is supported on Apple Silicon M2+ chips
torch_dtype = torch.bfloat16

# Model and tokenizer initialization
# --------------------------------------------------------------------------
MODEL_NAME = "google/gemma-3-4b-it" # "google/gemma-3-1b-it",

# Load base model
logger.info("Loading base model...")
# First load the model on CPU without any dtype or device specification
model = Gemma3ForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    attn_implementation="eager"  # Explicitly set eager attention during loading
)

# Then set dtype and move to appropriate device
logger.info(f"Converting model to {torch_dtype} and moving to device...")
model = model.to(device=device, dtype=torch_dtype)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Apply LoRA adapters for efficient fine-tuning
logger.info("Applying LoRA adapters...")
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    modules_to_save=["lm_head", "embed_tokens"]
)
model = get_peft_model(model, peft_config)
logger.info(model.print_trainable_parameters())

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos=True)
tokenizer.padding_side = 'right'

# Get dataset
dataset = get_gsm8k_questions()

# Set batch size based on device
if is_mps_available:
    per_device_batch = 2
    grad_accum = 4
else:
    per_device_batch = 4
    grad_accum = 4
    
dataset_size = len(dataset)

# Define training parameters
num_epochs = 3  # Define number of epochs upfront
effective_batch_size = per_device_batch * grad_accum
steps_per_epoch = dataset_size // effective_batch_size
total_steps = steps_per_epoch * num_epochs

logger.info("\nTraining configuration:")
logger.info(f"- Number of epochs: {num_epochs}")
logger.info(f"- Effective batch size: {effective_batch_size}")
logger.info(f"- Steps per epoch: {steps_per_epoch}")
logger.info(f"- Total steps: {total_steps}")

# Configure sequence lengths
max_prompt_length = 1024
max_seq_length = 2048

# Configure training arguments with MPS-compatible optimizer
logger.info("Configuring training arguments...")
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
    report_to="tensorboard",  # Changed from "none" to enable TensorBoard logging
    cache_implementation="hybrid",
    
    # Ensure correct gradient computation
    gradient_checkpointing=True,  # Enable gradient checkpointing in trainer arguments
    
    # Output directory
    output_dir=f"./gemma3_{MODEL_NAME}_thinking_mac"
)

# Make sure we're not using the cache during training to ensure gradient flow
# (Set this on the model instead of in config)
model.config.use_cache = False

logger.info(f"- Total steps for {training_args.num_train_epochs} epochs: {training_args.max_steps}")

# Initialize and start training
# --------------------------------------------------------------------------
logger.info("Initializing GRPO trainer...")
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

# Import reward functions configuration
from reward_config import REWARD_FUNCTIONS

# Get wrapped reward functions for logging
wrapped_reward_functions = get_wrapped_reward_functions(REWARD_FUNCTIONS)

# Initialize GRPO trainer with our enhanced logging callback
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_functions=wrapped_reward_functions,  # Use wrapped reward functions
    callbacks=[EnhancedLoggingCallback()]  # Use our enhanced callback
)

# Start training
logger.info("Starting training...")
trainer.train()

# Save model
logger.info("Saving model...")
trainer.save_model(f"./gemma3_{MODEL_NAME}_thinking_mac_final")

# Log final memory usage
if is_mps_available:
    try:
        log_memory_usage()
    except Exception as e:
        logger.warning("Unable to track MPS memory usage")

logger.info("Training completed successfully!")
