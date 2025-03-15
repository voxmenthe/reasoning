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

# Import reward functions from new module
from rewards import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    anti_repetition_reward_func,
    topic_relevance_reward_func
)

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
from reasoning_dataset import (
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
# Load base model
print("Loading base model...")
model = Gemma3ForCausalLM.from_pretrained(
    pretrained_model_name_or_path="google/gemma-3-1b-it",
    device_map=device if device.type != "mps" else "auto",  # handle MPS differently
    attn_implementation="eager",  # Use eager implementation during training for better compatibility
    torch_dtype=torch_dtype
)

# Enable gradient checkpointing for memory efficiency
print("Enabling gradient checkpointing...")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})

# Load and configure tokenizer
print("Loading tokenizer...")
processor = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", add_bos=True)
processor.padding_side = 'right'

# Add chat template from the instruction-tuned model (improvement from Fine_Tuning script)
processor.chat_template = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", add_bos=True).chat_template

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
print(f"Dataset size: {len(dataset)} examples")
print(f"Sample example: {dataset[0]}")

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

# Configure sequence lengths
max_prompt_length = 1024
max_seq_length = 2048

# Create training arguments with MPS-compatible optimizer
print("Configuring training arguments...")
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    
    # Use adamw_torch instead of adamw_8bit for MPS compatibility
    optim="adamw_torch",  # Changed from "adamw_8bit" to work on MPS
    
    logging_steps=1,
    per_device_train_batch_size=per_device_batch,
    gradient_accumulation_steps=grad_accum,
    num_generations=2,  # This needs to evenly divide into the batch size
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    num_train_epochs=1,
    max_steps=5,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    cache_implementation="hybrid",
    
    # Output directory
    output_dir="./gemma3_1b_thinking_mac"
)

# Initialize and start training
# --------------------------------------------------------------------------
print("Initializing GRPO trainer...")
logger.info("Initializing GRPO trainer with the following configuration:")
logger.info(f"Model: google/gemma-3-1b-it")
logger.info(f"Learning rate: {training_args.learning_rate}")
logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
logger.info(f"Number of epochs: {training_args.num_train_epochs}")
logger.info(f"Max steps: {training_args.max_steps}")

# Custom callback to log model outputs during training
class OutputLoggingCallback(TrainerCallback):
    def __init__(self):
        self.log_counter = 0
    
    def on_init_end(self, args, state, control, **kwargs):
        logger.info("Training initialization completed")
        return control
        
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        self.log_counter += 1
        if self.log_counter % 5 == 0 and logs:  # Log every 5 steps
            if 'rewards/correctness_reward_func' in logs:
                logger.info(f"Step {state.global_step}: Correctness reward: {logs['rewards/correctness_reward_func']}")
            if 'rewards/anti_repetition_reward_func' in logs:
                logger.info(f"Step {state.global_step}: Anti-repetition reward: {logs['rewards/anti_repetition_reward_func']}")
            logger.info(f"Step {state.global_step}: Total reward: {logs.get('reward', 'N/A')}")
        
        return control

    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Training completed")
        return control

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        anti_repetition_reward_func,
        topic_relevance_reward_func,  # Add the new reward function
    ],
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
output_dir = "./gemma3_1b_thinking_mac_final"
adapter_output_dir = "./gemma3_1b_peft_adapters"

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
    peak_memory = torch.mps.max_memory_allocated() / (1024 * 1024)
    final_memory = torch.mps.current_allocated_memory() / (1024 * 1024)
    print(f"Peak MPS memory usage: {peak_memory:.2f} MB")
    print(f"Final MPS memory usage: {final_memory:.2f} MB")

# Test the trained model
# --------------------------------------------------------------------------
print("Testing the trained model...")
from transformers import pipeline

question = "The school principal decided that she wanted every class to have an equal number of boys and girls in each first-grade classroom. There are 4 classrooms. There are 56 boys and 44 girls. How many total students are in each classroom?"

generator = pipeline("text-generation", model=trainer.model, tokenizer=processor)
input_text = processor.apply_chat_template([{"role": "user", "content": question}])
output = generator(input_text, max_new_tokens=1024)

print("\nTest output:")
print(output[0]['generated_text'])

print("\nTraining complete!")
