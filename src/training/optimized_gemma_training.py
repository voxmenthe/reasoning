#!/usr/bin/env python
# coding: utf-8

# # Optimized Gemma 3 Training Script
# 
# This script is a highly optimized version of the original Gemma3 training script
# with the following key improvements:
# - Using torch.compile() instead of ForwardWrapper for model compilation
# - Optimized gradient accumulation with fewer logging operations
# - Pre-tokenized and cached dataset for faster loading
# - Reduced callback frequency with background threading for CSV saving
# - Memory-efficient caching implementation
# - Efficient checkpointing for resuming interrupted training
# - Integrated centralized logging system

import os
from src.training.rewards import xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, int_reward_func, correctness_reward_func, anti_repetition_reward_func, topic_relevance_reward_func
import torch
import warnings
import threading
import queue
import time
from transformers import Gemma3ForCausalLM, AutoTokenizer, GenerationConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
import re
import csv
import json
from pathlib import Path
import numpy as np
from datasets import Dataset, load_from_disk
import functools
import datetime

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
    log_generation_metrics,
    log_generation_with_rewards
)

# Import our custom callback
from src.training.optimized_logging_callback import OptimizedLoggingCallback
from src.training.wrapped_rewards import get_wrapped_reward_functions

# reward functions configuration
REWARD_FUNCTIONS = [
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    anti_repetition_reward_func,
    topic_relevance_reward_func,
] 

# Initialize the logging system with Gemma-specific configuration
init_logging("src/logging/config/gemma_logging_config.yaml")
logger = get_training_logger()
system_logger = get_metrics_logger()

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

# Constants and configuration
# --------------------------------------------------------------------------
MODEL_NAME = "google/gemma-3-4b-it"  # "google/gemma-3-1b-it" # 
DATASET_CACHE_PATH = "./dataset_cache"
CHECKPOINT_DIR = "./checkpoints"
TENSORBOARD_DIR = "./logs/gemma_tensorboard"  # Updated to match logging config
CSV_OUTPUT_DIR = "./logs"  # Updated to match logging config

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# Extract reasoning and answer (moved from callback to function section)
# --------------------------------------------------------------------------
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

# Dataset preparation with caching
# --------------------------------------------------------------------------
def prepare_cached_dataset(force_rebuild=False):
    """
    Prepare and cache the dataset to disk for faster loading
    
    Args:
        force_rebuild: If True, rebuild the cached dataset even if it exists
        
    Returns:
        Cached dataset ready for training
    """
    # Check if cached dataset exists
    if not force_rebuild and os.path.exists(DATASET_CACHE_PATH):
        logger.info(f"Loading cached dataset from {DATASET_CACHE_PATH}")
        return load_from_disk(DATASET_CACHE_PATH)
    
    logger.info("Building and caching dataset...")
    
    # Get raw dataset
    raw_dataset = get_gsm8k_questions()
    
    # Load tokenizer for pre-tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos=True)
    tokenizer.padding_side = 'right'
    tokenizer.chat_template = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos=True).chat_template
    
    # Extract key fields needed for training
    processed_data = []
    
    for item in raw_dataset:
        question = item['question']
        answer = item['answer']
        
        # Format as a chat message
        chat_message = [{"role": "user", "content": question}]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(chat_message, tokenize=False)
        
        # Store processed sample
        processed_data.append({
            'prompt': prompt,
            'question': question,  # Store original for reference
            'answer': answer,
            'tokenized_prompt': tokenizer(prompt, truncation=True, max_length=1024).input_ids,
        })
    
    # Convert to HF Dataset format
    dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
    
    # Save to disk
    dataset.save_to_disk(DATASET_CACHE_PATH)
    
    logger.info(f"Dataset cached to {DATASET_CACHE_PATH}")
    return dataset

# Model and tokenizer initialization
# --------------------------------------------------------------------------
def load_compiled_model():
    """
    Load and compile the model with torch.compile() for better performance
    
    Returns:
        Compiled model with LoRA adapters
    """
    logger.info("Loading base model...")
    # Load model without dtype to avoid compatibility issues
    model = Gemma3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        attn_implementation="eager"
    )
    
    # Convert to dtype and move to device
    logger.info(f"Converting model to {torch_dtype} and moving to {device}...")
    model = model.to(device=device, dtype=torch_dtype)
    
    # Enable gradient checkpointing for memory efficiency before compilation
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    
    # Configure LoRA
    logger.info("Configuring LoRA adapters...")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )
    
    # Apply LoRA adapters
    model = get_peft_model(model, peft_config)
    logger.info(model.print_trainable_parameters())
    
    # Disable cache during training to ensure gradient flow
    model.config.use_cache = False
    
    # Compile the model with torch.compile() for better performance
    # Use mode='reduce-overhead' which works well with training
    logger.info("Compiling model with torch.compile()...")
    # Only apply torch.compile when not on MPS as it might not be fully supported
    if device.type != 'mps':
        model = torch.compile(model, mode='reduce-overhead')
    
    return model

# Optimized trainer with checkpointing
# --------------------------------------------------------------------------
class OptimizedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        # Extract checkpoint info before passing to parent
        self.checkpoint_dir = kwargs.pop('checkpoint_dir', None)
        self.checkpoint_steps = kwargs.pop('checkpoint_steps', 100)
        self.should_resume = kwargs.pop('should_resume', True)
        
        super().__init__(*args, **kwargs)
        self.current_batch_info = None
        
        # Don't resume in __init__ - optimizer isn't ready yet
        # We'll resume in train() method instead
        
    def try_resume_from_checkpoint(self):
        """Try to resume training from the latest checkpoint"""
        if not self.checkpoint_dir or not self.should_resume:
            return
            
        # Find the latest checkpoint
        checkpoint_paths = list(Path(self.checkpoint_dir).glob("checkpoint-*"))
        if not checkpoint_paths:
            logger.info("No checkpoints found to resume from")
            return
            
        # Sort by checkpoint step number
        checkpoint_paths.sort(key=lambda path: int(path.name.split("-")[1]))
        latest_checkpoint = checkpoint_paths[-1]
        
        try:
            # Load checkpoint state
            checkpoint_file = latest_checkpoint / "trainer_state.json"
            if not checkpoint_file.exists():
                logger.warning(f"No trainer state found in {latest_checkpoint}")
                return
                
            with open(checkpoint_file, 'r') as f:
                state = json.load(f)
                
            # Set trainer state
            self.state.global_step = state.get('global_step', 0)
            self.state.epoch = state.get('epoch', 0)
            
            # Load model weights - properly handled for PEFT models
            logger.info(f"Resuming from checkpoint {latest_checkpoint}")
            if hasattr(self.model, 'is_peft_model') and self.model.is_peft_model:
                # For PEFT/LoRA models, load adapters directly
                if os.path.exists(latest_checkpoint):
                    logger.info(f"Loading PEFT adapters from {latest_checkpoint}")
                    self.model = PeftModel.from_pretrained(
                        self.model.get_base_model(),
                        latest_checkpoint,
                        is_trainable=True,
                        device_map={"": device}
                    )
                else:
                    logger.warning(f"PEFT adapter checkpoint not found at {latest_checkpoint}")
            else:
                # For non-PEFT models, use regular loading
                model_path = latest_checkpoint / "pytorch_model.bin"
                if model_path.exists():
                    self.model.load_state_dict(torch.load(model_path, map_location=device))
                else:
                    logger.warning(f"Model weights not found at {model_path}")
                    
            # Load optimizer state - only if optimizer has been initialized
            optimizer_path = latest_checkpoint / "optimizer.pt"
            if optimizer_path.exists() and hasattr(self, 'optimizer') and self.optimizer is not None:
                try:
                    logger.info(f"Loading optimizer state from {optimizer_path}")
                    self.optimizer.load_state_dict(
                        torch.load(optimizer_path, map_location=device)
                    )
                except Exception as opt_error:
                    logger.warning(f"Error loading optimizer state: {str(opt_error)}")
                    logger.warning("Continuing without loading optimizer state")
                
            logger.info(f"Successfully resumed from step {self.state.global_step}")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {str(e)}")
            logger.exception("Detailed traceback:")
            # Continue without resuming if there's an error
            return False
        
    def save_checkpoint(self):
        """Save a checkpoint that can be resumed from"""
        if not self.checkpoint_dir:
            return
            
        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint-{self.state.global_step}"
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        
        # Save model - different handling for PEFT models
        try:
            if hasattr(self.model, 'is_peft_model') and self.model.is_peft_model:
                # For PEFT models, save only the adapters
                logger.info(f"Saving PEFT adapters to {checkpoint_path}")
                self.model.save_pretrained(checkpoint_path)
            else:
                # For regular models use save_pretrained or save model state dict directly
                self.model.save_pretrained(checkpoint_path)
                
            # Save optimizer state - only if initialized
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                try:
                    torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
                except Exception as opt_error:
                    logger.warning(f"Error saving optimizer state: {str(opt_error)}")
                
            # Save trainer state
            with open(checkpoint_path / "trainer_state.json", 'w') as f:
                json.dump({
                    'global_step': self.state.global_step,
                    'epoch': self.state.epoch,
                    'best_metric': getattr(self.state, 'best_metric', None),
                }, f)
                
            logger.info(f"Saved checkpoint at step {self.state.global_step}")
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            logger.exception("Detailed traceback:")
            return False
        
    def training_step(self, *args, **kwargs):
        """Override to add checkpoint saving"""
        result = super().training_step(*args, **kwargs)
        
        # Save checkpoint at specified intervals
        if self.checkpoint_dir and self.state.global_step % self.checkpoint_steps == 0:
            self.save_checkpoint()
            
        return result
        
    def train(self, *args, **kwargs):
        # Now try to resume from checkpoint - after optimizer is created by parent class
        if self.should_resume:
            # For GRPOTrainer, optimizer should be created by this point
            self.try_resume_from_checkpoint()
            
        result = super().train(*args, **kwargs)
        
        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint()
            
        return result
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Store current batch information for the callback
        if hasattr(inputs, 'questions'):
            self.current_batch_info = {
                'questions': inputs.questions,
                'true_answers': inputs.answers if hasattr(inputs, 'answers') else None,
                'model_outputs': None  # Will be populated after generation
            }
            
        # Call the parent method - omit num_items_in_batch since parent GRPOTrainer doesn't expect it
        result = super().compute_loss(model, inputs, return_outputs)
        
        # Update with model outputs if available
        if hasattr(self, 'last_outputs') and self.current_batch_info:
            self.current_batch_info['model_outputs'] = self.last_outputs
            
        return result

# Main training function
# --------------------------------------------------------------------------
def train_model():
    """Main function to train the model with all optimizations"""
    # Prepare cached dataset
    dataset = prepare_cached_dataset(force_rebuild=False)
    dataset_size = len(dataset)
    logger.info(f"Dataset size: {dataset_size} examples")
    
    # Load and compile model
    model = load_compiled_model()
    
    # Load tokenizer
    processor = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos=True)
    processor.padding_side = 'right'
    processor.chat_template = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos=True).chat_template
    
    # Training parameters
    if is_mps_available:
        per_device_batch = 2
        grad_accum = 4
    else:
        per_device_batch = 4
        grad_accum = 4
        
    # Define training parameters
    num_epochs = 3
    effective_batch_size = per_device_batch * grad_accum
    steps_per_epoch = dataset_size // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    
    logger.info("\nTraining configuration:")
    logger.info(f"- Number of epochs: {num_epochs}")
    logger.info(f"- Effective batch size: {effective_batch_size}")
    logger.info(f"- Steps per epoch: {steps_per_epoch}")
    logger.info(f"- Total steps: {total_steps}")
    
    # Configure sequence lengths
    max_prompt_length = 1160 # 1024
    max_seq_length = 2608 # 2048
    
    # Configure training arguments with MPS-compatible optimizer
    # Replace "hybrid" cache with "memory" for better performance
    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        
        # Use adamw_torch which is MPS-compatible
        optim="adamw_torch",
        
        # Reduce logging frequency for better performance
        logging_steps=20,
        logging_first_step=True,
        log_level="info",
        
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=2,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=num_epochs,
        max_steps=total_steps,
        save_steps=250,
        max_grad_norm=0.1,
        
        # Enable TensorBoard reporting
        report_to="tensorboard",
        
        # Replace "hybrid" cache with something else for better performance?
        cache_implementation="hybrid",
        
        # Enable gradient checkpointing in trainer arguments
        gradient_checkpointing=True,
        
        # Output directory
        output_dir=f"./optimized_gemma3_{MODEL_NAME.split('/')[-1]}"
    )
    
    # Get wrapped reward functions for logging
    wrapped_reward_functions = get_wrapped_reward_functions(REWARD_FUNCTIONS)
    
    # Initialize optimized trainer with checkpointing
    logger.info("Initializing optimized GRPO trainer...")
    trainer = OptimizedGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=wrapped_reward_functions,  # Use wrapped reward functions
        args=training_args,
        train_dataset=dataset,
        callbacks=[OptimizedLoggingCallback(checkpoint_dir=CHECKPOINT_DIR)],  # Use our new callback
        
        # Add checkpoint directory for resumable training
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_steps=100,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer_output = trainer.train()
    logger.info(f"Training completed in {trainer_output.metrics['train_runtime']:.2f} seconds")
    
    # Save model
    output_dir = f"./optimized_gemma3_{MODEL_NAME.split('/')[-1]}_final"
    adapter_output_dir = f"./optimized_gemma3_{MODEL_NAME.split('/')[-1]}_peft_adapters"
    
    logger.info(f"Saving full model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Save PEFT adapters separately for easier loading
    logger.info(f"Saving PEFT adapters to {adapter_output_dir}")
    trainer.model.save_pretrained(adapter_output_dir)
    
    # Track final memory usage on MPS
    if is_mps_available:
        try:
            log_memory_usage()
        except Exception as e:
            logger.warning("Unable to track MPS memory usage")
            
    return trainer, processor

# Test function
# --------------------------------------------------------------------------
def test_model(model, tokenizer, questions=None):
    """Test the trained model on sample questions"""
    if questions is None:
        questions = [
            "The school principal decided that she wanted every class to have an equal number of boys and girls in each first-grade classroom. There are 4 classrooms. There are 56 boys and 44 girls. How many total students are in each classroom?"
        ]
    
    logger.info("\nTesting trained model:")
    
    # Generate text function
    def generate_text(prompt, max_new_tokens=1024):
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
    
    # Test on each question
    test_results = []
    for i, question in enumerate(questions):
        logger.info(f"\nQuestion {i+1}: {question}")
        output = generate_text(question)
        logger.info("\nModel output:")
        logger.info(output)
        
        # Extract reasoning and answer
        reasoning, answer = extract_reasoning_and_answer(output)
        logger.info("\nExtracted reasoning: " + reasoning)
        logger.info("Extracted answer: " + answer)
        
        # Evaluate with reward functions to log values
        from src.training.rewards import (
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
            anti_repetition_reward_func,
            topic_relevance_reward_func
        )
        
        # Extract test answer (for correctness testing)
        if i == 0:  # First sample hardcoded answer
            true_answer = "25"
        else:
            true_answer = ""
            
        # Create unique question ID with source identifier
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        question_id = f"test_run_{run_timestamp}_sample_{i}"
        
        # Calculate rewards
        rewards = {}
        rewards["xmlcount_reward_func"] = xmlcount_reward_func([output])[0]
        rewards["soft_format_reward_func"] = soft_format_reward_func([output])[0]
        rewards["strict_format_reward_func"] = strict_format_reward_func([output])[0]
        rewards["int_reward_func"] = int_reward_func([output])[0]
        
        # Only calculate correctness if we have a true answer
        if true_answer:
            rewards["correctness_reward_func"] = correctness_reward_func([question], [output], [true_answer])[0]
        
        rewards["anti_repetition_reward_func"] = anti_repetition_reward_func([output])[0]
        rewards["topic_relevance_reward_func"] = topic_relevance_reward_func([question], [output])[0]
        
        # Log to CSV with rewards
        log_generation_with_rewards(
            question=question,
            true_answer=true_answer,
            model_output=output,
            rewards=rewards,
            reasoning=reasoning,
            answer=answer,
            step=0,  # No training step for test samples
            question_id=question_id
        )
        
        # Store results
        test_results.append({
            "question": question,
            "model_output": output,
            "reasoning": reasoning,
            "answer": answer,
            "rewards": rewards
        })
        
        # Log to the model logger
        log_model_output(
            question=question,
            true_answer=true_answer,
            model_output=output,
            reasoning=reasoning,
            answer=answer,
            question_id=question_id
        )
    
    logger.info("\nTesting complete!")
    return test_results

# Main execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Train the model
        trainer, processor = train_model()
        
        # Test the model
        test_model(trainer.model, processor)
        
        logger.info("\nTraining and testing completed successfully!")
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        raise 