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

import os
import torch
import warnings
import logging
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

# Import reward functions configuration
from reward_config import REWARD_FUNCTIONS

# Set up logging with less frequent file updates
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_process.log", mode='a'),
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

# Constants and configuration
# --------------------------------------------------------------------------
MODEL_NAME = "google/gemma-3-4b-it"  # "google/gemma-3-1b-it"
DATASET_CACHE_PATH = "./dataset_cache"
CHECKPOINT_DIR = "./checkpoints"
TENSORBOARD_DIR = "./tensorboard_logs"
CSV_OUTPUT_DIR = "./training_outputs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# Background thread for saving CSV data
# --------------------------------------------------------------------------
class CSVSaverThread(threading.Thread):
    def __init__(self, save_path, flush_interval=10):
        """
        Initialize the CSV saver thread
        
        Args:
            save_path: Path where CSV will be saved
            flush_interval: How often to check for new data (seconds)
        """
        super().__init__(daemon=True)
        self.save_path = save_path
        self.flush_interval = flush_interval
        self.data_queue = queue.Queue()
        self.running = True
        
    def add_data(self, data_list):
        """Add data to be saved in background"""
        if data_list:
            self.data_queue.put(data_list)
    
    def run(self):
        """Main thread execution loop"""
        while self.running:
            try:
                # Check if there's data to save, with timeout
                try:
                    data_to_save = self.data_queue.get(timeout=self.flush_interval)
                    self._save_to_csv(data_to_save)
                    self.data_queue.task_done()
                except queue.Empty:
                    # No data to save, continue waiting
                    pass
                    
            except Exception as e:
                logger.error(f"Error in CSV saver thread: {str(e)}")
                # Continue running despite errors
                
    def stop(self):
        """Stop the thread and save any remaining data"""
        self.running = False
        
        # Save any remaining items in the queue
        remaining_items = []
        while not self.data_queue.empty():
            remaining_items.extend(self.data_queue.get())
            self.data_queue.task_done()
            
        if remaining_items:
            self._save_to_csv(remaining_items)
            
    def _save_to_csv(self, data_list):
        """Save data to CSV file"""
        if not data_list:
            return
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            
            # If file exists, append without headers
            if os.path.exists(self.save_path) and os.path.getsize(self.save_path) > 0:
                df.to_csv(self.save_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.save_path, index=False)
                
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")

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
        print(f"Loading cached dataset from {DATASET_CACHE_PATH}")
        return load_from_disk(DATASET_CACHE_PATH)
    
    print("Building and caching dataset...")
    
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
    
    print(f"Dataset cached to {DATASET_CACHE_PATH}")
    return dataset

# Model and tokenizer initialization
# --------------------------------------------------------------------------
def load_compiled_model():
    """
    Load and compile the model with torch.compile() for better performance
    
    Returns:
        Compiled model with LoRA adapters
    """
    print("Loading base model...")
    # Load model without dtype to avoid compatibility issues
    model = Gemma3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        attn_implementation="eager"
    )
    
    # Convert to dtype and move to device
    print(f"Converting model to {torch_dtype} and moving to {device}...")
    model = model.to(device=device, dtype=torch_dtype)
    
    # Enable gradient checkpointing for memory efficiency before compilation
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    
    # Configure LoRA
    print("Configuring LoRA adapters...")
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
    print(model.print_trainable_parameters())
    
    # Disable cache during training to ensure gradient flow
    model.config.use_cache = False
    
    # Compile the model with torch.compile() for better performance
    # Use mode='reduce-overhead' which works well with training
    print("Compiling model with torch.compile()...")
    # Only apply torch.compile when not on MPS as it might not be fully supported
    if device.type != 'mps':
        model = torch.compile(model, mode='reduce-overhead')
    
    return model

# Function to extract reasoning and answer
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

# Optimized trainer with checkpointing
# --------------------------------------------------------------------------
class OptimizedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        # Extract checkpoint info before passing to parent
        self.checkpoint_dir = kwargs.pop('checkpoint_dir', None)
        self.checkpoint_steps = kwargs.pop('checkpoint_steps', 100)
        
        super().__init__(*args, **kwargs)
        self.current_batch_info = None
        
        # Attempt to resume from checkpoint if it exists
        self.try_resume_from_checkpoint()
        
    def try_resume_from_checkpoint(self):
        """Try to resume training from the latest checkpoint"""
        if not self.checkpoint_dir:
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
            
            # Load model weights
            logger.info(f"Resuming from checkpoint {latest_checkpoint}")
            self.model.load_state_dict(
                torch.load(latest_checkpoint / "pytorch_model.bin", map_location=device)
            )
            
            # Load optimizer state
            optimizer_path = latest_checkpoint / "optimizer.pt"
            if optimizer_path.exists() and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=device)
                )
                
            logger.info(f"Successfully resumed from step {self.state.global_step}")
            
        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {str(e)}")
            # Continue without resuming if there's an error
        
    def save_checkpoint(self):
        """Save a checkpoint that can be resumed from"""
        if not self.checkpoint_dir:
            return
            
        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint-{self.state.global_step}"
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        
        # Save optimizer state
        if hasattr(self, 'optimizer'):
            torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
            
        # Save trainer state
        with open(checkpoint_path / "trainer_state.json", 'w') as f:
            json.dump({
                'global_step': self.state.global_step,
                'epoch': self.state.epoch,
                'best_metric': getattr(self.state, 'best_metric', None),
            }, f)
            
        logger.info(f"Saved checkpoint at step {self.state.global_step}")
        
    def training_step(self, *args, **kwargs):
        """Override to add checkpoint saving"""
        result = super().training_step(*args, **kwargs)
        
        # Save checkpoint at specified intervals
        if self.checkpoint_dir and self.state.global_step % self.checkpoint_steps == 0:
            self.save_checkpoint()
            
        return result
        
    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        
        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint()
            
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

# Optimized callback with reduced frequency and background processing
# --------------------------------------------------------------------------
class OptimizedOutputCallback(TrainerCallback):
    def __init__(self, checkpoint_dir=None):
        self.log_counter = 0
        self.total_answers = 0
        self.correct_answers = 0
        
        # Set up CSV tracking with background thread
        self.csv_path = os.path.join(CSV_OUTPUT_DIR, "training_outputs.csv")
        self.csv_save_frequency = 50  # Save every 50 steps
        self.training_data_buffer = []
        
        # Create background thread for CSV saving
        self.csv_thread = CSVSaverThread(self.csv_path)
        self.csv_thread.start()
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['question', 'answer', 'llm_reasoning', 'llm_answer'])
                
        # For checkpointing (if provided)
        self.checkpoint_dir = checkpoint_dir
    
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
        
        # Process training outputs with REDUCED FREQUENCY (every 10 steps)
        if hasattr(kwargs.get('trainer', None), 'current_batch_info') and self.log_counter % 10 == 0:
            try:
                batch_info = kwargs['trainer'].current_batch_info
                
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
                        
                        # Store data in buffer
                        self.training_data_buffer.append({
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
            
        # Print lightweight progress to stdout every 5 steps
        if self.log_counter % 5 == 0:
            progress_pct = (state.global_step / args.max_steps) * 100
            print(f"\rTraining progress: {state.global_step}/{args.max_steps} steps ({progress_pct:.1f}%) | Correct: {self.correct_answers}/{self.total_answers} ({accuracy:.1f}%)", end="", flush=True)
        
        # Save CSV periodically by sending to background thread
        if self.log_counter % self.csv_save_frequency == 0 and self.training_data_buffer:
            self.csv_thread.add_data(self.training_data_buffer)
            self.training_data_buffer = []  # Clear buffer after sending to thread
        
        # Log metrics less frequently (every 20 steps)
        if self.log_counter % 20 == 0 and logs:
            if 'loss' in logs:
                logger.info(f"Step {state.global_step}: Loss: {logs['loss']:.6f}")
            if 'rewards/correctness_reward_func' in logs:
                logger.info(f"Step {state.global_step}: Correctness reward: {logs['rewards/correctness_reward_func']:.6f}")
            logger.info(f"Step {state.global_step}: Accuracy so far: {self.correct_answers}/{self.total_answers} ({accuracy:.2f}%)")
        
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # Print final progress and add a newline
        accuracy = 0.0
        if self.total_answers > 0:
            accuracy = (self.correct_answers / self.total_answers) * 100
            
        print(f"\rTraining progress: {state.global_step}/{args.max_steps} steps (100.0%) | Correct: {self.correct_answers}/{self.total_answers} ({accuracy:.1f}%)")
        print("\nTraining completed!")
        
        # Save any remaining data and stop the thread
        if self.training_data_buffer:
            self.csv_thread.add_data(self.training_data_buffer)
        self.csv_thread.stop()
        self.csv_thread.join(timeout=10)  # Wait up to 10 seconds for thread to finish
        
        logger.info(f"Final accuracy: {self.correct_answers}/{self.total_answers} ({accuracy:.2f}%)")
        logger.info("Training completed")
        return control

# Main training function
# --------------------------------------------------------------------------
def train_model():
    """Main function to train the model with all optimizations"""
    # Prepare cached dataset
    dataset = prepare_cached_dataset(force_rebuild=False)
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} examples")
    
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
    
    print(f"\nTraining configuration:")
    print(f"- Number of epochs: {num_epochs}")
    print(f"- Effective batch size: {effective_batch_size}")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Total steps: {total_steps}")
    
    # Configure sequence lengths
    max_prompt_length = 1024
    max_seq_length = 2048
    
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
        
        # Replace "hybrid" cache with "memory" for better performance
        cache_implementation="memory",
        
        # Enable gradient checkpointing in trainer arguments
        gradient_checkpointing=True,
        
        # Output directory
        output_dir=f"./optimized_gemma3_{MODEL_NAME.split('/')[-1]}"
    )
    
    # Initialize optimized trainer with checkpointing
    print("Initializing optimized GRPO trainer...")
    trainer = OptimizedGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=REWARD_FUNCTIONS,
        args=training_args,
        train_dataset=dataset,
        callbacks=[OptimizedOutputCallback()],
        
        # Add checkpoint directory for resumable training
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_steps=100,
    )
    
    # Start training
    print("Starting training...")
    trainer_output = trainer.train()
    print(f"Training completed in {trainer_output.metrics['train_runtime']:.2f} seconds")
    
    # Save model
    output_dir = f"./optimized_gemma3_{MODEL_NAME.split('/')[-1]}_final"
    adapter_output_dir = f"./optimized_gemma3_{MODEL_NAME.split('/')[-1]}_peft_adapters"
    
    print(f"Saving full model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Save PEFT adapters separately for easier loading
    print(f"Saving PEFT adapters to {adapter_output_dir}")
    trainer.model.save_pretrained(adapter_output_dir)
    
    # Track final memory usage on MPS
    if is_mps_available:
        try:
            final_memory = torch.mps.current_allocated_memory() / (1024 * 1024)
            print(f"Final MPS memory usage: {final_memory:.2f} MB")
        except Exception as e:
            print("Unable to track MPS memory usage")
            
    return trainer, processor

# Test function
# --------------------------------------------------------------------------
def test_model(model, tokenizer, questions=None):
    """Test the trained model on sample questions"""
    if questions is None:
        questions = [
            "The school principal decided that she wanted every class to have an equal number of boys and girls in each first-grade classroom. There are 4 classrooms. There are 56 boys and 44 girls. How many total students are in each classroom?"
        ]
    
    print("\nTesting trained model:")
    
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
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        output = generate_text(question)
        print("\nModel output:")
        print(output)
        
        # Extract reasoning and answer
        reasoning, answer = extract_reasoning_and_answer(output)
        print("\nExtracted reasoning:", reasoning)
        print("Extracted answer:", answer)
    
    print("\nTesting complete!")

# Main execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Train the model
    trainer, processor = train_model()
    
    # Test the model
    test_model(trainer.model, processor)
    
    print("\nTraining and testing completed successfully!") 