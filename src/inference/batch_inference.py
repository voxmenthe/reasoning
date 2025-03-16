#!/usr/bin/env python
# coding: utf-8

"""
Batch inference script for the fine-tuned Gemma 3 1b model with PEFT adapters
Loads saved model adapters and runs inference on multiple examples from a file
"""

import os
import torch
import argparse
import json
import csv
from tqdm import tqdm
from datetime import datetime
from transformers import Gemma3ForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from peft import PeftModel, PeftConfig

def read_questions(file_path):
    """Read questions from a file.
    Supports JSON, CSV, and plain text formats.
    """
    questions = []
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    questions = data
                elif all(isinstance(item, dict) and 'question' in item for item in data):
                    questions = [item['question'] for item in data]
            elif isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
    elif ext == '.csv':
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            question_idx = 0
            if 'question' in header:
                question_idx = header.index('question')
            questions = [row[question_idx] for row in reader]
    else:  # Assume plain text, one question per line
        with open(file_path, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
    
    return questions

# Parse arguments
parser = argparse.ArgumentParser(description="Run batch inference with fine-tuned Gemma 3 1b model")
parser.add_argument("--model_path", type=str, default="./gemma3_1b_peft_adapters",
                   help="Path to the saved PEFT adapters directory")
parser.add_argument("--base_model", type=str, default="google/gemma-3-1b-it",
                   help="Base model name or path")
parser.add_argument("--input_file", type=str, required=True,
                   help="File containing questions (JSON, CSV, or text)")
parser.add_argument("--max_new_tokens", type=int, default=256,
                   help="Maximum number of new tokens to generate")
parser.add_argument("--output_dir", type=str, default="./inference_results",
                   help="Directory to save inference outputs")
parser.add_argument("--temperature", type=float, default=0.7,
                   help="Temperature for sampling")
parser.add_argument("--top_p", type=float, default=0.9,
                   help="Top-p sampling parameter")
parser.add_argument("--do_sample", action="store_true",
                   help="Whether to use sampling; if False, uses greedy decoding")
parser.add_argument("--append_reasoning", action="store_true",
                   help="Whether to append <reasoning> tag to the input")
parser.add_argument("--batch_size", type=int, default=1,
                   help="Number of questions to process in parallel (if supported)")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Create a timestamped output file for combined results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
combined_output_path = os.path.join(args.output_dir, f"batch_results_{timestamp}.jsonl")

# Check for MPS (Apple Silicon) availability
is_mps_available = torch.backends.mps.is_available()
if is_mps_available:
    device = torch.device("mps")
    print("Using MPS device for inference")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for inference")

# Set dtype
torch_dtype = torch.bfloat16

# Load the base model
print(f"Loading base model: {args.base_model}")
model = Gemma3ForCausalLM.from_pretrained(
    args.base_model,
    device_map=device if device.type != "mps" else "auto",
    attn_implementation="sdpa",  # Changed from eager to sdpa for faster inference
    torch_dtype=torch_dtype,
)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.base_model, add_bos=True)
tokenizer.padding_side = 'right'

# Add chat template from the instruction-tuned model
tokenizer.chat_template = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", add_bos=True).chat_template

# Load PEFT adapters
print(f"Loading PEFT adapters from: {args.model_path}")
model = PeftModel.from_pretrained(model, args.model_path)

# Set up generation pipeline
print("Setting up generation pipeline...")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Create generation config for faster generation
generation_config = GenerationConfig(
    max_new_tokens=args.max_new_tokens,
    do_sample=args.do_sample,
    temperature=args.temperature if args.do_sample else 1.0,
    top_p=args.top_p if args.do_sample else 1.0,
    num_beams=1,  # Greedy decoding is faster
    early_stopping=True
)

# System prompt
system_message = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

# Read questions
print(f"Reading questions from {args.input_file}")
questions = read_questions(args.input_file)
print(f"Found {len(questions)} questions")

# Process each question
print("Starting batch inference...")
with open(combined_output_path, 'w') as combined_file:
    for i, question in enumerate(tqdm(questions, desc="Processing questions")):
        # Format the input
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        # Apply chat template with tokenize=False to get string instead of token IDs
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Optionally append <reasoning> tag to help guide the model's output format
        if args.append_reasoning:
            input_text = input_text + "<reasoning>"
        
        # Generate output
        output = generator(
            input_text,
            generation_config=generation_config
        )
        
        # Extract the model's specific output
        generated_text = output[0]['generated_text']
        model_output = generated_text[len(input_text):]
        
        # Save individual result
        result = {
            "id": i,
            "question": question,
            "full_response": generated_text,
            "model_output": model_output
        }
        
        # Write to the combined file
        combined_file.write(json.dumps(result) + '\n')
        
        # Also save individual files if needed
        individual_output_path = os.path.join(args.output_dir, f"question_{i}.txt")
        with open(individual_output_path, 'w') as f:
            f.write(f"Question: {question}\n\n")
            f.write(f"System message: {system_message}\n\n")
            f.write(f"Complete response:\n{generated_text}\n\n")
            f.write(f"Model's output only:\n{model_output}\n")

print(f"Batch inference completed! Results saved to {args.output_dir}")
print(f"Combined results file: {combined_output_path}")

# Run this script with:
# python batch_inference.py --input_file questions.json --output_dir ./results --append_reasoning 