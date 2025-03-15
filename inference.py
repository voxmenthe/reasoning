#!/usr/bin/env python
# coding: utf-8

"""
Inference script for the fine-tuned Gemma 3 1b model with PEFT adapters
Loads saved model adapters and runs inference on examples
"""

import os
import torch
import argparse
from transformers import Gemma3ForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from peft import PeftModel, PeftConfig

# Parse arguments
parser = argparse.ArgumentParser(description="Run inference with fine-tuned Gemma 3 1b model")
parser.add_argument("--model_path", type=str, default="./gemma3_1b_peft_adapters",
                   help="Path to the saved PEFT adapters directory")
parser.add_argument("--base_model", type=str, default="google/gemma-3-1b-it",
                   help="Base model name or path")
parser.add_argument("--question", type=str, 
                   default="If you had 12 apples and gave 3 to your friend, then ate 4 yourself, how many would you have left?",
                   help="Question to run inference on")
parser.add_argument("--max_new_tokens", type=int, default=256,
                   help="Maximum number of new tokens to generate")
parser.add_argument("--output_path", type=str, default="inference_output.txt",
                   help="Path to save inference output")
parser.add_argument("--append_reasoning", action="store_true",
                   help="Whether to append <reasoning> tag to the input")
parser.add_argument("--temperature", type=float, default=0.7,
                   help="Temperature for sampling")
parser.add_argument("--do_sample", action="store_true",
                   help="Whether to use sampling; if False, uses greedy decoding")
args = parser.parse_args()

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

# Format the input
print(f"\nRunning inference on question: {args.question}")
system_message = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": args.question}
]

# Apply chat template with tokenize=False to get string instead of token IDs
input_text = tokenizer.apply_chat_template(messages, tokenize=False)

# Optionally append <reasoning> tag to help guide the model's output format
if args.append_reasoning:
    input_text = input_text + "<reasoning>"
    print("Appended <reasoning> tag to input")

# Create generation config for faster generation
generation_config = GenerationConfig(
    max_new_tokens=args.max_new_tokens,
    do_sample=args.do_sample,
    temperature=args.temperature if args.do_sample else 1.0,
    top_p=0.9 if args.do_sample else 1.0,
    num_beams=1,  # Greedy decoding is faster
    early_stopping=True
)

# Generate output
print("Generating response...")
output = generator(
    input_text,
    generation_config=generation_config
)

# Print and save the result
generated_text = output[0]['generated_text']
print("\n--- Generated Response ---")
print(generated_text)

# Extract the model's specific output (not including the prompt)
model_output = generated_text[len(input_text):]
print("\n--- Model's Output Only ---")
print(model_output)

# Save to file
print(f"\nSaving output to {args.output_path}")
with open(args.output_path, "w") as f:
    f.write(f"Question: {args.question}\n\n")
    f.write(f"System message: {system_message}\n\n")
    f.write(f"Complete response:\n{generated_text}\n\n")
    f.write(f"Model's output only:\n{model_output}\n")

print("Inference completed successfully!")

# Run this script with:
# python inference.py --question "Your math question here" --output_path "your_output.txt" --append_reasoning 