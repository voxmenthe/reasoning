#!/usr/bin/env python
# coding: utf-8

import os
import torch
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import Gemma3ForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from reasoning_dataset import get_gsm8k_questions

# Import credentials
from creds import all_creds
os.environ["HUGGING_FACE_HUB_TOKEN"] = all_creds['HUGGINGFACE_ACCESS_TOKEN_Gemma']

def setup_model_and_tokenizer(model_name, peft_model_path=None):
    """Setup model and tokenizer with proper configurations."""
    print(f"Loading model: {model_name}")
    
    # Check for MPS (Apple Silicon) availability
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    model = Gemma3ForCausalLM.from_pretrained(
        model_name,
        device_map=device if device.type != "mps" else "auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    
    # Load PEFT adapters if specified
    if peft_model_path:
        print(f"Loading PEFT adapters from: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos=True)
    tokenizer.padding_side = 'right'
    
    return model, tokenizer, device

def generate_answer(model, tokenizer, question, device):
    """Generate an answer for a given question."""
    # Prepare the prompt
    input_text = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_numerical_answer(text):
    """Extract the final numerical answer from the generated text."""
    try:
        # Look for the last number in the text
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])
    except:
        pass
    return None

def evaluate_model(model, tokenizer, dataset, device, model_name="Unknown"):
    """Evaluate model performance on the dataset."""
    results = []
    correct = 0
    total = 0
    
    print(f"\nEvaluating {model_name}...")
    for example in tqdm(dataset[:100]):  # Test on first 100 examples for speed
        question = example['question']
        true_answer = float(example['answer'])
        
        # Generate and get predicted answer
        generated_text = generate_answer(model, tokenizer, question, device)
        predicted_answer = extract_numerical_answer(generated_text)
        
        # Check correctness (allowing for small numerical differences)
        is_correct = False
        if predicted_answer is not None:
            is_correct = abs(predicted_answer - true_answer) < 0.01
        
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        results.append({
            'question': question,
            'true_answer': true_answer,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    return results, accuracy

def generate_report(base_results, tuned_results, base_accuracy, tuned_accuracy):
    """Generate a comparison report between base and tuned models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"model_comparison_report_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'base_model_accuracy': base_accuracy,
        'tuned_model_accuracy': tuned_accuracy,
        'accuracy_improvement': tuned_accuracy - base_accuracy,
        'number_of_examples': len(base_results),
        'example_comparisons': []
    }
    
    # Add detailed comparisons for a few examples
    for i in range(min(5, len(base_results))):
        report['example_comparisons'].append({
            'question': base_results[i]['question'],
            'true_answer': base_results[i]['true_answer'],
            'base_model_output': base_results[i]['generated_text'],
            'base_model_answer': base_results[i]['predicted_answer'],
            'base_model_correct': base_results[i]['is_correct'],
            'tuned_model_output': tuned_results[i]['generated_text'],
            'tuned_model_answer': tuned_results[i]['predicted_answer'],
            'tuned_model_correct': tuned_results[i]['is_correct']
        })
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n=== Model Performance Comparison ===")
    print(f"Base Model Accuracy: {base_accuracy:.2%}")
    print(f"Tuned Model Accuracy: {tuned_accuracy:.2%}")
    print(f"Improvement: {(tuned_accuracy - base_accuracy):.2%}")
    print(f"\nDetailed report saved to: {report_path}")
    
    return report

def main():
    # Configuration
    MODEL_NAME = "google/gemma-3-4b-it"
    PEFT_MODEL_PATH = "./gemma3_google/gemma-3-4b-it_peft_adapters"  # Path to saved PEFT adapters
    
    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = get_gsm8k_questions()
    
    # Test base model
    base_model, base_tokenizer, device = setup_model_and_tokenizer(MODEL_NAME)
    base_results, base_accuracy = evaluate_model(base_model, base_tokenizer, dataset, device, "Base Model")
    
    # Test fine-tuned model
    tuned_model, tuned_tokenizer, device = setup_model_and_tokenizer(MODEL_NAME, PEFT_MODEL_PATH)
    tuned_results, tuned_accuracy = evaluate_model(tuned_model, tuned_tokenizer, dataset, device, "Fine-tuned Model")
    
    # Generate and save report
    report = generate_report(base_results, tuned_results, base_accuracy, tuned_accuracy)

if __name__ == "__main__":
    main() 