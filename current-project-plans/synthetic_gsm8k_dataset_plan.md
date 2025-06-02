# Synthetic GSM8K Dataset Generation Plan

## Project Objective
Create a synthetic version of the GSM8K dataset that maintains the same mathematical reasoning structure but replaces:
- Numbers
- Person names
- Item/entity names
- Variable names

## Data Flow Overview

1. Load the original GSM8K dataset
2. Process each problem through Gemini API to create:
   - Mathematical equation representation of the problem
   - Synthetic version with new values
3. Verify mathematical correctness using Python-based equation evaluation
4. Implement feedback loop for correction if needed
5. Save the verified synthetic dataset

## Detailed Implementation Steps

### 1. Dataset Loading and Analysis

```python
# Load the original GSM8K dataset using the existing utility
from src.datasets.reasoning_dataset import get_gsm8k_questions

# Load both training and test splits
train_data = get_gsm8k_questions("train")
test_data = get_gsm8k_questions("test")

# Analyze structure to understand patterns
# - Question formats
# - Answer formats
# - Types of mathematical operations involved
```

### 2. Gemini API Integration

```python
# Setup Gemini API client
from google.generativeai import configure, GenerativeModel
import os

# Configure API
configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = GenerativeModel("gemini-1.5-pro")
```

### 3. Prompt Engineering for Mathematical Equation Extraction

First, we'll have Gemini extract a mathematical equation from the original problem:

```
You are tasked with analyzing a mathematical word problem and creating a formal mathematical equation that represents it.

Original problem: {original_problem}
Original solution: {original_solution}
Final answer: {original_answer}

Your response MUST follow this format:
<equation>
[Provide a Python-executable mathematical equation/formula that fully represents the problem. Use clear variable names that match entities in the problem. The equation should be valid Python code that can be executed to get the final answer.]
</equation>
<variable_mapping>
[List each variable in your equation and what it represents in the original problem]
</variable_mapping>
<execution_steps>
[Provide step-by-step Python code showing how to calculate the final answer using your equation]
</execution_steps>
```

### 4. Equation Verification System

Create a module to verify the equation using Python's `exec()` function:

```python
def verify_equation(equation, variable_mapping, execution_steps, expected_answer):
    """
    Verify that the mathematical equation correctly solves the problem.
    
    Args:
        equation: The mathematical equation/formula as Python code
        variable_mapping: Dictionary mapping variables to their meanings
        execution_steps: Python code showing how to execute the equation
        expected_answer: The expected final answer
    
    Returns:
        dict: Verification results including success status and details
    """
    # Create a safe execution environment
    local_vars = {}
    
    try:
        # Execute the equation and calculation steps
        full_code = f"{equation}\n\n{execution_steps}"
        exec(full_code, {"__builtins__": __builtins__}, local_vars)
        
        # Extract the final answer from the execution
        if 'final_answer' in local_vars:
            calculated_answer = local_vars['final_answer']
        else:
            # Try to find the answer in the last line of execution
            last_executed_line = execution_steps.strip().split('\n')[-1]
            if '=' in last_executed_line:
                answer_var = last_executed_line.split('=')[0].strip()
                calculated_answer = local_vars.get(answer_var)
            else:
                raise ValueError("Could not determine final answer variable")
        
        # Convert to float for numeric comparison (with tolerance)
        try:
            numeric_expected = float(expected_answer)
            numeric_calculated = float(calculated_answer)
            is_correct = abs(numeric_expected - numeric_calculated) < 1e-6
        except ValueError:
            # For non-numeric answers, use string comparison
            is_correct = str(calculated_answer).strip() == str(expected_answer).strip()
        
        return {
            "is_valid": is_correct,
            "calculated_answer": calculated_answer,
            "expected_answer": expected_answer,
            "execution_log": full_code,
            "errors": [] if is_correct else ["Answer mismatch"]
        }
        
    except Exception as e:
        # Capture execution errors
        import traceback
        return {
            "is_valid": False,
            "errors": [str(e)],
            "traceback": traceback.format_exc(),
            "execution_log": full_code if 'full_code' in locals() else ""
        }
```

### 5. Synthetic Problem Generation with Equation Validation

After verifying the equation works for the original problem, use it to generate and validate synthetic problems:

```python
def generate_synthetic_problem(original_problem, original_answer, equation, variable_mapping, execution_steps):
    """
    Generate a synthetic problem and verify it using the equation.
    
    Args:
        original_problem: Original GSM8K problem text
        original_answer: Original answer
        equation: Verified mathematical equation
        variable_mapping: Dictionary mapping variables to their meanings
        execution_steps: Python code showing how to execute the equation
    
    Returns:
        dict: Synthetic problem data with verification results
    """
    # Prompt Gemini to create a synthetic version
    prompt = f"""
    You are tasked with creating a variation of a mathematical reasoning problem.
    
    Original problem: {original_problem}
    
    I have extracted this mathematical equation that solves the problem:
    {equation}
    
    With these variable mappings:
    {variable_mapping}
    
    Create a new problem by:
    1. Replacing all numbers with different values
    2. Replacing all person names with different names (maintain diversity)
    3. Replacing item types/entities with different but conceptually similar items
    4. Ensuring the problem can still be solved using the SAME mathematical equation
    
    Your response MUST follow this format:
    <synthetic_problem>
    [Your generated problem here]
    </synthetic_problem>
    <new_variable_values>
    [For each variable in the original equation, provide the new numeric value it should have]
    </new_variable_values>
    <expected_answer>
    [The final numeric answer based on the new values]
    </expected_answer>
    """
    
    # Get response from Gemini
    response = model.generate_content(prompt)
    
    # Extract components from response
    synthetic_problem = extract_tag_content(response, "synthetic_problem")
    new_variable_values = extract_tag_content(response, "new_variable_values")
    expected_answer = extract_tag_content(response, "expected_answer")
    
    # Convert new variable values to a dictionary
    variable_value_dict = parse_variable_values(new_variable_values)
    
    # Create modified execution steps with new variable values
    modified_execution = modify_execution_steps(execution_steps, variable_value_dict)
    
    # Verify the synthetic problem using the equation
    verification = verify_equation(
        equation, 
        variable_mapping,
        modified_execution,
        expected_answer
    )
    
    # If verification fails, request regeneration with feedback
    max_attempts = 3
    attempts = 1
    
    while not verification["is_valid"] and attempts < max_attempts:
        feedback_prompt = f"""
        The synthetic problem you generated has calculation issues:
        
        Problem: {synthetic_problem}
        Variable values: {new_variable_values}
        Expected answer: {expected_answer}
        
        Execution errors: {verification["errors"]}
        Execution log: {verification["execution_log"]}
        
        Please regenerate with correct values that will work with this equation:
        {equation}
        
        Your response MUST follow this format:
        <synthetic_problem>
        [Your corrected problem here]
        </synthetic_problem>
        <new_variable_values>
        [Corrected values for each variable]
        </new_variable_values>
        <expected_answer>
        [The correct final answer based on the new values]
        </expected_answer>
        """
        
        # Get new response from Gemini
        response = model.generate_content(feedback_prompt)
        
        # Extract and verify again
        synthetic_problem = extract_tag_content(response, "synthetic_problem")
        new_variable_values = extract_tag_content(response, "new_variable_values")
        expected_answer = extract_tag_content(response, "expected_answer")
        
        variable_value_dict = parse_variable_values(new_variable_values)
        modified_execution = modify_execution_steps(execution_steps, variable_value_dict)
        
        verification = verify_equation(
            equation,
            variable_mapping,
            modified_execution,
            expected_answer
        )
        
        attempts += 1
    
    return {
        "synthetic_problem": synthetic_problem,
        "answer": expected_answer if verification["is_valid"] else None,
        "original_equation": equation,
        "variable_mapping": variable_mapping,
        "new_variable_values": variable_value_dict,
        "verification": verification,
        "is_valid": verification["is_valid"]
    }
```

### 6. Helper Functions for Equation Processing

```python
def extract_tag_content(response, tag_name):
    """Extract content between specified XML-like tags from Gemini response"""
    import re
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, str(response), re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_variable_values(variable_values_text):
    """Parse variable values text into a dictionary"""
    result = {}
    lines = variable_values_text.strip().split('\n')
    
    for line in lines:
        if ':' in line:
            var_name, value = line.split(':', 1)
            result[var_name.strip()] = value.strip()
        elif '=' in line:
            var_name, value = line.split('=', 1)
            result[var_name.strip()] = value.strip()
    
    return result

def modify_execution_steps(execution_steps, new_values):
    """Modify execution steps with new variable values"""
    modified_steps = execution_steps
    
    # Replace variable initializations with new values
    for var_name, new_value in new_values.items():
        # Pattern to find variable initialization
        pattern = rf"{var_name}\s*=\s*[0-9.]+|{var_name}\s*=\s*\w+"
        replacement = f"{var_name} = {new_value}"
        
        # Replace the first occurrence (initialization)
        import re
        modified_steps = re.sub(pattern, replacement, modified_steps, count=1)
    
    return modified_steps
```

### 7. Complete Dataset Generation Pipeline

```python
def generate_synthetic_dataset(original_dataset, sample_size=None):
    """
    Generate synthetic versions of GSM8K problems using Gemini API and equation verification.
    
    Args:
        original_dataset: Original GSM8K dataset
        sample_size: Optional number of samples to generate (for testing)
    
    Returns:
        Synthetic dataset in the same format as the original
    """
    synthetic_problems = []
    equation_cache = {}  # Cache equations to avoid regenerating for similar problems
    
    # Process each problem (or a subset for sample_size)
    for i, problem in enumerate(original_dataset):
        if sample_size and i >= sample_size:
            break
            
        # Extract original question and answer
        original_question = problem["question"]
        original_answer = problem["answer"]
        
        # Step 1: Generate mathematical equation for the original problem
        equation_response = model.generate_content(
            create_equation_extraction_prompt(original_question, original_answer)
        )
        
        # Extract equation components
        equation = extract_tag_content(equation_response, "equation")
        variable_mapping = extract_tag_content(equation_response, "variable_mapping")
        execution_steps = extract_tag_content(equation_response, "execution_steps")
        
        # Step 2: Verify the equation works for the original problem
        verification = verify_equation(
            equation,
            variable_mapping,
            execution_steps,
            original_answer
        )
        
        # If equation verification fails for the original problem, try regenerating
        attempts = 1
        max_attempts = 3
        
        while not verification["is_valid"] and attempts < max_attempts:
            # Regenerate equation with feedback
            feedback_prompt = f"""
            The mathematical equation you provided doesn't produce the correct answer for the original problem.
            
            Original problem: {original_question}
            Expected answer: {original_answer}
            
            Your equation: {equation}
            Variable mapping: {variable_mapping}
            Execution steps: {execution_steps}
            
            Execution errors: {verification["errors"]}
            Execution log: {verification["execution_log"]}
            
            Please revise your equation and execution steps to correctly produce the answer {original_answer}.
            
            Your response MUST follow this format:
            <equation>
            [Revised equation]
            </equation>
            <variable_mapping>
            [Revised variable mapping]
            </variable_mapping>
            <execution_steps>
            [Revised execution steps]
            </execution_steps>
            """
            
            equation_response = model.generate_content(feedback_prompt)
            
            equation = extract_tag_content(equation_response, "equation")
            variable_mapping = extract_tag_content(equation_response, "variable_mapping")
            execution_steps = extract_tag_content(equation_response, "execution_steps")
            
            verification = verify_equation(
                equation,
                variable_mapping,
                execution_steps,
                original_answer
            )
            
            attempts += 1
        
        # If we have a valid equation, generate synthetic problem
        if verification["is_valid"]:
            # Cache the equation for potential reuse
            equation_cache[i] = {
                "equation": equation,
                "variable_mapping": variable_mapping,
                "execution_steps": execution_steps
            }
            
            # Step 3: Generate synthetic problem
            synthetic_result = generate_synthetic_problem(
                original_question,
                original_answer,
                equation,
                variable_mapping,
                execution_steps
            )
            
            if synthetic_result["is_valid"]:
                # Add to synthetic dataset
                synthetic_problems.append({
                    "question": synthetic_result["synthetic_problem"],
                    "answer": synthetic_result["answer"],
                    "prompt": [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': synthetic_result["synthetic_problem"]}
                    ],
                    "original_question_id": problem.get("id", i),
                    "original_equation": equation,
                    "variable_mapping": variable_mapping,
                    "verification_score": 1.0  # Perfect verification through equation
                })
            else:
                # Log verification issues
                print(f"Synthetic problem verification failed for problem {i}")
        else:
            # Log equation generation issues
            print(f"Equation verification failed for original problem {i}: {verification['errors']}")
            
    # Create dataset from collected problems
    from datasets import Dataset
    return Dataset.from_dict({
        'question': [p["question"] for p in synthetic_problems],
        'answer': [p["answer"] for p in synthetic_problems],
        'prompt': [p["prompt"] for p in synthetic_problems],
        'original_question_id': [p["original_question_id"] for p in synthetic_problems],
        'original_equation': [p["original_equation"] for p in synthetic_problems],
        'verification_score': [p["verification_score"] for p in synthetic_problems]
    })
```

### 8. Quality Control and Refinement

1. Generate a small pilot dataset (e.g., 20 problems)
2. Manual review process:
   - Check equation correctness and completeness
   - Verify execution in isolated Python environment
   - Ensure problems are understandable and well-formed
3. Refine equation extraction and verification based on pilot results
4. Generate the full synthetic dataset

### 9. Dataset Export with Equation Documentation

```python
def export_synthetic_dataset(synthetic_dataset, output_path):
    """Export the synthetic dataset in multiple formats with equation documentation"""
    
    # Save as HuggingFace dataset
    synthetic_dataset.save_to_disk(f"{output_path}/synthetic_gsm8k")
    
    # Export as CSV
    df = synthetic_dataset.to_pandas()
    df.to_csv(f"{output_path}/synthetic_gsm8k.csv", index=False)
    
    # Export as JSON
    import json
    with open(f"{output_path}/synthetic_gsm8k.json", 'w') as f:
        json.dump(synthetic_dataset.to_dict(), f)
    
    # Generate equation documentation
    with open(f"{output_path}/equations_documentation.md", 'w') as f:
        f.write("# Mathematical Equations for Synthetic GSM8K Dataset\n\n")
        
        for i, (question, answer, eq) in enumerate(zip(
            synthetic_dataset["question"], 
            synthetic_dataset["answer"],
            synthetic_dataset["original_equation"]
        )):
            f.write(f"## Problem {i+1}\n\n")
            f.write(f"**Question:** {question}\n\n")
            f.write(f"**Answer:** {answer}\n\n")
            f.write(f"**Mathematical Equation:**\n```python\n{eq}\n```\n\n")
            f.write("---\n\n")
    
    # Generate dataset statistics
    generate_dataset_statistics(synthetic_dataset, f"{output_path}/statistics.md")
```

## Implementation Timeline

1. **Day 1-2**: Setup and equation extraction system
   - Configure Gemini API access
   - Develop equation extraction prompts
   - Create Python-based equation verification system

2. **Day 3-4**: Pilot testing of equation approach
   - Generate equations for 20-30 problems
   - Test verification on original problems
   - Refine equation extraction and verification

3. **Day 5-6**: Synthetic problem generation
   - Implement synthetic problem generation with equation constraints
   - Test regeneration feedback loop
   - Sample validation of synthetic problems

4. **Day 7-9**: Full dataset generation
   - Process complete dataset with equation verification
   - Parallel processing for efficiency
   - Continuous monitoring of verification success rate

5. **Day 10-11**: Validation and quality assurance
   - Execute all equations in isolated environment
   - Verify mathematical correctness
   - Sample-based human evaluation

6. **Day 12**: Documentation and delivery
   - Complete equation documentation
   - Package dataset with verification metadata
   - Generate dataset card with equation statistics

## Resources Required

1. **API Access**:
   - Gemini API key with sufficient quota
   - Backup model API (e.g., OpenAI) for verification

2. **Compute**:
   - Processing for equation extraction and verification
   - Python execution environment for testing
   - Storage for equations and verification logs

3. **Human Review**:
   - Mathematics experts for equation validation
   - Python developers for verification system testing

## Risk Management

1. **Equation extraction challenges**:
   - Complex problems with multiple steps
   - Mitigation: Structured prompting and extraction verification

2. **Execution safety**:
   - Python `exec()` security considerations
   - Mitigation: Sandboxed execution with controlled scope

3. **Mathematical edge cases**:
   - Problems with non-numeric answers or complex logic
   - Mitigation: Special case handling in verification system

4. **API limitations**:
   - Rate limiting and token context limits
   - Mitigation: Caching and batched processing 