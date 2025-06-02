"""
Training script for fine-tuning a model on the TIGER-Lab/WebInstruct-verified dataset
using the 'verifiers' library with Group-Relative Policy Optimization (GRPO).

Usage:
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model <your_base_model_name> --tensor-parallel-size 2
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file <path_to_accelerate_config.yaml> train_webinstruct_verified.py --model_name_or_path <your_base_model_name> --output_dir ./webinstruct_output
"""
import argparse
import logging
import os

import argparse
import logging
import os
import ast
import math
import re

import verifiers as vf
from datasets import load_dataset
from transformers import TrainingArguments

logger = logging.getLogger(__name__)


# Helper functions for parsing and comparison
def _normalize_str(s: str) -> str:
    return s.strip().lower()

def _parse_boolean(s: str) -> bool | None:
    s_norm = _normalize_str(s)
    if s_norm in ["true", "yes", "1"]:
        return True
    if s_norm in ["false", "no", "0"]:
        return False
    return None

def _parse_num(s: str) -> float | int | None:
    s_norm = _normalize_str(s)
    try:
        return int(s_norm)
    except ValueError:
        try:
            return float(s_norm)
        except ValueError:
            return None

def _parse_percentage(s: str) -> float | None:
    s_norm = _normalize_str(s.replace('%', ''))
    num = _parse_num(s_norm)
    return num / 100 if num is not None else None

def _parse_fraction(s: str) -> float | None:
    try:
        # Be cautious with eval, ensure input is somewhat controlled or use safer alternatives
        # For simple fractions like "1/2", eval might be okay.
        # For more complex expressions, sympy or other libraries are better.
        return float(eval(s.strip()))
    except Exception:
        return None

def _parse_list_matrix(s: str) -> list | None:
    try:
        return ast.literal_eval(s.strip())
    except (ValueError, SyntaxError):
        return None

def verify_answer_reward(prompt: str, completion: str, target_answer: str, parsed_completion: dict, **kwargs) -> float:
    """
    Custom reward function to verify the model's answer against the target answer,
    considering the answer_type from the dataset.

    Args:
        prompt: The input prompt (question).
        max_completion_length: Maximum length of the completion.
        num_iterations: Number of iterations for GRPO.
        log_completions: Whether to log completions during training.
        completion: The raw completion string from the model.
        target_answer: The ground truth answer from the dataset (string).
        parsed_completion: The parsed output from the model (e.g., {'think': '...', 'answer': 'model_actual_answer'}).
        **kwargs: Additional data from the dataset item, including 'answer_type'.

    Returns:
        A reward score (e.g., 1.0 for correct, 0.0 for incorrect).
    """
    model_answer_str = parsed_completion.get('answer', '').strip()
    target_answer_str = str(target_answer).strip() # Ensure target is also stripped
    answer_type = kwargs.get('answer_type', 'String') # Default to String if not provided

    if not model_answer_str: # Penalize empty answers
        return 0.0

    # TODO: This function should be significantly enhanced:
    # 1. Use TIGER-Lab/general-verifier model for complex types like 'Expression', 'Matrix', 'List' (especially with symbolic content).
    # 2. For mathematical 'Expression', use sympy for robust equivalence checking (e.g., sympy.simplify(expr1 - expr2) == 0).
    # 3. For chemical 'Expression', implement or use a library for chemical equation balancing and comparison.
    # 4. Handle units and more complex numerical comparisons.
    # 5. Consider partial rewards for partially correct answers (e.g., correct format but wrong value).

    try:
        if answer_type == 'Float':
            m_ans = _parse_num(model_answer_str)
            t_ans = _parse_num(target_answer_str)
            return 1.0 if m_ans is not None and t_ans is not None and math.isclose(float(m_ans), float(t_ans), rel_tol=1e-5) else 0.0
        elif answer_type == 'Integer':
            m_ans = _parse_num(model_answer_str)
            t_ans = _parse_num(target_answer_str)
            return 1.0 if m_ans is not None and t_ans is not None and int(m_ans) == int(t_ans) else 0.0
        elif answer_type == 'Boolean':
            m_ans = _parse_boolean(model_answer_str)
            t_ans = _parse_boolean(target_answer_str)
            return 1.0 if m_ans is not None and m_ans == t_ans else 0.0
        elif answer_type == 'Percentage':
            m_ans = _parse_percentage(model_answer_str)
            t_ans = _parse_percentage(target_answer_str)
            return 1.0 if m_ans is not None and t_ans is not None and math.isclose(m_ans, t_ans, rel_tol=1e-5) else 0.0
        elif answer_type == 'Fraction':
            # For fractions, comparing float evaluations can be tricky due to precision.
            # A more robust way would be to compare canonical forms (e.g. using sympy.Rational)
            m_ans_eval = _parse_fraction(model_answer_str)
            t_ans_eval = _parse_fraction(target_answer_str)
            if m_ans_eval is not None and t_ans_eval is not None and math.isclose(m_ans_eval, t_ans_eval, rel_tol=1e-5):
                return 1.0
            # Fallback to string comparison if eval fails or differs, as eval might not be robust enough
            return 1.0 if _normalize_str(model_answer_str) == _normalize_str(target_answer_str) else 0.0
        elif answer_type in ['List', 'Matrix']:
            m_ans = _parse_list_matrix(model_answer_str)
            t_ans = _parse_list_matrix(target_answer_str)
            # This is a simple equality check. For matrices/lists of numbers, element-wise tolerance might be needed.
            return 1.0 if m_ans is not None and m_ans == t_ans else 0.0
        elif answer_type in ['Multiple Choice', 'String', 'Expression', 'Other']:
            # For 'Expression', this is a very basic check. Robust checking needs sympy or similar.
            # For 'Other', direct string match is the best we can do without more info.
            return 1.0 if _normalize_str(model_answer_str) == _normalize_str(target_answer_str) else 0.0
        else: # Unknown answer_type, fallback to string comparison
            logger.warning(f"Unknown answer_type '{answer_type}', falling back to string comparison.")
            return 1.0 if _normalize_str(model_answer_str) == _normalize_str(target_answer_str) else 0.0
    except Exception as e:
        logger.error(f"Error during answer verification (type: {answer_type}, model_ans: '{model_answer_str}', target_ans: '{target_answer_str}'): {e}")
        return 0.0 # Penalize if any error occurs during verification of model's answer

def main():
    parser = argparse.ArgumentParser(description="Train a model on WebInstruct-verified using verifiers GRPO.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from Hugging Face Hub.")
    parser.add_argument("--dataset_name", type=str, default="TIGER-Lab/WebInstruct-verified", help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default="./webinstruct_trained_model", help="Output directory for saving model checkpoints and results.")
    parser.add_argument("--run_name", type=str, default="webinstruct_grpo_run", help="Name for the W&B run.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--num_iterations", type=int, default=2, help="Number of GRPO iterations over the dataset within an epoch. Default: 2 from verifiers.grpo_defaults.")
    parser.add_argument("--log_completions", action=argparse.BooleanOptionalAction, default=True, help="Log prompt completions during training. Default: True from verifiers.grpo_defaults.")
    parser.add_argument("--max_grad_norm", type=float, default=0.001, help="Maximum gradient norm for clipping. Default: 0.001 from verifiers.grpo_defaults.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: set total number of training steps to perform. Overrides num_train_epochs.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    # Add more GRPOTrainer/TrainingArguments as needed

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.info(f"Starting training with arguments: {args}")

    # 1. Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    train_dataset = dataset.get("train")
    # eval_dataset = dataset.get("test") # Optional: if GRPOTrainer or env uses it

    if train_dataset is None:
        logger.error(f"'train' split not found in dataset {args.dataset_name}. Available splits: {list(dataset.keys())}")
        return

    # 2. Define Parser, Rubric, and Environment
    # Expects model output like: <think>...</think>\n<answer>...</answer>
    xml_parser = vf.XMLParser(fields=['think', 'answer'], answer_field='answer')

    system_prompt = (
        "You are an expert assistant. Please think step by step to solve the following question. "
        "Provide your thought process within <think> tags and your final answer within <answer> tags. "
        f"Respond in the following format:\n{xml_parser.get_format_str()}"
    )

    # Ensure the dataset has 'question' and 'answer' columns as expected by SingleTurnEnv default prompt template
    # and our custom reward function.
    # If column names differ, rename them:
    # train_dataset = train_dataset.rename_column("original_question_column", "question")
    # train_dataset = train_dataset.rename_column("original_answer_column", "answer")

    rubric = vf.Rubric(funcs=[verify_answer_reward, xml_parser.get_format_reward_func()])

    # According to verifiers README, SingleTurnEnv expects 'question' and 'answer' columns.
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=system_prompt,
        parser=xml_parser,
        rubric=rubric,
        # prompt_template can be customized if needed, default is:
        # "{system_prompt}\n\nQuestion: {question}\n\nAnswer:"
    )
    logger.info("Verifier environment created.")

    # 3. Load Model and Tokenizer
    logger.info(f"Loading model and tokenizer: {args.model_name_or_path}")
    # Note: GRPOTrainer might require the model to be on a specific device or in a specific format.
    # vf.get_model_and_tokenizer handles some of this.
    model, tokenizer = vf.get_model_and_tokenizer(args.model_name_or_path, trust_remote_code=True)
    # Ensure tokenizer has pad_token if it's missing for some models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # 4. Configure GRPOTrainer
    # vf.grpo_defaults provides a base set of TrainingArguments for GRPO.
    # We can override them with our script arguments.
    training_args_dict = vf.grpo_defaults(
        run_name=args.run_name,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_iterations=args.num_iterations,
        log_completions=args.log_completions,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="wandb",  # Default, or set to None/other integrations
        remove_unused_columns=False, # Important for custom columns in dataset
        max_grad_norm=args.max_grad_norm,
    )
    # Convert dict to TrainingArguments object
    training_args = TrainingArguments(**training_args_dict)

    trainer = vf.GRPOTrainer(
        model=model,
        # processing_class=tokenizer, # As per verifiers README example
        tokenizer=tokenizer, # GRPOTrainer might take tokenizer directly, or processing_class
        env=vf_env,
        args=training_args,
        # train_dataset=train_dataset, # GRPOTrainer might take dataset directly
        # eval_dataset=eval_dataset,   # if supported
    )
    logger.info("GRPOTrainer initialized.")

    # 5. Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")

    # 6. Save final model (optional, as checkpoints are saved during training)
    # final_model_path = os.path.join(args.output_dir, "final_model")
    # trainer.save_model(final_model_path)
    # tokenizer.save_pretrained(final_model_path)
    # logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()

