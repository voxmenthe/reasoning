"""
Validation runner for periodic evaluation on held-out data.

This module provides functionality to run periodic validation
on a held-out dataset to measure model performance.
"""

import json
import threading
import time
import os
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import logging
import traceback

# Get the logger
logger = logging.getLogger(__name__)


class ValidationRunner:
    """Run validation on a held-out dataset in a background thread.
    
    This class manages periodic validation of the model on a held-out dataset,
    running in a background thread to avoid blocking the training process.
    """
    
    def __init__(
        self,
        dataset_path: str,
        max_samples: int = 100,
        timeout_per_sample: float = 5.0,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize the validation runner.
        
        Args:
            dataset_path: Path to the validation dataset.
            max_samples: Maximum number of samples to use for validation.
            timeout_per_sample: Maximum time (in seconds) to spend on each sample.
            metrics_callback: Callback function to receive validation metrics.
        """
        self.dataset_path = dataset_path
        self.max_samples = max_samples
        self.timeout_per_sample = timeout_per_sample
        self.metrics_callback = metrics_callback
        
        # Validation results
        self.last_results: Dict[str, Any] = {}
        self.is_running = False
        
        # Thread for background validation
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the validation dataset.
        
        Returns:
            List of validation samples.
            
        Raises:
            FileNotFoundError: If the dataset file is not found.
            json.JSONDecodeError: If the dataset file is not valid JSON.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Validation dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Limit the number of samples
        if isinstance(dataset, list):
            return dataset[:self.max_samples]
        else:
            # If it's a dict with a 'samples' or 'data' key
            for key in ['samples', 'data', 'examples', 'validation']:
                if key in dataset and isinstance(dataset[key], list):
                    return dataset[key][:self.max_samples]
        
        raise ValueError(f"Invalid dataset format: {self.dataset_path}")
    
    def run_validation(
        self,
        model: Any,
        tokenizer: Any,
        step: int = 0,
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Run validation on the held-out dataset.
        
        This method starts validation in a background thread if it's not
        already running, or returns the last results if validation is
        currently running.
        
        Args:
            model: The model to evaluate.
            tokenizer: The tokenizer to use.
            step: Current training step.
            force: Force validation even if it's already running.
            
        Returns:
            Dictionary with validation metrics, or None if validation
            is already running.
        """
        with self.lock:
            if self.is_running and not force:
                logger.info("Validation already running, skipping")
                return None
            
            self.is_running = True
        
        # Start validation in a background thread
        self.thread = threading.Thread(
            target=self._run_validation_thread,
            args=(model, tokenizer, step),
            daemon=True
        )
        self.thread.start()
        
        # Return the last results immediately
        return self.last_results
    
    def _run_validation_thread(self, model: Any, tokenizer: Any, step: int) -> None:
        """Run validation in a background thread.
        
        Args:
            model: The model to evaluate.
            tokenizer: The tokenizer to use.
            step: Current training step.
        """
        start_time = time.time()
        logger.info(f"Starting validation at step {step}")
        
        try:
            # Load dataset
            dataset = self.load_dataset()
            logger.info(f"Loaded {len(dataset)} validation samples")
            
            # Run validation
            results = self._validate_dataset(model, tokenizer, dataset)
            
            # Add metadata
            results['step'] = step
            results['duration'] = time.time() - start_time
            results['timestamp'] = time.time()
            results['num_samples'] = len(dataset)
            
            # Store results
            with self.lock:
                self.last_results = results
            
            # Call callback
            if self.metrics_callback:
                try:
                    self.metrics_callback(results)
                except Exception as e:
                    logger.error(f"Error in validation metrics callback: {e}")
                    logger.error(traceback.format_exc())
            
            logger.info(f"Validation completed in {results['duration']:.2f}s: "
                        f"accuracy={results.get('accuracy', 0):.4f}")
        
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            with self.lock:
                self.is_running = False
    
    def _validate_dataset(
        self,
        model: Any,
        tokenizer: Any,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate the model on a dataset.
        
        Args:
            model: The model to evaluate.
            tokenizer: The tokenizer to use.
            dataset: List of validation samples.
            
        Returns:
            Dictionary with validation metrics.
        """
        correct = 0
        total = 0
        errors = []
        
        generation_times = []
        output_lengths = []
        
        for sample in dataset:
            try:
                # Extract input and expected output
                input_text = sample.get('question', sample.get('input', sample.get('prompt', '')))
                expected = sample.get('answer', sample.get('output', sample.get('target', '')))
                
                if not input_text:
                    logger.warning(f"Empty input in validation sample: {sample}")
                    continue
                
                # Generate output with timeout
                generation_start = time.time()
                output = self._generate_with_timeout(model, tokenizer, input_text)
                generation_time = time.time() - generation_start
                
                # Record metrics
                generation_times.append(generation_time)
                output_lengths.append(len(output.split()))
                
                # Extract answer from output
                answer = self._extract_answer(output)
                
                # Check correctness
                is_correct = self._check_correctness(answer, expected)
                
                if is_correct:
                    correct += 1
                else:
                    # Record error
                    errors.append({
                        'input': input_text,
                        'expected': expected,
                        'output': output,
                        'answer': answer
                    })
                
                total += 1
            
            except Exception as e:
                logger.error(f"Error validating sample: {e}")
                errors.append({
                    'input': sample.get('question', ''),
                    'error': str(e)
                })
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': errors[:10],  # Only keep a few errors to avoid bloat
            'avg_generation_time': sum(generation_times) / len(generation_times) if generation_times else 0,
            'avg_output_length': sum(output_lengths) / len(output_lengths) if output_lengths else 0,
            'format_issues': self._count_format_issues(errors)
        }
    
    def _generate_with_timeout(self, model: Any, tokenizer: Any, input_text: str) -> str:
        """Generate output with a timeout.
        
        Args:
            model: The model to use.
            tokenizer: The tokenizer to use.
            input_text: Input text.
            
        Returns:
            Generated output.
            
        Raises:
            TimeoutError: If generation takes too long.
        """
        # This is a placeholder implementation that should be adapted to the actual model API
        result = [""]  # Using a list to allow modification from the thread
        exception = [None]  # To capture exceptions from the thread
        
        def generate():
            try:
                # Placeholder for actual generation code
                # In reality, this would use the model and tokenizer
                # Something like:
                #   inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                #   outputs = model.generate(**inputs, max_length=512)
                #   result[0] = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Mock implementation for now
                result[0] = f"<reasoning>Thinking about {input_text}</reasoning>\n<answer>42</answer>"
                time.sleep(0.1)  # Simulate generation time
            except Exception as e:
                exception[0] = e
        
        # Start generation in a thread
        thread = threading.Thread(target=generate)
        thread.daemon = True
        thread.start()
        
        # Wait for completion with timeout
        thread.join(self.timeout_per_sample)
        
        if thread.is_alive():
            raise TimeoutError(f"Generation timed out after {self.timeout_per_sample}s")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def _extract_answer(self, output: str) -> str:
        """Extract answer from output text.
        
        Args:
            output: Generated output text.
            
        Returns:
            Extracted answer.
        """
        # Look for <answer>...</answer> format
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
        
        if answer_match:
            return answer_match.group(1).strip()
        
        # If no tags, return the full output
        return output.strip()
    
    def _check_correctness(self, answer: str, expected: str) -> bool:
        """Check if an answer is correct.
        
        This is a simple exact match check. In a real implementation,
        this would be more sophisticated, potentially using semantic
        similarity or other metrics.
        
        Args:
            answer: Generated answer.
            expected: Expected answer.
            
        Returns:
            True if the answer is correct, False otherwise.
        """
        # Simple exact match for now
        return answer.strip().lower() == expected.strip().lower()
    
    def _count_format_issues(self, errors: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count different types of format issues in errors.
        
        Args:
            errors: List of error records.
            
        Returns:
            Dictionary mapping issue types to counts.
        """
        import re
        
        issues = {
            'missing_reasoning_tag': 0,
            'missing_answer_tag': 0,
            'malformed_tags': 0
        }
        
        for error in errors:
            output = error.get('output', '')
            
            if '<reasoning>' not in output or '</reasoning>' not in output:
                issues['missing_reasoning_tag'] += 1
            
            if '<answer>' not in output or '</answer>' not in output:
                issues['missing_answer_tag'] += 1
            
            # Check for malformed tags (e.g., unclosed tags)
            if re.search(r'<(reasoning|answer)[^>]*>[^<]*$', output) or \
               re.search(r'^[^<]*</(reasoning|answer)>', output):
                issues['malformed_tags'] += 1
        
        return issues


# Global validation runner instance
default_validation_runner: Optional[ValidationRunner] = None


def get_validation_runner(config: Optional[Dict[str, Any]] = None) -> ValidationRunner:
    """Get or create the default validation runner.
    
    Args:
        config: Optional configuration for the validation runner.
        
    Returns:
        The default validation runner instance.
    """
    global default_validation_runner
    
    if default_validation_runner is None and config is not None:
        dataset_path = config.get('dataset_path', 'data/validation_set.json')
        max_samples = config.get('max_samples', 100)
        timeout_per_sample = config.get('timeout_per_sample', 5.0)
        
        default_validation_runner = ValidationRunner(
            dataset_path=dataset_path,
            max_samples=max_samples,
            timeout_per_sample=timeout_per_sample
        )
    
    if default_validation_runner is None:
        # Default configuration if none provided
        default_validation_runner = ValidationRunner(
            dataset_path='data/validation_set.json',
            max_samples=100,
            timeout_per_sample=5.0
        )
    
    return default_validation_runner


def run_validation(
    step: int,
    model: Any,
    tokenizer: Any,
    num_samples: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Run validation on the held-out dataset.
    
    Args:
        step: Current training step.
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        num_samples: Number of validation samples to use.
        config: Optional configuration for the validation runner.
        
    Returns:
        Dictionary with validation metrics, or None if validation
        is already running.
    """
    runner = get_validation_runner(config)
    
    if num_samples is not None:
        runner.max_samples = num_samples
    
    return runner.run_validation(model, tokenizer, step) 