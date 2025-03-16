"""
Model evaluation metrics for training evaluation.

This module provides functions to calculate various metrics for evaluating
model training progress, especially for RL-based training approaches.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import threading
import time
import re


class MetricsTracker:
    """Track and calculate metrics for model training.
    
    This class maintains a history of metrics and calculates statistics
    and trends over time. It's designed for RL-based training approaches
    where traditional loss metrics may not be as informative.
    """
    
    def __init__(self, window_size: int = 100, max_history: int = 1000):
        """Initialize the metrics tracker.
        
        Args:
            window_size: Size of the sliding window for trend calculations.
            max_history: Maximum number of values to keep in history.
        """
        self.window_size = window_size
        self.max_history = max_history
        self.history: Dict[str, List[Tuple[int, float]]] = {}
        self.lock = threading.RLock()
    
    def add_metric(self, name: str, value: float, step: int) -> None:
        """Add a metric value.
        
        Args:
            name: Name of the metric.
            value: Value of the metric.
            step: Training step.
        """
        with self.lock:
            if name not in self.history:
                self.history[name] = []
            
            self.history[name].append((step, value))
            
            # Trim history if needed
            if len(self.history[name]) > self.max_history:
                self.history[name] = self.history[name][-self.max_history:]
    
    def get_recent_values(self, name: str, window: Optional[int] = None) -> List[float]:
        """Get recent values for a metric.
        
        Args:
            name: Name of the metric.
            window: Number of recent values to return. If None, use window_size.
            
        Returns:
            List of recent values.
        """
        with self.lock:
            if name not in self.history:
                return []
            
            window = window or self.window_size
            values = [v for _, v in self.history[name][-window:]]
            return values
    
    def get_trend(self, name: str, window: Optional[int] = None) -> Dict[str, float]:
        """Calculate trend metrics for a given metric.
        
        Args:
            name: Name of the metric.
            window: Window size for trend calculation. If None, use window_size.
            
        Returns:
            Dictionary with trend metrics (mean, median, min, max, std, etc.).
        """
        values = self.get_recent_values(name, window)
        
        if not values:
            return {}
        
        values_array = np.array(values)
        return {
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "std": float(np.std(values_array)),
            "count": len(values),
            "trend": self._calculate_trend_score(values_array)
        }
    
    def _calculate_trend_score(self, values: np.ndarray) -> float:
        """Calculate a trend score.
        
        A positive score indicates an improving trend, negative indicates declining.
        
        Args:
            values: Array of values.
            
        Returns:
            Trend score.
        """
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        return float(slope)
    
    def get_all_trends(self) -> Dict[str, Dict[str, float]]:
        """Get trends for all tracked metrics.
        
        Returns:
            Dictionary mapping metric names to trend dictionaries.
        """
        with self.lock:
            return {name: self.get_trend(name) for name in self.history.keys()}


def calculate_token_entropy(texts: List[str], tokenize: Callable[[str], List[str]] = None) -> float:
    """Calculate token entropy to measure output diversity.
    
    Args:
        texts: List of generated texts.
        tokenize: Function to tokenize text. If None, simple whitespace tokenization is used.
        
    Returns:
        Token entropy value.
    """
    if not texts:
        return 0.0
    
    # Default tokenization
    if tokenize is None:
        tokenize = lambda x: x.split()
    
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    
    if not all_tokens:
        return 0.0
    
    # Calculate token frequencies
    token_freqs = {}
    for token in all_tokens:
        token_freqs[token] = token_freqs.get(token, 0) + 1
    
    # Calculate entropy
    total_tokens = len(all_tokens)
    entropy = 0.0
    for count in token_freqs.values():
        prob = count / total_tokens
        entropy -= prob * np.log2(prob)
    
    return float(entropy)


def calculate_format_adherence_rate(texts: List[str], 
                                    pattern: str = r'<reasoning>.*?</reasoning>.*?<answer>.*?</answer>') -> float:
    """Calculate the rate of format adherence in generated texts.
    
    Args:
        texts: List of generated texts.
        pattern: Regex pattern to match the expected format.
        
    Returns:
        Format adherence rate (0.0 to 1.0).
    """
    if not texts:
        return 0.0
    
    pattern_re = re.compile(pattern, re.DOTALL)
    matches = sum(1 for text in texts if pattern_re.search(text))
    
    return matches / len(texts)


def calculate_reasoning_answer_ratio(texts: List[str]) -> Dict[str, Union[float, int]]:
    """Calculate the ratio of tokens spent on reasoning vs. answers.
    
    Args:
        texts: List of generated texts.
        
    Returns:
        Dictionary with reasoning/answer statistics.
    """
    if not texts:
        return {
            "reasoning_to_answer_ratio": 0.0,
            "avg_reasoning_length": 0,
            "avg_answer_length": 0
        }
    
    reasoning_lengths = []
    answer_lengths = []
    
    reasoning_pattern = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    
    for text in texts:
        reasoning_match = reasoning_pattern.search(text)
        answer_match = answer_pattern.search(text)
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            reasoning_lengths.append(len(reasoning.split()))
        
        if answer_match:
            answer = answer_match.group(1).strip()
            answer_lengths.append(len(answer.split()))
    
    if not reasoning_lengths or not answer_lengths:
        return {
            "reasoning_to_answer_ratio": 0.0,
            "avg_reasoning_length": 0,
            "avg_answer_length": 0
        }
    
    avg_reasoning = sum(reasoning_lengths) / len(reasoning_lengths)
    avg_answer = sum(answer_lengths) / len(answer_lengths)
    
    ratio = avg_reasoning / avg_answer if avg_answer > 0 else 0.0
    
    return {
        "reasoning_to_answer_ratio": ratio,
        "avg_reasoning_length": avg_reasoning,
        "avg_answer_length": avg_answer
    }


def calculate_kl_divergence(samples1: List[float], samples2: List[float], 
                           num_bins: int = 20) -> float:
    """Calculate KL divergence between two sets of samples.
    
    Args:
        samples1: First set of samples.
        samples2: Second set of samples.
        num_bins: Number of bins for histogram.
        
    Returns:
        KL divergence value.
    """
    if not samples1 or not samples2:
        return 0.0
    
    # Create histograms
    min_val = min(min(samples1), min(samples2))
    max_val = max(max(samples1), max(samples2))
    
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    
    hist1, _ = np.histogram(samples1, bins=num_bins, range=(min_val, max_val))
    hist2, _ = np.histogram(samples2, bins=num_bins, range=(min_val, max_val))
    
    # Normalize
    hist1 = hist1.astype(float) / (sum(hist1) + eps)
    hist2 = hist2.astype(float) / (sum(hist2) + eps)
    
    # Replace zeros with epsilon
    hist1 = np.maximum(hist1, eps)
    hist2 = np.maximum(hist2, eps)
    
    # Calculate KL divergence
    kl_div = np.sum(hist1 * np.log(hist1 / hist2))
    
    return float(kl_div)


# Global metrics tracker instance
global_metrics = MetricsTracker() 