"""
TensorBoard logging handler.

This handler writes log records to TensorBoard for visualization.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Union, Tuple
import os


class TensorBoardHandler(logging.Handler):
    """A logging handler that writes metrics to TensorBoard.
    
    This handler extracts metrics from log records and writes them to TensorBoard
    for visualization. It handles scalar values, histograms, and text data.
    """
    
    def __init__(
        self,
        log_dir: str,
        flush_secs: int = 10,
        max_queue: int = 10,
        purge_step: Optional[int] = None
    ):
        """Initialize the handler.
        
        Args:
            log_dir: Directory where TensorBoard logs are saved.
            flush_secs: How often to flush the metrics.
            max_queue: Size of the queue for asynchronous logging.
            purge_step: Optional step at which to purge old data.
        """
        super().__init__()
        
        self.log_dir = log_dir
        self.flush_secs = flush_secs
        self.max_queue = max_queue
        self.purge_step = purge_step
        
        # Create directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Lazy import to avoid importing TensorFlow unnecessarily
        # TensorBoard will be initialized when needed
        self._writer = None
        self._writer_lock = threading.RLock()
        
        # Cache steps for different tags
        self._steps = {}
    
    @property
    def writer(self):
        """Get or create the TensorBoard SummaryWriter."""
        # Lazy initialization
        if self._writer is None:
            with self._writer_lock:
                if self._writer is None:
                    # Import TensorBoard here to avoid import dependency
                    try:
                        # First try to import PyTorch's SummaryWriter
                        from torch.utils.tensorboard import SummaryWriter
                        self._writer = SummaryWriter(
                            log_dir=self.log_dir,
                            flush_secs=self.flush_secs,
                            max_queue=self.max_queue,
                            purge_step=self.purge_step
                        )
                    except ImportError:
                        # Fall back to TensorFlow's SummaryWriter
                        try:
                            from tensorboard.summary.writer.writer import SummaryWriter
                            self._writer = SummaryWriter(
                                logdir=self.log_dir,
                                max_queue=self.max_queue,
                                flush_secs=self.flush_secs
                            )
                        except ImportError:
                            raise ImportError(
                                "Neither PyTorch nor TensorFlow TensorBoard is available. "
                                "Please install either 'torch' or 'tensorboard'."
                            )
        
        return self._writer
    
    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record.
        
        Args:
            record: The log record to process.
        """
        try:
            # Extract the message
            msg = self.format(record)
            
            # If the message is a dict, use it directly
            if isinstance(record.msg, dict):
                data = record.msg
            else:
                # Try to interpret as JSON or dict
                try:
                    if isinstance(msg, str):
                        import json
                        data = json.loads(msg)
                    else:
                        data = msg
                except (ValueError, TypeError):
                    # If not a dict or JSON, use the formatted message as text
                    self._log_text(f"{record.name}", msg, None)
                    return
            
            # Look for the step in the data
            step = data.get('step', None)
            
            # Process metrics in the data
            self._process_metrics(data, step)
            
        except Exception:
            self.handleError(record)
    
    def _process_metrics(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Process metrics in the data.
        
        Args:
            data: Dictionary containing metrics.
            step: Optional step for the metrics. If None, an auto-incrementing
                counter will be used.
        """
        # Handle special case for model outputs
        if 'model_output' in data and 'question' in data:
            self._log_text(
                f"model/output/{data.get('question', 'unknown')[:50]}",
                data['model_output'],
                step
            )
            return
        
        # Handle reward logging
        if 'reward_name' in data and 'value' in data:
            self._log_scalar(
                f"rewards/{data['reward_name']}",
                data['value'],
                step
            )
            return
        
        # Handle general metrics
        for key, value in data.items():
            if key == 'step':
                continue
                
            if isinstance(value, (int, float)):
                self._log_scalar(f"metrics/{key}", value, step)
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                self._log_histogram(f"distributions/{key}", value, step)
            elif isinstance(value, str):
                # Only log certain string values as text
                if key in ['generation', 'prediction', 'output']:
                    self._log_text(f"text/{key}", value, step)
    
    def _log_scalar(self, tag: str, value: Union[int, float], step: Optional[int] = None) -> None:
        """Log a scalar value to TensorBoard.
        
        Args:
            tag: The tag for the scalar.
            value: The scalar value.
            step: Optional step for the scalar. If None, an auto-incrementing
                counter will be used.
        """
        # Use auto-incrementing counter if step is None
        if step is None:
            step = self._steps.get(tag, 0)
            self._steps[tag] = step + 1
        
        try:
            self.writer.add_scalar(tag, value, global_step=step)
        except:
            # Fallback if add_scalar is not available
            try:
                import numpy as np
                self.writer.add_scalars(tag.split('/')[0], {tag.split('/')[-1]: value}, global_step=step)
            except:
                pass
    
    def _log_histogram(self, tag: str, values: List[Union[int, float]], step: Optional[int] = None) -> None:
        """Log a histogram to TensorBoard.
        
        Args:
            tag: The tag for the histogram.
            values: The values for the histogram.
            step: Optional step for the histogram. If None, an auto-incrementing
                counter will be used.
        """
        # Use auto-incrementing counter if step is None
        if step is None:
            step = self._steps.get(tag, 0)
            self._steps[tag] = step + 1
        
        try:
            import numpy as np
            self.writer.add_histogram(tag, np.array(values), global_step=step)
        except:
            # If histogram fails, try to add as scalars
            try:
                import numpy as np
                self.writer.add_scalars(
                    tag, 
                    {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    },
                    global_step=step
                )
            except:
                pass
    
    def _log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Log text to TensorBoard.
        
        Args:
            tag: The tag for the text.
            text: The text value.
            step: Optional step for the text. If None, an auto-incrementing
                counter will be used.
        """
        # Use auto-incrementing counter if step is None
        if step is None:
            step = self._steps.get(tag, 0)
            self._steps[tag] = step + 1
        
        try:
            self.writer.add_text(tag, text, global_step=step)
        except:
            # No fallback for text
            pass
    
    def close(self) -> None:
        """Close the handler and flush all pending data."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        
        super().close() 