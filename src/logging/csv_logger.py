"""
CSV Logger for Gemma training outputs and rewards.

This module provides thread-safe CSV logging functionality for model generations
and reward values with batched writes for better performance.
"""

import os
import csv
import time
import queue
import threading
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class CSVLogger:
    """Thread-safe CSV logger with batch processing and background writing.
    
    This class provides an efficient way to log model generations and rewards
    to CSV files without blocking the main training loop.
    """
    
    def __init__(self, 
                 csv_path: str, 
                 fieldnames: Optional[List[str]] = None,
                 max_queue_size: int = 1000,
                 batch_size: int = 50,
                 flush_interval: float = 5.0):
        """Initialize the CSV logger.
        
        Args:
            csv_path: Path to the CSV file
            fieldnames: List of column names (if None, will be set from first record)
            max_queue_size: Maximum size of the queue before blocking
            batch_size: Maximum number of records to write in a single batch
            flush_interval: Time in seconds to wait for new records before flushing
        """
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Create a thread-safe queue for records
        self.queue = queue.Queue(maxsize=max_queue_size)
        
        # Track if header has been written
        self.header_written = False
        
        # Start the background worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # For safe shutdown
        self._shutdown = False
    
    def log_generation(self, data_dict: Dict[str, Any]):
        """Add a generation record to the CSV queue.
        
        Args:
            data_dict: Dictionary containing the data to log
        """
        # Add timestamp if not present
        if 'timestamp' not in data_dict:
            data_dict['timestamp'] = datetime.datetime.now().isoformat()
            
        # Put in queue (with timeout to avoid blocking forever)
        try:
            self.queue.put(data_dict, block=True, timeout=1.0)
        except queue.Full:
            # Log a warning and continue (better to lose a log than block training)
            print(f"WARNING: CSV logging queue is full - dropping record")
    
    def _worker(self):
        """Background thread that processes the queue and writes to CSV."""
        while not self._shutdown:
            # Batch process for efficiency
            batch = []
            try:
                # Get at least one item
                batch.append(self.queue.get(block=True, timeout=self.flush_interval))
                
                # Get more items if available (up to batch_size-1 more)
                for _ in range(self.batch_size - 1):
                    try:
                        batch.append(self.queue.get(block=False))
                    except queue.Empty:
                        break
                        
                if batch:
                    self._write_batch_to_csv(batch)
                    
                # Mark tasks as done
                for _ in range(len(batch)):
                    self.queue.task_done()
                    
            except queue.Empty:
                # No items available, but we still need to flush any partial batch
                if batch:
                    self._write_batch_to_csv(batch)
                    # Mark tasks as done
                    for _ in range(len(batch)):
                        self.queue.task_done()
                # No items available, just continue
                continue
                
    def _write_batch_to_csv(self, batch):
        """Write a batch of records to the CSV file.
        
        Args:
            batch: List of dictionaries to write
        """
        if not batch:
            return
            
        # Determine field names if not provided
        if self.fieldnames is None:
            # Use keys from first record, ensuring timestamp is first
            fieldnames = list(batch[0].keys())
            if 'timestamp' in fieldnames:
                fieldnames.remove('timestamp')
                fieldnames = ['timestamp'] + fieldnames
            self.fieldnames = fieldnames
            
        # Create or append to file
        mode = 'a' if os.path.exists(self.csv_path) and self.header_written else 'w'
        
        try:
            with open(self.csv_path, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                
                # Write header if needed
                if not self.header_written or mode == 'w':
                    writer.writeheader()
                    self.header_written = True
                
                # Write all records
                for record in batch:
                    # Ensure all fields exist (with empty values for missing ones)
                    row = {field: record.get(field, '') for field in self.fieldnames}
                    writer.writerow(row)
        except Exception as e:
            print(f"ERROR writing to CSV {self.csv_path}: {str(e)}")
    
    def flush(self):
        """Force flush any pending records to disk."""
        # Simply wait for the queue to be processed
        self.queue.join()
    
    def shutdown(self):
        """Shutdown the worker thread after flushing the queue."""
        # Signal shutdown
        self._shutdown = True
        
        # Wait for queue to be processed
        self.flush()
        
        # Wait for worker thread to exit (with timeout)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10.0)
            

# Global CSV logger instance (will be initialized by the logging system)
_csv_logger = None

def get_csv_logger():
    """Get the global CSV logger instance."""
    return _csv_logger

def initialize_csv_logger(csv_path, fieldnames=None, **kwargs):
    """Initialize the global CSV logger instance.
    
    Args:
        csv_path: Path to the CSV file
        fieldnames: List of column names
        **kwargs: Additional arguments for CSVLogger
    """
    global _csv_logger
    _csv_logger = CSVLogger(csv_path, fieldnames, **kwargs)
    return _csv_logger

def log_generation_with_rewards(generation_data):
    """Log a generation with reward values to the CSV file.
    
    Args:
        generation_data: Dictionary containing the generation data and rewards
    """
    global _csv_logger
    if _csv_logger is None:
        raise RuntimeError("CSV logger not initialized. Call initialize_csv_logger first.")
    
    _csv_logger.log_generation(generation_data) 