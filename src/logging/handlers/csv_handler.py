"""
CSV logging handler with asynchronous file writes.

This handler writes log records to a CSV file asynchronously in a background thread.
"""

import csv
import logging
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Any, TextIO


class CSVHandler(logging.Handler):
    """A logging handler that writes records to a CSV file asynchronously.
    
    This handler buffers log records in memory and writes them to a CSV file
    in a background thread, to avoid blocking the main thread.
    """
    
    def __init__(
        self,
        filename: str,
        columns: List[str],
        delimiter: str = ',',
        buffer_size: int = 100,
        flush_interval: float = 5.0,
        encoding: str = 'utf-8',
        mode: str = 'a'
    ):
        """Initialize the handler.
        
        Args:
            filename: Path to the CSV file.
            columns: List of column names.
            delimiter: CSV delimiter.
            buffer_size: Maximum number of records to buffer before writing.
            flush_interval: Maximum time (in seconds) to wait before flushing buffer.
            encoding: File encoding.
            mode: File open mode ('a' for append, 'w' for write).
        """
        super().__init__()
        
        self.filename = filename
        self.columns = columns
        self.delimiter = delimiter
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.encoding = encoding
        self.mode = mode
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        # Queue for log records
        self.queue: queue.Queue = queue.Queue()
        
        # Thread for background processing
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        
        # Flag for thread control
        self._shutdown = threading.Event()
        
        # Keep track of file existence to handle headers correctly
        self._file_exists = os.path.exists(self.filename) and os.path.getsize(self.filename) > 0
        
        self.lock = threading.RLock()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record.
        
        Args:
            record: The log record to process.
        """
        try:
            if self._shutdown.is_set():
                return
                
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
                    # If not a dict or JSON, use the formatted message
                    data = {"message": msg}
            
            # Put the record in the queue
            self.queue.put(data)
            
        except Exception:
            self.handleError(record)
    
    def _process_queue(self) -> None:
        """Background thread that processes the queue and writes to the CSV file."""
        buffer: List[Dict[str, Any]] = []
        last_flush_time = time.time()
        file_handle: Optional[TextIO] = None
        csv_writer = None
        
        while not self._shutdown.is_set():
            try:
                # Try to get a record from the queue
                try:
                    record = self.queue.get(timeout=0.1)
                    buffer.append(record)
                    self.queue.task_done()
                except queue.Empty:
                    # If no records, check if we need to flush the buffer
                    if not buffer:
                        continue
                
                # Flush the buffer if it's full or the flush interval has elapsed
                if (len(buffer) >= self.buffer_size or 
                    time.time() - last_flush_time >= self.flush_interval):
                    
                    # Open the file if it's not already open
                    if file_handle is None:
                        file_exists = os.path.exists(self.filename) and os.path.getsize(self.filename) > 0
                        file_handle = open(self.filename, self.mode, newline='', encoding=self.encoding)
                        csv_writer = csv.DictWriter(
                            file_handle, 
                            fieldnames=self.columns,
                            delimiter=self.delimiter,
                            extrasaction='ignore'
                        )
                        
                        # Write header if the file is new
                        if not file_exists:
                            csv_writer.writeheader()
                    
                    # Write all records
                    for record in buffer:
                        csv_writer.writerow(record)
                    
                    # Flush to disk
                    file_handle.flush()
                    
                    # Clear buffer and update flush time
                    buffer.clear()
                    last_flush_time = time.time()
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                # Clear buffer and continue
                buffer.clear()
                last_flush_time = time.time()
            
            finally:
                # Close the file if it's open
                if file_handle is not None and len(buffer) == 0:
                    file_handle.close()
                    file_handle = None
                    csv_writer = None
    
    def close(self) -> None:
        """Close the handler and flush all buffered data."""
        with self.lock:
            if self._shutdown.is_set():
                return
                
            self._shutdown.set()
            
            # Wait for the thread to finish
            if self.thread.is_alive():
                self.thread.join(timeout=10.0)
            
            # Process any remaining records
            buffer: List[Dict[str, Any]] = []
            while not self.queue.empty():
                buffer.append(self.queue.get())
                self.queue.task_done()
            
            if buffer:
                # Open the file
                file_exists = os.path.exists(self.filename) and os.path.getsize(self.filename) > 0
                with open(self.filename, self.mode, newline='', encoding=self.encoding) as f:
                    writer = csv.DictWriter(
                        f, 
                        fieldnames=self.columns,
                        delimiter=self.delimiter,
                        extrasaction='ignore'
                    )
                    
                    # Write header if the file is new
                    if not file_exists:
                        writer.writeheader()
                    
                    # Write all records
                    for record in buffer:
                        writer.writerow(record)
            
            super().close() 