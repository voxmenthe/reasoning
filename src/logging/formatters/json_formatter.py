"""
JSON log formatter.

This formatter converts log records to JSON format for structured logging.
"""

import json
import logging
import datetime
from typing import Dict, Any, List, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON strings.
    
    This formatter converts log records to JSON format, making them
    suitable for structured logging and easier to parse by log
    analysis tools.
    """
    
    def __init__(
        self,
        fields: Optional[List[str]] = None,
        time_format: str = "%Y-%m-%d %H:%M:%S",
        msec_format: str = "%s.%03d",
        indent: Optional[int] = None
    ):
        """Initialize the formatter.
        
        Args:
            fields: List of fields to include in the JSON output. If None,
                   include all fields.
            time_format: Format for the timestamp.
            msec_format: Format for the milliseconds part of the timestamp.
            indent: Indentation level for the JSON output. If None, the output
                   will be compact.
        """
        super().__init__()
        self.fields = fields
        self.time_format = time_format
        self.msec_format = msec_format
        self.indent = indent
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.
        
        Args:
            record: The log record to format.
            
        Returns:
            A JSON string representation of the log record.
        """
        # Create a dictionary with the base fields
        log_data = {
            "timestamp": self.formatTime(record, self.time_format),
            "level": record.levelname,
            "name": record.name,
            "file": record.pathname,
            "line": record.lineno,
            "func": record.funcName
        }
        
        # Add the message
        if isinstance(record.msg, dict):
            # If the message is already a dict, use it directly
            log_data["message"] = record.msg
        else:
            # Otherwise, use the formatted message
            log_data["message"] = record.getMessage()
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add stack info if present
        if record.stack_info:
            log_data["stack"] = self.formatStack(record.stack_info)
        
        # Filter fields if requested
        if self.fields:
            log_data = {k: v for k, v in log_data.items() if k in self.fields}
        
        # Convert to JSON
        return json.dumps(log_data, default=self._json_default, indent=self.indent)
    
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Format the timestamp.
        
        Args:
            record: The log record.
            datefmt: Format for the timestamp.
            
        Returns:
            Formatted timestamp.
        """
        ct = self.converter(record.created)
        if datefmt:
            s = datetime.datetime.fromtimestamp(record.created).strftime(datefmt)
        else:
            s = datetime.datetime.fromtimestamp(record.created).strftime(self.time_format)
        
        if self.msec_format:
            s = self.msec_format % (s, record.msecs)
        
        return s
    
    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Convert non-serializable objects to a serializable format.
        
        Args:
            obj: The object to convert.
            
        Returns:
            A serializable representation of the object.
        """
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj) 