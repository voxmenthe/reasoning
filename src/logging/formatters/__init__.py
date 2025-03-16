"""
Custom log formatters for the logging system.

This package provides custom formatters beyond the standard library.
"""

from typing import Dict, Any, Type

# Formatter registry
FORMATTERS: Dict[str, Type[Any]] = {}

# Import and register formatters
from .json_formatter import JSONFormatter
FORMATTERS["JSONFormatter"] = JSONFormatter

def get_formatter_class(formatter_name: str) -> Type[Any]:
    """Get a formatter class by name.
    
    Args:
        formatter_name: Name of the formatter class.
        
    Returns:
        The formatter class.
        
    Raises:
        ValueError: If the formatter is not found.
    """
    if formatter_name not in FORMATTERS:
        raise ValueError(f"Unknown formatter: {formatter_name}")
    
    return FORMATTERS[formatter_name] 