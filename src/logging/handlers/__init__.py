"""
Custom logging handlers for the logging system.

This package provides custom logging handlers beyond the standard library.
"""

from typing import Dict, Any, Type

# Handler registry
HANDLERS: Dict[str, Type[Any]] = {}

# Import and register handlers
from .csv_handler import CSVHandler
HANDLERS["CSVHandler"] = CSVHandler

from .tensorboard_handler import TensorBoardHandler
HANDLERS["TensorBoardHandler"] = TensorBoardHandler

def get_handler_class(handler_name: str) -> Type[Any]:
    """Get a handler class by name.
    
    Args:
        handler_name: Name of the handler class.
        
    Returns:
        The handler class.
        
    Raises:
        ValueError: If the handler is not found.
    """
    if handler_name not in HANDLERS:
        raise ValueError(f"Unknown handler: {handler_name}")
    
    return HANDLERS[handler_name] 