"""
Configuration handling for the logging system.

This module handles loading, validation, and application of logging configuration.
"""

import os
import logging
import logging.config
from typing import Dict, Optional, Any
import yaml
import importlib.resources
import importlib.util

# Default configuration path - using the importlib.resources approach for more reliable package resource loading
try:
    # Python 3.9+
    import importlib.resources as pkg_resources
    from importlib.resources import files
    DEFAULT_CONFIG_PATH = str(files('src.logging').joinpath('config/default_logging_config.yaml'))
except (ImportError, AttributeError):
    # Fallback for older Python versions
    import pkg_resources
    DEFAULT_CONFIG_PATH = pkg_resources.resource_filename("src.logging", "config/default_logging_config.yaml")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load logging configuration from a YAML file.
    
    Args:
        config_path: Path to a YAML configuration file. If None, default config is used.
        
    Returns:
        Dict containing logging configuration.
        
    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    if config_path is None:
        # Use default config if it exists
        if os.path.exists(DEFAULT_CONFIG_PATH):
            config_path = DEFAULT_CONFIG_PATH
        else:
            # Fallback to embedded default config
            return _get_default_config()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            validate_config(config)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file: {e}")

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the logging configuration.
    
    Args:
        config: Dictionary containing logging configuration.
        
    Raises:
        ValueError: If the configuration is invalid.
    """
    # Check required sections
    required_sections = ["version", "handlers", "formatters", "loggers"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: {section}")
    
    # Ensure all referenced handlers exist
    handlers = set(config.get("handlers", {}).keys())
    
    # Check root logger
    if "root" in config:
        root_handlers = set(config["root"].get("handlers", []))
        if not root_handlers.issubset(handlers):
            unknown = root_handlers - handlers
            raise ValueError(f"Root logger references unknown handlers: {unknown}")
    
    # Check all loggers
    for logger_name, logger_config in config.get("loggers", {}).items():
        logger_handlers = set(logger_config.get("handlers", []))
        if not logger_handlers.issubset(handlers):
            unknown = logger_handlers - handlers
            raise ValueError(f"Logger '{logger_name}' references unknown handlers: {unknown}")
    
    # TODO: Add more validation as needed

def _get_default_config() -> Dict[str, Any]:
    """Get the default logging configuration.
    
    Returns:
        Dict containing default logging configuration.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "root": {
            "level": "INFO",
            "handlers": ["console"]
        },
        "loggers": {
            "logging.training": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "logging.model": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "logging.reward": {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False
            },
            "logging.system": {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False
            },
            "logging.metrics": {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "filename": "logs/training.log",
                "maxBytes": 10485760,
                "backupCount": 5
            }
        },
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        }
    }

def apply_config(config: Dict[str, Any]) -> None:
    """Apply the logging configuration.
    
    Args:
        config: Dictionary containing logging configuration.
    """
    # Create log directory if it doesn't exist
    for handler_config in config.get("handlers", {}).values():
        if "filename" in handler_config:
            log_dir = os.path.dirname(handler_config["filename"])
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
    
    # Handle custom handlers before applying config
    processed_config = _process_custom_handlers(config)
    
    # Apply configuration
    logging.config.dictConfig(processed_config)

def _process_custom_handlers(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process custom handlers in the configuration.
    
    This function replaces custom handler classes with actual class references
    that Python's logging system can use.
    
    Args:
        config: Dictionary containing logging configuration.
        
    Returns:
        Processed configuration with custom handler classes.
    """
    # Make a deep copy of the config to avoid modifying the original
    import copy
    processed_config = copy.deepcopy(config)
    
    # Process handlers
    for handler_name, handler_config in processed_config.get("handlers", {}).items():
        if "class" in handler_config:
            class_path = handler_config["class"]
            
            # Check if it's a custom handler
            if class_path.startswith("logging.custom."):
                # Extract the handler name
                handler_class_name = class_path.split(".")[-1]
                
                # Import the handler
                try:
                    from .handlers import get_handler_class
                    handler_class = get_handler_class(handler_class_name)
                    
                    # Replace the class path with the actual class
                    handler_config["class"] = handler_class
                except (ImportError, ValueError) as e:
                    logging.warning(f"Failed to load custom handler {handler_class_name}: {e}")
                    # Fallback to a standard handler
                    handler_config["class"] = "logging.StreamHandler"
    
    # Process formatters
    for formatter_name, formatter_config in processed_config.get("formatters", {}).items():
        if formatter_config.get("format") == "json":
            # Replace with JSONFormatter
            try:
                from .formatters import JSONFormatter
                formatter_config["()"] = JSONFormatter
                
                # Get fields if specified
                if "fields" in formatter_config:
                    formatter_config["fields"] = formatter_config["fields"]
                
                # Remove format key as it's not needed
                formatter_config.pop("format", None)
            except ImportError as e:
                logging.warning(f"Failed to load JSONFormatter: {e}")
    
    return processed_config

def update_log_level(logger_name: str, level: str) -> None:
    """Update the log level for a specific logger.
    
    Args:
        logger_name: Name of the logger to update.
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Raises:
        ValueError: If the level is invalid.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level) 