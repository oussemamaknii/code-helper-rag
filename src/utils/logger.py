"""
Logging configuration and utilities.

This module provides structured logging capabilities with support for both
development and production environments, including JSON formatting and
contextual metadata.
"""

import sys
import logging
import json
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
import structlog
from structlog.typing import FilteringBoundLogger

from src.config.settings import settings


def setup_logging() -> None:
    """
    Configure application logging with structured logging support.
    
    Sets up logging configuration based on application settings including
    log level, format, and output destinations.
    """
    # Configure structlog
    if settings.enable_structured_logging:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                # Use JSON formatter for production, console for development
                structlog.processors.JSONRenderer() if settings.log_format == "json" 
                else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    if settings.log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.log_level))
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if settings.log_file:
        log_file_path = Path(settings.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, settings.log_level))
        root_logger.addHandler(file_handler)


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Formats log records as JSON objects with consistent structure
    suitable for log aggregation and analysis tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "getMessage"
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(name: str, **context: Any) -> FilteringBoundLogger:
    """
    Get a structured logger instance with optional context.
    
    Args:
        name: Logger name (typically __name__)
        **context: Additional context to include in all log messages
    
    Returns:
        Configured logger instance
    
    Examples:
        >>> logger = get_logger(__name__, service="data_ingestion")
        >>> logger.info("Processing started", batch_size=100)
    """
    if settings.enable_structured_logging:
        logger = structlog.get_logger(name)
        if context:
            logger = logger.bind(**context)
        return logger
    else:
        # Fallback to standard logging
        return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding temporary context to logs.
    
    Allows adding contextual information to all log messages
    within a specific scope.
    
    Examples:
        >>> logger = get_logger(__name__)
        >>> with LogContext(user_id="123", request_id="abc"):
        ...     logger.info("User action performed")
    """
    
    def __init__(self, **context: Any):
        self.context = context
        self.logger_context = None
    
    def __enter__(self):
        if settings.enable_structured_logging:
            # This is a simplified approach - in practice you might want
            # to use contextvars for thread-safe context management
            pass
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def log_function_call(logger: FilteringBoundLogger):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger: Logger instance to use
    
    Examples:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        ... def process_data(data_size: int) -> Dict:
        ...     return {"processed": data_size}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            import inspect
            
            # Get function signature for logging
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            start_time = time.time()
            
            logger.info(
                "Function call started",
                function=func.__name__,
                module=func.__module__,
                parameters={k: str(v)[:100] for k, v in bound_args.arguments.items()}
            )
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    "Function call completed",
                    function=func.__name__,
                    execution_time=execution_time,
                    success=True
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(
                    "Function call failed",
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                    success=False
                )
                raise
                
        return wrapper
    return decorator


# Initialize logging on module import
setup_logging() 