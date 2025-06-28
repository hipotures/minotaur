"""
Database logging configuration.

This module sets up database-specific logging that respects the log level
configured in mcts_config.yaml while providing detailed database operation tracking.
"""

import logging
import sys
from typing import Dict, Any
from pathlib import Path

class SafeFormatter(logging.Formatter):
    """Custom formatter that provides default values for missing fields."""
    
    def format(self, record):
        # Provide default values for missing fields
        if not hasattr(record, 'database_context'):
            record.database_context = 'db'
        if not hasattr(record, 'dataset_name'):
            record.dataset_name = 'dataset'
        
        return super().format(record)

def setup_db_logging(main_config: Dict[str, Any]) -> logging.Logger:
    """
    Setup database logging based on main configuration.
    
    The logging level is controlled by mcts_config.yaml, but database-specific
    formatting and handlers are configured here.
    
    Args:
        main_config: Main MCTS configuration dictionary
        
    Returns:
        Configured logger for database operations
    """
    # Get logging configuration from main config
    logging_config = main_config.get('logging', {})
    log_level = logging_config.get('level', 'INFO').upper()
    
    # Create database logger
    logger = logging.getLogger('db')
    logger.setLevel(getattr(logging, log_level))
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Ensure logs directory exists
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure handlers based on log level
    if log_level == 'DEBUG':
        # DEBUG level: Detailed database operation logging
        debug_handler = logging.FileHandler('logs/db_debug.log')
        debug_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d\n'
            'Message: %(message)s\n'
            '%(query)s%(params)s%(duration)s%(separator)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        debug_handler.setFormatter(debug_formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)
        
        # Also add console handler for DEBUG
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - DB - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)  # Only INFO and above to console
        logger.addHandler(console_handler)
        
    else:
        # INFO level and above: Summary logging only
        info_handler = logging.FileHandler('logs/db.log')
        info_formatter = SafeFormatter(
            '%(asctime)s.%(msecs)03d - [%(database_context)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        info_handler.setFormatter(info_formatter)
        info_handler.setLevel(logging.INFO)
        logger.addHandler(info_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger

def log_query(logger: logging.Logger, query: str, params: tuple = None, 
              duration: float = None, level: str = 'DEBUG') -> None:
    """
    Log database query with appropriate detail level.
    
    Args:
        logger: Database logger instance
        query: SQL query string
        params: Query parameters tuple
        duration: Query execution time in seconds
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    log_level = getattr(logging, level.upper())
    
    if logger.isEnabledFor(logging.DEBUG):
        # DEBUG level: Log full query details
        extra = {
            'query': f'\nQuery: {query}',
            'params': f'\nParams: {params}' if params else '',
            'duration': f'\nDuration: {duration:.3f}s' if duration else '',
            'separator': '\n' + '-' * 80
        }
        logger.log(log_level, f"Database operation executed", extra=extra)
    elif logger.isEnabledFor(logging.INFO) and level in ['INFO', 'WARNING', 'ERROR']:
        # INFO level: Log summary only
        if duration:
            logger.log(log_level, f"Query executed in {duration:.3f}s")
        else:
            logger.log(log_level, "Database operation completed")

def log_transaction(logger: logging.Logger, operation: str, success: bool, 
                   duration: float = None, error: str = None) -> None:
    """
    Log database transaction outcome.
    
    Args:
        logger: Database logger instance
        operation: Transaction operation description
        success: Whether transaction succeeded
        duration: Transaction duration in seconds
        error: Error message if transaction failed
    """
    if success:
        message = f"Transaction '{operation}' completed successfully"
        if duration:
            message += f" in {duration:.3f}s"
        logger.info(message)
    else:
        message = f"Transaction '{operation}' failed"
        if error:
            message += f": {error}"
        logger.error(message)

def log_connection_event(logger: logging.Logger, event: str, details: str = None) -> None:
    """
    Log connection pool events.
    
    Args:
        logger: Database logger instance
        event: Event type ('acquired', 'released', 'created', 'closed', 'error')
        details: Additional event details
    """
    if logger.isEnabledFor(logging.DEBUG):
        message = f"Connection {event}"
        if details:
            message += f": {details}"
        logger.debug(message)
    elif event == 'error':
        logger.error(f"Connection error: {details}")

def log_performance_metric(logger: logging.Logger, metric_name: str, 
                          value: float, unit: str = '') -> None:
    """
    Log performance metrics.
    
    Args:
        logger: Database logger instance
        metric_name: Name of the performance metric
        value: Metric value
        unit: Unit of measurement
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Performance metric - {metric_name}: {value}{unit}")

class DatabaseLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds database-specific context to log messages.
    
    This adapter automatically includes query information in DEBUG mode
    and provides convenient methods for database operation logging.
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Add database context to log records."""
        # Extract database name from path for context
        db_path = self.extra.get('db_path', 'unknown')
        if db_path != 'unknown':
            # Extract database name from path (e.g., data/minotaur.duckdb -> minotaur)
            from pathlib import Path
            db_name = Path(db_path).stem
        else:
            db_name = 'unknown'
        
        # For LoggerAdapter, we need to add the context to 'extra' in kwargs
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['database_context'] = db_name
        
        return msg, kwargs
    
    def query(self, query: str, params: tuple = None, duration: float = None) -> None:
        """Log a database query."""
        log_query(self.logger, query, params, duration)
    
    def transaction(self, operation: str, success: bool, duration: float = None, 
                   error: str = None) -> None:
        """Log a database transaction."""
        log_transaction(self.logger, operation, success, duration, error)
    
    def connection_event(self, event: str, details: str = None) -> None:
        """Log a connection pool event."""
        log_connection_event(self.logger, event, details)
    
    def performance(self, metric_name: str, value: float, unit: str = '') -> None:
        """Log a performance metric."""
        log_performance_metric(self.logger, metric_name, value, unit)