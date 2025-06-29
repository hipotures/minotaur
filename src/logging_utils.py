"""
Logging utilities for MCTS Feature Discovery System.

Provides session-aware logging capabilities using Python's contextvars
for thread-safe session tracking across the application.
"""

import logging
from contextvars import ContextVar
from typing import Optional

# Thread-safe context variable for storing session name
_session_context: ContextVar[Optional[str]] = ContextVar('session_name', default=None)


class SessionFilter(logging.Filter):
    """
    Logging filter that adds session name to log records.
    
    This filter enriches log records with the current session name from the context,
    allowing all log messages to include session identification for easy filtering.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add session_name to the log record."""
        # Get session name from context, default to "no_session" if not set
        session_name = _session_context.get()
        record.session_name = session_name if session_name else "no_session"
        return True


def set_session_context(session_name: str) -> None:
    """
    Set the current session name in the logging context.
    
    Args:
        session_name: The session name to use for all subsequent log messages
    """
    _session_context.set(session_name)


def clear_session_context() -> None:
    """Clear the current session name from the logging context."""
    _session_context.set(None)


def get_session_context() -> Optional[str]:
    """
    Get the current session name from the logging context.
    
    Returns:
        The current session name or None if not set
    """
    return _session_context.get()


def setup_session_logging(root_logger: Optional[logging.Logger] = None) -> None:
    """
    Add SessionFilter to all handlers of the specified logger.
    
    Args:
        root_logger: Logger to configure. If None, uses the root logger.
    """
    if root_logger is None:
        root_logger = logging.getLogger()
    
    session_filter = SessionFilter()
    
    # Add filter to all existing handlers
    for handler in root_logger.handlers:
        handler.addFilter(session_filter)
    
    # Also add to any child loggers that have handlers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.addFilter(session_filter)


def setup_main_logging(config):
    """
    Setup main application logging to logs/minotaur.log.
    
    This is separate from database logging which goes to logs/db.log.
    """
    from pathlib import Path
    from logging.handlers import RotatingFileHandler
    
    log_config = config['logging']
    
    # Ensure logs directory exists
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Main application log format with session context
    log_format = '%(asctime)s - [%(session_name)s] - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config['level']))
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler for main application logs
    file_handler = RotatingFileHandler(
        log_config['log_file'],
        maxBytes=log_config['max_log_size_mb'] * 1024 * 1024,
        backupCount=log_config['backup_count']
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Only file handler - no console output
    root_logger.addHandler(file_handler)
    
    # Reduce verbosity of some libraries
    logging.getLogger('autogluon').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Setup session-aware logging filters
    setup_session_logging()
    
    # Get main logger and log initialization
    logger = logging.getLogger(__name__)
    logger.info("Main application logging configuration initialized")
    
    return root_logger


def setup_dataset_logging(dataset_name: str, config):
    """
    Setup contextual logging for dataset operations that goes to main log.
    
    This creates a logger that uses the dataset name as session context
    and logs to the main application log (minotaur.log), not database log.
    """
    # Set the dataset name as session context
    original_session = get_session_context()
    set_session_context(dataset_name)
    
    # Get logger that will use the main application logging system
    logger = logging.getLogger('dataset_importer')
    
    # Return a wrapper that restores original session context when done
    class DatasetLogger:
        def __init__(self, logger, original_session):
            self.logger = logger
            self.original_session = original_session
        
        def info(self, msg, *args, **kwargs):
            self.logger.info(msg, *args, **kwargs)
        
        def warning(self, msg, *args, **kwargs):
            self.logger.warning(msg, *args, **kwargs)
        
        def error(self, msg, *args, **kwargs):
            self.logger.error(msg, *args, **kwargs)
        
        def debug(self, msg, *args, **kwargs):
            self.logger.debug(msg, *args, **kwargs)
        
        def __del__(self):
            # Restore original session context when logger is destroyed
            set_session_context(self.original_session)
    
    return DatasetLogger(logger, original_session)