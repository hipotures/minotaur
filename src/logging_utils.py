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