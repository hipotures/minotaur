"""
Utility modules for Minotaur MCTS System.

Provides common functionality used across the system including:
- Session resolution and identification
- Standard command-line argument patterns
- Common helper functions
"""

from .session_resolver import SessionResolver, SessionInfo, SessionResolutionError, create_session_resolver
from .session_args import (
    add_session_argument,
    add_multiple_sessions_argument, 
    add_optional_session_argument,
    add_legacy_resume_argument,
    resolve_session_from_args,
    resolve_multiple_sessions_from_args,
    validate_session_args,
    validate_optional_session_args,
    validate_multiple_sessions_args,
    add_standard_session_args,
    add_analysis_session_args,
    add_universal_optional_session_args,
    add_comparison_session_args,
    add_mcts_session_args
)

__all__ = [
    # Session resolution
    'SessionResolver',
    'SessionInfo', 
    'SessionResolutionError',
    'create_session_resolver',
    
    # Session arguments
    'add_session_argument',
    'add_multiple_sessions_argument',
    'add_optional_session_argument', 
    'add_legacy_resume_argument',
    'resolve_session_from_args',
    'resolve_multiple_sessions_from_args',
    'validate_session_args',
    'validate_optional_session_args',
    'validate_multiple_sessions_args',
    'add_standard_session_args',
    'add_analysis_session_args',
    'add_universal_optional_session_args',
    'add_comparison_session_args',
    'add_mcts_session_args'
]