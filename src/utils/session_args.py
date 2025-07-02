"""
Standard Session Arguments for Minotaur MCTS System

Provides unified command-line argument patterns for session handling
across all tools and scripts.
"""

import argparse
from typing import Optional


def add_session_argument(parser: argparse.ArgumentParser, 
                        required: bool = True,
                        allow_latest: bool = True,
                        help_suffix: str = "") -> None:
    """
    Add standardized --session argument to ArgumentParser.
    
    Args:
        parser: ArgumentParser instance to add argument to
        required: Whether session argument is required
        allow_latest: Whether to mention "latest" keyword in help
        help_suffix: Additional help text to append
    """
    help_text = "Session identifier: UUID (full/partial), session name"
    
    if allow_latest:
        help_text += ', or "latest"'
        
    if help_suffix:
        help_text += f". {help_suffix}"
        
    parser.add_argument(
        '--session',
        type=str,
        required=required,
        help=help_text
    )


def add_multiple_sessions_argument(parser: argparse.ArgumentParser,
                                 min_sessions: int = 2,
                                 help_suffix: str = "") -> None:
    """
    Add argument for multiple sessions (e.g., for comparison commands).
    
    Args:
        parser: ArgumentParser instance to add argument to
        min_sessions: Minimum number of sessions required
        help_suffix: Additional help text to append
    """
    help_text = f"Session identifiers (minimum {min_sessions}): UUIDs (full/partial) or session names"
    
    if help_suffix:
        help_text += f". {help_suffix}"
        
    parser.add_argument(
        '--sessions',
        type=str,
        nargs='+',
        required=True,
        help=help_text
    )


def add_optional_session_argument(parser: argparse.ArgumentParser,
                                help_suffix: str = "") -> None:
    """
    Add optional --session argument with automatic fallback to latest.
    
    Args:
        parser: ArgumentParser instance to add argument to  
        help_suffix: Additional help text to append
    """
    help_text = "Session identifier (optional, defaults to latest): UUID (full/partial), session name, or \"latest\""
    
    if help_suffix:
        help_text += f". {help_suffix}"
        
    parser.add_argument(
        '--session',
        type=str,
        required=False,
        default='latest',
        help=help_text
    )


def add_legacy_resume_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add legacy --resume argument for backwards compatibility.
    Maps to --session internally.
    
    Args:
        parser: ArgumentParser instance to add argument to
    """
    parser.add_argument(
        '--resume',
        type=str,
        nargs='?',
        const='latest',
        metavar='SESSION_ID',
        help='[DEPRECATED] Resume session: --resume (latest) or --resume SESSION_ID. Use --session instead.'
    )


def resolve_session_from_args(args) -> Optional[str]:
    """
    Extract session identifier from parsed arguments.
    Handles both new --session and legacy --resume arguments.
    
    Args:
        args: Parsed arguments from ArgumentParser
        
    Returns:
        Session identifier string or None if not provided
    """
    # Priority: --session > --resume > None
    if hasattr(args, 'session') and args.session:
        return args.session
        
    if hasattr(args, 'resume') and args.resume is not None:
        # --resume with no argument means 'latest'
        return args.resume if args.resume else 'latest'
        
    return None


def resolve_multiple_sessions_from_args(args) -> Optional[list]:
    """
    Extract multiple session identifiers from parsed arguments.
    
    Args:
        args: Parsed arguments from ArgumentParser
        
    Returns:
        List of session identifiers or None if not provided
    """
    if hasattr(args, 'sessions') and args.sessions:
        return args.sessions
        
    return None


def validate_session_args(args) -> str:
    """
    Validate and extract session identifier from arguments.
    Raises appropriate errors for missing or invalid arguments.
    
    Args:
        args: Parsed arguments from ArgumentParser
        
    Returns:
        Valid session identifier
        
    Raises:
        SystemExit: If session argument is missing or invalid
    """
    session_id = resolve_session_from_args(args)
    
    if not session_id:
        print("❌ Error: Session identifier is required")
        print("💡 Use --session SESSION_ID or --session latest")
        raise SystemExit(1)
        
    return session_id


def validate_optional_session_args(args) -> str:
    """
    Validate and extract session identifier from arguments with optional fallback.
    Always returns a valid session identifier (defaults to 'latest').
    
    Args:
        args: Parsed arguments from ArgumentParser
        
    Returns:
        Valid session identifier (defaults to 'latest' if not provided)
    """
    session_id = resolve_session_from_args(args)
    
    # Default to 'latest' if no session provided
    if not session_id:
        session_id = 'latest'
        
    return session_id


def validate_multiple_sessions_args(args, min_sessions: int = 2) -> list:
    """
    Validate and extract multiple session identifiers from arguments.
    
    Args:
        args: Parsed arguments from ArgumentParser
        min_sessions: Minimum number of sessions required
        
    Returns:
        List of valid session identifiers
        
    Raises:
        SystemExit: If insufficient sessions provided
    """
    sessions = resolve_multiple_sessions_from_args(args)
    
    if not sessions:
        print(f"❌ Error: At least {min_sessions} session identifiers required")
        print("💡 Use --sessions SESSION1 SESSION2 ...")
        raise SystemExit(1)
        
    if len(sessions) < min_sessions:
        print(f"❌ Error: At least {min_sessions} sessions required, got {len(sessions)}")
        raise SystemExit(1)
        
    return sessions


# Convenience functions for common patterns
def add_standard_session_args(parser: argparse.ArgumentParser) -> None:
    """Add standard session argument for most tools."""
    add_session_argument(parser, required=True, allow_latest=True)


def add_analysis_session_args(parser: argparse.ArgumentParser) -> None:
    """Add session arguments for analysis tools (optional with latest fallback)."""
    add_optional_session_argument(parser, "Analysis will use most recent session if not specified")


def add_universal_optional_session_args(parser: argparse.ArgumentParser, 
                                       tool_name: str = "Tool") -> None:
    """Add universal optional session argument - defaults to latest if not specified."""
    help_text = f"Session identifier (optional): UUID (full/partial), session name, or \"latest\". {tool_name} will use most recent session if not specified"
    
    parser.add_argument(
        '--session',
        type=str,
        required=False,
        default='latest',
        help=help_text
    )


def add_comparison_session_args(parser: argparse.ArgumentParser, min_sessions: int = 2) -> None:
    """Add session arguments for comparison tools."""
    add_multiple_sessions_argument(parser, min_sessions, "All sessions must be same identifier type")


def add_mcts_session_args(parser: argparse.ArgumentParser) -> None:
    """Add session arguments for MCTS runner (supports both new and legacy)."""
    add_legacy_resume_argument(parser)  # For backwards compatibility
    add_session_argument(parser, required=False, allow_latest=True, 
                        help_suffix="Preferred over --resume")