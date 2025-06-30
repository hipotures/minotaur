"""
Unified formatting utilities for the Minotaur project.

This module consolidates all formatting functions from across the codebase
to eliminate duplication and provide a single source of truth.
"""

from datetime import datetime, timedelta
from typing import Union, Optional, Any
import re
import json


def format_number(number: Union[int, float], decimals: int = 2) -> str:
    """Format number with thousands separator and optional decimals.
    
    Args:
        number: Number to format
        decimals: Number of decimal places for floats
        
    Returns:
        Formatted string
    """
    if isinstance(number, int):
        return f"{number:,}"
    else:
        return f"{number:,.{decimals}f}"


def format_duration(seconds: float, precision: str = 'auto') -> str:
    """Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        precision: 'auto', 'seconds', 'minutes', 'hours', 'days', 'full'
        
    Returns:
        Formatted duration string
    """
    if seconds < 0:
        return "N/A"
    
    if precision == 'auto':
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"
    elif precision == 'seconds':
        return f"{seconds:.1f}s"
    elif precision == 'minutes':
        return f"{seconds/60:.1f}m"
    elif precision == 'hours':
        return f"{seconds/3600:.1f}h"
    elif precision == 'days':
        return f"{seconds/86400:.1f}d"
    else:
        # Full format
        td = timedelta(seconds=int(seconds))
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")
        
        return " ".join(parts)


def format_datetime(dt: Union[datetime, str], format: str = 'short') -> str:
    """Format datetime in various formats.
    
    Args:
        dt: Datetime object or ISO string
        format: 'short', 'long', 'iso', 'relative', 'date_only'
        
    Returns:
        Formatted datetime string
    """
    if isinstance(dt, str):
        # Parse ISO format
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    
    if format == 'short':
        return dt.strftime('%Y-%m-%d %H:%M')
    elif format == 'long':
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    elif format == 'date_only':
        return dt.strftime('%Y-%m-%d')
    elif format == 'iso':
        return dt.isoformat()
    elif format == 'relative':
        now = datetime.now(dt.tzinfo)
        delta = now - dt
        
        if delta.days > 365:
            years = delta.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        elif delta.days > 30:
            months = delta.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        elif delta.days > 0:
            return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"
    else:
        return str(dt)


def format_percentage(value: float, decimals: int = 1, show_sign: bool = False, 
                     as_ratio: bool = True) -> str:
    """Format value as percentage.
    
    Args:
        value: Value to format (0.5 = 50% if as_ratio=True, 50 = 50% if as_ratio=False)
        decimals: Number of decimal places
        show_sign: Whether to show + sign for positive values
        as_ratio: Whether input is a ratio (0-1) or already percentage (0-100)
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    
    percentage = value * 100 if as_ratio else value
    formatted = f"{percentage:.{decimals}f}%"
    
    if show_sign and value > 0:
        formatted = f"+{formatted}"
    
    return formatted


def format_bytes(bytes_size: int, precision: int = 2) -> str:
    """Format bytes in human-readable format.
    
    Args:
        bytes_size: Number of bytes
        precision: Decimal precision
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.{precision}f} {unit}"
        bytes_size /= 1024.0
    
    return f"{bytes_size:.{precision}f} PB"


def format_file_size(bytes_size: int) -> str:
    """Alias for format_bytes with default precision of 1."""
    return format_bytes(bytes_size, precision=1)


def format_impact(impact: float, precision: int = 5) -> str:
    """Format impact value with appropriate sign and precision.
    
    Args:
        impact: Impact value to format
        precision: Number of decimal places
        
    Returns:
        Formatted impact string
    """
    if impact is None:
        return "N/A"
    
    if impact == 0:
        return f"0.{'0' * precision}"
    
    return f"{impact:+.{precision}f}"


def format_score(score: float, precision: int = 5) -> str:
    """Format score value with specified precision.
    
    Args:
        score: Score value to format
        precision: Number of decimal places
        
    Returns:
        Formatted score string
    """
    if score is None:
        return "N/A"
    
    return f"{score:.{precision}f}"


def format_status(status: str, use_emoji: bool = True) -> str:
    """Format status with optional emoji.
    
    Args:
        status: Status string (completed, failed, running, pending)
        use_emoji: Whether to include emoji
        
    Returns:
        Formatted status string
    """
    status_map = {
        'completed': ('✅', 'Completed'),
        'failed': ('❌', 'Failed'),
        'running': ('🔄', 'Running'),
        'pending': ('⏳', 'Pending'),
        'interrupted': ('⚠️', 'Interrupted'),
        'active': ('🟢', 'Active'),
        'inactive': ('🔴', 'Inactive')
    }
    
    status_lower = status.lower()
    if status_lower in status_map:
        emoji, text = status_map[status_lower]
        return f"{emoji} {text}" if use_emoji else text
    
    return status


def truncate_string(text: str, max_length: int, ellipsis: str = "...") -> str:
    """Truncate string to maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including ellipsis
        ellipsis: String to append when truncating
        
    Returns:
        Truncated string
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(ellipsis)] + ellipsis


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Alias for truncate_string for compatibility."""
    return truncate_string(text, max_length, suffix)


def parse_time_range(time_str: str) -> Optional[timedelta]:
    """Parse time range string to timedelta.
    
    Args:
        time_str: Time string like "1d", "2h", "30m", "1w"
        
    Returns:
        timedelta object or None if invalid
    """
    pattern = r'^(\d+)([dhmsw])$'
    match = re.match(pattern, time_str.lower())
    
    if not match:
        return None
    
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit == 's':
        return timedelta(seconds=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    elif unit == 'w':
        return timedelta(weeks=value)
    
    return None


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Sanitize filename by replacing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, replacement)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"
    
    return filename


def format_list(items: list, separator: str = ", ", max_items: int = None,
                overflow_suffix: str = "...") -> str:
    """Format list of items as string with optional truncation.
    
    Args:
        items: List of items to format
        separator: String to join items with
        max_items: Maximum number of items to show
        overflow_suffix: String to append when truncated
        
    Returns:
        Formatted string
    """
    if not items:
        return ""
    
    str_items = [str(item) for item in items]
    
    if max_items and len(str_items) > max_items:
        displayed = str_items[:max_items]
        remaining = len(str_items) - max_items
        return f"{separator.join(displayed)} {overflow_suffix} (+{remaining} more)"
    
    return separator.join(str_items)


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as JSON string with proper handling of special types.
    
    Args:
        data: Data to format
        indent: JSON indent level
        
    Returns:
        Formatted JSON string
    """
    try:
        return json.dumps(data, indent=indent, default=str)
    except Exception:
        return str(data)


def format_table_row(values: list, widths: list, separator: str = " | ") -> str:
    """Format a single table row with specified column widths.
    
    Args:
        values: List of cell values
        widths: List of column widths
        separator: Column separator
        
    Returns:
        Formatted row string
    """
    formatted_cells = []
    for i, value in enumerate(values):
        if i < len(widths):
            formatted_cells.append(str(value).ljust(widths[i]))
        else:
            formatted_cells.append(str(value))
    
    return separator.join(formatted_cells)


def calculate_column_widths(headers: list, rows: list, min_width: int = 10) -> list:
    """Calculate optimal column widths for table display.
    
    Args:
        headers: List of header values
        rows: List of row data (list of lists)
        min_width: Minimum column width
        
    Returns:
        List of column widths
    """
    col_widths = []
    
    for i, header in enumerate(headers):
        max_width = max(len(str(header)), min_width)
        for row in rows:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width)
    
    return col_widths