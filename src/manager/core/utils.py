"""
Shared utility functions for the manager system
"""

from datetime import datetime, timedelta
from typing import Union, Optional
import re


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
        precision: 'auto', 'seconds', 'minutes', 'hours', 'days'
        
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
        format: 'short', 'long', 'iso', 'relative'
        
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


def format_percentage(value: float, decimals: int = 1, show_sign: bool = False) -> str:
    """Format value as percentage.
    
    Args:
        value: Value to format (0.5 = 50%)
        decimals: Number of decimal places
        show_sign: Whether to show + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    percentage = value * 100
    formatted = f"{percentage:.{decimals}f}%"
    
    if show_sign and value > 0:
        formatted = f"+{formatted}"
    
    return formatted


def format_bytes(bytes: int, precision: int = 2) -> str:
    """Format bytes in human-readable format.
    
    Args:
        bytes: Number of bytes
        precision: Decimal precision
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.{precision}f} {unit}"
        bytes /= 1024.0
    
    return f"{bytes:.{precision}f} PB"


def truncate_string(text: str, max_length: int, ellipsis: str = "...") -> str:
    """Truncate string to maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including ellipsis
        ellipsis: String to append when truncating
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(ellipsis)] + ellipsis


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