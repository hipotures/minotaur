"""
Console output utilities for standardized messaging.

This module provides consistent console output formatting with emoji support
and color-coded messages for different message types.
"""

import sys
from typing import Any, List, Optional
from .formatting import format_json, format_table_row, calculate_column_widths


class Console:
    """Unified console output manager."""
    
    # Standard emoji mappings for different message types
    EMOJIS = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'debug': '🐛',
        'question': '❓',
        'bullet': '•',
        'arrow': '→',
        'check': '✓',
        'cross': '✗',
        'star': '⭐',
        'fire': '🔥',
        'rocket': '🚀',
        'clock': '⏰',
        'folder': '📁',
        'file': '📄',
        'database': '🗄️',
        'chart': '📊',
        'gear': '⚙️',
        'magnifier': '🔍',
        'lightbulb': '💡',
        'package': '📦',
        'lock': '🔒',
        'key': '🔑',
        'shield': '🛡️',
        'running': '🔄',
        'pending': '⏳',
        'completed': '✅',
        'failed': '❌',
        'active': '🟢',
        'inactive': '🔴',
        'interrupted': '⚠️'
    }
    
    @classmethod
    def print_success(cls, message: str, emoji: bool = True) -> None:
        """Print success message with green checkmark."""
        prefix = f"{cls.EMOJIS['success']} " if emoji else ""
        print(f"{prefix}{message}")
    
    @classmethod
    def print_error(cls, message: str, emoji: bool = True) -> None:
        """Print error message with red X."""
        prefix = f"{cls.EMOJIS['error']} " if emoji else ""
        print(f"{prefix}{message}", file=sys.stderr)
    
    @classmethod
    def print_warning(cls, message: str, emoji: bool = True) -> None:
        """Print warning message with yellow triangle."""
        prefix = f"{cls.EMOJIS['warning']}  " if emoji else ""
        print(f"{prefix}{message}")
    
    @classmethod
    def print_info(cls, message: str, emoji: bool = True) -> None:
        """Print info message with blue info icon."""
        prefix = f"{cls.EMOJIS['info']}  " if emoji else ""
        print(f"{prefix}{message}")
    
    @classmethod
    def print_debug(cls, message: str, emoji: bool = True) -> None:
        """Print debug message with bug icon."""
        prefix = f"{cls.EMOJIS['debug']} " if emoji else ""
        print(f"{prefix}{message}")
    
    @classmethod
    def print_question(cls, message: str, emoji: bool = True) -> None:
        """Print question message."""
        prefix = f"{cls.EMOJIS['question']} " if emoji else ""
        print(f"{prefix}{message}")
    
    @classmethod
    def print_bullet(cls, message: str, level: int = 0, emoji: bool = True) -> None:
        """Print bullet point message with indentation."""
        indent = "  " * level
        prefix = f"{cls.EMOJIS['bullet']} " if emoji else "- "
        print(f"{indent}{prefix}{message}")
    
    @classmethod
    def print_status(cls, status: str, message: str, emoji: bool = True) -> None:
        """Print status message with appropriate emoji."""
        status_lower = status.lower()
        if emoji and status_lower in cls.EMOJIS:
            prefix = f"{cls.EMOJIS[status_lower]} "
        else:
            prefix = f"[{status.upper()}] "
        print(f"{prefix}{message}")
    
    @classmethod
    def print_header(cls, title: str, char: str = "=", emoji: Optional[str] = None) -> None:
        """Print section header with optional emoji."""
        if emoji and emoji in cls.EMOJIS:
            title = f"{cls.EMOJIS[emoji]} {title}"
        
        print(f"\n{title}")
        print(char * len(title))
    
    @classmethod
    def print_subheader(cls, title: str, char: str = "-", emoji: Optional[str] = None) -> None:
        """Print subsection header with optional emoji."""
        if emoji and emoji in cls.EMOJIS:
            title = f"{cls.EMOJIS[emoji]} {title}"
        
        print(f"\n{title}")
        print(char * len(title))
    
    @classmethod
    def print_json(cls, data: Any, title: Optional[str] = None, emoji: bool = True) -> None:
        """Print data in JSON format with optional title."""
        if title:
            prefix = f"{cls.EMOJIS['file']} " if emoji else ""
            print(f"\n{prefix}{title.upper()}")
            print("=" * (len(title) + (4 if emoji else 0)))
        
        print(format_json(data))
    
    @classmethod
    def print_table(cls, headers: List[str], rows: List[List[str]], 
                   min_width: int = 10, title: Optional[str] = None) -> None:
        """Print data in formatted table.
        
        Args:
            headers: List of column headers
            rows: List of row data (list of lists)
            min_width: Minimum column width
            title: Optional table title
        """
        if not headers or not rows:
            return
        
        if title:
            cls.print_header(title, emoji='chart')
            print()
        
        # Calculate column widths
        col_widths = calculate_column_widths(headers, rows, min_width)
        
        # Print header
        header_row = format_table_row(headers, col_widths)
        print(header_row)
        print("-" * len(header_row))
        
        # Print rows
        for row in rows:
            print(format_table_row(row, col_widths))
    
    @classmethod
    def print_separator(cls, char: str = "-", width: int = 80) -> None:
        """Print horizontal separator line."""
        print(char * width)
    
    @classmethod
    def print_blank_line(cls, count: int = 1) -> None:
        """Print blank line(s)."""
        for _ in range(count):
            print()
    
    @classmethod
    def print_progress(cls, current: int, total: int, message: str = "", 
                      bar_width: int = 40, emoji: bool = True) -> None:
        """Print progress bar.
        
        Args:
            current: Current progress value
            total: Total value
            message: Optional message to display
            bar_width: Width of the progress bar
            emoji: Whether to use emoji
        """
        if total == 0:
            percentage = 0
        else:
            percentage = (current / total) * 100
        
        filled = int(bar_width * current // total)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        prefix = f"{cls.EMOJIS['running']} " if emoji else ""
        status = f"{prefix}[{bar}] {percentage:.1f}%"
        
        if message:
            status += f" - {message}"
        
        # Use carriage return to update the same line
        print(f"\r{status}", end="", flush=True)
        
        # Print newline when complete
        if current >= total:
            print()
    
    @classmethod
    def print_tree(cls, items: List[tuple], indent: str = "  ", 
                  last_item_chars: tuple = ("└─", "├─"), 
                  continuation_char: str = "│") -> None:
        """Print items in tree structure.
        
        Args:
            items: List of (level, text) tuples
            indent: Indentation string
            last_item_chars: Characters for last and non-last items
            continuation_char: Character for tree continuation
        """
        for i, (level, text) in enumerate(items):
            # Check if this is the last item at this level
            is_last = (i == len(items) - 1 or 
                      (i < len(items) - 1 and items[i + 1][0] <= level))
            
            # Build the prefix
            prefix = ""
            for l in range(level):
                if l < level - 1:
                    prefix += continuation_char + indent
                else:
                    prefix += last_item_chars[0] if is_last else last_item_chars[1]
                    prefix += " "
            
            print(f"{prefix}{text}")
    
    @classmethod
    def confirm(cls, message: str, default: bool = False, emoji: bool = True) -> bool:
        """Ask for user confirmation.
        
        Args:
            message: Confirmation message
            default: Default answer if user presses Enter
            emoji: Whether to use emoji
            
        Returns:
            True if confirmed, False otherwise
        """
        prefix = f"{cls.EMOJIS['question']} " if emoji else ""
        suffix = " [Y/n]" if default else " [y/N]"
        
        try:
            response = input(f"{prefix}{message}{suffix}: ").strip().lower()
            
            if not response:
                return default
            
            return response in ('y', 'yes', 'true', '1')
        except (KeyboardInterrupt, EOFError):
            print()  # New line after interrupt
            return False
    
    @classmethod
    def prompt(cls, message: str, default: Optional[str] = None, 
              emoji: bool = True) -> Optional[str]:
        """Prompt user for input.
        
        Args:
            message: Prompt message
            default: Default value if user presses Enter
            emoji: Whether to use emoji
            
        Returns:
            User input or default value
        """
        prefix = f"{cls.EMOJIS['arrow']} " if emoji else ""
        suffix = f" [{default}]" if default else ""
        
        try:
            response = input(f"{prefix}{message}{suffix}: ").strip()
            return response if response else default
        except (KeyboardInterrupt, EOFError):
            print()  # New line after interrupt
            return None


# Convenience functions for backward compatibility
def print_success(message: str, emoji: bool = True) -> None:
    """Print success message."""
    Console.print_success(message, emoji)


def print_error(message: str, emoji: bool = True) -> None:
    """Print error message."""
    Console.print_error(message, emoji)


def print_warning(message: str, emoji: bool = True) -> None:
    """Print warning message."""
    Console.print_warning(message, emoji)


def print_info(message: str, emoji: bool = True) -> None:
    """Print info message."""
    Console.print_info(message, emoji)


def print_table(headers: List[str], rows: List[List[str]], **kwargs) -> None:
    """Print formatted table."""
    Console.print_table(headers, rows, **kwargs)