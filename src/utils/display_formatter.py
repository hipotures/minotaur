"""
Universal Display Formatter for Minotaur MCTS System

Provides unified formatting system with two modes:
- Rich mode (default): Beautiful formatting with Rich library and emoji
- Plain mode: Clean text output for sharing/debugging

Usage:
    # Set environment variable for plain mode
    export MINOTAUR_PLAIN_OUTPUT=1
    
    # Or in code
    import os
    os.environ['MINOTAUR_PLAIN_OUTPUT'] = '1'
"""

import os
import re
from typing import Any, Dict, List, Optional, Union
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Check environment variable for plain output mode
PLAIN_OUTPUT = os.environ.get('MINOTAUR_PLAIN_OUTPUT', '').lower() in ('1', 'true', 'yes', 'on')


class DisplayFormatter:
    """Universal formatter that adapts output based on environment."""
    
    def __init__(self, force_plain: bool = None):
        """
        Initialize formatter.
        
        Args:
            force_plain: Override environment setting for plain output
        """
        self.plain_mode = PLAIN_OUTPUT if force_plain is None else force_plain
        self.console = Console() if not self.plain_mode else None
        
    def is_plain_mode(self) -> bool:
        """Check if formatter is in plain text mode."""
        return self.plain_mode
    
    def emoji(self, emoji_char: str, fallback: str = "") -> str:
        """Return emoji in rich mode, fallback text in plain mode."""
        if self.plain_mode:
            return fallback
        return emoji_char
    
    def header(self, title: str, style: str = "bold blue") -> str:
        """Format a header title."""
        if self.plain_mode:
            clean_title = self._clean_text(title)
            return f"\n{clean_title}\n"
        else:
            return Panel.fit(title, style=style)
    
    def section_header(self, title: str, style: str = "bold green") -> str:
        """Format a section header."""
        if self.plain_mode:
            clean_title = self._clean_text(title)
            return f"\n{clean_title}\n"
        else:
            return Panel.fit(title, style=style)
    
    def table(self, data: List[Dict[str, Any]], title: str = "", 
              headers: Optional[List[str]] = None) -> str:
        """Format tabular data."""
        if not data:
            return self._format_text("No data to display", "italic")
            
        if self.plain_mode:
            return self._plain_table(data, title, headers)
        else:
            return self._rich_table(data, title, headers)
    
    def key_value_pairs(self, pairs: Dict[str, Any], title: str = "", 
                       title_style: str = "blue") -> Union[str, Table]:
        """Format key-value pairs."""
        if self.plain_mode:
            return self._plain_key_value(pairs, title)
        else:
            return self._rich_key_value(pairs, title, title_style)
    
    def info_block(self, content: str, title: str = "", 
                   border_style: str = "blue") -> str:
        """Format an information block."""
        if self.plain_mode:
            clean_content = self._clean_text(content)
            if title:
                clean_title = self._clean_text(title)
                return f"\n[{clean_title}]\n{clean_content}\n"
            return f"{clean_content}"
        else:
            if title:
                return Panel(content, title=title, border_style=border_style)
            return Panel(content, border_style=border_style)
    
    def success(self, message: str) -> str:
        """Format success message."""
        prefix = self.emoji("✅", "[SUCCESS]")
        return f"{prefix} {self._clean_text(message)}"
    
    def error(self, message: str) -> str:
        """Format error message."""
        prefix = self.emoji("❌", "[ERROR]")
        return f"{prefix} {self._clean_text(message)}"
    
    def warning(self, message: str) -> str:
        """Format warning message."""
        prefix = self.emoji("⚠️", "[WARNING]")
        return f"{prefix} {self._clean_text(message)}"
    
    def info(self, message: str) -> str:
        """Format info message."""
        prefix = self.emoji("ℹ️", "[INFO]")
        return f"{prefix} {self._clean_text(message)}"
    
    def progress(self, message: str) -> str:
        """Format progress message."""
        prefix = self.emoji("🔄", "[PROGRESS]")
        return f"{prefix} {self._clean_text(message)}"
    
    def print(self, content: Any, **kwargs) -> None:
        """Print content using appropriate method."""
        if self.plain_mode:
            if isinstance(content, str):
                print(self._clean_text(content))
            else:
                print(content)
        else:
            if self.console:
                self.console.print(content, **kwargs)
            else:
                print(content)
    
    def _clean_text(self, text: str) -> str:
        """Remove emoji and special characters for plain text output."""
        if not isinstance(text, str):
            return str(text)
            
        # Remove emoji (basic pattern - matches most common emoji)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        text = emoji_pattern.sub('', text)
        
        # Remove Rich markup tags (basic pattern)
        markup_pattern = re.compile(r'\[/?[^\]]*\]')
        text = markup_pattern.sub('', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _plain_table(self, data: List[Dict[str, Any]], title: str, 
                    headers: Optional[List[str]]) -> str:
        """Format table for plain text output."""
        if not data:
            return "No data"
            
        # Determine headers
        if headers is None:
            headers = list(data[0].keys()) if data else []
        
        # Calculate column widths
        col_widths = {}
        for header in headers:
            col_widths[header] = len(str(header))
            
        for row in data:
            for header in headers:
                value = str(row.get(header, ''))
                col_widths[header] = max(col_widths[header], len(value))
        
        # Build table
        result = []
        if title:
            result.append(f"\n{title}")
        
        # Header row
        header_row = " | ".join(h.ljust(col_widths[h]) for h in headers)
        result.append(header_row)
        
        # Data rows
        for row in data:
            data_row = " | ".join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers)
            result.append(data_row)
        
        return "\n".join(result)
    
    def _rich_table(self, data: List[Dict[str, Any]], title: str, 
                   headers: Optional[List[str]]) -> Table:
        """Format table for Rich output."""
        if headers is None:
            headers = list(data[0].keys()) if data else []
            
        table = Table(show_header=True, box=None)
        
        for header in headers:
            table.add_column(header, style="white")
        
        for row in data:
            table.add_row(*[str(row.get(h, '')) for h in headers])
        
        if title:
            return Panel(table, title=title, border_style="blue")
        return table
    
    def _plain_key_value(self, pairs: Dict[str, Any], title: str) -> str:
        """Format key-value pairs for plain text."""
        result = []
        if title:
            result.append(f"\n{self._clean_text(title)}")
        
        max_key_len = max(len(str(k)) for k in pairs.keys()) if pairs else 0
        
        for key, value in pairs.items():
            result.append(f"{str(key).ljust(max_key_len)}: {value}")
        
        return "\n".join(result)
    
    def _rich_key_value(self, pairs: Dict[str, Any], title: str, 
                       title_style: str) -> Table:
        """Format key-value pairs for Rich output."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in pairs.items():
            table.add_row(str(key), str(value))
        
        if title:
            return Panel(table, title=title, border_style=title_style)
        return table
    
    def _format_text(self, text: str, style: str = "") -> str:
        """Format text with optional style."""
        if self.plain_mode:
            return self._clean_text(text)
        else:
            if style:
                return Text(text, style=style)
            return text


# Global formatter instance
_global_formatter = DisplayFormatter()


def get_formatter(force_plain: bool = None) -> DisplayFormatter:
    """Get global formatter instance or create with specific mode."""
    if force_plain is not None:
        return DisplayFormatter(force_plain)
    return _global_formatter


def set_plain_mode(enabled: bool = True) -> None:
    """Set global plain mode for all formatters."""
    global _global_formatter
    os.environ['MINOTAUR_PLAIN_OUTPUT'] = '1' if enabled else '0'
    _global_formatter = DisplayFormatter()


def is_plain_mode() -> bool:
    """Check if global formatter is in plain mode."""
    return _global_formatter.is_plain_mode()