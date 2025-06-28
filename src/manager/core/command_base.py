"""
Base Command - Common functionality for all module commands.

Provides shared utilities including:
- Output formatting and styling
- JSON output handling
- Table printing utilities
- Error and status messaging
- Common data formatting methods
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json


class BaseCommand(ABC):
    """Base class for all module commands."""
    
    def __init__(self):
        self.config = None
        self.db_pool = None
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """Inject required services (config, db_pool, etc.)."""
        self.config = services.get('config')
        self.db_pool = services.get('db_pool')
    
    @abstractmethod
    def execute(self, args) -> None:
        """Execute the command with given arguments."""
        pass
    
    # Output formatting utilities
    def print_success(self, message: str) -> None:
        """Print success message with green checkmark."""
        print(f"✅ {message}")
    
    def print_error(self, message: str) -> None:
        """Print error message with red X."""
        print(f"❌ {message}")
    
    def print_warning(self, message: str) -> None:
        """Print warning message with yellow triangle."""
        print(f"⚠️  {message}")
    
    def print_info(self, message: str) -> None:
        """Print info message with blue info icon."""
        print(f"ℹ️  {message}")
    
    def print_json(self, data: Any, title: str = None) -> None:
        """Print data in JSON format with optional title."""
        if title:
            print(f"📄 {title.upper()}")
            print("=" * (len(title) + 4))
        
        try:
            print(json.dumps(data, indent=2, default=str))
        except Exception as e:
            self.print_error(f"Failed to format JSON output: {e}")
    
    def print_table(self, headers: List[str], rows: List[List[str]], 
                   min_width: int = 10) -> None:
        """Print data in formatted table."""
        if not headers or not rows:
            return
        
        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = max(len(header), min_width)
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width)
        
        # Print header
        header_row = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
        print(header_row)
        print("-" * len(header_row))
        
        # Print rows
        for row in rows:
            formatted_row = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    formatted_row.append(str(cell).ljust(col_widths[i]))
                else:
                    formatted_row.append(str(cell))
            print(" | ".join(formatted_row))
    
    # Data formatting utilities
    def truncate_text(self, text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to specified length with suffix."""
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    def format_impact(self, impact: float) -> str:
        """Format impact value with appropriate sign and precision."""
        if impact is None:
            return "N/A"
        
        if impact == 0:
            return "0.00000"
        
        return f"{impact:+.5f}"
    
    def format_percentage(self, value: float, total: float) -> str:
        """Format percentage with one decimal place."""
        if total == 0:
            return "0.0%"
        
        percentage = (value / total) * 100
        return f"{percentage:.1f}%"
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def format_file_size(self, bytes_size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f}{unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f}TB"
    
    # Data validation utilities
    def validate_required_services(self, required: List[str]) -> None:
        """Validate that required services are available."""
        missing = []
        
        for service_name in required:
            if not hasattr(self, service_name) or getattr(self, service_name) is None:
                missing.append(service_name)
        
        if missing:
            raise ValueError(f"Missing required services: {', '.join(missing)}")
    
    def safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get value from dictionary with default."""
        try:
            return data.get(key, default)
        except (AttributeError, TypeError):
            return default