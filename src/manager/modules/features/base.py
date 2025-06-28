"""
Base command class for features module.

Provides common functionality for all feature commands including:
- Service injection and access
- Common validation and utilities  
- Output formatting helpers
- Feature identifier resolution
- Dataset resolution utilities
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class BaseFeaturesCommand(ABC):
    """Base class for all features commands."""
    
    def __init__(self):
        self._services = {}
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """Inject required services."""
        self._services = services
    
    @property
    def feature_service(self):
        """Get feature service."""
        return self._services.get('feature_service')
    
    @property
    def session_service(self):
        """Get session service."""
        return self._services.get('session_service')
    
    @property
    def dataset_service(self):
        """Get dataset service."""
        return self._services.get('dataset_service')
    
    @abstractmethod
    def execute(self, args) -> None:
        """Execute the command with given arguments."""
        pass
    
    def resolve_dataset_identifier(self, identifier: str) -> Optional[str]:
        """
        Resolve dataset identifier (hash or name) to dataset hash.
        
        Args:
            identifier: Dataset hash or name
            
        Returns:
            Dataset hash if found, None otherwise
        """
        if not identifier:
            return None
        
        try:
            # Try direct hash lookup first
            datasets = self.dataset_service.repository.get_all_datasets(include_inactive=True)
            
            # Check if it's a hash (starts with matching characters)
            for dataset in datasets:
                if dataset['dataset_id'].startswith(identifier.lower()):
                    return dataset['dataset_id']
            
            # Check if it's a name
            for dataset in datasets:
                if dataset['dataset_name'] == identifier:
                    return dataset['dataset_id']
            
            return None
            
        except Exception:
            return None
    
    def build_feature_filters(self, args) -> Dict[str, Any]:
        """
        Build filter dictionary from command line arguments.
        
        Args:
            args: Command line arguments
            
        Returns:
            Dictionary of filters for feature queries
        """
        filters = {}
        
        # Session filter
        if hasattr(args, 'session') and args.session:
            filters['session_id'] = args.session
        
        # Category filter
        if hasattr(args, 'category') and args.category:
            filters['category'] = args.category
        
        # Dataset filter
        if hasattr(args, 'dataset') and args.dataset:
            dataset_hash = self.resolve_dataset_identifier(args.dataset)
            if dataset_hash:
                filters['dataset_hash'] = dataset_hash
        
        # Dataset name filter (alternative to dataset)
        if hasattr(args, 'dataset_name') and args.dataset_name:
            dataset_hash = self.resolve_dataset_identifier(args.dataset_name)
            if dataset_hash:
                filters['dataset_hash'] = dataset_hash
        
        # Minimum impact filter
        if hasattr(args, 'min_impact') and args.min_impact is not None:
            filters['min_impact'] = args.min_impact
        
        # Limit filter
        if hasattr(args, 'limit') and args.limit:
            filters['limit'] = args.limit
        
        return filters
    
    def format_impact(self, impact: float) -> str:
        """Format impact value for display."""
        if impact is None:
            return "N/A"
        return f"{impact:+.5f}"
    
    def format_percentage(self, value: float) -> str:
        """Format percentage value for display."""
        if value is None:
            return "N/A"
        return f"{value:.1f}%"
    
    def format_number(self, value: int) -> str:
        """Format number with thousands separators."""
        if value is None:
            return "N/A"
        return f"{value:,}"
    
    def truncate_text(self, text: str, max_length: int = 30) -> str:
        """Truncate text to maximum length with ellipsis."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def print_table(self, headers: list, rows: list, title: str = None) -> None:
        """Print a formatted table."""
        if title:
            print(f"\n📊 {title.upper()}")
            print("=" * 60)
        
        if not rows:
            print("No data to display.")
            return
        
        # Calculate column widths
        widths = [len(str(header)) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        
        # Print header
        header_row = " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
        print(f"\n{header_row}")
        print("-" * len(header_row))
        
        # Print rows
        for row in rows:
            data_row = " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row)))
            print(data_row)
    
    def print_json(self, data: Any, title: str = None) -> None:
        """Print data as formatted JSON."""
        import json
        if title:
            print(f"\n📊 {title.upper()}")
            print("=" * 60)
        print(json.dumps(data, indent=2, default=str))
    
    def save_to_file(self, content: str, filename: str) -> None:
        """Save content to file."""
        try:
            with open(filename, 'w') as f:
                f.write(content)
            self.print_success(f"Output saved to {filename}")
        except Exception as e:
            self.print_error(f"Failed to save to {filename}: {e}")
    
    def print_error(self, message: str) -> None:
        """Print error message."""
        print(f"❌ {message}")
    
    def print_success(self, message: str) -> None:
        """Print success message."""
        print(f"✅ {message}")
    
    def print_warning(self, message: str) -> None:
        """Print warning message."""
        print(f"⚠️  {message}")
    
    def print_info(self, message: str) -> None:
        """Print info message."""
        print(f"💡 {message}")