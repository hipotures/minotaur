"""
Base command class for datasets module.

Provides common functionality for all dataset commands including:
- Service injection and access
- Common validation and utilities
- Output formatting helpers
- Error handling patterns
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseDatasetsCommand(ABC):
    """Base class for all datasets commands."""
    
    def __init__(self):
        self._services = {}
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """Inject required services."""
        self._services = services
    
    @property
    def dataset_service(self):
        """Get dataset service."""
        return self._services.get('dataset_service')
    
    @property
    def session_service(self):
        """Get session service."""
        return self._services.get('session_service')
    
    @property
    def feature_service(self):
        """Get feature service."""
        return self._services.get('feature_service')
    
    @property
    def backup_service(self):
        """Get backup service."""
        return self._services.get('backup_service')
    
    @abstractmethod
    def execute(self, args) -> None:
        """Execute the command with given arguments."""
        pass
    
    def resolve_dataset_identifier(self, identifier: str) -> Optional[str]:
        """
        Resolve dataset identifier (ID or name) to dataset ID.
        
        Args:
            identifier: Dataset ID (hash) or name
            
        Returns:
            Dataset ID if found, None otherwise
        """
        if not identifier:
            return None
            
        # Get datasets directly from repository
        datasets = self.dataset_service.repository.get_all_datasets(include_inactive=True)
        
        # Check if it's already a valid dataset ID (hash format)
        if len(identifier) >= 8 and all(c in '0123456789abcdef' for c in identifier.lower()):
            # Try to find by ID first
            for dataset in datasets:
                if dataset['dataset_id'].startswith(identifier.lower()):
                    return dataset['dataset_id']
        
        # Try to find by name
        for dataset in datasets:
            if dataset['dataset_name'] == identifier:
                return dataset['dataset_id']
        
        # Try partial name match
        matches = []
        for dataset in datasets:
            if identifier.lower() in dataset['dataset_name'].lower():
                matches.append(dataset)
        
        if len(matches) == 1:
            return matches[0]['dataset_id']
        elif len(matches) > 1:
            print(f"❌ Multiple datasets match '{identifier}':")
            for match in matches:
                print(f"   {match['dataset_id'][:8]} - {match['dataset_name']}")
            print("Please use a more specific identifier.")
            return None
        
        return None
    
    def find_dataset_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Find dataset by identifier and return full dataset info.
        
        Args:
            identifier: Dataset ID or name
            
        Returns:
            Dataset dictionary if found, None otherwise
        """
        dataset_id = self.resolve_dataset_identifier(identifier)
        if not dataset_id:
            return None
        
        # Find dataset by ID from repository
        datasets = self.dataset_service.repository.get_all_datasets(include_inactive=True)
        for dataset in datasets:
            if dataset['dataset_id'] == dataset_id:
                return dataset
        return None
    
    def format_size(self, size_bytes: int) -> str:
        """Format byte size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def print_table(self, headers: list, rows: list, title: str = None) -> None:
        """Print a formatted table."""
        if title:
            print(f"\n📊 {title.upper()}")
            print("=" * 50)
        
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
            print("=" * 50)
        print(json.dumps(data, indent=2, default=str))
    
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