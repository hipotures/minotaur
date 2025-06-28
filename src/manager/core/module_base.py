"""
Enhanced base module interface with service injection support
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
import argparse
import logging


class ModuleInterface(ABC):
    """Enhanced base interface for all manager modules with service support."""
    
    def __init__(self):
        """Initialize module with logger."""
        self.logger = logging.getLogger(self.__class__.__module__)
        self._services: Dict[str, Any] = {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name (used for command routing)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Module description for help text."""
        pass
    
    @property
    def dependencies(self) -> List[str]:
        """List of module names this module depends on."""
        return []
    
    @property
    def required_services(self) -> Dict[str, Type]:
        """Dictionary of service_name: ServiceClass this module requires."""
        return {}
    
    @property
    @abstractmethod
    def commands(self) -> Dict[str, str]:
        """Dictionary of command_name: description for this module."""
        pass
    
    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add module-specific arguments to argument parser."""
        pass
    
    @abstractmethod
    def execute(self, args: argparse.Namespace, manager) -> None:
        """Execute module command with parsed arguments and manager instance."""
        pass
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """Inject required services into the module.
        
        Args:
            services: Dictionary of service instances
        """
        for service_name, service_class in self.required_services.items():
            if service_name in services:
                self._services[service_name] = services[service_name]
            else:
                raise ValueError(f"Required service '{service_name}' not provided")
    
    def get_service(self, service_name: str) -> Any:
        """Get an injected service by name.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not found
        """
        if service_name not in self._services:
            raise KeyError(f"Service '{service_name}' not found. Available: {list(self._services.keys())}")
        return self._services[service_name]
    
    def validate_dependencies(self, available_modules: List[str]) -> bool:
        """Check if all dependencies are available."""
        return all(dep in available_modules for dep in self.dependencies)
    
    def format_output(self, data: Any, format: str = 'text') -> str:
        """Format output data based on requested format.
        
        Args:
            data: Data to format
            format: Output format (text, json, csv, etc.)
            
        Returns:
            Formatted string
        """
        if format == 'json':
            import json
            return json.dumps(data, indent=2, default=str)
        elif format == 'csv':
            # Basic CSV formatting for lists of dicts
            if isinstance(data, list) and data and isinstance(data[0], dict):
                import csv
                import io
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
                return output.getvalue()
        
        # Default text formatting
        return str(data)