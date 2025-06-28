#!/usr/bin/env python3
"""
DuckDB Manager Modules

Dynamic module system for database management operations.
Each module provides specific functionality with standardized interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import argparse

class ModuleInterface(ABC):
    """Base interface for all DuckDB manager modules."""
    
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
    
    def validate_dependencies(self, available_modules: List[str]) -> bool:
        """Check if all dependencies are available."""
        return all(dep in available_modules for dep in self.dependencies)

__all__ = ['ModuleInterface']