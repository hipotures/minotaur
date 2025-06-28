"""
Manager Package - Modular database management system

This package provides a flexible, extensible system for managing DuckDB databases
with a plugin-based architecture for different management tasks.
"""

from .core.module_base import ModuleInterface
from .core.config import Config
from .core.database import DatabaseConnection

__all__ = ['ModuleInterface', 'Config', 'DatabaseConnection']