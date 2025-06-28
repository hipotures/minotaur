"""
Core database components.

This module contains the fundamental building blocks for database operations:
- Connection management and pooling
- Base repository implementation
- Query building utilities
"""

from .connection import DuckDBConnectionManager
from .base_repository import BaseRepository

__all__ = ["DuckDBConnectionManager", "BaseRepository"]