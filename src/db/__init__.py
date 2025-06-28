"""
Database abstraction layer for Minotaur MCTS Feature Discovery System.

This module provides a clean, maintainable database layer that separates
database operations from business logic using the Repository pattern.

Key components:
- Connection management with pooling
- Repository pattern for data access
- Pydantic models for type safety
- Migration system for schema management
- Centralized logging configuration
"""

from .core.connection import DuckDBConnectionManager
from .core.base_repository import BaseRepository, ReadOnlyRepository
from .config import get_duckdb_config, setup_db_logging, DatabaseLoggerAdapter

# Import all models
from . import models
from .models import *

# Import all repositories
from . import repositories
from .repositories import *

# Import migration system
from .migrations import MigrationRunner, Migration

__version__ = "1.0.0"
__all__ = [
    # Core components
    "DuckDBConnectionManager",
    "BaseRepository", 
    "ReadOnlyRepository",
    
    # Configuration
    "get_duckdb_config",
    "setup_db_logging",
    "DatabaseLoggerAdapter",
    
    # Migration system
    "MigrationRunner",
    "Migration",
    
    # All models and repositories are exported through their __init__.py files
] + models.__all__ + repositories.__all__