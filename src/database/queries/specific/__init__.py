"""
Database-specific queries
"""

from .duckdb_queries import DuckDBSpecificQueries
from .postgres_queries import PostgreSQLSpecificQueries  
from .sqlite_queries import SQLiteSpecificQueries

__all__ = [
    'DuckDBSpecificQueries',
    'PostgreSQLSpecificQueries', 
    'SQLiteSpecificQueries'
]