"""
Base repository class for common database operations
"""

from typing import Any, Dict, List, Optional, Union
import logging
from ..core.database import DatabasePool


class BaseRepository:
    """Base class for all repository implementations."""
    
    def __init__(self, db_pool: DatabasePool):
        """Initialize repository with database connection pool.
        
        Args:
            db_pool: Database connection pool instance
        """
        self.db_pool = db_pool
        self.logger = logging.getLogger(self.__class__.__module__)
    
    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a query through the connection pool."""
        with self.db_pool.get_connection() as conn:
            return conn.execute(query, params)
    
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one row as dictionary."""
        with self.db_pool.get_connection() as conn:
            result = conn.fetch_one(query, params)
            if result:
                # Convert to dictionary
                cursor = conn.connection.execute(query, params or ())
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return None
    
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute query and fetch all rows as dictionaries."""
        with self.db_pool.get_connection() as conn:
            rows = conn.fetch_all(query, params)
            if rows:
                # Convert to list of dictionaries
                cursor = conn.connection.execute(query, params or ())
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
            return []
    
    def fetch_df(self, query: str, params: Optional[tuple] = None) -> 'pd.DataFrame':
        """Execute query and return as pandas DataFrame."""
        with self.db_pool.get_connection() as conn:
            return conn.fetch_df(query, params)
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_name = ? AND table_schema = 'main'
        """
        result = self.fetch_one(query, (table_name,))
        return list(result.values())[0] > 0 if result else False
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column info dictionaries
        """
        query = f"PRAGMA table_info('{table_name}')"
        columns = []
        
        with self.db_pool.get_connection() as conn:
            rows = conn.fetch_all(query)
            for col in rows:
                columns.append({
                    'name': col[1],
                    'type': col[2],
                    'nullable': not col[3],
                    'default': col[4],
                    'primary_key': bool(col[5])
                })
        
        return columns
    
    def get_row_count(self, table_name: str, where_clause: str = "") -> int:
        """Get row count for a table with optional filter.
        
        Args:
            table_name: Name of the table
            where_clause: Optional WHERE clause (without 'WHERE' keyword)
            
        Returns:
            Row count
        """
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        result = self.fetch_one(query)
        return result['count'] if result else 0
    
    def build_where_clause(self, conditions: Dict[str, Any]) -> tuple:
        """Build WHERE clause from conditions dictionary.
        
        Args:
            conditions: Dictionary of column: value conditions
            
        Returns:
            Tuple of (where_clause, params)
        """
        if not conditions:
            return "", ()
        
        clauses = []
        params = []
        
        for column, value in conditions.items():
            if value is None:
                clauses.append(f"{column} IS NULL")
            elif isinstance(value, (list, tuple)):
                placeholders = ','.join(['?' for _ in value])
                clauses.append(f"{column} IN ({placeholders})")
                params.extend(value)
            elif isinstance(value, dict):
                # Handle operators like {'$gt': 10, '$lt': 20}
                for op, val in value.items():
                    if op == '$gt':
                        clauses.append(f"{column} > ?")
                        params.append(val)
                    elif op == '$gte':
                        clauses.append(f"{column} >= ?")
                        params.append(val)
                    elif op == '$lt':
                        clauses.append(f"{column} < ?")
                        params.append(val)
                    elif op == '$lte':
                        clauses.append(f"{column} <= ?")
                        params.append(val)
                    elif op == '$ne':
                        clauses.append(f"{column} != ?")
                        params.append(val)
                    elif op == '$like':
                        clauses.append(f"{column} LIKE ?")
                        params.append(val)
            else:
                clauses.append(f"{column} = ?")
                params.append(value)
        
        where_clause = " AND ".join(clauses)
        return where_clause, tuple(params)