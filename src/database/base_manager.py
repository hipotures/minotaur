"""
Base database manager using SQLAlchemy Core
"""

from sqlalchemy import MetaData, Table, select, insert, update, delete, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Base database manager using SQLAlchemy Core"""
    
    def __init__(self, engine: Engine):
        """
        Initialize database manager with SQLAlchemy engine
        
        Args:
            engine: SQLAlchemy Engine instance
        """
        self.engine = engine
        self.metadata = MetaData()
        self._tables = {}
    
    def reflect_table(self, table_name: str) -> Table:
        """
        Load metadata for an existing table
        
        Args:
            table_name: Name of the table to reflect
            
        Returns:
            SQLAlchemy Table object
        """
        if table_name not in self._tables:
            self._tables[table_name] = Table(
                table_name, 
                self.metadata, 
                autoload_with=self.engine
            )
        return self._tables[table_name]
    
    def execute_query(self, query: Union[str, Any], params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute SELECT query using SQLAlchemy Core
        
        Args:
            query: SQL query string or SQLAlchemy selectable
            params: Optional query parameters
            
        Returns:
            List of result dictionaries
        """
        with self.engine.connect() as conn:
            if isinstance(query, str):
                result = conn.execute(text(query), params or {})
            else:
                result = conn.execute(query)
            return [dict(row._mapping) for row in result]
    
    def execute_ddl(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Execute DDL (CREATE, DROP, ALTER) statement
        
        Args:
            query: DDL SQL statement
            params: Optional query parameters
        """
        with self.engine.begin() as conn:
            conn.execute(text(query), params or {})
    
    def execute_dml(self, query: str, params: Optional[Dict[str, Any]] = None) -> int:
        """
        Execute DML (INSERT, UPDATE, DELETE) statement
        
        Args:
            query: DML SQL statement
            params: Optional query parameters
            
        Returns:
            Number of affected rows
        """
        with self.engine.begin() as conn:
            result = conn.execute(text(query), params or {})
            return result.rowcount if result.rowcount is not None else 0
    
    def execute_query_df(self, query: Union[str, Any], params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute SELECT query and return as DataFrame
        
        Args:
            query: SQL query string or SQLAlchemy selectable
            params: Optional query parameters
            
        Returns:
            Pandas DataFrame with results
        """
        with self.engine.connect() as conn:
            if isinstance(query, str):
                return pd.read_sql(text(query), conn, params=params or {})
            else:
                return pd.read_sql(query, conn)
    
    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> int:
        """
        Insert data into table
        
        Args:
            table_name: Name of target table
            data: List of dictionaries with data to insert
            
        Returns:
            Number of rows inserted
        """
        table = self.reflect_table(table_name)
        with self.engine.begin() as conn:
            result = conn.execute(insert(table), data)
            return result.rowcount
    
    def update_data(self, table_name: str, values: Dict[str, Any], 
                   conditions: Dict[str, Any]) -> int:
        """
        Update data in table
        
        Args:
            table_name: Name of target table
            values: Dictionary of column values to update
            conditions: Dictionary of WHERE conditions
            
        Returns:
            Number of rows updated
        """
        table = self.reflect_table(table_name)
        stmt = update(table).values(**values)
        
        # Add WHERE conditions
        for col, val in conditions.items():
            stmt = stmt.where(getattr(table.c, col) == val)
        
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount
    
    def delete_data(self, table_name: str, conditions: Dict[str, Any]) -> int:
        """
        Delete data from table
        
        Args:
            table_name: Name of target table
            conditions: Dictionary of WHERE conditions
            
        Returns:
            Number of rows deleted
        """
        table = self.reflect_table(table_name)
        stmt = delete(table)
        
        # Add WHERE conditions
        for col, val in conditions.items():
            stmt = stmt.where(getattr(table.c, col) == val)
        
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount
    
    def bulk_insert_from_pandas(self, df: pd.DataFrame, table_name: str, 
                               if_exists: str = 'append', index: bool = False) -> None:
        """
        Bulk insert DataFrame using pandas to_sql
        
        Args:
            df: Pandas DataFrame to insert
            table_name: Name of target table
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            index: Whether to write DataFrame index
        """
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists in database
        
        Args:
            table_name: Name of table to check
            
        Returns:
            True if table exists, False otherwise
        """
        with self.engine.connect() as conn:
            return conn.dialect.has_table(conn, table_name)
    
    def get_table_names(self) -> List[str]:
        """
        Get list of all table names in database
        
        Returns:
            List of table names
        """
        with self.engine.connect() as conn:
            return conn.dialect.get_table_names(conn)
    
    def get_column_names(self, table_name: str) -> List[str]:
        """
        Get list of column names for a table
        
        Args:
            table_name: Name of table
            
        Returns:
            List of column names
        """
        table = self.reflect_table(table_name)
        return [col.name for col in table.columns]
    
    def count_rows(self, table_name: str, where_clause: Optional[str] = None) -> int:
        """
        Count rows in table
        
        Args:
            table_name: Name of table
            where_clause: Optional WHERE clause
            
        Returns:
            Number of rows
        """
        if where_clause:
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}"
        else:
            query = f"SELECT COUNT(*) FROM {table_name}"
        
        result = self.execute_query(query)
        return result[0]['count_star()'] if result else 0
    
    def create_table_from_df(self, df: pd.DataFrame, table_name: str, 
                           if_exists: str = 'fail') -> None:
        """
        Create table from DataFrame structure
        
        Args:
            df: DataFrame to use as template
            table_name: Name of new table
            if_exists: What to do if table exists
        """
        df.head(0).to_sql(table_name, self.engine, if_exists=if_exists, index=False)
    
    def drop_table(self, table_name: str, if_exists: bool = True) -> None:
        """
        Drop table from database
        
        Args:
            table_name: Name of table to drop
            if_exists: Don't raise error if table doesn't exist
        """
        if if_exists:
            query = f"DROP TABLE IF EXISTS {table_name}"
        else:
            query = f"DROP TABLE {table_name}"
        
        with self.engine.begin() as conn:
            conn.execute(text(query))
        
        # Remove from cache
        self._tables.pop(table_name, None)
    
    def vacuum(self) -> None:
        """Run database vacuum/analyze (if supported)"""
        try:
            with self.engine.begin() as conn:
                # Try common vacuum commands
                for cmd in ["VACUUM", "ANALYZE", "VACUUM ANALYZE"]:
                    try:
                        conn.execute(text(cmd))
                        logger.info(f"Executed {cmd}")
                        break
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Vacuum operation not supported or failed: {e}")
    
    def close(self) -> None:
        """Close database connections"""
        if hasattr(self.engine, 'dispose'):
            self.engine.dispose()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()