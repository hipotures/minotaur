"""
PostgreSQL-specific queries and operations
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PostgreSQLSpecificQueries:
    """PostgreSQL-specific query operations"""
    
    @staticmethod
    def create_extension(engine: Engine, extension_name: str) -> None:
        """
        Create PostgreSQL extension
        
        Args:
            engine: SQLAlchemy engine
            extension_name: Name of extension to create
        """
        query = text(f"CREATE EXTENSION IF NOT EXISTS {extension_name}")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Created extension {extension_name}")
    
    @staticmethod
    def vacuum_analyze(engine: Engine, table_name: Optional[str] = None) -> None:
        """
        Run VACUUM ANALYZE
        
        Args:
            engine: SQLAlchemy engine
            table_name: Optional specific table name
        """
        if table_name:
            query = text(f"VACUUM ANALYZE {table_name}")
        else:
            query = text("VACUUM ANALYZE")
        
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Ran VACUUM ANALYZE on {table_name or 'all tables'}")
    
    @staticmethod
    def create_index_concurrently(engine: Engine, index_name: str, table_name: str, 
                                 columns: str) -> None:
        """
        Create index concurrently
        
        Args:
            engine: SQLAlchemy engine
            index_name: Name of index
            table_name: Target table
            columns: Column specification
        """
        query = text(f"CREATE INDEX CONCURRENTLY {index_name} ON {table_name} ({columns})")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Created index {index_name} on {table_name}({columns})")
    
    @staticmethod
    def drop_index_concurrently(engine: Engine, index_name: str) -> None:
        """
        Drop index concurrently
        
        Args:
            engine: SQLAlchemy engine
            index_name: Name of index to drop
        """
        query = text(f"DROP INDEX CONCURRENTLY {index_name}")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Dropped index {index_name}")
    
    @staticmethod
    def sample_tablesample(engine: Engine, table_name: str, percentage: float, 
                          method: str = "BERNOULLI") -> str:
        """
        Create TABLESAMPLE query
        
        Args:
            engine: SQLAlchemy engine
            table_name: Source table name
            percentage: Sampling percentage (0-100)
            method: Sampling method (BERNOULLI or SYSTEM)
            
        Returns:
            SQL query string for sampling
        """
        return f"SELECT * FROM {table_name} TABLESAMPLE {method} ({percentage})"
    
    @staticmethod
    def copy_from_csv(engine: Engine, table_name: str, file_path: str, 
                     header: bool = True, delimiter: str = ',') -> None:
        """
        Copy data from CSV file using COPY command
        
        Args:
            engine: SQLAlchemy engine
            table_name: Target table name
            file_path: Path to CSV file
            header: CSV has header row
            delimiter: CSV delimiter
        """
        header_opt = "HEADER" if header else ""
        query = text(f"""
            COPY {table_name} 
            FROM '{file_path}' 
            WITH (FORMAT CSV, {header_opt}, DELIMITER '{delimiter}')
        """)
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Copied data from {file_path} to table {table_name}")
    
    @staticmethod
    def copy_to_csv(engine: Engine, table_name: str, file_path: str, 
                   header: bool = True, delimiter: str = ',') -> None:
        """
        Copy table data to CSV file using COPY command
        
        Args:
            engine: SQLAlchemy engine
            table_name: Source table name
            file_path: Output CSV file path
            header: Include header row
            delimiter: CSV delimiter
        """
        header_opt = "HEADER" if header else ""
        query = text(f"""
            COPY {table_name} 
            TO '{file_path}' 
            WITH (FORMAT CSV, {header_opt}, DELIMITER '{delimiter}')
        """)
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Copied table {table_name} to {file_path}")
    
    @staticmethod
    def explain_analyze(engine: Engine, query: str) -> str:
        """
        Create EXPLAIN ANALYZE query
        
        Args:
            engine: SQLAlchemy engine
            query: Query to analyze
            
        Returns:
            EXPLAIN ANALYZE query string
        """
        return f"EXPLAIN ANALYZE {query}"
    
    @staticmethod
    def get_table_size(engine: Engine, table_name: str) -> str:
        """
        Get table size query
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table
            
        Returns:
            Query to get table size
        """
        return f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))"
    
    @staticmethod
    def get_database_size(engine: Engine) -> str:
        """
        Get database size query
        
        Args:
            engine: SQLAlchemy engine
            
        Returns:
            Query to get database size
        """
        return "SELECT pg_size_pretty(pg_database_size(current_database()))"
    
    @staticmethod
    def create_sequence(engine: Engine, sequence_name: str, start: int = 1, 
                       increment: int = 1) -> None:
        """
        Create sequence
        
        Args:
            engine: SQLAlchemy engine
            sequence_name: Name of sequence
            start: Starting value
            increment: Increment value
        """
        query = text(f"""
            CREATE SEQUENCE {sequence_name} 
            START {start} 
            INCREMENT {increment}
        """)
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Created sequence {sequence_name}")
    
    @staticmethod
    def drop_sequence(engine: Engine, sequence_name: str, if_exists: bool = True) -> None:
        """
        Drop sequence
        
        Args:
            engine: SQLAlchemy engine
            sequence_name: Name of sequence
            if_exists: Use IF EXISTS clause
        """
        if_exists_clause = "IF EXISTS" if if_exists else ""
        query = text(f"DROP SEQUENCE {if_exists_clause} {sequence_name}")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Dropped sequence {sequence_name}")