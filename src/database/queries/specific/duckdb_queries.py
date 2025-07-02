"""
DuckDB-specific queries and operations
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DuckDBSpecificQueries:
    """DuckDB-specific query operations"""
    
    @staticmethod
    def read_parquet(engine: Engine, file_path: str, table_name: str) -> None:
        """
        Read Parquet file into DuckDB table
        
        Args:
            engine: SQLAlchemy engine
            file_path: Path to Parquet file
            table_name: Target table name
        """
        query = text(f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * FROM read_parquet('{file_path}')
        """)
        with engine.connect() as conn:
            conn.execute(query)
            conn.commit()
        logger.info(f"Loaded Parquet file {file_path} into table {table_name}")
    
    @staticmethod
    def read_csv_auto(engine: Engine, file_path: str, table_name: str) -> None:
        """
        Read CSV file with automatic schema detection
        
        Args:
            engine: SQLAlchemy engine
            file_path: Path to CSV file
            table_name: Target table name
        """
        query = text(f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * FROM read_csv_auto('{file_path}')
        """)
        with engine.connect() as conn:
            conn.execute(query)
            conn.commit()
        logger.info(f"Loaded CSV file {file_path} into table {table_name}")
    
    @staticmethod
    def export_to_parquet(engine: Engine, table_name: str, file_path: str) -> None:
        """
        Export table to Parquet file
        
        Args:
            engine: SQLAlchemy engine
            table_name: Source table name
            file_path: Output Parquet file path
        """
        query = text(f"COPY {table_name} TO '{file_path}' (FORMAT PARQUET)")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Exported table {table_name} to Parquet file {file_path}")
    
    @staticmethod
    def export_to_csv(engine: Engine, table_name: str, file_path: str, 
                     header: bool = True) -> None:
        """
        Export table to CSV file
        
        Args:
            engine: SQLAlchemy engine
            table_name: Source table name
            file_path: Output CSV file path
            header: Include column headers
        """
        header_opt = "HEADER" if header else ""
        query = text(f"COPY {table_name} TO '{file_path}' (FORMAT CSV, {header_opt})")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Exported table {table_name} to CSV file {file_path}")
    
    @staticmethod
    def sample_reservoir(engine: Engine, table_name: str, sample_size: int, 
                        seed: Optional[int] = None) -> str:
        """
        Create reservoir sampling query
        
        Args:
            engine: SQLAlchemy engine
            table_name: Source table name
            sample_size: Number of samples
            seed: Random seed
            
        Returns:
            SQL query string for sampling
        """
        if seed is not None:
            return f"SELECT * FROM {table_name} TABLESAMPLE RESERVOIR({sample_size}) REPEATABLE ({seed})"
        else:
            return f"SELECT * FROM {table_name} TABLESAMPLE RESERVOIR({sample_size})"
    
    @staticmethod
    def sample_bernoulli(engine: Engine, table_name: str, percentage: float, 
                        seed: Optional[int] = None) -> str:
        """
        Create Bernoulli sampling query
        
        Args:
            engine: SQLAlchemy engine
            table_name: Source table name
            percentage: Sampling percentage (0-100)
            seed: Random seed
            
        Returns:
            SQL query string for sampling
        """
        if seed is not None:
            return f"SELECT * FROM {table_name} USING SAMPLE {percentage}% (bernoulli, {seed})"
        else:
            return f"SELECT * FROM {table_name} USING SAMPLE {percentage}% (bernoulli)"
    
    @staticmethod
    def attach_database(engine: Engine, db_path: str, alias: str) -> None:
        """
        Attach another DuckDB database
        
        Args:
            engine: SQLAlchemy engine
            db_path: Path to database file
            alias: Alias for attached database
        """
        query = text(f"ATTACH '{db_path}' AS {alias}")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Attached database {db_path} as {alias}")
    
    @staticmethod
    def detach_database(engine: Engine, alias: str) -> None:
        """
        Detach database
        
        Args:
            engine: SQLAlchemy engine
            alias: Alias of attached database
        """
        query = text(f"DETACH {alias}")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Detached database {alias}")
    
    @staticmethod
    def set_memory_limit(engine: Engine, memory_gb: int) -> None:
        """
        Set DuckDB memory limit
        
        Args:
            engine: SQLAlchemy engine
            memory_gb: Memory limit in GB
        """
        query = text(f"SET memory_limit = '{memory_gb}GB'")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Set memory limit to {memory_gb}GB")
    
    @staticmethod
    def set_threads(engine: Engine, thread_count: int) -> None:
        """
        Set DuckDB thread count
        
        Args:
            engine: SQLAlchemy engine
            thread_count: Number of threads
        """
        query = text(f"SET threads = {thread_count}")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Set thread count to {thread_count}")
    
    @staticmethod
    def describe_table(engine: Engine, table_name: str) -> str:
        """
        Get table description query
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table
            
        Returns:
            DESCRIBE query string
        """
        return f"DESCRIBE {table_name}"
    
    @staticmethod
    def show_tables(engine: Engine) -> str:
        """
        Get show tables query
        
        Args:
            engine: SQLAlchemy engine
            
        Returns:
            SHOW TABLES query string
        """
        return "SHOW TABLES"