"""
SQLite-specific queries and operations
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SQLiteSpecificQueries:
    """SQLite-specific query operations"""
    
    @staticmethod
    def vacuum(engine: Engine) -> None:
        """
        Run VACUUM command
        
        Args:
            engine: SQLAlchemy engine
        """
        query = text("VACUUM")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info("Ran VACUUM command")
    
    @staticmethod
    def analyze(engine: Engine, table_name: Optional[str] = None) -> None:
        """
        Run ANALYZE command
        
        Args:
            engine: SQLAlchemy engine
            table_name: Optional specific table name
        """
        if table_name:
            query = text(f"ANALYZE {table_name}")
        else:
            query = text("ANALYZE")
        
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Ran ANALYZE on {table_name or 'all tables'}")
    
    @staticmethod
    def attach_database(engine: Engine, db_path: str, alias: str) -> None:
        """
        Attach another SQLite database
        
        Args:
            engine: SQLAlchemy engine
            db_path: Path to database file
            alias: Alias for attached database
        """
        query = text(f"ATTACH DATABASE '{db_path}' AS {alias}")
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
        query = text(f"DETACH DATABASE {alias}")
        with engine.connect() as conn:
            conn.execute(query)
        logger.info(f"Detached database {alias}")
    
    @staticmethod
    def pragma_query(engine: Engine, pragma_name: str, value: Optional[str] = None) -> Any:
        """
        Execute PRAGMA command
        
        Args:
            engine: SQLAlchemy engine
            pragma_name: Name of pragma
            value: Optional value to set
            
        Returns:
            Result of pragma query if no value set
        """
        if value is not None:
            query = text(f"PRAGMA {pragma_name} = {value}")
            with engine.connect() as conn:
                conn.execute(query)
            logger.info(f"Set pragma {pragma_name} = {value}")
        else:
            query = text(f"PRAGMA {pragma_name}")
            with engine.connect() as conn:
                result = conn.execute(query).fetchone()
            return result[0] if result else None
    
    @staticmethod
    def set_journal_mode(engine: Engine, mode: str = "WAL") -> None:
        """
        Set journal mode
        
        Args:
            engine: SQLAlchemy engine
            mode: Journal mode (DELETE, TRUNCATE, PERSIST, MEMORY, WAL, OFF)
        """
        SQLiteSpecificQueries.pragma_query(engine, "journal_mode", mode)
    
    @staticmethod
    def set_synchronous(engine: Engine, level: str = "NORMAL") -> None:
        """
        Set synchronous mode
        
        Args:
            engine: SQLAlchemy engine
            level: Synchronous level (OFF, NORMAL, FULL, EXTRA)
        """
        SQLiteSpecificQueries.pragma_query(engine, "synchronous", level)
    
    @staticmethod
    def set_cache_size(engine: Engine, size: int = -2000) -> None:
        """
        Set cache size
        
        Args:
            engine: SQLAlchemy engine
            size: Cache size in pages (negative for KB)
        """
        SQLiteSpecificQueries.pragma_query(engine, "cache_size", str(size))
    
    @staticmethod
    def sample_random(engine: Engine, table_name: str, sample_size: int, 
                     seed: Optional[int] = None) -> str:
        """
        Create random sampling query using ORDER BY RANDOM()
        
        Args:
            engine: SQLAlchemy engine
            table_name: Source table name
            sample_size: Number of samples
            seed: Random seed (not directly supported in SQLite)
            
        Returns:
            SQL query string for sampling
        """
        # SQLite doesn't support seeded random, but we can use ORDER BY RANDOM()
        return f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
    
    @staticmethod
    def import_csv(engine: Engine, table_name: str, file_path: str, 
                  header: bool = True) -> None:
        """
        Import CSV data (requires csv mode in SQLite CLI)
        Note: This is primarily for reference, as it requires SQLite CLI
        
        Args:
            engine: SQLAlchemy engine
            table_name: Target table name
            file_path: Path to CSV file
            header: CSV has header row
        """
        # This would typically be done via SQLite CLI, not SQLAlchemy
        logger.warning("CSV import in SQLite typically requires CLI commands")
        logger.info(f"To import {file_path} to {table_name}, use SQLite CLI:")
        logger.info(f".mode csv")
        if header:
            logger.info(f".headers on")
        logger.info(f".import {file_path} {table_name}")
    
    @staticmethod
    def export_csv(engine: Engine, table_name: str, file_path: str) -> None:
        """
        Export to CSV (requires SQLite CLI)
        Note: This is primarily for reference, as it requires SQLite CLI
        
        Args:
            engine: SQLAlchemy engine
            table_name: Source table name
            file_path: Output CSV file path
        """
        # This would typically be done via SQLite CLI, not SQLAlchemy
        logger.warning("CSV export in SQLite typically requires CLI commands")
        logger.info(f"To export {table_name} to {file_path}, use SQLite CLI:")
        logger.info(f".mode csv")
        logger.info(f".headers on")
        logger.info(f".output {file_path}")
        logger.info(f"SELECT * FROM {table_name};")
    
    @staticmethod
    def get_table_info(engine: Engine, table_name: str) -> str:
        """
        Get table info query
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table
            
        Returns:
            PRAGMA table_info query string
        """
        return f"PRAGMA table_info({table_name})"
    
    @staticmethod
    def get_database_list(engine: Engine) -> str:
        """
        Get database list query
        
        Args:
            engine: SQLAlchemy engine
            
        Returns:
            PRAGMA database_list query string
        """
        return "PRAGMA database_list"
    
    @staticmethod
    def get_table_list(engine: Engine) -> str:
        """
        Get table list query
        
        Args:
            engine: SQLAlchemy engine
            
        Returns:
            Query to list all tables
        """
        return "SELECT name FROM sqlite_master WHERE type='table'"