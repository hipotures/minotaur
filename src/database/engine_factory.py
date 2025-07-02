"""
Database engine factory for creating database managers with specific functionality
"""

from typing import Dict, Any, Callable
import logging

from .config import DatabaseConfig
from .base_manager import DatabaseManager
from .queries.specific import DuckDBSpecificQueries, PostgreSQLSpecificQueries, SQLiteSpecificQueries

logger = logging.getLogger(__name__)


class DatabaseFactory:
    """Factory for creating database managers with database-specific functionality"""
    
    @staticmethod
    def create_manager(db_type: str, connection_params: Dict[str, Any]) -> DatabaseManager:
        """
        Create database manager with database-specific functionality
        
        Args:
            db_type: Database type ('duckdb', 'sqlite', 'postgresql')
            connection_params: Database connection parameters
            
        Returns:
            DatabaseManager instance with database-specific methods
        """
        # Create engine
        engine = DatabaseConfig.get_engine(db_type, connection_params)
        
        # Create base manager
        manager = DatabaseManager(engine)
        
        # Add database-specific functionality
        if db_type == 'duckdb':
            DatabaseFactory._add_duckdb_methods(manager, engine)
        elif db_type == 'sqlite':
            DatabaseFactory._add_sqlite_methods(manager, engine)
        elif db_type == 'postgresql':
            DatabaseFactory._add_postgresql_methods(manager, engine)
        else:
            logger.warning(f"Unknown database type {db_type}, using base functionality only")
        
        # Store database type for reference
        manager.db_type = db_type
        
        logger.info(f"Created {db_type} database manager")
        return manager
    
    @staticmethod
    def _add_duckdb_methods(manager: DatabaseManager, engine) -> None:
        """Add DuckDB-specific methods to manager"""
        queries = DuckDBSpecificQueries()
        
        # File operations
        manager.read_parquet = lambda fp, tn: queries.read_parquet(engine, fp, tn)
        manager.read_csv_auto = lambda fp, tn: queries.read_csv_auto(engine, fp, tn)
        manager.export_to_parquet = lambda tn, fp: queries.export_to_parquet(engine, tn, fp)
        manager.export_to_csv = lambda tn, fp, h=True: queries.export_to_csv(engine, tn, fp, h)
        
        # Sampling operations
        manager.sample_reservoir = lambda tn, size, seed=None: queries.sample_reservoir(engine, tn, size, seed)
        manager.sample_bernoulli = lambda tn, pct, seed=None: queries.sample_bernoulli(engine, tn, pct, seed)
        
        # Database operations
        manager.attach_database = lambda path, alias: queries.attach_database(engine, path, alias)
        manager.detach_database = lambda alias: queries.detach_database(engine, alias)
        
        # Configuration
        manager.set_memory_limit = lambda gb: queries.set_memory_limit(engine, gb)
        manager.set_threads = lambda count: queries.set_threads(engine, count)
        
        # Introspection
        manager.describe_table = lambda tn: queries.describe_table(engine, tn)
        manager.show_tables = lambda: queries.show_tables(engine)
        
        logger.debug("Added DuckDB-specific methods")
    
    @staticmethod
    def _add_sqlite_methods(manager: DatabaseManager, engine) -> None:
        """Add SQLite-specific methods to manager"""
        queries = SQLiteSpecificQueries()
        
        # Database operations
        manager.vacuum_db = lambda: queries.vacuum(engine)
        manager.analyze_db = lambda tn=None: queries.analyze(engine, tn)
        
        # Database attachment
        manager.attach_database = lambda path, alias: queries.attach_database(engine, path, alias)
        manager.detach_database = lambda alias: queries.detach_database(engine, alias)
        
        # Pragma operations
        manager.pragma_query = lambda name, val=None: queries.pragma_query(engine, name, val)
        manager.set_journal_mode = lambda mode="WAL": queries.set_journal_mode(engine, mode)
        manager.set_synchronous = lambda level="NORMAL": queries.set_synchronous(engine, level)
        manager.set_cache_size = lambda size=-2000: queries.set_cache_size(engine, size)
        
        # Sampling
        manager.sample_random = lambda tn, size, seed=None: queries.sample_random(engine, tn, size, seed)
        
        # Introspection
        manager.get_table_info = lambda tn: queries.get_table_info(engine, tn)
        manager.get_database_list = lambda: queries.get_database_list(engine)
        manager.get_table_list = lambda: queries.get_table_list(engine)
        
        logger.debug("Added SQLite-specific methods")
    
    @staticmethod
    def _add_postgresql_methods(manager: DatabaseManager, engine) -> None:
        """Add PostgreSQL-specific methods to manager"""
        queries = PostgreSQLSpecificQueries()
        
        # Extensions
        manager.create_extension = lambda ext: queries.create_extension(engine, ext)
        
        # Maintenance
        manager.vacuum_analyze = lambda tn=None: queries.vacuum_analyze(engine, tn)
        
        # Index operations
        manager.create_index_concurrently = lambda idx, tn, cols: queries.create_index_concurrently(engine, idx, tn, cols)
        manager.drop_index_concurrently = lambda idx: queries.drop_index_concurrently(engine, idx)
        
        # Sampling
        manager.sample_tablesample = lambda tn, pct, method="BERNOULLI": queries.sample_tablesample(engine, tn, pct, method)
        
        # File operations
        manager.copy_from_csv = lambda tn, fp, h=True, d=',': queries.copy_from_csv(engine, tn, fp, h, d)
        manager.copy_to_csv = lambda tn, fp, h=True, d=',': queries.copy_to_csv(engine, tn, fp, h, d)
        
        # Query analysis
        manager.explain_analyze = lambda q: queries.explain_analyze(engine, q)
        
        # Size queries
        manager.get_table_size = lambda tn: queries.get_table_size(engine, tn)
        manager.get_database_size = lambda: queries.get_database_size(engine)
        
        # Sequences
        manager.create_sequence = lambda name, start=1, inc=1: queries.create_sequence(engine, name, start, inc)
        manager.drop_sequence = lambda name, exists=True: queries.drop_sequence(engine, name, exists)
        
        logger.debug("Added PostgreSQL-specific methods")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> DatabaseManager:
        """
        Create database manager from configuration dictionary
        
        Args:
            config: Configuration with 'db_type' and 'connection_params' keys
            
        Returns:
            DatabaseManager instance
        """
        db_type = config.get('db_type')
        connection_params = config.get('connection_params', {})
        
        if not db_type:
            raise ValueError("Configuration must include 'db_type'")
        
        return DatabaseFactory.create_manager(db_type, connection_params)
    
    @staticmethod
    def get_supported_databases() -> list:
        """
        Get list of supported database types
        
        Returns:
            List of supported database type strings
        """
        return ['duckdb', 'sqlite', 'postgresql']