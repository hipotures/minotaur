"""
Database configuration and engine management
"""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Configuration manager for database connections"""
    
    @staticmethod
    def get_engine(db_type: str, connection_params: Dict[str, Any]) -> Engine:
        """
        Create SQLAlchemy engine based on database type and parameters
        
        Args:
            db_type: Database type ('duckdb', 'sqlite', 'postgresql')
            connection_params: Database connection parameters
            
        Returns:
            SQLAlchemy Engine instance
        """
        if db_type == 'duckdb':
            # Requires duckdb-engine
            database = connection_params.get('database', ':memory:')
            conn_string = f"duckdb:///{database}"
        elif db_type == 'sqlite':
            database = connection_params.get('database', ':memory:')
            conn_string = f"sqlite:///{database}"
        elif db_type == 'postgresql':
            user = connection_params.get('user', 'postgres')
            password = connection_params.get('password', '')
            host = connection_params.get('host', 'localhost')
            port = connection_params.get('port', 5432)
            database = connection_params.get('database', 'postgres')
            conn_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        # Get engine arguments
        engine_args = connection_params.get('engine_args', {})
        
        # Set default engine arguments based on database type
        if db_type == 'duckdb':
            engine_args.setdefault('pool_pre_ping', True)
            engine_args.setdefault('echo', False)
        elif db_type == 'sqlite':
            engine_args.setdefault('pool_pre_ping', True)
            engine_args.setdefault('echo', False)
        elif db_type == 'postgresql':
            engine_args.setdefault('pool_size', 10)
            engine_args.setdefault('max_overflow', 20)
            engine_args.setdefault('pool_pre_ping', True)
            engine_args.setdefault('echo', False)
        
        logger.info(f"Creating {db_type} engine: {conn_string}")
        return create_engine(conn_string, **engine_args)
    
    @staticmethod
    def get_default_config(db_type: str, database_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get default configuration for a database type
        
        Args:
            db_type: Database type
            database_path: Optional database file path
            
        Returns:
            Default configuration dictionary
        """
        if db_type == 'duckdb':
            return {
                'db_type': 'duckdb',
                'connection_params': {
                    'database': database_path or ':memory:',
                    'engine_args': {
                        'pool_pre_ping': True,
                        'echo': False
                    }
                }
            }
        elif db_type == 'sqlite':
            return {
                'db_type': 'sqlite',
                'connection_params': {
                    'database': database_path or ':memory:',
                    'engine_args': {
                        'pool_pre_ping': True,
                        'echo': False
                    }
                }
            }
        elif db_type == 'postgresql':
            return {
                'db_type': 'postgresql',
                'connection_params': {
                    'user': 'postgres',
                    'password': '',
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'postgres',
                    'engine_args': {
                        'pool_size': 10,
                        'max_overflow': 20,
                        'pool_pre_ping': True,
                        'echo': False
                    }
                }
            }
        else:
            raise ValueError(f"Unsupported database type: {db_type}")