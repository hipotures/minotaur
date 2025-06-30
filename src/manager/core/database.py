"""
Database connection management with pooling and query utilities
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading
from contextlib import contextmanager
import time

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None


class DatabaseConnection:
    """Manages a single DuckDB connection with query utilities."""
    
    def __init__(self, db_path: Union[str, Path], settings: Optional[Dict[str, Any]] = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to database file
            settings: DuckDB configuration settings
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is not installed. Please install with: pip install duckdb")
        
        self.db_path = Path(db_path)
        self.settings = settings or {}
        self.connection = None
        self._query_count = 0
        self._total_query_time = 0.0
    
    def connect(self) -> 'duckdb.DuckDBPyConnection':
        """Establish database connection."""
        if self.connection is None:
            # Use consistent DuckDB connection (no config dict to avoid conflicts)
            self.connection = duckdb.connect(str(self.db_path))
            
            # Apply performance settings manually after connection
            if 'max_memory' in self.settings:
                self.connection.execute(f"SET memory_limit = '{self.settings['max_memory']}'")
            if 'threads' in self.settings:
                self.connection.execute(f"SET threads = {self.settings['threads']}")
        
        return self.connection
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            # Note: DatabaseConnection doesn't have logger, so we skip logging here
    
    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a query and return results.
        
        Args:
            query: SQL query to execute
            params: Optional query parameters
            
        Returns:
            Query results
        """
        conn = self.connect()
        start_time = time.time()
        
        try:
            if params:
                result = conn.execute(query, params)
            else:
                result = conn.execute(query)
            
            self._query_count += 1
            self._total_query_time += time.time() - start_time
            
            return result
        except Exception as e:
            # Note: DatabaseConnection doesn't have logger setup
            raise
    
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """Execute query and fetch one row."""
        result = self.execute(query, params)
        return result.fetchone() if result else None
    
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """Execute query and fetch all rows."""
        result = self.execute(query, params)
        return result.fetchall() if result else []
    
    def fetch_df(self, query: str, params: Optional[tuple] = None) -> 'pd.DataFrame':
        """Execute query and return as pandas DataFrame."""
        result = self.execute(query, params)
        return result.df() if result else None
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'query_count': self._query_count,
            'total_query_time': self._total_query_time,
            'avg_query_time': self._total_query_time / max(1, self._query_count)
        }


class DatabasePool:
    """Connection pool for managing multiple database connections."""
    
    def __init__(self, db_path: Union[str, Path], settings: Optional[Dict[str, Any]] = None, 
                 max_connections: int = 5):
        """Initialize connection pool.
        
        Args:
            db_path: Path to database file
            settings: DuckDB configuration settings
            max_connections: Maximum number of connections in pool
        """
        self.db_path = Path(db_path)
        self.settings = settings or {}
        self.max_connections = max_connections
        
        self._connections: List[DatabaseConnection] = []
        self._available: List[DatabaseConnection] = []
        self._in_use: Dict[int, DatabaseConnection] = {}
        self._lock = threading.Lock()
        self._migrations_run = False
        
        # Setup advanced logging with database context
        self.logger = self._setup_database_logging()
        
        # Note: Migrations are now handled by manager.py to avoid duplicate runs
        # self._ensure_migrations()
    
    def _setup_database_logging(self):
        """Setup advanced database logging with context."""
        try:
            # Try to load the advanced logging configuration
            src_path = Path(__file__).parent.parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            from db.config.logging_config import setup_db_logging, DatabaseLoggerAdapter
            
            # Load main config for logging setup
            try:
                import yaml
                config_path = src_path.parent / 'config' / 'mcts_config.yaml'
                with open(config_path, 'r') as f:
                    main_config = yaml.safe_load(f)
            except Exception:
                # Fallback config
                main_config = {'logging': {'level': 'INFO'}}
            
            # Setup database logger
            base_logger = setup_db_logging(main_config)
            
            # Create adapter with database context
            db_name = self.db_path.stem  # Extract database name from path
            adapter = DatabaseLoggerAdapter(
                base_logger, 
                {'component': 'manager', 'db_path': str(self.db_path)}
            )
            
            return adapter
            
        except Exception as e:
            # Fallback to simple logger if advanced logging fails
            fallback_logger = logging.getLogger('manager.database')
            fallback_logger.warning(f"Could not setup advanced logging: {e}")
            return fallback_logger
    
    def _ensure_migrations(self):
        """Ensure database migrations are run."""
        if self._migrations_run:
            return
            
        try:
            # Import migration runner
            from pathlib import Path
            import sys
            
            # Add src to path if needed
            src_path = Path(__file__).parent.parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            from db.migrations.migration_runner import MigrationRunner
            from db.core.connection import DuckDBConnectionManager
            
            # Create connection manager for migrations
            config = {'database': {'db_path': str(self.db_path)}}
            conn_manager = DuckDBConnectionManager(config)
            
            # Run migrations
            migration_runner = MigrationRunner(conn_manager)
            applied = migration_runner.run_migrations()
            
            if applied:
                self.logger.info(f"Applied {len(applied)} database migrations")
            
            self._migrations_run = True
            
        except Exception as e:
            self.logger.warning(f"Could not run migrations: {e}")
            # Don't fail - maybe table exists from old system
    
    @contextmanager
    def get_connection(self) -> DatabaseConnection:
        """Get a connection from the pool."""
        conn = self._acquire_connection()
        try:
            yield conn
        finally:
            self._release_connection(conn)
    
    def _acquire_connection(self) -> DatabaseConnection:
        """Acquire a connection from the pool."""
        with self._lock:
            # Try to get an available connection
            if self._available:
                conn = self._available.pop()
                self._in_use[id(conn)] = conn
                return conn
            
            # Create new connection if under limit
            if len(self._connections) < self.max_connections:
                conn = DatabaseConnection(self.db_path, self.settings)
                self._connections.append(conn)
                self._in_use[id(conn)] = conn
                return conn
            
            # Wait for a connection to become available
            # In a real implementation, this would use a condition variable
            raise RuntimeError("No connections available in pool")
    
    def _release_connection(self, conn: DatabaseConnection) -> None:
        """Release a connection back to the pool."""
        with self._lock:
            if id(conn) in self._in_use:
                del self._in_use[id(conn)]
                self._available.append(conn)
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._connections:
                conn.close()
            self._connections.clear()
            self._available.clear()
            self._in_use.clear()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_stats = {
                'total_connections': len(self._connections),
                'available_connections': len(self._available),
                'in_use_connections': len(self._in_use),
                'total_queries': sum(conn.stats['query_count'] for conn in self._connections),
                'total_query_time': sum(conn.stats['total_query_time'] for conn in self._connections)
            }
            
            if total_stats['total_queries'] > 0:
                total_stats['avg_query_time'] = total_stats['total_query_time'] / total_stats['total_queries']
            else:
                total_stats['avg_query_time'] = 0.0
            
            return total_stats