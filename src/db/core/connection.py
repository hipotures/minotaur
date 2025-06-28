"""
DuckDB Connection Manager with pooling and performance optimization.

This module provides centralized connection management for the Minotaur database layer.
It includes connection pooling, automatic retry logic, performance monitoring,
and integration with the main configuration system.
"""

import os
import time
import threading
from queue import Queue, Empty, Full
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None

from ..config.db_config import get_duckdb_config, validate_config
from ..config.logging_config import setup_db_logging, DatabaseLoggerAdapter


class ConnectionPool:
    """
    Thread-safe connection pool for DuckDB connections.
    
    Manages a pool of database connections with automatic cleanup,
    health checking, and resource management.
    """
    
    def __init__(self, db_path: str, pool_size: int, timeout: float, logger: logging.Logger):
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self.logger = logger
        
        # Thread-safe connection pool
        self._pool = Queue(maxsize=pool_size)
        self._all_connections = []  # Track all connections for cleanup
        self._lock = threading.RLock()
        self._created_count = 0
        self._active_count = 0
        
        # Statistics
        self.stats = {
            'connections_created': 0,
            'connections_acquired': 0,
            'connections_released': 0,
            'connections_failed': 0,
            'pool_waits': 0,
            'total_wait_time': 0.0
        }
        
        # Initialize pool with initial connections
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Create initial connections for the pool."""
        try:
            # Create initial connections (start with half of pool size)
            initial_size = max(1, self.pool_size // 2)
            
            for _ in range(initial_size):
                conn = self._create_connection()
                if conn:
                    self._pool.put(conn, block=False)
                    
            self.logger.info(f"Connection pool initialized with {initial_size} connections")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def _create_connection(self) -> Optional[duckdb.DuckDBPyConnection]:
        """Create a new DuckDB connection with optimized settings."""
        if not DUCKDB_AVAILABLE:
            raise RuntimeError("DuckDB is not available")
        
        try:
            with self._lock:
                if self._created_count >= self.pool_size:
                    return None
                
                # Create connection
                conn = duckdb.connect(database=self.db_path)
                
                # Apply performance optimizations
                self._configure_connection(conn)
                
                # Track connection
                self._all_connections.append(conn)
                self._created_count += 1
                self.stats['connections_created'] += 1
                
                self.logger.debug(f"Created new connection #{self._created_count}")
                return conn
                
        except Exception as e:
            self.stats['connections_failed'] += 1
            self.logger.error(f"Failed to create connection: {e}")
            return None
    
    def _configure_connection(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Apply DuckDB performance optimizations to connection."""
        try:
            config = get_duckdb_config()
            perf_config = config['performance']
            
            # Memory and performance settings
            conn.execute(f"SET memory_limit = '{perf_config['memory_limit']}'")
            
            # Set threads (ensure at least 1)
            threads = perf_config['threads']
            if threads == 'auto':
                threads = max(1, os.cpu_count() or 1)
            conn.execute(f"SET threads = {threads}")
            
            # Other optimizations
            conn.execute(f"SET enable_progress_bar = {perf_config['enable_progress_bar']}")
            conn.execute(f"SET enable_object_cache = {perf_config['enable_object_cache']}")
            conn.execute(f"SET force_compression = '{perf_config['force_compression']}'")
            
            # Set temp directory if specified
            temp_dir = perf_config.get('temp_directory')
            if temp_dir and Path(temp_dir).exists():
                conn.execute(f"SET temp_directory = '{temp_dir}'")
            
        except Exception as e:
            self.logger.warning(f"Failed to configure connection: {e}")
    
    def acquire(self) -> Optional[duckdb.DuckDBPyConnection]:
        """
        Acquire a connection from the pool.
        
        Returns:
            DuckDB connection or None if unable to acquire
        """
        start_time = time.time()
        
        try:
            # Try to get existing connection from pool
            try:
                conn = self._pool.get(timeout=self.timeout)
                
                # Validate connection health
                if self._is_connection_healthy(conn):
                    with self._lock:
                        self._active_count += 1
                    self.stats['connections_acquired'] += 1
                    
                    wait_time = time.time() - start_time
                    if wait_time > 0.1:  # Log significant waits
                        self.stats['pool_waits'] += 1
                        self.stats['total_wait_time'] += wait_time
                        self.logger.debug(f"Pool wait: {wait_time:.3f}s")
                    
                    return conn
                else:
                    # Connection is unhealthy, discard it
                    self.logger.warning("Discarded unhealthy connection")
                    self._remove_connection(conn)
                    
            except Empty:
                # Pool is empty, try to create new connection
                pass
            
            # Try to create new connection if pool has space
            conn = self._create_connection()
            if conn:
                with self._lock:
                    self._active_count += 1
                self.stats['connections_acquired'] += 1
                return conn
            
            # Unable to acquire connection
            self.logger.warning("Unable to acquire database connection")
            return None
            
        except Exception as e:
            self.logger.error(f"Error acquiring connection: {e}")
            return None
    
    def release(self, conn: duckdb.DuckDBPyConnection) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            conn: Connection to release
        """
        if not conn:
            return
        
        try:
            # Check if connection is still healthy
            if self._is_connection_healthy(conn):
                # Return to pool if there's space
                try:
                    self._pool.put(conn, block=False)
                    with self._lock:
                        self._active_count = max(0, self._active_count - 1)
                    self.stats['connections_released'] += 1
                    
                except Full:
                    # Pool is full, close this connection
                    self._remove_connection(conn)
            else:
                # Connection is unhealthy, remove it
                self._remove_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Error releasing connection: {e}")
            self._remove_connection(conn)
    
    def _is_connection_healthy(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """Check if connection is healthy and responsive."""
        try:
            # Simple health check query
            result = conn.execute("SELECT 1").fetchone()
            return result is not None and result[0] == 1
        except:
            return False
    
    def _remove_connection(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Remove and close a connection."""
        try:
            if conn in self._all_connections:
                self._all_connections.remove(conn)
            
            conn.close()
            
            with self._lock:
                self._created_count = max(0, self._created_count - 1)
                self._active_count = max(0, self._active_count - 1)
            
        except Exception as e:
            self.logger.warning(f"Error removing connection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                'pool_size': self.pool_size,
                'created_connections': self._created_count,
                'active_connections': self._active_count,
                'pooled_connections': self._pool.qsize(),
                'stats': self.stats.copy()
            }
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        self.logger.info("Closing all connections in pool")
        
        # Close pooled connections
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                conn.close()
            except Empty:
                break
            except Exception as e:
                self.logger.warning(f"Error closing pooled connection: {e}")
        
        # Close all tracked connections
        for conn in self._all_connections[:]:
            try:
                conn.close()
            except Exception as e:
                self.logger.warning(f"Error closing tracked connection: {e}")
        
        self._all_connections.clear()
        with self._lock:
            self._created_count = 0
            self._active_count = 0


class DuckDBConnectionManager:
    """
    Main connection manager for DuckDB database operations.
    
    This class provides the primary interface for database connections in the
    Minotaur system. It integrates with the main configuration system and
    provides connection pooling, transaction management, and performance monitoring.
    """
    
    def __init__(self, main_config: Dict[str, Any]):
        """
        Initialize connection manager with configuration.
        
        Args:
            main_config: Main MCTS configuration dictionary
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is not installed. Please install with: pip install duckdb")
        
        self.main_config = main_config
        self.db_config = get_duckdb_config()
        
        # Validate configuration
        if not validate_config(self.db_config):
            raise ValueError("Invalid database configuration")
        
        # Database path from main config
        db_config = main_config.get('database', {})
        self.db_path = str(Path(db_config.get('path', 'data/minotaur.duckdb')))
        
        # Setup logging
        self.logger = DatabaseLoggerAdapter(
            setup_db_logging(main_config),
            {'component': 'connection_manager', 'db_path': self.db_path}
        )
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection pool
        conn_config = self.db_config['connection']
        self.pool = ConnectionPool(
            db_path=self.db_path,
            pool_size=conn_config['pool_size'],
            timeout=conn_config['timeout'],
            logger=self.logger.logger
        )
        
        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'total_query_time': 0.0,
            'slow_queries': 0,
            'failed_queries': 0
        }
        
        self.logger.info(f"DuckDB connection manager initialized: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool using context manager.
        
        Usage:
            with manager.get_connection() as conn:
                result = conn.execute("SELECT * FROM table").fetchall()
        """
        conn = None
        start_time = time.time()
        
        try:
            conn = self.pool.acquire()
            if not conn:
                raise RuntimeError("Unable to acquire database connection")
            
            self.logger.connection_event('acquired', f"Pool stats: {self.pool.get_stats()}")
            yield conn
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            raise
        finally:
            if conn:
                self.pool.release(conn)
                duration = time.time() - start_time
                self.logger.connection_event('released', f"Duration: {duration:.3f}s")
    
    def execute_query(self, query: str, params: tuple = None, fetch: str = 'all') -> Any:
        """
        Execute a query with automatic connection management.
        
        Args:
            query: SQL query string
            params: Query parameters tuple
            fetch: Fetch method ('all', 'one', 'none')
            
        Returns:
            Query results based on fetch method
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                # Execute query
                if params:
                    cursor = conn.execute(query, params)
                else:
                    cursor = conn.execute(query)
                
                # Fetch results based on method
                if fetch == 'all':
                    result = cursor.fetchall()
                elif fetch == 'one':
                    result = cursor.fetchone()
                elif fetch == 'none':
                    result = None
                else:
                    raise ValueError(f"Invalid fetch method: {fetch}")
                
                # Track performance
                duration = time.time() - start_time
                self._track_query_performance(query, duration)
                
                # Log query details
                self.logger.query(query, params, duration)
                
                return result
                
        except Exception as e:
            duration = time.time() - start_time
            self.query_stats['failed_queries'] += 1
            self.logger.error(f"Query failed after {duration:.3f}s: {e}")
            raise
    
    def execute_transaction(self, operations: List[tuple]) -> List[Any]:
        """
        Execute multiple operations in a single transaction.
        
        Args:
            operations: List of (query, params) tuples
            
        Returns:
            List of results for each operation
        """
        start_time = time.time()
        results = []
        
        try:
            with self.get_connection() as conn:
                # Begin transaction
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    for query, params in operations:
                        if params:
                            result = conn.execute(query, params).fetchall()
                        else:
                            result = conn.execute(query).fetchall()
                        results.append(result)
                    
                    # Commit transaction
                    conn.execute("COMMIT")
                    
                    duration = time.time() - start_time
                    self.logger.transaction(
                        f"{len(operations)} operations", 
                        True, 
                        duration
                    )
                    
                    return results
                    
                except Exception as e:
                    # Rollback on error
                    conn.execute("ROLLBACK")
                    raise e
                    
        except Exception as e:
            duration = time.time() - start_time
            self.logger.transaction(
                f"{len(operations)} operations", 
                False, 
                duration, 
                str(e)
            )
            raise
    
    def execute_script(self, script: str) -> None:
        """
        Execute a SQL script (multiple statements).
        
        Args:
            script: SQL script with multiple statements
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                # Split script into individual statements
                statements = [stmt.strip() for stmt in script.split(';') if stmt.strip()]
                
                for statement in statements:
                    conn.execute(statement)
                
                duration = time.time() - start_time
                self.logger.info(f"Executed script with {len(statements)} statements in {duration:.3f}s")
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Script execution failed after {duration:.3f}s: {e}")
            raise
    
    def _track_query_performance(self, query: str, duration: float) -> None:
        """Track query performance metrics."""
        self.query_stats['total_queries'] += 1
        self.query_stats['total_query_time'] += duration
        
        # Check for slow queries
        slow_threshold = self.db_config['query']['slow_query_threshold']
        if duration > slow_threshold:
            self.query_stats['slow_queries'] += 1
            self.logger.warning(f"Slow query detected: {duration:.3f}s > {slow_threshold}s")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the connection manager."""
        stats = self.query_stats.copy()
        
        if stats['total_queries'] > 0:
            stats['avg_query_time'] = stats['total_query_time'] / stats['total_queries']
        else:
            stats['avg_query_time'] = 0.0
        
        stats['pool_stats'] = self.pool.get_stats()
        stats['database_path'] = self.db_path
        
        # Get database file size
        try:
            stats['database_size_mb'] = Path(self.db_path).stat().st_size / 1024 / 1024
        except:
            stats['database_size_mb'] = 0.0
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection."""
        try:
            with self.get_connection() as conn:
                # Basic connectivity test
                result = conn.execute("SELECT 1").fetchone()
                if not result or result[0] != 1:
                    raise RuntimeError("Basic query failed")
                
                # Get DuckDB version
                version_result = conn.execute("SELECT version()").fetchone()
                duckdb_version = version_result[0] if version_result else "unknown"
                
                return {
                    'status': 'healthy',
                    'database_path': self.db_path,
                    'duckdb_version': duckdb_version,
                    'pool_stats': self.pool.get_stats(),
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_path': self.db_path,
                'timestamp': time.time()
            }
    
    def close(self) -> None:
        """Close all connections and cleanup resources."""
        self.logger.info("Shutting down connection manager")
        
        if hasattr(self, 'pool'):
            self.pool.close_all()
        
        self.logger.info("Connection manager shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()