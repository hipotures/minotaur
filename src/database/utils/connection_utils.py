"""
Connection utilities for database operations
"""

from sqlalchemy.engine import Engine
from typing import Dict, Any, Optional, List
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ConnectionUtils:
    """Utilities for database connection management and monitoring"""
    
    @staticmethod
    def test_connection(engine: Engine) -> Dict[str, Any]:
        """
        Test database connection and return status information
        
        Args:
            engine: SQLAlchemy engine to test
            
        Returns:
            Dictionary with connection test results
        """
        result = {
            'success': False,
            'error': None,
            'response_time_ms': None,
            'database_type': None,
            'database_version': None
        }
        
        start_time = time.time()
        
        try:
            with engine.connect() as conn:
                # Basic connectivity test
                conn.execute("SELECT 1")
                
                result['success'] = True
                result['response_time_ms'] = round((time.time() - start_time) * 1000, 2)
                result['database_type'] = engine.dialect.name
                
                # Try to get database version
                try:
                    if engine.dialect.name == 'duckdb':
                        version_result = conn.execute("SELECT version()")
                        result['database_version'] = version_result.scalar()
                    elif engine.dialect.name == 'sqlite':
                        version_result = conn.execute("SELECT sqlite_version()")
                        result['database_version'] = version_result.scalar()
                    elif engine.dialect.name == 'postgresql':
                        version_result = conn.execute("SELECT version()")
                        result['database_version'] = version_result.scalar()
                except Exception as e:
                    logger.debug(f"Could not get database version: {e}")
                
        except Exception as e:
            result['error'] = str(e)
            result['response_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        return result
    
    @staticmethod
    def get_connection_info(engine: Engine) -> Dict[str, Any]:
        """
        Get detailed connection information
        
        Args:
            engine: SQLAlchemy engine
            
        Returns:
            Dictionary with connection details
        """
        info = {
            'url': str(engine.url),
            'dialect': engine.dialect.name,
            'driver': engine.dialect.driver,
            'pool_size': getattr(engine.pool, 'size', None),
            'pool_checked_out': getattr(engine.pool, 'checkedout', None),
            'pool_overflow': getattr(engine.pool, 'overflow', None),
            'pool_checked_in': getattr(engine.pool, 'checkedin', None),
        }
        
        # Remove password from URL for security
        if hasattr(engine.url, 'password') and engine.url.password:
            safe_url = engine.url._replace(password='***')
            info['url'] = str(safe_url)
        
        return info
    
    @staticmethod
    @contextmanager
    def connection_with_timeout(engine: Engine, timeout_seconds: int = 30):
        """
        Context manager for database connection with timeout
        
        Args:
            engine: SQLAlchemy engine
            timeout_seconds: Connection timeout in seconds
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Database connection timed out after {timeout_seconds} seconds")
        
        # Set timeout alarm (Unix only)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            with engine.connect() as conn:
                yield conn
                
        finally:
            signal.alarm(0)  # Cancel alarm
            if old_handler:
                signal.signal(signal.SIGALRM, old_handler)
    
    @staticmethod
    def execute_with_retry(engine: Engine, query: str, params: Optional[Dict] = None,
                          max_retries: int = 3, retry_delay: float = 1.0) -> Any:
        """
        Execute query with retry logic
        
        Args:
            engine: SQLAlchemy engine
            query: SQL query to execute
            params: Query parameters
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Query result
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                with engine.connect() as conn:
                    if params:
                        result = conn.execute(query, params)
                    else:
                        result = conn.execute(query)
                    return result
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Query failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Query failed after {max_retries + 1} attempts: {e}")
                    raise last_error
    
    @staticmethod
    def optimize_connection_pool(engine: Engine, db_type: str) -> Dict[str, Any]:
        """
        Get recommended connection pool settings for database type
        
        Args:
            engine: SQLAlchemy engine
            db_type: Database type
            
        Returns:
            Recommended pool settings
        """
        recommendations = {}
        
        if db_type == 'sqlite':
            # SQLite doesn't benefit from connection pooling
            recommendations = {
                'pool_size': 1,
                'max_overflow': 0,
                'pool_pre_ping': True,
                'pool_recycle': -1,
                'reason': 'SQLite is single-threaded, minimal pooling recommended'
            }
        elif db_type == 'duckdb':
            # DuckDB can handle some concurrency but is primarily single-threaded
            recommendations = {
                'pool_size': 5,
                'max_overflow': 10,
                'pool_pre_ping': True,
                'pool_recycle': 3600,
                'reason': 'DuckDB benefits from small pool for analytical workloads'
            }
        elif db_type == 'postgresql':
            # PostgreSQL can handle high concurrency
            recommendations = {
                'pool_size': 20,
                'max_overflow': 30,
                'pool_pre_ping': True,
                'pool_recycle': 3600,
                'reason': 'PostgreSQL supports high concurrency'
            }
        else:
            # Default settings
            recommendations = {
                'pool_size': 10,
                'max_overflow': 20,
                'pool_pre_ping': True,
                'pool_recycle': 3600,
                'reason': 'Default settings for unknown database type'
            }
        
        return recommendations
    
    @staticmethod
    def monitor_connection_health(engine: Engine, check_interval: int = 60) -> Dict[str, Any]:
        """
        Monitor connection pool health
        
        Args:
            engine: SQLAlchemy engine
            check_interval: Check interval in seconds
            
        Returns:
            Health status information
        """
        health_info = {
            'timestamp': time.time(),
            'pool_status': 'unknown',
            'active_connections': 0,
            'total_connections': 0,
            'errors': []
        }
        
        try:
            # Test basic connectivity
            test_result = ConnectionUtils.test_connection(engine)
            
            if test_result['success']:
                health_info['pool_status'] = 'healthy'
            else:
                health_info['pool_status'] = 'unhealthy'
                health_info['errors'].append(test_result['error'])
            
            # Get pool statistics if available
            if hasattr(engine.pool, 'checkedout'):
                health_info['active_connections'] = engine.pool.checkedout()
            
            if hasattr(engine.pool, 'size'):
                health_info['total_connections'] = engine.pool.size()
            
        except Exception as e:
            health_info['pool_status'] = 'error'
            health_info['errors'].append(str(e))
        
        return health_info
    
    @staticmethod
    def close_all_connections(engine: Engine) -> None:
        """
        Close all connections in the pool
        
        Args:
            engine: SQLAlchemy engine
        """
        try:
            engine.dispose()
            logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")


class DatabaseHealthChecker:
    """Class for continuous database health monitoring"""
    
    def __init__(self, engine: Engine, check_interval: int = 60):
        """
        Initialize health checker
        
        Args:
            engine: SQLAlchemy engine to monitor
            check_interval: Check interval in seconds
        """
        self.engine = engine
        self.check_interval = check_interval
        self.health_history: List[Dict[str, Any]] = []
        self.max_history = 100  # Keep last 100 checks
    
    def check_health(self) -> Dict[str, Any]:
        """Perform health check and store result"""
        health_info = ConnectionUtils.monitor_connection_health(
            self.engine, self.check_interval
        )
        
        # Add to history
        self.health_history.append(health_info)
        
        # Trim history if needed
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]
        
        return health_info
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of recent health checks"""
        if not self.health_history:
            return {'status': 'no_data', 'checks': 0}
        
        recent_checks = self.health_history[-10:]  # Last 10 checks
        healthy_checks = sum(1 for check in recent_checks if check['pool_status'] == 'healthy')
        
        return {
            'status': 'healthy' if healthy_checks >= len(recent_checks) * 0.8 else 'degraded',
            'checks': len(recent_checks),
            'healthy_ratio': healthy_checks / len(recent_checks),
            'last_check': self.health_history[-1],
            'total_history': len(self.health_history)
        }