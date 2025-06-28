"""
Database configuration settings.

This module defines all database-specific configuration parameters that are separate
from the main MCTS configuration. Logging levels are still controlled by mcts_config.yaml.
"""

from typing import Dict, Any

# DuckDB-specific configuration
DUCKDB_CONFIG = {
    'connection': {
        'pool_size': 5,                    # Maximum number of connections in pool
        'timeout': 30,                     # Connection timeout in seconds
        'retry_attempts': 3,               # Number of retry attempts for failed connections
        'retry_delay': 1.0,                # Delay between retry attempts in seconds
        'max_idle_time': 300,              # Max idle time before connection cleanup (seconds)
    },
    'performance': {
        'memory_limit': '4GB',             # DuckDB memory limit
        'threads': 'auto',                 # Number of threads ('auto' = CPU count)
        'enable_object_cache': True,       # Enable DuckDB object cache
        'force_compression': 'zstd',       # Compression algorithm
        'enable_progress_bar': False,      # Disable progress bar for better logging
        'temp_directory': '/tmp/duckdb',   # Temporary directory for DuckDB
    },
    'query': {
        'timeout': 30,                     # Query timeout in seconds
        'max_batch_size': 1000,            # Maximum batch size for bulk operations
        'enable_query_cache': True,        # Enable query result caching
        'cache_size_mb': 256,              # Query cache size in MB
        'slow_query_threshold': 1.0,       # Log queries slower than this (seconds)
    },
    'maintenance': {
        'vacuum_interval_hours': 24,       # Run VACUUM every N hours
        'analyze_interval_hours': 6,       # Run ANALYZE every N hours
        'checkpoint_interval': 1000,       # Checkpoint every N operations
        'backup_retention_days': 30,       # Keep backups for N days
    },
    'transaction': {
        'isolation_level': 'READ_COMMITTED', # Transaction isolation level
        'lock_timeout': 30,                # Lock timeout in seconds
        'max_transaction_size': 10000,     # Max operations per transaction
    }
}

def get_duckdb_config() -> Dict[str, Any]:
    """Get DuckDB configuration dictionary."""
    return DUCKDB_CONFIG.copy()

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate database configuration parameters."""
    try:
        # Validate connection settings
        conn_config = config.get('connection', {})
        if conn_config.get('pool_size', 0) <= 0:
            raise ValueError("pool_size must be positive")
        
        if conn_config.get('timeout', 0) <= 0:
            raise ValueError("timeout must be positive")
        
        # Validate performance settings
        perf_config = config.get('performance', {})
        memory_limit = perf_config.get('memory_limit', '4GB')
        if not isinstance(memory_limit, str) or not memory_limit.endswith(('GB', 'MB')):
            raise ValueError("memory_limit must be a string ending with 'GB' or 'MB'")
        
        # Validate query settings
        query_config = config.get('query', {})
        if query_config.get('timeout', 0) <= 0:
            raise ValueError("query timeout must be positive")
        
        return True
        
    except Exception as e:
        print(f"❌ Database configuration validation failed: {e}")
        return False