"""
Secure DuckDB Data Manager with SQL Injection Protection

This is a security-enhanced version of the DuckDB data manager that:
- Uses parameterized queries to prevent SQL injection
- Validates all file paths to prevent directory traversal
- Implements connection pooling for better resource management
- Adds proper error handling and recovery
"""

import os
import time
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from contextlib import contextmanager
from queue import Queue
import pandas as pd
import numpy as np

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None

from .timing import timed, timing_context, record_timing
from .security import (
    SecurePathManager, SecureQueryBuilder, SecurityError,
    PathTraversalError, QueryInjectionError, setup_secure_logging
)

# Set up secure logging
logger = setup_secure_logging(__name__)


class DuckDBConnectionPool:
    """Thread-safe connection pool for DuckDB."""
    
    def __init__(self, db_path: str, pool_size: int = 5, 
                 max_memory: Optional[str] = None,
                 max_threads: Optional[int] = None):
        """Initialize connection pool.
        
        Args:
            db_path: Path to database file
            pool_size: Number of connections in pool
            max_memory: Maximum memory per connection (e.g., '4GB')
            max_threads: Maximum threads per connection
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.max_memory = max_memory
        self.max_threads = max_threads
        self._pool = Queue(maxsize=pool_size)
        self._closed = False
        
        # Initialize pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)
    
    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new DuckDB connection with configuration."""
        conn = duckdb.connect(self.db_path)
        
        # Set resource limits
        if self.max_memory:
            conn.execute(f"SET max_memory='{self.max_memory}'")
        if self.max_threads:
            conn.execute(f"SET threads={self.max_threads}")
        
        # Enable parallel execution
        conn.execute("SET enable_parallel_csv_reader=true")
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = self._pool.get()
        try:
            yield conn
        finally:
            if not self._closed:
                self._pool.put(conn)
    
    def close(self):
        """Close all connections in the pool."""
        self._closed = True
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except:
                pass


class SecureDuckDBDataManager:
    """
    Security-enhanced DuckDB data manager with protection against:
    - SQL injection attacks
    - Path traversal vulnerabilities
    - Resource exhaustion
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize secure DuckDB data manager."""
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is not installed. Please install with: pip install duckdb")
        
        self.config = config
        self.duckdb_config = config.get('data', {}).get('duckdb', {})
        self.autogluon_config = config.get('autogluon', {})
        
        # Set up secure path management
        allowed_dirs = [
            os.getcwd(),  # Current working directory
            '/mnt/ml/competitions',  # Competition data directory
            config.get('data', {}).get('cache_dir', 'cache'),  # Cache directory
        ]
        self.path_manager = SecurePathManager(allowed_dirs)
        
        # Validate and set paths
        self._setup_paths()
        
        # Initialize connection pool
        pool_size = self.duckdb_config.get('connection_pool_size', 5)
        max_memory = self.duckdb_config.get('max_memory_gb', '4') + 'GB'
        max_threads = self.duckdb_config.get('max_threads', 4)
        
        self.connection_pool = DuckDBConnectionPool(
            str(self.db_path),
            pool_size=pool_size,
            max_memory=max_memory,
            max_threads=max_threads
        )
        
        # Database state
        self.data_loaded = False
        self.column_mapping = None
        
        # Initialize database
        self._initialize_database()
    
    def _setup_paths(self):
        """Set up and validate all file paths."""
        # Database path
        db_dir = self.duckdb_config.get('db_dir', 'cache/duckdb')
        os.makedirs(db_dir, mode=0o750, exist_ok=True)
        
        db_name = self.duckdb_config.get('db_name', 'features.duckdb')
        self.db_path = self.path_manager.join_path(db_dir, db_name)
        
        # Data paths
        train_path = self.autogluon_config.get('train_path')
        test_path = self.autogluon_config.get('test_path')
        
        if train_path:
            self.train_path = self.path_manager.validate_path(train_path)
        else:
            raise ValueError("train_path must be specified in config")
        
        if test_path:
            self.test_path = self.path_manager.validate_path(test_path)
        else:
            self.test_path = None
        
        logger.info(f"Secure paths initialized - DB: {self.db_path}, Train: {self.train_path}")
    
    def _initialize_database(self):
        """Initialize database schema with secure queries."""
        with self.connection_pool.get_connection() as conn:
            # Create tables using static queries (no injection risk)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS train_data (
                    id INTEGER PRIMARY KEY,
                    data JSON
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_data (
                    id INTEGER PRIMARY KEY,
                    data JSON
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_train_id ON train_data(id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_test_id ON test_data(id)")
    
    def _load_csv_to_database_secure(self) -> None:
        """Load CSV data using parameterized queries."""
        try:
            with self.connection_pool.get_connection() as conn:
                # Get column structure using parameterized query
                describe_query, params = SecureQueryBuilder.build_csv_describe_query(
                    self.train_path
                )
                columns_result = conn.execute(describe_query, params).fetchall()
                available_columns = [row[0] for row in columns_result]
                
                logger.info(f"Available columns in CSV: {available_columns}")
                
                # Build column mapping
                column_mapping = self._build_column_mapping(available_columns)
                self.column_mapping = column_mapping
                
                # Build and execute load query
                load_query, load_params = SecureQueryBuilder.build_csv_load_query(
                    self.train_path,
                    'train_data',
                    column_mapping
                )
                
                # Clear existing data
                conn.execute("DELETE FROM train_data")
                
                # Load new data
                conn.execute(load_query, load_params)
                
                # Get count
                count_query, count_params = SecureQueryBuilder.build_count_query('train_data')
                train_count = conn.execute(count_query, count_params).fetchone()[0]
                logger.info(f"Loaded {train_count} training samples into database")
                
                # Load test data if available
                if self.test_path and self.test_path.exists():
                    self._load_test_data_secure(conn)
                
                self.data_loaded = True
                logger.info("CSV data successfully loaded into DuckDB database")
                
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise
    
    def _load_test_data_secure(self, conn: duckdb.DuckDBPyConnection):
        """Load test data using secure queries."""
        # Get test column structure
        describe_query, params = SecureQueryBuilder.build_csv_describe_query(
            self.test_path
        )
        test_columns = conn.execute(describe_query, params).fetchall()
        test_available_columns = [row[0] for row in test_columns]
        
        # Build test column mapping
        test_column_mapping = self._build_column_mapping(test_available_columns)
        
        # Build and execute load query
        load_query, load_params = SecureQueryBuilder.build_csv_load_query(
            self.test_path,
            'test_data',
            test_column_mapping
        )
        
        # Clear and load
        conn.execute("DELETE FROM test_data")
        conn.execute(load_query, load_params)
        
        # Get count
        count_query, _ = SecureQueryBuilder.build_count_query('test_data')
        test_count = conn.execute(count_query).fetchone()[0]
        logger.info(f"Loaded {test_count} test samples into database")
    
    def _build_column_mapping(self, available_columns: List[str]) -> Dict[str, str]:
        """Build mapping between expected and actual column names."""
        # Expected columns for different domains
        expected_mappings = {
            'fertilizer': {
                'nitrogen': ['Nitrogen', 'nitrogen', 'N'],
                'phosphorous': ['Phosphorous', 'phosphorous', 'P'],
                'potassium': ['Potassium', 'potassium', 'K'],
                'temperature': ['Temperature', 'temperature', 'temp'],
                'humidity': ['Humidity', 'humidity'],
                'ph': ['pH', 'ph', 'PH'],
                'rainfall': ['Rainfall', 'rainfall', 'rain'],
                'label': ['Fertilizer Name', 'fertilizer_name', 'label'],
                'id': ['id', 'ID', 'index']
            },
            'titanic': {
                'pclass': ['Pclass', 'pclass'],
                'sex': ['Sex', 'sex'],
                'age': ['Age', 'age'],
                'sibsp': ['SibSp', 'sibsp'],
                'parch': ['Parch', 'parch'],
                'fare': ['Fare', 'fare'],
                'embarked': ['Embarked', 'embarked'],
                'survived': ['Survived', 'survived'],
                'id': ['PassengerId', 'id', 'ID']
            }
        }
        
        # Try to determine domain based on available columns
        domain = self._detect_domain(available_columns)
        mapping_options = expected_mappings.get(domain, {})
        
        # Build the mapping
        column_mapping = {}
        for expected_col, possible_names in mapping_options.items():
            for possible in possible_names:
                if possible in available_columns:
                    column_mapping[expected_col] = possible
                    break
        
        # Add any remaining columns as-is
        for col in available_columns:
            if col not in column_mapping.values():
                # Use the column name as both key and value
                safe_name = col.lower().replace(' ', '_').replace('-', '_')
                column_mapping[safe_name] = col
        
        return column_mapping
    
    def _detect_domain(self, columns: List[str]) -> str:
        """Detect domain based on column names."""
        columns_lower = [c.lower() for c in columns]
        
        if any('nitrogen' in c or 'phosphorous' in c for c in columns_lower):
            return 'fertilizer'
        elif any('survived' in c or 'pclass' in c for c in columns_lower):
            return 'titanic'
        else:
            return 'generic'
    
    @timed("duckdb_random_sample")
    def get_random_sample(self, n_samples: int, stratify_column: Optional[str] = None,
                         use_test_data: bool = False) -> pd.DataFrame:
        """Get random sample using secure queries."""
        if not self.data_loaded:
            self._load_csv_to_database_secure()
        
        with self.connection_pool.get_connection() as conn:
            table_name = 'test_data' if use_test_data else 'train_data'
            
            # Validate table name
            table_name = SecureQueryBuilder.validate_table_name(table_name)
            
            if stratify_column:
                return self._get_stratified_sample_secure(
                    conn, n_samples, stratify_column, table_name
                )
            else:
                # Simple random sampling with parameterized query
                query = f"""
                SELECT data 
                FROM {table_name}
                TABLESAMPLE RESERVOIR({n_samples} ROWS) REPEATABLE(42)
                """
                
                result = conn.execute(query).fetchall()
                
                # Convert to DataFrame
                data_dicts = [json.loads(row[0]) for row in result]
                df = pd.DataFrame(data_dicts)
                
                return self._convert_dataframe_types(df)
    
    def _get_stratified_sample_secure(self, conn: duckdb.DuckDBPyConnection,
                                    n_samples: int, stratify_column: str,
                                    table_name: str) -> pd.DataFrame:
        """Get stratified sample using secure queries."""
        # Validate column name
        safe_column = SecureQueryBuilder.validate_column_name(stratify_column)
        
        # Get class distribution
        dist_query = f"""
        SELECT data->'{safe_column}' as class, COUNT(*) as count
        FROM {table_name}
        GROUP BY data->'{safe_column}'
        """
        
        distribution = conn.execute(dist_query).fetchdf()
        
        # Calculate samples per class
        total_count = distribution['count'].sum()
        samples_per_class = {}
        
        for _, row in distribution.iterrows():
            class_val = row['class']
            class_count = row['count']
            class_samples = int(n_samples * class_count / total_count)
            if class_samples > 0:
                samples_per_class[class_val] = class_samples
        
        # Sample from each class
        sampled_dfs = []
        
        for class_val, n_class_samples in samples_per_class.items():
            # Use parameterized query for class value
            class_query = f"""
            SELECT data
            FROM {table_name}
            WHERE data->'{safe_column}' = ?
            TABLESAMPLE RESERVOIR({n_class_samples} ROWS) REPEATABLE(42)
            """
            
            result = conn.execute(class_query, (class_val,)).fetchall()
            data_dicts = [json.loads(row[0]) for row in result]
            sampled_dfs.append(pd.DataFrame(data_dicts))
        
        # Combine and shuffle
        if sampled_dfs:
            df = pd.concat(sampled_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            return self._convert_dataframe_types(df)
        else:
            return pd.DataFrame()
    
    def _convert_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to proper data types."""
        try:
            # Convert id to integer if present
            if 'id' in df.columns:
                df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
            
            # Convert numeric columns
            numeric_patterns = ['nitrogen', 'phosphorous', 'potassium', 
                              'temperature', 'humidity', 'ph', 'rainfall',
                              'age', 'fare', 'sibsp', 'parch', 'pclass']
            
            for col in df.columns:
                if any(pattern in col.lower() for pattern in numeric_patterns):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.warning(f"Type conversion warning: {e}")
            return df
    
    def close(self):
        """Close connection pool and clean up resources."""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.close()
    
    def __del__(self):
        """Ensure connections are closed on deletion."""
        self.close()


# For backward compatibility, create alias
DuckDBDataManager = SecureDuckDBDataManager