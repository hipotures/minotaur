"""
SQLAlchemy-based data manager as a drop-in replacement for DuckDBDataManager
"""

import os
import time
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from .database.engine_factory import DatabaseFactory
from .database.config import DatabaseConfig
from .timing import timed, timing_context, record_timing

logger = logging.getLogger(__name__)


class SQLAlchemyDataManager:
    """
    SQLAlchemy-based data manager that provides the same interface as DuckDBDataManager
    but supports multiple database backends (DuckDB, SQLite, PostgreSQL).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SQLAlchemy data manager with database configuration."""
        self.config = config
        # Get database-specific config based on current database type
        db_config = config.get('database', {})
        db_type = db_config.get('type', 'duckdb')
        self.database_config = config.get('data', {}).get('database_configs', {}).get(db_type, {})
        self.autogluon_config = config.get('autogluon', {})
        
        # Database configuration
        db_config = config.get('database', {})
        self.db_type = db_config.get('type', 'duckdb')  # Default to DuckDB for compatibility
        
        # Dataset path configuration - support both old and new systems
        dataset_name = self.autogluon_config.get('dataset_name')
        if dataset_name:
            # New system: use cached dataset
            cache_dir = Path(self.config.get('project_root', '.')) / 'cache' / dataset_name
            cache_db_path = cache_dir / 'dataset.duckdb'
            
            if cache_db_path.exists():
                self.train_path = str(cache_db_path)
                self.test_path = str(cache_db_path)  # Same file contains both tables
                self.use_cached_dataset = True
                self.dataset_name = dataset_name
                self.db_path = str(cache_db_path)
                self.db_dir = str(cache_dir)
            else:
                raise ValueError(f"Cached dataset not found: {cache_db_path}")
        else:
            # Legacy system: direct file paths
            self.train_path = self.autogluon_config.get('train_path')
            self.test_path = self.autogluon_config.get('test_path')
            self.use_cached_dataset = False
            self.dataset_name = None
            
            if not self.train_path:
                raise ValueError("Either 'dataset_name' or 'train_path' must be specified in autogluon configuration")
            
            # Generate database path based on data paths for legacy mode
            self.db_path = self._generate_database_path()
            self.db_dir = os.path.dirname(self.db_path)
        
        # Ensure database directory exists
        Path(self.db_dir).mkdir(parents=True, exist_ok=True)
        
        # Create database manager
        db_connection_config = self._build_database_config()
        self.db_manager = DatabaseFactory.create_manager(**db_connection_config)
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
        
        # Database state tracking
        self.tables_initialized = False
        self.data_loaded = False
        
        # Dataset metadata cache (for backward compatibility)
        self.dataset_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Column mapping cache - will be populated after data loading
        self.column_mapping: Dict[str, Optional[str]] = {}
        
        self._initialize_database_schema()
        self._check_and_load_data()
        
        logger.info(f"Initialized SQLAlchemy data manager ({self.db_type}) at: {self.db_path}")
    
    def _build_database_config(self) -> Dict[str, Any]:
        """Build database configuration for the manager."""
        if self.use_cached_dataset and self.db_type == 'duckdb':
            # Use the cached DuckDB file directly
            return {
                'db_type': 'duckdb',
                'connection_params': {
                    'database': self.db_path,
                    'engine_args': {
                        'echo': False,
                        'pool_pre_ping': True
                    }
                }
            }
        else:
            # Use in-memory or file-based database based on configuration
            return DatabaseConfig.get_default_config(self.db_type, self.db_path)
    
    def _generate_database_path(self) -> str:
        """Generate database path based on MD5 hash of data paths."""
        # Create hash from train and test paths
        path_string = f"{self.train_path}|{self.test_path or ''}"
        path_hash = hashlib.md5(path_string.encode()).hexdigest()
        
        # Use cache directory for dataset-specific database files
        project_root = Path(__file__).parent.parent
        db_dir = project_root / "cache" / path_hash
        
        # Choose file extension based on database type
        if self.db_type == 'sqlite':
            db_path = db_dir / "features.sqlite"
        elif self.db_type == 'duckdb':
            db_path = db_dir / "features.duckdb"
        else:
            # For PostgreSQL, use database name instead of file
            return f"features_{path_hash}"
        
        logger.info(f"Generated database path: {db_path} (hash: {path_hash})")
        return str(db_path)
    
    def _initialize_database_schema(self) -> None:
        """Initialize database schema with required tables."""
        try:
            # If using cached dataset, skip schema creation (tables already exist)
            if self.use_cached_dataset:
                logger.info("Using cached dataset - skipping schema initialization")
                return
            
            # Create tables if they don't exist (for legacy mode only)
            schema_sql = [
                """
                CREATE TABLE IF NOT EXISTS train_data (
                    id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL,
                    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS test_data (
                    id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL,
                    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS features_cache (
                    feature_hash VARCHAR(255) PRIMARY KEY,
                    feature_name VARCHAR(255) NOT NULL,
                    feature_data TEXT NOT NULL,
                    feature_params TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    evaluation_score DOUBLE PRECISION,
                    node_depth INTEGER
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    id INTEGER PRIMARY KEY,
                    feature_hash VARCHAR(255) NOT NULL,
                    operation_type VARCHAR(255) NOT NULL,
                    operation_params TEXT,
                    execution_time DOUBLE PRECISION,
                    memory_usage_mb DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            ]
            
            for sql in schema_sql:
                self.db_manager.execute_query(sql)
            
            # Create indexes for performance
            index_queries = [
                "CREATE INDEX IF NOT EXISTS idx_features_cache_score ON features_cache(evaluation_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_features_cache_created ON features_cache(created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_feature_metadata_hash ON feature_metadata(feature_hash)"
            ]
            
            for query in index_queries:
                try:
                    self.db_manager.execute_query(query)
                except Exception as e:
                    logger.debug(f"Index creation failed (may already exist): {e}")
            
            logger.info("Database schema initialized successfully")
            self.tables_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise
    
    def _check_and_load_data(self) -> None:
        """Check if data exists in database, load from files if needed."""
        try:
            if self.use_cached_dataset:
                # For cached datasets, check for train_features/test_features tables
                tables = self.db_manager.get_table_names()
                
                if 'train_features' in tables:
                    train_count = self.db_manager.count_rows('train_features')
                    logger.info(f"Found train_features table with {train_count} rows")
                    self.data_loaded = True
                elif 'train' in tables:
                    # train table exists but train_features doesn't - normal during feature generation
                    train_count = self.db_manager.count_rows('train')
                    logger.info(f"Using train table with {train_count} rows (features not generated yet)")
                    self.data_loaded = True
                elif len(tables) == 0:
                    # No tables at all - normal during initial dataset creation
                    logger.debug("No tables found in database - dataset creation in progress")
                    self.data_loaded = False
                else:
                    # Some tables exist but neither train nor train_features
                    logger.warning(f"Dataset contains tables {tables} but no train/train_features table")
                    raise ValueError("Neither train_features nor train table found in cached dataset")
                        
                if 'test_features' in tables:
                    test_count = self.db_manager.count_rows('test_features')
                    logger.info(f"Found test_features table with {test_count} rows")
                elif 'test' in tables:
                    test_count = self.db_manager.count_rows('test')
                    logger.info(f"Using test table with {test_count} rows (features not generated yet)")
            else:
                # Legacy mode - check train_data/test_data tables
                if self.db_manager.table_exists('train_data'):
                    train_count = self.db_manager.count_rows('train_data')
                    test_count = self.db_manager.count_rows('test_data') if self.db_manager.table_exists('test_data') else 0
                    
                    if train_count == 0:
                        logger.info("No training data found in database, loading from CSV...")
                        self._load_csv_to_database()
                    else:
                        logger.info(f"Found existing data: train={train_count}, test={test_count} rows")
                        self.data_loaded = True
                else:
                    logger.info("No data tables found, loading from CSV...")
                    self._load_csv_to_database()
                
        except Exception as e:
            logger.error(f"Failed to check/load data: {e}")
            raise
    
    def _load_csv_to_database(self) -> None:
        """Load data into database tables from cache or CSV files."""
        try:
            if self.use_cached_dataset:
                # New system: copy data from cached dataset
                self._load_from_cached_dataset()
            else:
                # Legacy system: load from CSV files
                self._load_from_csv_files()
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _load_from_cached_dataset(self) -> None:
        """Load data from cached dataset.duckdb file."""
        try:
            logger.info(f"Loading from cached dataset: {self.dataset_name}")
            
            # If we're using DuckDB and have access to the cached file, use attach/copy
            if self.db_type == 'duckdb' and hasattr(self.db_manager, 'attach_database'):
                self._load_from_cached_duckdb()
            else:
                # For other databases, read via pandas and load
                self._load_from_cached_via_pandas()
                
        except Exception as e:
            logger.error(f"Failed to load from cached dataset: {e}")
            raise
    
    def _load_from_cached_duckdb(self) -> None:
        """Load from cached DuckDB using database attachment."""
        cached_db_path = self.train_path  # Points to dataset.duckdb
        
        # Attach and copy tables
        self.db_manager.attach_database(cached_db_path, 'cached_db')
        
        try:
            # Copy train table
            train_copy_sql = """
                CREATE TABLE train_data AS 
                SELECT ROW_NUMBER() OVER () as id, 
                       to_json(struct_pack(*)) as data
                FROM cached_db.train
            """
            self.db_manager.execute_query(train_copy_sql)
            logger.info("Copied train table as train_data")
            
            # Copy test table if it exists
            try:
                test_copy_sql = """
                    CREATE TABLE test_data AS 
                    SELECT ROW_NUMBER() OVER () as id, 
                           to_json(struct_pack(*)) as data
                    FROM cached_db.test
                """
                self.db_manager.execute_query(test_copy_sql)
                logger.info("Copied test table as test_data")
            except Exception as e:
                logger.warning(f"No test table found in cached dataset: {e}")
            
        finally:
            self.db_manager.detach_database('cached_db')
    
    def _load_from_cached_via_pandas(self) -> None:
        """Load from cached dataset via pandas (for non-DuckDB targets)."""
        # Use a temporary DuckDB connection to read the cached data
        temp_config = DatabaseConfig.get_default_config('duckdb', self.train_path)
        temp_manager = DatabaseFactory.create_manager(**temp_config)
        
        try:
            # Read train data
            if temp_manager.table_exists('train'):
                train_df = temp_manager.execute_query_df("SELECT * FROM train")
                # Convert to JSON format expected by legacy system
                train_json_data = []
                for _, row in train_df.iterrows():
                    row_dict = row.to_dict()
                    train_json_data.append({
                        'id': len(train_json_data) + 1,
                        'data': json.dumps(row_dict)
                    })
                
                train_json_df = pd.DataFrame(train_json_data)
                self.db_manager.bulk_insert_from_pandas(train_json_df, 'train_data', if_exists='replace')
                logger.info(f"Loaded {len(train_json_df)} train rows via pandas")
            
            # Read test data if available
            if temp_manager.table_exists('test'):
                test_df = temp_manager.execute_query_df("SELECT * FROM test")
                test_json_data = []
                for _, row in test_df.iterrows():
                    row_dict = row.to_dict()
                    test_json_data.append({
                        'id': len(test_json_data) + 1,
                        'data': json.dumps(row_dict)
                    })
                
                test_json_df = pd.DataFrame(test_json_data)
                self.db_manager.bulk_insert_from_pandas(test_json_df, 'test_data', if_exists='replace')
                logger.info(f"Loaded {len(test_json_df)} test rows via pandas")
                
        finally:
            temp_manager.close()
    
    def _load_from_csv_files(self) -> None:
        """Load data from CSV files (legacy system)."""
        try:
            logger.info(f"Loading training data from: {self.train_path}")
            
            # Load train data
            train_df = pd.read_csv(self.train_path)
            train_json_data = []
            
            for _, row in train_df.iterrows():
                row_dict = row.to_dict()
                # Convert NaN to None for JSON serialization
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
                train_json_data.append({
                    'id': len(train_json_data) + 1,
                    'data': json.dumps(row_dict)
                })
            
            train_json_df = pd.DataFrame(train_json_data)
            self.db_manager.bulk_insert_from_pandas(train_json_df, 'train_data', if_exists='replace')
            
            # Build column mapping
            self.column_mapping = {col: col for col in train_df.columns}
            logger.info(f"Loaded {len(train_json_df)} training samples into database")
            
            # Load test data if available
            if self.test_path and os.path.exists(self.test_path):
                logger.info(f"Loading test data from: {self.test_path}")
                
                test_df = pd.read_csv(self.test_path)
                test_json_data = []
                
                for _, row in test_df.iterrows():
                    row_dict = row.to_dict()
                    row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
                    test_json_data.append({
                        'id': len(test_json_data) + 1,
                        'data': json.dumps(row_dict)
                    })
                
                test_json_df = pd.DataFrame(test_json_data)
                self.db_manager.bulk_insert_from_pandas(test_json_df, 'test_data', if_exists='replace')
                logger.info(f"Loaded {len(test_json_df)} test samples into database")
            
            self.data_loaded = True
            logger.info("CSV data successfully loaded into database")
            
        except Exception as e:
            logger.error(f"Failed to load CSV data to database: {e}")
            raise
    
    # Public API methods (matching DuckDBDataManager interface)
    
    @timed("sqlalchemy.sample_dataset", include_memory=True)
    def sample_dataset(self, 
                      file_path: str, 
                      train_size: Union[int, float], 
                      stratify_column: Optional[str] = None,
                      random_seed: int = 42) -> pd.DataFrame:
        """Sample dataset efficiently using the database backend."""
        start_time = time.time()
        
        try:
            # Determine which table to use based on file_path
            if file_path == self.train_path:
                table_name = self._get_train_table_name()
            elif file_path == self.test_path:
                table_name = self._get_test_table_name()
            else:
                # Fallback to file-based loading
                logger.warning(f"Unknown file path {file_path}, loading from file")
                return pd.read_csv(file_path).sample(n=int(train_size), random_state=random_seed)
            
            # Get total row count
            total_rows = self.db_manager.count_rows(table_name)
            
            # Determine sample size
            if isinstance(train_size, float) and 0 <= train_size <= 1:
                n_samples = int(total_rows * train_size)
                logger.info(f"🎲 Sampling {train_size*100:.1f}% = {n_samples}/{total_rows} rows from {table_name}")
            else:
                n_samples = min(int(train_size), total_rows)
                logger.info(f"🎲 Sampling {n_samples}/{total_rows} rows from {table_name}")
            
            # If sample size >= total rows, load everything
            if n_samples >= total_rows:
                logger.info("Sample size >= dataset size, loading full dataset")
                return self.load_full_dataset(file_path)
            
            # Use database-specific sampling
            if self.use_cached_dataset and table_name in ['train_features', 'test_features', 'train', 'test']:
                # Direct sampling for feature tables
                if hasattr(self.db_manager, 'sample_reservoir'):
                    sample_query = self.db_manager.sample_reservoir(table_name, n_samples, random_seed)
                elif hasattr(self.db_manager, 'sample_random'):
                    sample_query = self.db_manager.sample_random(table_name, n_samples, random_seed)
                else:
                    sample_query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {n_samples}"
                
                result_df = self.db_manager.execute_query_df(sample_query)
            else:
                # For JSON storage tables, extract and sample
                if hasattr(self.db_manager, 'sample_reservoir'):
                    sample_sql = f"""
                        SELECT data FROM (
                            {self.db_manager.sample_reservoir(table_name, n_samples, random_seed)}
                        )
                    """
                else:
                    sample_sql = f"""
                        SELECT data FROM {table_name} 
                        ORDER BY RANDOM() 
                        LIMIT {n_samples}
                    """
                
                sample_data = self.db_manager.execute_query(sample_sql)
                
                # Convert JSON data back to DataFrame
                rows = []
                for record in sample_data:
                    row_data = json.loads(record['data'])
                    rows.append(row_data)
                
                result_df = pd.DataFrame(rows)
            
            # Convert columns to proper data types
            result_df = self._convert_dataframe_types(result_df)
            
            query_time = time.time() - start_time
            self.total_query_time += query_time
            self.query_count += 1
            
            actual_rows = len(result_df)
            logger.info(f"✅ Sampled {actual_rows} rows in {query_time:.3f}s")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Sampling failed for {file_path}: {e}")
            # Fallback to pandas
            return pd.read_csv(file_path).sample(n=int(train_size), random_state=random_seed)
    
    def _get_train_table_name(self) -> str:
        """Get the appropriate train table name."""
        if self.use_cached_dataset:
            tables = self.db_manager.get_table_names()
            return 'train_features' if 'train_features' in tables else 'train'
        else:
            return 'train_data'
    
    def _get_test_table_name(self) -> str:
        """Get the appropriate test table name."""
        if self.use_cached_dataset:
            tables = self.db_manager.get_table_names()
            return 'test_features' if 'test_features' in tables else 'test'
        else:
            return 'test_data'
    
    @timed("sqlalchemy.load_full_dataset")
    def load_full_dataset(self, file_path: str) -> pd.DataFrame:
        """Load full dataset from database."""
        try:
            start_time = time.time()
            
            # Determine table name
            if file_path == self.train_path:
                table_name = self._get_train_table_name()
            elif file_path == self.test_path:
                table_name = self._get_test_table_name()
            else:
                # Fallback to file loading
                logger.warning(f"Unknown file path {file_path}, loading from file")
                return pd.read_csv(file_path)
            
            # Load data based on table type
            if self.use_cached_dataset and table_name in ['train_features', 'test_features', 'train', 'test']:
                # Direct query for feature tables
                result_df = self.db_manager.execute_query_df(f"SELECT * FROM {table_name}")
            else:
                # Extract from JSON storage
                data_records = self.db_manager.execute_query(f"SELECT data FROM {table_name}")
                
                rows = []
                for record in data_records:
                    row_data = json.loads(record['data'])
                    rows.append(row_data)
                
                result_df = pd.DataFrame(rows)
            
            # Convert columns to proper data types
            result_df = self._convert_dataframe_types(result_df)
            
            query_time = time.time() - start_time
            self.total_query_time += query_time
            self.query_count += 1
            
            logger.info(f"Loaded full dataset from {table_name}: {len(result_df)} rows in {query_time:.3f}s")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to load full dataset from {file_path}: {e}")
            raise
    
    def _convert_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to proper data types dynamically."""
        try:
            # Convert id to integer if present
            if 'id' in df.columns:
                df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')
            
            # For other columns, try to convert numeric where possible
            for col in df.columns:
                if col == 'id':
                    continue
                
                # Check if column likely contains categorical/string data
                if any(keyword in col.lower() for keyword in ['name', 'type', 'category', 'class', 'label']):
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace('None', 'Unknown').fillna('Unknown')
                else:
                    # Try to convert to numeric
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].fillna(0.0)
                    except:
                        # If conversion fails, keep as string
                        df[col] = df[col].astype(str).fillna('Unknown')
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to convert data types: {e}")
            return df
    
    # Feature caching methods (delegate to database manager)
    
    def cache_features(self, 
                      feature_hash: str,
                      feature_name: str, 
                      features_df: pd.DataFrame,
                      feature_params: Dict[str, Any] = None,
                      evaluation_score: float = None,
                      node_depth: int = None) -> None:
        """Cache generated features in database."""
        try:
            feature_data = features_df.to_json(orient='records')
            
            data_to_insert = [{
                'feature_hash': feature_hash,
                'feature_name': feature_name,
                'feature_data': feature_data,
                'feature_params': json.dumps(feature_params) if feature_params else None,
                'evaluation_score': evaluation_score,
                'node_depth': node_depth
            }]
            
            # Use INSERT OR REPLACE for SQLite, UPSERT for PostgreSQL
            if self.db_manager.table_exists('features_cache'):
                # Delete existing entry first, then insert
                self.db_manager.delete_data('features_cache', {'feature_hash': feature_hash})
            
            self.db_manager.insert_data('features_cache', data_to_insert)
            logger.debug(f"Cached features: {feature_name} (hash: {feature_hash[:8]}...)")
            
        except Exception as e:
            logger.error(f"Failed to cache features {feature_name}: {e}")
    
    def get_cached_features(self, feature_hash: str) -> Optional[pd.DataFrame]:
        """Retrieve cached features from database."""
        try:
            results = self.db_manager.execute_query(
                "SELECT feature_data FROM features_cache WHERE feature_hash = :hash",
                {'hash': feature_hash}
            )
            
            if results:
                feature_data_json = results[0]['feature_data']
                features_df = pd.read_json(feature_data_json, orient='records')
                self.cache_hits += 1
                logger.debug(f"Cache hit for feature hash: {feature_hash[:8]}...")
                return features_df
            else:
                logger.debug(f"Cache miss for feature hash: {feature_hash[:8]}...")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve cached features: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_query_time = self.total_query_time / self.query_count if self.query_count > 0 else 0
        
        # Get feature cache stats
        try:
            cache_count = self.db_manager.count_rows('features_cache') if self.db_manager.table_exists('features_cache') else 0
        except:
            cache_count = 0
        
        return {
            'total_queries': self.query_count,
            'total_query_time': self.total_query_time,
            'avg_query_time': avg_query_time,
            'cache_hits': self.cache_hits,
            'cached_datasets': len(self.dataset_metadata),
            'database_type': self.db_type,
            'database_path': self.db_path,
            'tables_initialized': self.tables_initialized,
            'data_loaded': self.data_loaded,
            'total_cached_features': cache_count
        }
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute arbitrary SQL query and return DataFrame."""
        try:
            start_time = time.time()
            result = self.db_manager.execute_query_df(query)
            query_time = time.time() - start_time
            
            self.total_query_time += query_time
            self.query_count += 1
            
            logger.debug(f"Executed custom query in {query_time:.3f}s, returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_query_dict(self, query: str) -> List[Dict[str, Any]]:
        """Execute arbitrary SQL query and return list of dictionaries."""
        try:
            start_time = time.time()
            result = self.db_manager.execute_query(query)
            query_time = time.time() - start_time
            
            self.total_query_time += query_time
            self.query_count += 1
            
            logger.debug(f"Executed custom query in {query_time:.3f}s, returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self, 'db_manager'):
            self.db_manager.close()
            logger.debug("SQLAlchemy database connections closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()