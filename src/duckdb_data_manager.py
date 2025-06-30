"""
DuckDB Data Manager for Efficient Random Sampling

Provides high-performance data loading with true random sampling capabilities
without loading entire datasets into memory. Designed for large-scale 
feature discovery with minimal memory footprint.
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

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None

from .timing import timed, timing_context, record_timing

logger = logging.getLogger(__name__)

class DuckDBDataManager:
    """
    High-performance data manager using DuckDB for efficient sampling.
    
    Key advantages:
    - True random sampling without loading full dataset
    - Columnar processing for fast access
    - SQL-based operations for complex queries
    - Memory-efficient streaming processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DuckDB data manager with persistent database."""
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is not installed. Please install with: pip install duckdb")
        
        self.config = config
        self.duckdb_config = config.get('data', {}).get('duckdb', {})
        self.autogluon_config = config.get('autogluon', {})
        
        # Database path configuration - support both old and new systems
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
                # Use the dataset.duckdb directly instead of creating a new features.duckdb
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
        
        # DuckDB connection settings
        self.connection = None
        self.max_memory_gb = self.duckdb_config.get('max_memory_gb', 4)
        self.enable_sampling = self.duckdb_config.get('enable_sampling', True)
        
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
        
        self._initialize_connection()
        self._initialize_database_schema()
        self._check_and_load_data()
        
        logger.info(f"Initialized persistent DuckDBDataManager at: {self.db_path}")
    
    def _build_column_mapping(self, available_columns: List[str]) -> Dict[str, Optional[str]]:
        """Build flexible mapping - use all available columns dynamically."""
        mapping = {}
        
        # Always try to find an ID column
        id_variants = ['id', 'Id', 'ID', 'index', 'PassengerId', 'passenger_id']
        id_column = None
        for variant in id_variants:
            if variant in available_columns:
                id_column = variant
                break
        mapping['id'] = id_column
        
        # Map all other columns directly (no fixed schema)
        for col in available_columns:
            if col != id_column:  # Skip ID column
                mapping[col] = col  # Direct mapping
        
        logger.info(f"Dynamic column mapping: {len(mapping)} columns detected")
        return mapping
    
    def _initialize_connection(self) -> None:
        """Initialize DuckDB connection with persistent database."""
        try:
            self.connection = duckdb.connect(database=self.db_path)
            
            # Configure memory and performance settings
            self.connection.execute(f"SET memory_limit = '{self.max_memory_gb}GB'")
            
            # Set threads (DuckDB requires at least 1)
            import os
            cpu_count = os.cpu_count() or 1
            self.connection.execute(f"SET threads = {max(1, cpu_count)}")
            
            self.connection.execute("SET enable_progress_bar = false")
            
            # Enable optimizations
            self.connection.execute("SET enable_object_cache = true")
            self.connection.execute("SET force_compression = 'zstd'")
            
            logger.debug(f"DuckDB persistent connection initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB connection: {e}")
            raise
    
    def _generate_database_path(self) -> str:
        """Generate database path based on MD5 hash of data paths."""
        # Create hash from train and test paths
        path_string = f"{self.train_path}|{self.test_path or ''}"
        path_hash = hashlib.md5(path_string.encode()).hexdigest()
        
        # Use cache directory for dataset-specific DuckDB files
        project_root = Path(__file__).parent.parent
        db_dir = project_root / "cache" / path_hash
        db_path = db_dir / "features.duckdb"
        
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
            schema_queries = [
                """
                CREATE TABLE IF NOT EXISTS train_data (
                    id INTEGER PRIMARY KEY,
                    data JSON NOT NULL,
                    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS test_data (
                    id INTEGER PRIMARY KEY,
                    data JSON NOT NULL,
                    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS features_cache (
                    feature_hash VARCHAR PRIMARY KEY,
                    feature_name VARCHAR NOT NULL,
                    feature_data JSON NOT NULL,
                    feature_params JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    evaluation_score DOUBLE,
                    node_depth INTEGER
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    id INTEGER PRIMARY KEY,
                    feature_hash VARCHAR NOT NULL,
                    operation_type VARCHAR NOT NULL,
                    operation_params JSON,
                    execution_time DOUBLE,
                    memory_usage_mb DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            ]
            
            for query in schema_queries:
                self.connection.execute(query)
            
            # Create indexes for performance
            index_queries = [
                "CREATE INDEX IF NOT EXISTS idx_features_cache_score ON features_cache(evaluation_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_features_cache_created ON features_cache(created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_feature_metadata_hash ON feature_metadata(feature_hash)"
            ]
            
            for query in index_queries:
                self.connection.execute(query)
            
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
                tables = self.connection.execute("SHOW TABLES").fetchall()
                table_names = [t[0] for t in tables]
                
                if 'train_features' in table_names:
                    train_count = self.connection.execute("SELECT COUNT(*) FROM train_features").fetchone()[0]
                    logger.info(f"Found train_features table with {train_count} rows")
                    self.data_loaded = True
                elif 'train' in table_names:
                    # train table exists but train_features doesn't - normal during feature generation
                    train_count = self.connection.execute("SELECT COUNT(*) FROM train").fetchone()[0]
                    logger.info(f"Using train table with {train_count} rows (features not generated yet)")
                    self.data_loaded = True
                elif len(table_names) == 0:
                    # No tables at all - normal during initial dataset creation
                    logger.debug("No tables found in database - dataset creation in progress")
                    self.data_loaded = False
                else:
                    # Some tables exist but neither train nor train_features
                    logger.warning(f"Dataset contains tables {table_names} but no train/train_features table")
                    raise ValueError("Neither train_features nor train table found in cached dataset")
                        
                if 'test_features' in table_names:
                    test_count = self.connection.execute("SELECT COUNT(*) FROM test_features").fetchone()[0]
                    logger.info(f"Found test_features table with {test_count} rows")
                elif 'test' in table_names:
                    test_count = self.connection.execute("SELECT COUNT(*) FROM test").fetchone()[0]
                    logger.info(f"Using test table with {test_count} rows (features not generated yet)")
            else:
                # Legacy mode - check train_data/test_data tables
                train_count = self.connection.execute("SELECT COUNT(*) FROM train_data").fetchone()[0]
                test_count = self.connection.execute("SELECT COUNT(*) FROM test_data").fetchone()[0] if self.test_path else 0
                
                if train_count == 0:
                    logger.info("No training data found in database, loading from CSV...")
                    self._load_csv_to_database()
                else:
                    logger.info(f"Found existing data: train={train_count}, test={test_count} rows")
                    self.data_loaded = True
                
        except Exception as e:
            logger.error(f"Failed to check/load data: {e}")
            raise
    
    def _load_csv_to_database(self) -> None:
        """Load data into DuckDB database tables from cache or CSV files."""
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
            
            # Connect to the cached dataset database
            cached_db_path = self.train_path  # Points to dataset.duckdb
            logger.info(f"Opening cached database: {cached_db_path}")
            
            # Copy data from cached database to working database
            attach_query = f"ATTACH '{cached_db_path}' AS cached_db"
            self.connection.execute(attach_query)
            
            # Try to copy train table directly (assume it exists)
            try:
                # First check what columns are available
                desc_query = "DESCRIBE cached_db.train"
                columns_info = self.connection.execute(desc_query).fetchall()
                column_names = [col[0] for col in columns_info]
                logger.info(f"Train table columns: {column_names}")
                
                # Build column list for JSON conversion
                json_columns = []
                for col in column_names:
                    json_columns.append(f"'{col}': COALESCE(CAST(\"{col}\" AS VARCHAR), 'NULL')")
                
                json_construction = '{' + ', '.join(json_columns) + '}'
                
                # Drop and recreate table to ensure fresh data
                self.connection.execute("DROP TABLE IF EXISTS train_data")
                copy_train_query = f"""
                    CREATE TABLE train_data AS 
                    SELECT ROW_NUMBER() OVER () as id, 
                           to_json({json_construction}) as data
                    FROM cached_db.train
                """
                self.connection.execute(copy_train_query)
                logger.info("Copied train table as train_data")
            except Exception as e:
                logger.error(f"Failed to copy train table: {e}")
                raise ValueError("No 'train' table found in cached dataset")
            
            # Try to copy test table if it exists
            try:
                # Check test table columns
                desc_test_query = "DESCRIBE cached_db.test"
                test_columns_info = self.connection.execute(desc_test_query).fetchall()
                test_column_names = [col[0] for col in test_columns_info]
                logger.info(f"Test table columns: {test_column_names}")
                
                # Build JSON for test table
                test_json_columns = []
                for col in test_column_names:
                    test_json_columns.append(f"'{col}': COALESCE(CAST(\"{col}\" AS VARCHAR), 'NULL')")
                
                test_json_construction = '{' + ', '.join(test_json_columns) + '}'
                
                # Drop and recreate table to ensure fresh data
                self.connection.execute("DROP TABLE IF EXISTS test_data")
                copy_test_query = f"""
                    CREATE TABLE test_data AS 
                    SELECT ROW_NUMBER() OVER () as id, 
                           to_json({test_json_construction}) as data
                    FROM cached_db.test
                """
                self.connection.execute(copy_test_query)
                logger.info("Copied test table as test_data")
            except Exception as e:
                logger.warning(f"No test table found in cached dataset: {e}")
            
            # Detach cached database
            self.connection.execute("DETACH cached_db")
            
            # Get column mapping from the first record
            sample_query = "SELECT data FROM train_data LIMIT 1"
            sample_result = self.connection.execute(sample_query).fetchone()
            
            if sample_result:
                import json
                sample_data = json.loads(sample_result[0])
                self.column_mapping = {col: col for col in sample_data.keys()}
                logger.info(f"Detected columns from cached data: {list(self.column_mapping.keys())}")
            else:
                logger.warning("No data found in cached dataset")
                
        except Exception as e:
            logger.error(f"Failed to load from cached dataset: {e}")
            raise
    
    def _load_from_csv_files(self) -> None:
        """Load data from CSV files (legacy system)."""
        try:
            # First, get the actual column structure of the CSV
            logger.info(f"Loading training data from: {self.train_path}")
            
            # Detect columns dynamically
            column_query = f"DESCRIBE SELECT * FROM read_csv_auto('{self.train_path}')"
            columns_result = self.connection.execute(column_query).fetchall()
            available_columns = [row[0] for row in columns_result]
            logger.info(f"Available columns in CSV: {available_columns}")
            
            # Build dynamic column mapping
            column_mapping = self._build_column_mapping(available_columns)
            self.column_mapping = column_mapping  # Cache for later use
            logger.info(f"Column mapping: {column_mapping}")
            
            # Build JSON construction dynamically for all columns
            json_fields = []
            for col_name, csv_column in column_mapping.items():
                if csv_column and col_name != 'id':  # Skip id, it's handled separately
                    # Store everything as VARCHAR in JSON to avoid type conflicts
                    # Use COALESCE with CAST to handle NULLs and convert everything to string
                    json_fields.append(f"'{col_name}': COALESCE(CAST(\"{csv_column}\" AS VARCHAR), 'Unknown')")
            
            # Always include id
            id_col = column_mapping.get('id', 'ROW_NUMBER() OVER ()')
            if column_mapping.get('id'):
                json_fields.insert(0, f"'id': COALESCE({id_col}, ROW_NUMBER() OVER ())")
            else:
                json_fields.insert(0, f"'id': ROW_NUMBER() OVER ()")
            
            json_construction = "{ " + ", ".join(json_fields) + " }"
            
            load_train_query = f"""
            INSERT INTO train_data (id, data)
            SELECT ROW_NUMBER() OVER () as id, 
                   to_json({json_construction}) as data
            FROM read_csv_auto('{self.train_path}')
            """
            
            logger.debug(f"Executing query: {load_train_query}")
            self.connection.execute(load_train_query)
            
            train_count = self.connection.execute("SELECT COUNT(*) FROM train_data").fetchone()[0]
            logger.info(f"Loaded {train_count} training samples into database")
            
            # Load test data if available
            if self.test_path and os.path.exists(self.test_path):
                logger.info(f"Loading test data from: {self.test_path}")
                
                # Detect test data columns
                test_column_query = f"DESCRIBE SELECT * FROM read_csv_auto('{self.test_path}')"
                test_columns_result = self.connection.execute(test_column_query).fetchall()
                test_available_columns = [row[0] for row in test_columns_result]
                logger.info(f"Available test columns: {test_available_columns}")
                
                # Build test column mapping
                test_column_mapping = self._build_column_mapping(test_available_columns)
                
                # Build JSON construction for test data - same logic as train
                test_json_fields = []
                for col_name, csv_column in test_column_mapping.items():
                    if csv_column and col_name != 'id':  # Skip id, handled separately
                        # Store everything as VARCHAR in JSON to avoid type conflicts
                        # Use COALESCE with CAST to handle NULLs and convert everything to string
                        test_json_fields.append(f"'{col_name}': COALESCE(CAST(\"{csv_column}\" AS VARCHAR), 'Unknown')")
                
                # Always include id
                test_id_col = test_column_mapping.get('id', 'ROW_NUMBER() OVER ()')
                if test_column_mapping.get('id'):
                    test_json_fields.insert(0, f"'id': COALESCE({test_id_col}, ROW_NUMBER() OVER ())")
                else:
                    test_json_fields.insert(0, f"'id': ROW_NUMBER() OVER ()")
                
                test_json_construction = "{ " + ", ".join(test_json_fields) + " }"
                
                load_test_query = f"""
                INSERT INTO test_data (id, data)
                SELECT ROW_NUMBER() OVER () as id,
                       to_json({test_json_construction}) as data
                FROM read_csv_auto('{self.test_path}')
                """
                self.connection.execute(load_test_query)
                
                test_count = self.connection.execute("SELECT COUNT(*) FROM test_data").fetchone()[0]
                logger.info(f"Loaded {test_count} test samples into database")
            
            self.data_loaded = True
            logger.info("CSV data successfully loaded into DuckDB database")
            
        except Exception as e:
            logger.error(f"Failed to load CSV data to database: {e}")
            raise
    
    def _convert_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to proper data types dynamically."""
        try:
            # Convert id to integer
            if 'id' in df.columns:
                df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')
            
            # For all other columns, try to convert numeric where possible
            for col in df.columns:
                if col == 'id':
                    continue
                    
                # Check if column contains string keywords
                if any(keyword in col.lower() for keyword in ['name', 'fertilizer', 'type', 'category']):
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace('None', 'Unknown').fillna('Unknown')
                else:
                    # Try to convert to numeric
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].fillna(0.0)  # Fill NULL with 0 as default
                    except:
                        # If conversion fails, keep as string
                        df[col] = df[col].astype(str).fillna('Unknown')
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to convert data types: {e}")
            return df
    
    def _fallback_file_sampling(self, 
                               file_path: str, 
                               train_size: Union[int, float],
                               stratify_column: Optional[str] = None,
                               random_seed: int = 42) -> pd.DataFrame:
        """Fallback to file-based sampling using original logic."""
        logger.warning("Using file-based fallback sampling (less efficient)")
        
        try:
            file_path = str(Path(file_path).resolve())
            
            # Get dataset info for validation
            dataset_info = self.get_dataset_info(file_path)
            total_rows = dataset_info['row_count']
            
            # Determine sample size
            if isinstance(train_size, float) and 0 <= train_size <= 1:
                n_samples = int(total_rows * train_size)
            else:
                n_samples = min(int(train_size), total_rows)
            
            # If sample size >= total rows, load everything
            if n_samples >= total_rows:
                return self.load_full_dataset(file_path)
            
            # Build sampling query
            if stratify_column and stratify_column in dataset_info['columns']:
                # Stratified sampling
                sampling_query = self._build_stratified_sampling_query(
                    file_path, n_samples, stratify_column, random_seed
                )
            else:
                # Simple random sampling
                sampling_query = f"""
                SELECT * FROM '{file_path}' 
                USING SAMPLE {n_samples} ROWS (bernoulli, {random_seed})
                """
            
            # Execute sampling query
            result = self.connection.execute(sampling_query).df()
            return result
            
        except Exception as e:
            logger.error(f"File-based fallback sampling also failed: {e}")
            # Final fallback to pandas
            return self._fallback_pandas_sampling(file_path, train_size, stratify_column)
    
    @timed("duckdb.get_dataset_info")
    def get_dataset_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get dataset information without loading full data.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Dict with dataset metadata
        """
        file_path = str(Path(file_path).resolve())
        
        # Check cache first
        if file_path in self.dataset_metadata:
            return self.dataset_metadata[file_path]
        
        try:
            start_time = time.time()
            
            # Get basic info efficiently
            info_query = f"""
            SELECT 
                COUNT(*) as row_count,
                COUNT(DISTINCT *) as unique_rows
            FROM '{file_path}'
            """
            
            result = self.connection.execute(info_query).fetchone()
            row_count, unique_rows = result
            
            # Get column information
            columns_query = f"DESCRIBE SELECT * FROM '{file_path}'"
            columns_result = self.connection.execute(columns_query).fetchall()
            
            columns_info = {}
            for col_name, col_type, null_count, key, default, extra in columns_result:
                columns_info[col_name] = {
                    'type': col_type,
                    'nullable': null_count != 'NO'
                }
            
            query_time = time.time() - start_time
            self.total_query_time += query_time
            self.query_count += 1
            
            metadata = {
                'file_path': file_path,
                'row_count': row_count,
                'unique_rows': unique_rows,
                'column_count': len(columns_info),
                'columns': columns_info,
                'file_size_mb': Path(file_path).stat().st_size / 1024 / 1024,
                'last_modified': os.path.getmtime(file_path),
                'query_time': query_time
            }
            
            # Cache metadata
            self.dataset_metadata[file_path] = metadata
            
            logger.debug(f"Dataset info: {row_count} rows, {len(columns_info)} columns in {query_time:.3f}s")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get dataset info for {file_path}: {e}")
            raise
    
    @timed("duckdb.sample_dataset", include_memory=True)
    def sample_dataset(self, 
                      file_path: str, 
                      train_size: Union[int, float], 
                      stratify_column: Optional[str] = None,
                      random_seed: int = 42) -> pd.DataFrame:
        """
        Efficiently sample dataset from DuckDB database.
        
        Args:
            file_path: Path identifier (train_path or test_path) 
            train_size: Number of samples (int) or percentage (float 0-1)
            stratify_column: Column for stratified sampling
            random_seed: Random seed for reproducibility
            
        Returns:
            Sampled DataFrame
        """
        if not self.enable_sampling:
            logger.warning("DuckDB sampling disabled, falling back to full load")
            return self.load_full_dataset(file_path)
        
        start_time = time.time()
        
        try:
            # Determine which table to use based on file_path
            if file_path == self.train_path:
                if self.use_cached_dataset:
                    # Check if train_features exists, otherwise use train
                    tables = self.connection.execute("SHOW TABLES").fetchall()
                    table_names = [t[0] for t in tables]
                    table_name = 'train_features' if 'train_features' in table_names else 'train'
                else:
                    table_name = 'train_data'
            elif file_path == self.test_path:
                if self.use_cached_dataset:
                    # Check if test_features exists, otherwise use test
                    tables = self.connection.execute("SHOW TABLES").fetchall()
                    table_names = [t[0] for t in tables]
                    table_name = 'test_features' if 'test_features' in table_names else 'test'
                else:
                    table_name = 'test_data'
            else:
                # Fallback to file-based loading for unknown paths
                logger.warning(f"Unknown file path {file_path}, using file-based sampling")
                return self._fallback_file_sampling(file_path, train_size, stratify_column, random_seed)
            
            # Get total row count from database
            total_rows = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            # Determine sample size
            if isinstance(train_size, float) and 0 <= train_size <= 1:
                n_samples = int(total_rows * train_size)
                sample_method = "percentage"
                logger.info(f"🎲 DuckDB sampling {train_size*100:.1f}% = {n_samples}/{total_rows} rows from {table_name}")
            else:
                n_samples = min(int(train_size), total_rows)
                sample_method = "absolute"
                logger.info(f"🎲 DuckDB sampling {n_samples}/{total_rows} rows from {table_name}")
            
            # If sample size >= total rows, load everything
            if n_samples >= total_rows:
                logger.info("Sample size >= dataset size, loading full dataset")
                return self.load_full_dataset(file_path)
            
            # For feature tables or cached datasets, use direct sampling
            if self.use_cached_dataset and table_name in ['train_features', 'test_features', 'train', 'test']:
                # Direct sampling for feature tables
                sampling_query = f"""
                SELECT * FROM {table_name} 
                TABLESAMPLE RESERVOIR({n_samples}) REPEATABLE ({random_seed})
                """
            else:
                # Build dynamic sampling query based on available columns (for JSON storage)
                if not self.column_mapping:
                    # If no column mapping available, get it from first row
                    sample_query = f"SELECT data FROM {table_name} LIMIT 1"
                    sample_result = self.connection.execute(sample_query).fetchone()
                    if sample_result:
                        import json
                        sample_data = json.loads(sample_result[0])
                        self.column_mapping = {col: col for col in sample_data.keys()}
                
                # Build SELECT clause dynamically
                select_fields = []
                for col_name in self.column_mapping.keys():
                    if col_name:
                        select_fields.append(f"json_extract_string(data, '$.{col_name}') as \"{col_name}\"")
                
                select_clause = ", ".join(select_fields) if select_fields else "json_extract_string(data, '$.id') as id"
                
                sampling_query = f"""
                SELECT {select_clause}
                FROM (
                    SELECT data FROM {table_name} 
                    TABLESAMPLE RESERVOIR({n_samples}) REPEATABLE ({random_seed})
                )
                """
            
            # Execute sampling query and convert to proper types
            logger.debug(f"Executing DuckDB sampling query from {table_name}")
            result_df = self.connection.execute(sampling_query).df()
            
            # Convert columns to proper data types
            result_df = self._convert_dataframe_types(result_df)
            
            # Skip feature filtering - MCTS uses all pre-built features from dataset registration
            
            query_time = time.time() - start_time
            self.total_query_time += query_time
            self.query_count += 1
            
            actual_rows = len(result_df)
            logger.info(f"✅ DuckDB sampled {actual_rows} rows in {query_time:.3f}s "
                       f"(efficiency: {actual_rows/n_samples:.1%})")
            
            return result_df
            
        except Exception as e:
            logger.error(f"DuckDB sampling failed for {file_path}: {e}")
            logger.info("Falling back to file-based sampling")
            return self._fallback_file_sampling(file_path, train_size, stratify_column, random_seed)
    
    def _build_stratified_sampling_query(self, 
                                       file_path: str, 
                                       n_samples: int, 
                                       stratify_column: str,
                                       random_seed: int) -> str:
        """Build stratified sampling query."""
        # Get stratum information
        strata_query = f"""
        SELECT {stratify_column}, COUNT(*) as stratum_size
        FROM '{file_path}'
        GROUP BY {stratify_column}
        ORDER BY {stratify_column}
        """
        
        strata_info = self.connection.execute(strata_query).df()
        total_rows = strata_info['stratum_size'].sum()
        
        # Calculate samples per stratum
        strata_info['samples_needed'] = (strata_info['stratum_size'] * n_samples / total_rows).round().astype(int)
        strata_info['samples_needed'] = strata_info['samples_needed'].clip(lower=1)  # At least 1 sample per stratum
        
        # Build UNION query for stratified sampling
        union_parts = []
        for _, row in strata_info.iterrows():
            stratum_value = row[stratify_column]
            stratum_samples = row['samples_needed']
            
            if isinstance(stratum_value, str):
                where_clause = f"{stratify_column} = '{stratum_value}'"
            else:
                where_clause = f"{stratify_column} = {stratum_value}"
            
            union_part = f"""
            (SELECT * FROM '{file_path}' 
             WHERE {where_clause}
             USING SAMPLE {stratum_samples} ROWS (bernoulli, {random_seed}))
            """
            union_parts.append(union_part)
        
        return " UNION ALL ".join(union_parts)
    
    @timed("duckdb.load_columns")
    def load_columns(self, 
                    file_path: str, 
                    columns: List[str],
                    where_clause: Optional[str] = None,
                    limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load only specific columns efficiently.
        
        Args:
            file_path: Path to data file
            columns: List of column names to load
            where_clause: Optional SQL WHERE clause
            limit: Optional row limit
            
        Returns:
            DataFrame with selected columns
        """
        file_path = str(Path(file_path).resolve())
        
        try:
            # Build column selection
            columns_str = ", ".join([f'"{col}"' for col in columns])
            
            # Build query
            query_parts = [f"SELECT {columns_str} FROM '{file_path}'"]
            
            if where_clause:
                query_parts.append(f"WHERE {where_clause}")
            
            if limit:
                query_parts.append(f"LIMIT {limit}")
            
            query = " ".join(query_parts)
            
            start_time = time.time()
            result = self.connection.execute(query).df()
            query_time = time.time() - start_time
            
            self.total_query_time += query_time
            self.query_count += 1
            
            logger.debug(f"Loaded {len(columns)} columns, {len(result)} rows in {query_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load columns {columns} from {file_path}: {e}")
            raise
    
    @timed("duckdb.load_full_dataset")
    def load_full_dataset(self, file_path: str) -> pd.DataFrame:
        """Load full dataset from DuckDB database."""
        try:
            start_time = time.time()
            
            # Determine which table to use based on file_path
            if file_path == self.train_path:
                if self.use_cached_dataset:
                    # Check if train_features exists, otherwise use train
                    tables = self.connection.execute("SHOW TABLES").fetchall()
                    table_names = [t[0] for t in tables]
                    table_name = 'train_features' if 'train_features' in table_names else 'train'
                else:
                    table_name = 'train_data'
            elif file_path == self.test_path:
                if self.use_cached_dataset:
                    # Check if test_features exists, otherwise use test
                    tables = self.connection.execute("SHOW TABLES").fetchall()
                    table_names = [t[0] for t in tables]
                    table_name = 'test_features' if 'test_features' in table_names else 'test'
                else:
                    table_name = 'test_data'
            else:
                # Fallback to file-based loading
                logger.warning(f"Unknown file path {file_path}, using file-based loading")
                file_path = str(Path(file_path).resolve())
                query = f"SELECT * FROM '{file_path}'"
                result = self.connection.execute(query).df()
                query_time = time.time() - start_time
                self.total_query_time += query_time
                self.query_count += 1
                logger.info(f"Loaded full dataset from file: {len(result)} rows in {query_time:.3f}s")
                return result
            
            # For feature tables or cached datasets, use direct SELECT
            if self.use_cached_dataset and table_name in ['train_features', 'test_features', 'train', 'test']:
                # Direct query for feature tables
                query = f"SELECT * FROM {table_name}"
                result_df = self.connection.execute(query).df()
            else:
                # Build dynamic query based on available columns (for JSON storage)
                if not self.column_mapping:
                    # Get column mapping from first row if not available
                    sample_query = f"SELECT data FROM {table_name} LIMIT 1"
                    sample_result = self.connection.execute(sample_query).fetchone()
                    if sample_result:
                        import json
                        sample_data = json.loads(sample_result[0])
                        self.column_mapping = {col: col for col in sample_data.keys()}
                
                # Build SELECT clause dynamically
                select_fields = []
                for col_name in self.column_mapping.keys():
                    if col_name:
                        select_fields.append(f"json_extract_string(data, '$.{col_name}') as \"{col_name}\"")
                
                select_clause = ", ".join(select_fields) if select_fields else "json_extract_string(data, '$.id') as id"
                
                query = f"""
                SELECT {select_clause}
                FROM {table_name}
                """
                
                result_df = self.connection.execute(query).df()
            
            # Convert columns to proper data types
            result_df = self._convert_dataframe_types(result_df)
            
            # Skip feature filtering - MCTS uses all pre-built features from dataset registration
            
            query_time = time.time() - start_time
            self.total_query_time += query_time
            self.query_count += 1
            
            logger.info(f"Loaded full dataset from {table_name}: {len(result_df)} rows in {query_time:.3f}s")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to load full dataset from {file_path}: {e}")
            raise
    
    def _fallback_pandas_sampling(self, 
                                 file_path: str, 
                                 train_size: Union[int, float],
                                 stratify_column: Optional[str] = None) -> pd.DataFrame:
        """Fallback to pandas sampling if DuckDB fails."""
        logger.warning("Using pandas fallback sampling (less efficient)")
        
        try:
            # Load with pandas
            if Path(file_path).suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Apply sampling
            if isinstance(train_size, float) and 0 <= train_size <= 1:
                n_samples = int(len(df) * train_size)
            else:
                n_samples = min(int(train_size), len(df))
            
            if n_samples >= len(df):
                return df
            
            if stratify_column and stratify_column in df.columns:
                # Stratified sampling with pandas
                return df.groupby(stratify_column, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, int(n_samples * len(x) / len(df)))),
                                     random_state=42)
                ).reset_index(drop=True)
            else:
                # Simple random sampling
                return df.sample(n=n_samples, random_state=42).reset_index(drop=True)
                
        except Exception as e:
            logger.error(f"Pandas fallback sampling also failed: {e}")
            raise
    
    # Feature Caching Methods
    
    def cache_features(self, 
                      feature_hash: str,
                      feature_name: str, 
                      features_df: pd.DataFrame,
                      feature_params: Dict[str, Any] = None,
                      evaluation_score: float = None,
                      node_depth: int = None) -> None:
        """Cache generated features in DuckDB database."""
        try:
            # Convert DataFrame to JSON for storage
            feature_data = features_df.to_json(orient='records')
            
            # Insert or replace feature cache entry
            insert_query = """
            INSERT OR REPLACE INTO features_cache 
            (feature_hash, feature_name, feature_data, feature_params, evaluation_score, node_depth)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            self.connection.execute(insert_query, [
                feature_hash,
                feature_name,
                feature_data,
                json.dumps(feature_params) if feature_params else None,
                evaluation_score,
                node_depth
            ])
            
            logger.debug(f"Cached features: {feature_name} (hash: {feature_hash[:8]}...)")
            
        except Exception as e:
            logger.error(f"Failed to cache features {feature_name}: {e}")
    
    def get_cached_features(self, feature_hash: str) -> Optional[pd.DataFrame]:
        """Retrieve cached features from DuckDB database."""
        try:
            query = "SELECT feature_data FROM features_cache WHERE feature_hash = ?"
            result = self.connection.execute(query, [feature_hash]).fetchone()
            
            if result:
                feature_data_json = result[0]
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
    
    def get_cached_features_by_score(self, min_score: float = 0.0, limit: int = 10) -> List[Dict[str, Any]]:
        """Get cached features ordered by evaluation score."""
        try:
            query = """
            SELECT feature_hash, feature_name, evaluation_score, node_depth, created_at
            FROM features_cache 
            WHERE evaluation_score >= ?
            ORDER BY evaluation_score DESC
            LIMIT ?
            """
            
            results = self.connection.execute(query, [min_score, limit]).fetchall()
            
            cached_features = []
            for row in results:
                cached_features.append({
                    'feature_hash': row[0],
                    'feature_name': row[1],
                    'evaluation_score': row[2],
                    'node_depth': row[3],
                    'created_at': row[4]
                })
            
            return cached_features
            
        except Exception as e:
            logger.error(f"Failed to get cached features by score: {e}")
            return []
    
    def clear_feature_cache(self, max_entries: int = None, min_score: float = None) -> None:
        """Clear feature cache with optional filtering."""
        try:
            if max_entries:
                # Keep only top N entries by score
                cleanup_query = """
                DELETE FROM features_cache 
                WHERE feature_hash NOT IN (
                    SELECT feature_hash FROM features_cache 
                    ORDER BY evaluation_score DESC 
                    LIMIT ?
                )
                """
                self.connection.execute(cleanup_query, [max_entries])
                
            elif min_score is not None:
                # Remove entries below score threshold
                cleanup_query = "DELETE FROM features_cache WHERE evaluation_score < ?"
                self.connection.execute(cleanup_query, [min_score])
                
            else:
                # Clear all cached features
                self.connection.execute("DELETE FROM features_cache")
            
            logger.info("Feature cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear feature cache: {e}")
    
    def get_feature_cache_stats(self) -> Dict[str, Any]:
        """Get feature cache statistics."""
        try:
            stats_query = """
            SELECT 
                COUNT(*) as total_features,
                AVG(evaluation_score) as avg_score,
                MAX(evaluation_score) as max_score,
                MIN(evaluation_score) as min_score
            FROM features_cache
            WHERE evaluation_score IS NOT NULL
            """
            
            result = self.connection.execute(stats_query).fetchone()
            
            return {
                'total_cached_features': result[0] if result else 0,
                'avg_score': result[1] if result and result[1] else 0.0,
                'max_score': result[2] if result and result[2] else 0.0,
                'min_score': result[3] if result and result[3] else 0.0,
                'cache_hits': self.cache_hits
            }
            
        except Exception as e:
            logger.error(f"Failed to get feature cache stats: {e}")
            return {'total_cached_features': 0, 'cache_hits': self.cache_hits}
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute arbitrary SQL query and return DataFrame."""
        try:
            start_time = time.time()
            result = self.connection.execute(query).df()
            query_time = time.time() - start_time
            
            self.total_query_time += query_time
            self.query_count += 1
            
            logger.debug(f"Executed custom query in {query_time:.3f}s, returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _filter_features_by_config(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter features based on MCTS config settings."""
        original_cols = len(df.columns)
        columns_to_keep = []
        
        # Get config settings
        feature_space_config = self.config.get('feature_space', {})
        excluded_columns = feature_space_config.get('excluded_columns', [])
        generic_operations = feature_space_config.get('generic_operations', {})
        
        # Always keep basic columns (non-feature columns)
        basic_cols = ['PassengerId', 'Survived', 'id', 'ID', 'target', 'label']
        
        for col in df.columns:
            # Skip excluded columns
            if col in excluded_columns:
                continue
                
            # Always keep basic columns
            if col in basic_cols or col.lower() in [c.lower() for c in basic_cols]:
                columns_to_keep.append(col)
                continue
            
            # Check if it's a generic feature
            is_generic_feature = False
            
            # Statistical aggregations
            if any(pattern in col for pattern in ['_mean_by_', '_std_by_', '_dev_from_']):
                if generic_operations.get('statistical_aggregations', False):
                    columns_to_keep.append(col)
                is_generic_feature = True
                
            # Polynomial features
            elif any(pattern in col for pattern in ['_squared', '_cubed', '_log', '_sqrt', '_x_']):
                if generic_operations.get('polynomial_features', False):
                    columns_to_keep.append(col)
                is_generic_feature = True
                
            # Binning features
            elif '_bin_' in col:
                if generic_operations.get('binning_features', False):
                    columns_to_keep.append(col)
                is_generic_feature = True
                
            # Ranking features
            elif any(pattern in col for pattern in ['_rank', '_rank_pct']):
                if generic_operations.get('ranking_features', False):
                    columns_to_keep.append(col)
                is_generic_feature = True
            
            # If it's not a generic feature, it's either original or custom - keep it
            if not is_generic_feature:
                columns_to_keep.append(col)
        
        # Apply filtering
        filtered_df = df[columns_to_keep]
        
        logger.info(f"Feature filtering: {original_cols} -> {len(filtered_df.columns)} columns "
                   f"(removed {original_cols - len(filtered_df.columns)} features)")
        
        return filtered_df
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_query_time = self.total_query_time / self.query_count if self.query_count > 0 else 0
        
        # Get feature cache stats
        feature_stats = self.get_feature_cache_stats()
        
        return {
            'total_queries': self.query_count,
            'total_query_time': self.total_query_time,
            'avg_query_time': avg_query_time,
            'cache_hits': self.cache_hits,
            'cached_datasets': len(self.dataset_metadata),
            'duckdb_version': duckdb.__version__ if DUCKDB_AVAILABLE else None,
            'max_memory_gb': self.max_memory_gb,
            'database_path': self.db_path,
            'database_size_mb': os.path.getsize(self.db_path) / 1024 / 1024 if os.path.exists(self.db_path) else 0,
            'tables_initialized': self.tables_initialized,
            'data_loaded': self.data_loaded,
            'total_cached_features': feature_stats['total_cached_features'],
            'feature_cache_hits': feature_stats['cache_hits']
        }
    
    def clear_cache(self) -> None:
        """Clear dataset metadata cache."""
        self.dataset_metadata.clear()
        logger.info("DuckDB dataset metadata cache cleared")
    
    def close(self) -> None:
        """Close DuckDB connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.debug("DuckDB connection closed")
            except:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Utility functions for DuckDB integration

def is_duckdb_available() -> bool:
    """Check if DuckDB is available and working."""
    if not DUCKDB_AVAILABLE:
        return False
    
    try:
        conn = duckdb.connect(database=':memory:')
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return True
    except:
        return False

def get_duckdb_version() -> Optional[str]:
    """Get DuckDB version string."""
    return duckdb.__version__ if DUCKDB_AVAILABLE else None

def estimate_sample_efficiency(dataset_size: int, sample_size: int) -> Dict[str, Any]:
    """Estimate efficiency gains from DuckDB sampling vs pandas."""
    if sample_size >= dataset_size:
        return {
            'efficiency_gain': 1.0,
            'memory_reduction': 1.0,
            'recommended_backend': 'either'
        }
    
    sample_ratio = sample_size / dataset_size
    efficiency_gain = 1.0 / sample_ratio  # Approximate
    memory_reduction = sample_ratio
    
    # Recommend DuckDB for large datasets with small samples
    if dataset_size > 100000 and sample_ratio < 0.1:
        recommended = 'duckdb'
    elif dataset_size < 10000:
        recommended = 'pandas'
    else:
        recommended = 'duckdb'
    
    return {
        'dataset_size': dataset_size,
        'sample_size': sample_size,
        'sample_ratio': sample_ratio,
        'efficiency_gain': efficiency_gain,
        'memory_reduction': memory_reduction,
        'recommended_backend': recommended
    }