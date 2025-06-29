#!/usr/bin/env python3
"""
Dataset Importer - Auto-detection and import of datasets

Handles auto-detection of dataset files and conversion to DuckDB format.
"""

import os
import re
import hashlib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatasetImporter:
    """Handles dataset auto-detection and import to DuckDB format."""
    
    def __init__(self, dataset_name: str):
        """Initialize importer with dataset name."""
        self.dataset_name = dataset_name
        self.base_cache_dir = Path("cache")
        self.dataset_dir = self.base_cache_dir / dataset_name
        
        # File pattern mappings
        self.file_patterns = {
            'train': [
                r'train\.csv$', r'train\.parquet$', r'training\.csv$',
                r'.*train.*\.csv$', r'.*training.*\.csv$',
                r'data\.csv$', r'dataset\.csv$'
            ],
            'test': [
                r'test\.csv$', r'test\.parquet$', r'testing\.csv$',
                r'.*test.*\.csv$', r'.*testing.*\.csv$'
            ],
            'submission': [
                r'sample_submission\.csv$', r'submission\.csv$',
                r'.*submission.*\.csv$', r'submit.*\.csv$'
            ],
            'validation': [
                r'val\.csv$', r'validation\.csv$', r'valid\.csv$',
                r'.*val.*\.csv$', r'.*validation.*\.csv$'
            ]
        }
    
    def auto_detect_files(self, search_path: str) -> Dict[str, str]:
        """Auto-detect dataset files in given directory.
        
        Args:
            search_path: Directory to search for files
            
        Returns:
            Dict mapping file types to detected file paths
        """
        search_dir = Path(search_path)
        if not search_dir.exists():
            raise FileNotFoundError(f"Search directory not found: {search_path}")
        
        detected_files = {}
        
        # Get all CSV/Parquet files in directory
        all_files = []
        for pattern in ['*.csv', '*.parquet']:
            all_files.extend(search_dir.glob(pattern))
            all_files.extend(search_dir.glob(f"**/{pattern}"))  # Search subdirectories too
        
        logger.info(f"Found {len(all_files)} data files in {search_path}")
        
        # Match files to types
        for file_type, patterns in self.file_patterns.items():
            for file_path in all_files:
                file_name = file_path.name.lower()
                
                for pattern in patterns:
                    if re.search(pattern, file_name):
                        if file_type not in detected_files:  # Use first match
                            detected_files[file_type] = str(file_path)
                            logger.info(f"Detected {file_type}: {file_path}")
                            break
                
                if file_type in detected_files:
                    break
        
        if not detected_files:
            raise ValueError(f"No recognizable dataset files found in {search_path}")
        
        if 'train' not in detected_files:
            raise ValueError("No training file detected - required for dataset registration")
        
        return detected_files
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a dataset file to extract metadata.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            Dict with file metadata (rows, columns, dtypes, etc.)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Load file
            if file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
                file_format = 'parquet'
            else:
                df = pd.read_csv(file_path, nrows=1000)  # Sample for analysis
                file_format = 'csv'
            
            # Get full row count for CSV files
            if file_format == 'csv':
                # Quick row count without loading entire file
                with open(file_path, 'r') as f:
                    row_count = sum(1 for line in f) - 1  # Subtract header
            else:
                row_count = len(df)
            
            # Analyze columns
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # File size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            metadata = {
                'records': row_count,
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'file_format': file_format,
                'file_size_mb': round(file_size_mb, 2),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            }
            
            logger.info(f"Analyzed {file_path}: {row_count:,} rows, {len(df.columns)} columns")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            raise ValueError(f"Could not analyze file {file_path}: {e}")
    
    def detect_target_column(self, train_path: str, hints: List[str] = None) -> Optional[str]:
        """Auto-detect target column from training file.
        
        Args:
            train_path: Path to training file
            hints: Optional list of column name hints
            
        Returns:
            Most likely target column name
        """
        try:
            df = pd.read_csv(train_path, nrows=100)  # Sample for analysis
            
            # Common target column patterns
            target_patterns = [
                r'^target$', r'^label$', r'^y$', r'^class$',
                r'^outcome$', r'^result$', r'^response$',
                r'.*target.*', r'.*label.*', r'.*class.*'
            ]
            
            # Add user hints to patterns
            if hints:
                for hint in hints:
                    target_patterns.append(f'^{re.escape(hint)}$')
                    target_patterns.append(f'.*{re.escape(hint)}.*')
            
            # Check patterns against column names
            for pattern in target_patterns:
                for col in df.columns:
                    if re.search(pattern, col.lower()):
                        logger.info(f"Detected target column: {col}")
                        return col
            
            # If no pattern match, look for columns with limited unique values
            # (likely categorical targets)
            categorical_candidates = []
            for col in df.columns:
                if df[col].dtype in ['object', 'category'] or df[col].nunique() <= 20:
                    categorical_candidates.append((col, df[col].nunique()))
            
            if categorical_candidates:
                # Return column with fewest unique values
                best_candidate = min(categorical_candidates, key=lambda x: x[1])
                logger.info(f"Detected likely target column: {best_candidate[0]} ({best_candidate[1]} unique values)")
                return best_candidate[0]
            
            logger.warning("Could not auto-detect target column")
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect target column: {e}")
            return None
    
    def detect_id_column(self, file_path: str) -> Optional[str]:
        """Auto-detect ID column from file.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            Most likely ID column name
        """
        try:
            df = pd.read_csv(file_path, nrows=100)
            
            # ID column patterns
            id_patterns = [
                r'^id$', r'^index$', r'^row_id$', r'^sample_id$',
                r'.*_id$', r'.*id$', r'^id_.*'
            ]
            
            for pattern in id_patterns:
                for col in df.columns:
                    if re.search(pattern, col.lower()):
                        # Verify it looks like an ID (unique, numeric or string)
                        if df[col].nunique() == len(df):  # All unique values
                            logger.info(f"Detected ID column: {col}")
                            return col
            
            # Look for first column if it's unique and looks like an index
            first_col = df.columns[0]
            if df[first_col].nunique() == len(df) and (
                df[first_col].dtype in ['int64', 'object'] and 
                str(df[first_col].iloc[0]).isdigit()
            ):
                logger.info(f"Detected ID column (first column): {first_col}")
                return first_col
            
            logger.info("No ID column detected")
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect ID column: {e}")
            return None
    
    def create_duckdb_dataset(self, file_mappings: Dict[str, str], 
                             target_column: str, id_column: str = None) -> str:
        """Convert dataset files to DuckDB format.
        
        Args:
            file_mappings: Dict mapping table names to file paths
            target_column: Name of target column
            id_column: Optional ID column name
            
        Returns:
            Path to created DuckDB file
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB not available - cannot create dataset")
        
        # Create dataset directory
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # DuckDB file path
        duckdb_path = self.dataset_dir / "dataset.duckdb"
        
        # Remove existing file if it exists
        if duckdb_path.exists():
            duckdb_path.unlink()
        
        logger.info(f"Creating DuckDB dataset: {duckdb_path}")
        
        try:
            # Connect to DuckDB
            conn = duckdb.connect(str(duckdb_path))
            
            # Import each file as a table
            for table_name, file_path in file_mappings.items():
                logger.info(f"Importing {table_name} from {file_path}")
                
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    logger.warning(f"File not found, skipping: {file_path}")
                    continue
                
                # Create table from file
                if file_path_obj.suffix.lower() == '.parquet':
                    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')")
                else:
                    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
                
                # Get record count
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                record_count = result[0] if result else 0
                
                logger.info(f"✅ Imported {record_count:,} records to table '{table_name}'")
            
            # Verify target column exists in train table
            if 'train' in file_mappings:
                columns = conn.execute("DESCRIBE train").fetchall()
                column_names = [col[0] for col in columns]
                
                if target_column not in column_names:
                    logger.warning(f"Target column '{target_column}' not found in train table")
                    logger.info(f"Available columns: {', '.join(column_names)}")
                
                if id_column and id_column not in column_names:
                    logger.warning(f"ID column '{id_column}' not found in train table")
            
            # Generate features for train and test tables
            logger.info("Generating features for dataset...")
            self._generate_and_save_features(conn, self.dataset_name, target_column, id_column)
            
            conn.close()
            
            logger.info(f"✅ Successfully created DuckDB dataset: {duckdb_path}")
            return str(duckdb_path)
            
        except Exception as e:
            # Cleanup on failure
            if duckdb_path.exists():
                duckdb_path.unlink()
            raise ValueError(f"Failed to create DuckDB dataset: {e}")
    
    def _generate_and_save_features(self, conn, dataset_name: str, target_column: str, id_column: str = None):
        """Generate features in separate tables: generic, custom, train_features, test_features."""
        import sys
        from pathlib import Path
        
        # Setup contextual logging for this dataset
        dataset_logger = self._setup_dataset_logging(dataset_name)
        
        # Import FeatureSpace using absolute import
        try:
            # Try absolute import first (when run as module)
            from src.feature_space import FeatureSpace
        except ImportError:
            # Fallback to adding src to path for standalone execution
            src_path = Path(__file__).parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from feature_space import FeatureSpace
        
        # Load configuration from main config file
        import yaml
        config_path = Path(__file__).parent.parent / 'config' / 'mcts_config.yaml'
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            dataset_logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            dataset_logger.error(f"Failed to load config from {config_path}: {e}")
            raise ValueError(f"Cannot load MCTS configuration: {e}")
        
        # Override dataset name in autogluon config
        config['autogluon']['dataset_name'] = dataset_name
        
        # Determine custom domain module based on dataset name
        domain_mapping = {
            's5e6': 'domains.fertilizer',
            'fertilizer': 'domains.fertilizer',
            'titanic': 'domains.titanic',
        }
        
        for key, domain in domain_mapping.items():
            if key in dataset_name.lower():
                config['feature_space']['custom_domain_module'] = domain
                dataset_logger.info(f"Using custom domain module: {domain}")
                break
        
        # Initialize FeatureSpace
        feature_space = FeatureSpace(config)
        
        # Process train data if exists
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='train'").fetchone():
            dataset_logger.info("📊 Generating features for train data...")
            
            # Load train data
            train_df = conn.execute("SELECT * FROM train").df()
            dataset_logger.info(f"Loaded {len(train_df)} train records with {len(train_df.columns)} columns")
            
            # 1. Generate GENERIC features only
            dataset_logger.info("🔧 Generating GENERIC features...")
            generic_df = feature_space.generate_generic_features(train_df)
            dataset_logger.info(f"Generated {len(generic_df.columns)} generic columns")
            
            # Save generic table
            conn.execute("DROP TABLE IF EXISTS generic")
            conn.register('generic_df', generic_df)
            conn.execute("CREATE TABLE generic AS SELECT * FROM generic_df")
            conn.unregister('generic_df')
            dataset_logger.info(f"✅ Created 'generic' table with {len(generic_df.columns)} columns")
            
            # 2. Generate CUSTOM features only
            dataset_logger.info("🎯 Generating CUSTOM domain features...")
            custom_df = feature_space.generate_custom_features(train_df, dataset_name)
            dataset_logger.info(f"Generated {len(custom_df.columns)} custom columns")
            
            # Save custom table
            conn.execute("DROP TABLE IF EXISTS custom")
            conn.register('custom_df', custom_df)
            conn.execute("CREATE TABLE custom AS SELECT * FROM custom_df")
            conn.unregister('custom_df')
            dataset_logger.info(f"✅ Created 'custom' table with {len(custom_df.columns)} columns")
            
            # 3. Create train_features by column concatenation in DuckDB
            dataset_logger.info("🔗 Creating train_features (train + generic + custom)...")
            conn.execute("""
                DROP TABLE IF EXISTS train_features;
                CREATE TABLE train_features AS 
                SELECT t.*, g.*, c.*
                FROM train t, generic g, custom c 
                WHERE t.rowid = g.rowid AND g.rowid = c.rowid
            """)
            
            result = conn.execute("SELECT COUNT(*) FROM pragma_table_info('train_features')").fetchone()
            dataset_logger.info(f"✅ Created 'train_features' table with {result[0]} columns")
        
        # Process test data if exists
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test'").fetchone():
            dataset_logger.info("📊 Generating features for test data...")
            
            # Load test data
            test_df = conn.execute("SELECT * FROM test").df()
            dataset_logger.info(f"Loaded {len(test_df)} test records with {len(test_df.columns)} columns")
            
            # Remove target column if exists in test
            if target_column in test_df.columns:
                test_df = test_df.drop(columns=[target_column])
            
            # 1. Generate GENERIC features for test
            dataset_logger.info("🔧 Generating GENERIC features for test...")
            test_generic_df = feature_space.generate_generic_features(test_df)
            dataset_logger.info(f"Generated {len(test_generic_df.columns)} generic columns for test")
            
            # Save test_generic table
            conn.execute("DROP TABLE IF EXISTS test_generic")
            conn.register('test_generic_df', test_generic_df)
            conn.execute("CREATE TABLE test_generic AS SELECT * FROM test_generic_df")
            conn.unregister('test_generic_df')
            dataset_logger.info(f"✅ Created 'test_generic' table with {len(test_generic_df.columns)} columns")
            
            # 2. Generate CUSTOM features for test
            dataset_logger.info("🎯 Generating CUSTOM domain features for test...")
            test_custom_df = feature_space.generate_custom_features(test_df, dataset_name)
            dataset_logger.info(f"Generated {len(test_custom_df.columns)} custom columns for test")
            
            # Save test_custom table
            conn.execute("DROP TABLE IF EXISTS test_custom")
            conn.register('test_custom_df', test_custom_df)
            conn.execute("CREATE TABLE test_custom AS SELECT * FROM test_custom_df")
            conn.unregister('test_custom_df')
            dataset_logger.info(f"✅ Created 'test_custom' table with {len(test_custom_df.columns)} columns")
            
            # 3. Create test_features by column concatenation in DuckDB
            dataset_logger.info("🔗 Creating test_features (test + test_generic + test_custom)...")
            conn.execute("""
                DROP TABLE IF EXISTS test_features;
                CREATE TABLE test_features AS 
                SELECT t.*, g.*, c.*
                FROM test t, test_generic g, test_custom c 
                WHERE t.rowid = g.rowid AND g.rowid = c.rowid
            """)
            
            result = conn.execute("SELECT COUNT(*) FROM pragma_table_info('test_features')").fetchone()
            dataset_logger.info(f"✅ Created 'test_features' table with {result[0]} columns")
    
    def generate_dataset_id(self, file_mappings: Dict[str, str]) -> str:
        """Generate unique dataset ID based on file contents.
        
        Args:
            file_mappings: Dict mapping table names to file paths
            
        Returns:
            MD5 hash-based dataset ID
        """
        # Create hash from file paths and modification times
        hash_input = []
        
        for table_name in sorted(file_mappings.keys()):
            file_path = Path(file_mappings[table_name])
            if file_path.exists():
                # Include file path, size, and modification time
                stat = file_path.stat()
                hash_input.extend([
                    table_name,
                    str(file_path),
                    str(stat.st_size),
                    str(stat.st_mtime)
                ])
        
        # Add dataset name to ensure uniqueness
        hash_input.append(self.dataset_name)
        
        # Generate MD5 hash
        hash_string = '|'.join(hash_input)
        dataset_id = hashlib.md5(hash_string.encode()).hexdigest()
        
        logger.info(f"Generated dataset ID: {dataset_id}")
        return dataset_id
    
    def _setup_dataset_logging(self, dataset_name: str):
        """Setup contextual logging for dataset operations."""
        try:
            import yaml
            from logging_utils import setup_dataset_logging, setup_main_logging
            
            # Load main config for logging setup
            try:
                config_path = Path(__file__).parent.parent / 'config' / 'mcts_config.yaml'
                with open(config_path, 'r') as f:
                    main_config = yaml.safe_load(f)
            except Exception:
                # Fallback config
                main_config = {'logging': {'level': 'INFO', 'log_file': 'logs/minotaur.log', 'max_log_size_mb': 100, 'backup_count': 5}}
            
            # Setup main logging system if not already configured
            try:
                setup_main_logging(main_config)
            except Exception:
                pass  # May already be configured
            
            # Create dataset logger that uses main application logging
            dataset_logger = setup_dataset_logging(dataset_name, main_config)
            
            return dataset_logger
            
        except Exception as e:
            # Fallback to global logger
            logger.warning(f"Could not setup dataset logging for {dataset_name}: {e}")
            return logger