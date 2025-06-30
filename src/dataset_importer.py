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
                        logger.info(f"Detected target column: {col.lower()}")
                        return col.lower()
            
            # If no pattern match, look for columns with limited unique values
            # (likely categorical targets)
            categorical_candidates = []
            for col in df.columns:
                if df[col].dtype in ['object', 'category'] or df[col].nunique() <= 20:
                    categorical_candidates.append((col, df[col].nunique()))
            
            if categorical_candidates:
                # Return column with fewest unique values
                best_candidate = min(categorical_candidates, key=lambda x: x[1])
                logger.info(f"Detected likely target column: {best_candidate[0].lower()} ({best_candidate[1]} unique values)")
                return best_candidate[0].lower()
            
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
            
            # ID column patterns - case insensitive, ends with 'id' or starts with 'id'
            for col in df.columns:
                col_lower = col.lower()
                if re.search(r'.*id$', col_lower) or re.search(r'^id.*', col_lower):
                    # Verify it looks like an ID (unique, numeric or string)
                    if df[col].nunique() == len(df):  # All unique values
                        logger.info(f"Detected ID column: {col.lower()}")
                        return col.lower()
            
            # Look for first column if it's unique and looks like an index
            first_col = df.columns[0]
            if df[first_col].nunique() == len(df) and (
                df[first_col].dtype in ['int64', 'object'] and 
                str(df[first_col].iloc[0]).isdigit()
            ):
                logger.info(f"Detected ID column (first column): {first_col.lower()}")
                return first_col.lower()
            
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
                
                # Create table from file with lowercase column names
                if file_path_obj.suffix.lower() == '.parquet':
                    # First create temp table, then create final table with lowercase columns
                    conn.execute(f"CREATE TABLE {table_name}_temp AS SELECT * FROM read_parquet('{file_path}')")
                else:
                    conn.execute(f"CREATE TABLE {table_name}_temp AS SELECT * FROM read_csv_auto('{file_path}')")
                
                # Get column names and create lowercase versions
                temp_columns = conn.execute(f"DESCRIBE {table_name}_temp").fetchall()
                column_mappings = []
                for col_info in temp_columns:
                    original_name = col_info[0]
                    lowercase_name = original_name.lower()
                    column_mappings.append(f'"{original_name}" AS "{lowercase_name}"')
                
                # Create final table with lowercase column names
                select_clause = ", ".join(column_mappings)
                conn.execute(f"CREATE TABLE {table_name} AS SELECT {select_clause} FROM {table_name}_temp")
                
                # Drop temporary table
                conn.execute(f"DROP TABLE {table_name}_temp")
                
                # Get record count
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                record_count = result[0] if result else 0
                
                logger.info(f"✅ Imported {record_count:,} records to table '{table_name}'")
            
            # Verify target column exists in train table (now with lowercase names)
            if 'train' in file_mappings:
                columns = conn.execute("DESCRIBE train").fetchall()
                column_names = [col[0] for col in columns]
                
                # Check if target and ID columns exist in database (parameters already lowercase)
                if target_column and target_column not in column_names:
                    logger.warning(f"Target column '{target_column}' not found in train table")
                    logger.info(f"Available columns: {', '.join(column_names)}")
                
                if id_column and id_column not in column_names:
                    logger.warning(f"ID column '{id_column}' not found in train table")
            
            # Generate features for train and test tables
            logger.info("Generating features for dataset...")
            # Pass target and ID columns (already lowercase from earlier conversion)
            self._generate_and_save_features(conn, self.dataset_name, target_column, id_column)
            
            conn.close()
            
            # Register original dataset features in the feature catalog
            try:
                from features.train_features import register_train_features
                train_file = file_mappings.get('train')
                if train_file and target_column:
                    logger.info(f"🏷️ Registering original dataset features as 'train' origin...")
                    register_train_features(
                        dataset_name=self.dataset_name,
                        train_path=train_file,
                        target_column=target_column,
                        id_column=id_column or 'passengerid'  # Fallback for datasets without ID
                    )
                    logger.info(f"✅ Original dataset features registered successfully")
            except Exception as e:
                logger.warning(f"Failed to register train features: {e}")
            
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
        
        # Add target and ID column information for feature filtering (lowercase to match database)
        config['autogluon']['target_column'] = target_column.lower() if target_column else None
        config['autogluon']['id_column'] = id_column.lower() if id_column else None
        
        # Lowercase ignore_columns if they exist
        existing_ignore = config['autogluon'].get('ignore_columns', []) or []
        config['autogluon']['ignore_columns'] = [col.lower() if isinstance(col, str) else col for col in existing_ignore]
        
        # Log dataset configuration
        dataset_logger.info(f"📊 Dataset configuration: target='{config['autogluon']['target_column']}', id='{config['autogluon']['id_column']}', ignore={config['autogluon']['ignore_columns']}")
        
        # Auto-detect custom domain module based on dataset name
        dataset_name_clean = dataset_name.lower().replace('-', '_').replace(' ', '_')
        
        # Check if custom domain module exists for this dataset
        custom_modules_dir = Path(__file__).parent / 'features' / 'custom'
        module_file = f"{dataset_name_clean}.py"
        
        if (custom_modules_dir / module_file).exists():
            module_name = module_file[:-3]  # Remove .py extension
            config['feature_space']['custom_domain_module'] = module_name
            dataset_logger.info(f"Auto-detected custom domain module: {module_name}")
        else:
            dataset_logger.info("No custom domain module found, using generic features only")
        
        # Initialize FeatureSpace
        feature_space = FeatureSpace(config)
        
        # Check if new pipeline is enabled
        use_new_pipeline = config.get('feature_space', {}).get('use_new_pipeline', False)
        
        # Process train data if exists
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='train'").fetchone():
            dataset_logger.info("📊 Generating features for train data...")
            
            # Load train data
            train_df = conn.execute("SELECT * FROM train").df()
            dataset_logger.info(f"Loaded {len(train_df)} train records with {len(train_df.columns)} columns")
            
            # NEW PIPELINE PATH - All features at once with signal detection during generation
            if use_new_pipeline:
                dataset_logger.info("🚀 Using new feature pipeline with signal detection during generation...")
                
                # Generate all features using pipeline
                features_df = feature_space.generate_all_features_pipeline(
                    train_df, 
                    dataset_name=dataset_name,
                    target_column=target_column,
                    id_column=id_column
                )
                
                # Save directly as train_features (no post-hoc filtering needed)
                conn.execute("DROP TABLE IF EXISTS train_features")
                conn.register('features_df', features_df)
                conn.execute("CREATE TABLE train_features AS SELECT * FROM features_df")
                conn.unregister('features_df')
                
                dataset_logger.info(f"✅ Created 'train_features' table with {len(features_df.columns)} columns (signal checked during generation)")
                
                # Save feature metadata
                metadata = feature_space.get_feature_metadata()
                dataset_logger.info(f"📊 Feature statistics: {len(metadata)} total, "
                                  f"{sum(1 for m in metadata.values() if m.has_signal)} with signal")
                
                # Skip the old logic
                valid_train_columns = list(features_df.columns)
            else:
                # OLD PIPELINE PATH - Generate features separately with post-hoc filtering
                # 1. Generate GENERIC features only
                dataset_logger.info("🔧 Generating GENERIC features...")
                generic_df = feature_space.generate_generic_features(train_df, target_column=target_column, id_column=id_column)
                dataset_logger.info(f"Generated {len(generic_df.columns)} generic columns")
                
                # Save train_generic table
                conn.execute("DROP TABLE IF EXISTS train_generic")
                conn.register('generic_df', generic_df)
                conn.execute("CREATE TABLE train_generic AS SELECT * FROM generic_df")
                conn.unregister('generic_df')
                dataset_logger.info(f"✅ Created 'train_generic' table with {len(generic_df.columns)} columns")
                
                # 2. Generate CUSTOM features only
                dataset_logger.info("🎯 Generating CUSTOM domain features...")
                custom_df = feature_space.generate_custom_features(train_df, dataset_name)
                dataset_logger.info(f"Generated {len(custom_df.columns)} custom columns")
                
                # Save train_custom table
                conn.execute("DROP TABLE IF EXISTS train_custom")
                conn.register('custom_df', custom_df)
                conn.execute("CREATE TABLE train_custom AS SELECT * FROM custom_df")
                conn.unregister('custom_df')
                dataset_logger.info(f"✅ Created 'train_custom' table with {len(custom_df.columns)} columns")
                
                # 3. Create train_features by column concatenation in DuckDB
                dataset_logger.info("🔗 Creating train_features (train + train_generic + train_custom)...")
                
                # Get column names from each table to handle duplicates
                train_cols = conn.execute("SELECT name FROM pragma_table_info('train')").fetchall()
                generic_cols = conn.execute("SELECT name FROM pragma_table_info('train_generic')").fetchall()
                custom_cols = conn.execute("SELECT name FROM pragma_table_info('train_custom')").fetchall()
                
                # Build deduplicated column list
                seen_columns = set()
                select_parts = []
                
                # Add all columns from train table
                for col in train_cols:
                    col_name = col[0]
                    select_parts.append(f't."{col_name}"')
                    seen_columns.add(col_name)
                
                # Add non-duplicate columns from generic table
                for col in generic_cols:
                    col_name = col[0]
                    if col_name not in seen_columns:
                        select_parts.append(f'g."{col_name}"')
                        seen_columns.add(col_name)
                
                # Add non-duplicate columns from custom table
                for col in custom_cols:
                    col_name = col[0]
                    if col_name not in seen_columns:
                        select_parts.append(f'c."{col_name}"')
                        seen_columns.add(col_name)
                
                # Build the SELECT statement
                select_clause = ", ".join(select_parts)
                
                conn.execute(f"""
                    DROP TABLE IF EXISTS train_features;
                    CREATE TABLE train_features AS 
                    SELECT {select_clause}
                    FROM train t, train_generic g, train_custom c 
                    WHERE t.rowid = g.rowid AND g.rowid = c.rowid
                """)
                
                result = conn.execute("SELECT COUNT(*) FROM pragma_table_info('train_features')").fetchone()
                dataset_logger.info(f"✅ Created 'train_features' table with {result[0]} columns")
                
                # Filter out no-signal columns from train_features and get valid column list
                valid_train_columns = self._filter_no_signal_columns(conn, 'train_features', target_column, id_column, dataset_logger)
        
        # Process test data if exists
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test'").fetchone():
            dataset_logger.info("📊 Generating features for test data...")
            
            # Load test data
            test_df = conn.execute("SELECT * FROM test").df()
            dataset_logger.info(f"Loaded {len(test_df)} test records with {len(test_df.columns)} columns")
            
            # Remove target column if exists in test
            if target_column in test_df.columns:
                test_df = test_df.drop(columns=[target_column])
            
            # NEW PIPELINE PATH for test data
            if use_new_pipeline:
                dataset_logger.info("🚀 Using new feature pipeline for test data...")
                
                # Configure to not check signal for test data
                feature_space._adapter.generator.check_signal = False
                
                # Generate all features using pipeline
                test_features_df = feature_space.generate_all_features_pipeline(
                    test_df, 
                    dataset_name=dataset_name,
                    target_column=None,  # No target in test
                    id_column=id_column
                )
                
                # Align with train columns
                if 'valid_train_columns' in locals():
                    # Keep only columns that exist in training
                    common_columns = [col for col in valid_train_columns if col in test_features_df.columns]
                    test_features_df = test_features_df[common_columns]
                
                # Save directly as test_features
                conn.execute("DROP TABLE IF EXISTS test_features")
                conn.register('test_features_df', test_features_df)
                conn.execute("CREATE TABLE test_features AS SELECT * FROM test_features_df")
                conn.unregister('test_features_df')
                
                dataset_logger.info(f"✅ Created 'test_features' table with {len(test_features_df.columns)} columns")
            else:
                # OLD PIPELINE PATH for test data
                # 1. Generate GENERIC features for test (without signal checking and auto-registration)
                dataset_logger.info("🔧 Generating GENERIC features for test...")
                test_generic_df = feature_space.generate_generic_features(test_df, check_signal=False, target_column=None, id_column=id_column, auto_register=False, origin='generic')
                dataset_logger.info(f"Generated {len(test_generic_df.columns)} generic columns for test")
                
                # Save test_generic table
                conn.execute("DROP TABLE IF EXISTS test_generic")
                conn.register('test_generic_df', test_generic_df)
                conn.execute("CREATE TABLE test_generic AS SELECT * FROM test_generic_df")
                conn.unregister('test_generic_df')
                dataset_logger.info(f"✅ Created 'test_generic' table with {len(test_generic_df.columns)} columns")
                
                # 2. Generate CUSTOM features for test (without signal checking and auto-registration)
                dataset_logger.info("🎯 Generating CUSTOM domain features for test...")
                test_custom_df = feature_space.generate_custom_features(test_df, dataset_name, check_signal=False, auto_register=False, origin='custom')
                dataset_logger.info(f"Generated {len(test_custom_df.columns)} custom columns for test")
                
                # Save test_custom table
                conn.execute("DROP TABLE IF EXISTS test_custom")
                conn.register('test_custom_df', test_custom_df)
                conn.execute("CREATE TABLE test_custom AS SELECT * FROM test_custom_df")
                conn.unregister('test_custom_df')
                dataset_logger.info(f"✅ Created 'test_custom' table with {len(test_custom_df.columns)} columns")
                
                # 3. Create test_features by column concatenation in DuckDB
                dataset_logger.info("🔗 Creating test_features (test + test_generic + test_custom)...")
                
                # Get column names from each table to handle duplicates
                test_cols = conn.execute("SELECT name FROM pragma_table_info('test')").fetchall()
                test_generic_cols = conn.execute("SELECT name FROM pragma_table_info('test_generic')").fetchall()
                test_custom_cols = conn.execute("SELECT name FROM pragma_table_info('test_custom')").fetchall()
                
                # Build deduplicated column list
                seen_columns = set()
                select_parts = []
                
                # Add all columns from test table
                for col in test_cols:
                    col_name = col[0]
                    select_parts.append(f't."{col_name}"')
                    seen_columns.add(col_name)
                
                # Add non-duplicate columns from generic table
                for col in test_generic_cols:
                    col_name = col[0]
                    if col_name not in seen_columns:
                        select_parts.append(f'g."{col_name}"')
                        seen_columns.add(col_name)
                
                # Add non-duplicate columns from custom table
                for col in test_custom_cols:
                    col_name = col[0]
                    if col_name not in seen_columns:
                        select_parts.append(f'c."{col_name}"')
                        seen_columns.add(col_name)
                
                # Build the SELECT statement
                select_clause = ", ".join(select_parts)
                
                conn.execute(f"""
                    DROP TABLE IF EXISTS test_features;
                    CREATE TABLE test_features AS 
                    SELECT {select_clause}
                    FROM test t, test_generic g, test_custom c 
                    WHERE t.rowid = g.rowid AND g.rowid = c.rowid
                """)
                
                result = conn.execute("SELECT COUNT(*) FROM pragma_table_info('test_features')").fetchone()
                dataset_logger.info(f"✅ Created 'test_features' table with {result[0]} columns")
                
                # Filter test_features to match valid train columns (don't remove based on test signal)
                self._align_test_features_with_train(conn, 'test_features', valid_train_columns, target_column, dataset_logger)
        
        # PART 5: Validate train/test feature column synchronization
        if (conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='train_features'").fetchone() and
            conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_features'").fetchone()):
            self._validate_feature_columns(conn, target_column, dataset_logger)
            
            # Save debug dumps if DEBUG is enabled
            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                self._save_debug_dumps(conn, dataset_name, dataset_logger)
    
    def _filter_no_signal_columns(self, conn, table_name: str, target_column: str, id_column: str, dataset_logger) -> list:
        """Remove columns with no signal (constant values) from feature table. Returns list of remaining columns."""
        try:
            # Get all columns
            columns_result = conn.execute(f"SELECT name FROM pragma_table_info('{table_name}')").fetchall()
            all_columns = [col[0] for col in columns_result]
            
            # Find columns to remove (no signal)
            columns_to_remove = []
            
            for col in all_columns:
                # Skip target and ID columns (use passed id_column parameter)
                if col == target_column or (id_column and col == id_column):
                    continue
                
                # Check if column has signal (more than 1 unique value)
                result = conn.execute(f'SELECT COUNT(DISTINCT "{col}") FROM {table_name}').fetchone()
                unique_count = result[0] if result else 0
                
                if unique_count <= 1:
                    columns_to_remove.append(col)
            
            remaining_columns = [col for col in all_columns if col not in columns_to_remove]
            
            if columns_to_remove:
                dataset_logger.info(f"🧹 Removing {len(columns_to_remove)} no-signal columns from {table_name}")
                
                # Log all removed columns
                for col in columns_to_remove:
                    dataset_logger.debug(f"  Removing: {col}")
                
                # Create new table without no-signal columns
                column_list = ', '.join([f'"{col}"' for col in remaining_columns])
                
                conn.execute(f"""
                    CREATE TABLE {table_name}_filtered AS
                    SELECT {column_list} FROM {table_name}
                """)
                
                # Replace original table
                conn.execute(f"DROP TABLE {table_name}")
                conn.execute(f"ALTER TABLE {table_name}_filtered RENAME TO {table_name}")
                
                final_count = len(remaining_columns)
                dataset_logger.info(f"✅ Filtered {table_name}: {len(all_columns)} -> {final_count} columns (removed {len(columns_to_remove)})")
            else:
                dataset_logger.info(f"✅ No no-signal columns found in {table_name}")
            
            return remaining_columns
                
        except Exception as e:
            dataset_logger.error(f"Failed to filter no-signal columns from {table_name}: {e}")
            return []
    
    def _align_test_features_with_train(self, conn, table_name: str, valid_train_columns: list, target_column: str, dataset_logger) -> None:
        """Filter test_features to only include columns that had signal in train (excluding target)."""
        try:
            # Get all test columns
            test_columns_result = conn.execute(f"SELECT name FROM pragma_table_info('{table_name}')").fetchall()
            all_test_columns = [col[0] for col in test_columns_result]
            
            # Remove target column from valid train columns for comparison
            valid_train_columns_no_target = [col for col in valid_train_columns if col != target_column]
            
            # Keep only columns that exist in both test and valid train columns
            columns_to_keep = [col for col in all_test_columns if col in valid_train_columns_no_target]
            
            # Log what we're keeping vs removing
            columns_to_remove = [col for col in all_test_columns if col not in columns_to_keep]
            
            if columns_to_remove:
                dataset_logger.info(f"🔄 Aligning test_features with train: keeping {len(columns_to_keep)}, removing {len(columns_to_remove)} columns")
                
                # Log all removed columns
                for col in columns_to_remove:
                    dataset_logger.debug(f"  Removing from test: {col}")
                
                # Check if we have columns to keep
                if not columns_to_keep:
                    dataset_logger.error("❌ No columns to keep after alignment - this should not happen")
                    raise ValueError("No valid columns remain after test feature alignment")
                
                # Create new table with only valid columns
                column_list = ', '.join([f'"{col}"' for col in columns_to_keep])
                
                conn.execute(f"""
                    CREATE TABLE {table_name}_aligned AS
                    SELECT {column_list} FROM {table_name}
                """)
                
                # Replace original table
                conn.execute(f"DROP TABLE {table_name}")
                conn.execute(f"ALTER TABLE {table_name}_aligned RENAME TO {table_name}")
                
                dataset_logger.info(f"✅ Aligned {table_name}: {len(all_test_columns)} -> {len(columns_to_keep)} columns")
            else:
                dataset_logger.info(f"✅ Test features already aligned with train")
                
        except Exception as e:
            dataset_logger.error(f"Failed to align test features with train: {e}")

    def _save_debug_dumps(self, conn, dataset_name: str, dataset_logger) -> None:
        """Save train_features and test_features to CSV files for debugging."""
        try:
            # Save train_features
            train_path = f"/tmp/{dataset_name}-train-0000.csv"
            train_df = conn.execute("SELECT * FROM train_features").df()
            train_df.to_csv(train_path, index=False)
            dataset_logger.info(f"📝 DEBUG: Saved train_features to {train_path} "
                              f"({len(train_df)} rows, {len(train_df.columns)} columns)")
            
            # Save test_features  
            test_path = f"/tmp/{dataset_name}-test-0000.csv"
            test_df = conn.execute("SELECT * FROM test_features").df()
            test_df.to_csv(test_path, index=False)
            dataset_logger.info(f"📝 DEBUG: Saved test_features to {test_path} "
                             f"({len(test_df)} rows, {len(test_df.columns)} columns)")
                             
        except Exception as e:
            dataset_logger.error(f"Failed to save debug dumps: {e}")
    
    def _validate_feature_columns(self, conn, target_column: str, dataset_logger):
        """Validate that train and test feature columns are synchronized."""
        try:
            # Get column names from both tables
            train_columns_result = conn.execute("SELECT name FROM pragma_table_info('train_features')").fetchall()
            test_columns_result = conn.execute("SELECT name FROM pragma_table_info('test_features')").fetchall()
            
            train_columns = {col[0] for col in train_columns_result}
            test_columns = {col[0] for col in test_columns_result}
            
            # Remove target column from train columns for comparison
            train_columns_no_target = train_columns - {target_column}
            
            # Check if columns are synchronized
            if train_columns_no_target == test_columns:
                dataset_logger.info(f"✅ Feature columns synchronized: {len(test_columns)} columns in both train (without target) and test")
                return
            
            # Calculate differences
            missing_in_test = train_columns_no_target - test_columns
            extra_in_test = test_columns - train_columns_no_target
            
            # Log detailed error information
            dataset_logger.error(f"❌ Feature column mismatch detected!")
            dataset_logger.error(f"  Train columns (without target): {len(train_columns_no_target)}")
            dataset_logger.error(f"  Test columns: {len(test_columns)}")
            
            if missing_in_test:
                dataset_logger.error(f"  Missing in test ({len(missing_in_test)}): {sorted(list(missing_in_test))}")
            
            if extra_in_test:
                dataset_logger.error(f"  Extra in test ({len(extra_in_test)}): {sorted(list(extra_in_test))}")
            
            raise ValueError(f"Feature columns must be identical between train and test! Train: {len(train_columns_no_target)}, Test: {len(test_columns)}")
            
        except Exception as e:
            dataset_logger.error(f"Failed to validate feature columns: {e}")
            raise ValueError(f"Feature column validation failed: {e}")
    
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