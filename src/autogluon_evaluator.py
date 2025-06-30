"""
AutoGluon Evaluator for MCTS Feature Discovery

Fast and adaptive AutoGluon wrapper optimized for MCTS exploration.
Supports different evaluation strategies for exploration vs exploitation phases.
"""

import os
import time
import tempfile
import shutil
import logging
import random
import string
import hashlib
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .timing import timed, timing_context, record_timing

logger = logging.getLogger(__name__)

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except ImportError:
    logger.error("AutoGluon not installed. Please install with: pip install autogluon")
    raise

class AutoGluonEvaluator:
    """
    AutoGluon evaluator optimized for MCTS feature discovery.
    
    Provides fast evaluation with adaptive configuration based on exploration phase.
    Manages model directories and caching for efficient repeated evaluations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AutoGluon evaluator with configuration."""
        self.config = config
        self.autogluon_config = config['autogluon']
        self.session_config = config['session']
        self.resource_config = config['resources']
        
        # Dataset configuration - support both old and new systems
        self.target_metric = self.autogluon_config['target_metric']
        
        # Resolve dataset information
        dataset_info = self._resolve_dataset_configuration()
        self.database_path = dataset_info['database_path']
        self.target_column = dataset_info['target_column']
        self.id_column = dataset_info['id_column']
        self.ignore_columns = self.autogluon_config.get('ignore_columns', []) or []
        self.use_duckdb_cache = dataset_info.get('use_duckdb_cache', False)
        self.dataset_name = dataset_info.get('dataset_name')
        
        # Load base data once
        self.base_train_data = None
        self.base_test_data = None
        self.validation_data = None
        
        # Note: Multi-phase evaluation system removed per user request
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_eval_time = 0.0
        self.best_score = 0.0
        self.evaluation_cache = {}  # Cache results for identical feature sets
        
        # Temporary directories management - generate unique suffix for each instance
        base_temp_dir = self.resource_config.get('temp_dir', '/tmp/mcts_features')
        random_suffix = self._generate_random_suffix()
        self.temp_base_dir = os.path.join(base_temp_dir, random_suffix)
        Path(self.temp_base_dir).mkdir(parents=True, exist_ok=True)
        self.model_dirs = []  # Track created model directories for cleanup
        logger.debug(f"Created unique temp directory: {self.temp_base_dir}")
        
        # Initialize data
        self._load_base_data()
        self._setup_evaluation_strategy()
        
        logger.info(f"Initialized AutoGluonEvaluator for {self.target_metric}")
    
    def _resolve_dataset_configuration(self) -> Dict[str, str]:
        """Resolve dataset configuration from registry - fail if dataset not found."""
        dataset_name = self.autogluon_config.get('dataset_name')
        if not dataset_name:
            raise ValueError("dataset_name must be specified in autogluon configuration")
        
        from .discovery_db import FeatureDiscoveryDB
        
        # Create temporary DB instance to access dataset registry (read-only mode)
        temp_db = FeatureDiscoveryDB(self.config, read_only=True)
        
        # Use the new database service API
        dataset_repo = temp_db.db_service.dataset_repo
        result = dataset_repo.find_by_name(dataset_name)
        
        if not result:
            raise ValueError(f"Dataset '{dataset_name}' not found in registry. Please register it first using: python manager.py datasets --register --dataset-name {dataset_name}")
        
        dataset_id, name, target_col, id_col = result.dataset_id, result.dataset_name, result.target_column, result.id_column
        
        # Build cache paths
        cache_dir = Path(self.config.get('project_root', '.')) / 'cache' / dataset_name
        cache_db_path = cache_dir / 'dataset.duckdb'
        
        if not cache_db_path.exists():
            raise FileNotFoundError(f"Dataset cache not found at {cache_db_path}. Please register the dataset properly.")
        
        logger.info(f"Using dataset: {dataset_name} from {cache_db_path}")
        return {
            'database_path': str(cache_db_path),
            'target_column': target_col,
            'id_column': id_col or 'id',
            'dataset_name': dataset_name,
            'use_duckdb_cache': True
        }
    
    def _load_base_data(self) -> None:
        """Load and prepare base training/test data using configured backend."""
        try:
            # Check configured backend
            data_config = self.config.get('data', {})
            backend = data_config.get('backend', 'duckdb').lower()
            
            logger.info(f"Loading base training and test data using backend: {backend}")
            
            # Initialize DuckDB data manager for persistent storage
            from .duckdb_data_manager import DuckDBDataManager, is_duckdb_available
            
            # Use DuckDB for persistent data storage and loading (required for registered datasets)
            if self.use_duckdb_cache:
                self.duckdb_manager = DuckDBDataManager(self.config)
                
                logger.info(f"Loading data from DuckDB database: {self.database_path}")
                # Load train and test data from same database using SQL queries
                self.base_train_data = self.duckdb_manager.execute_query("SELECT * FROM train_features")
                self.base_test_data = self.duckdb_manager.execute_query("SELECT * FROM test_features")
            else:
                raise ValueError("Only registered datasets with DuckDB cache are supported. Please register your dataset first.")
            
            # Use full training data - AutoGluon will handle validation split internally via holdout_frac
            self.base_train_data = self.base_train_data.reset_index(drop=True)
            self.validation_data = None  # AutoGluon handles validation internally
            
            logger.info(f"Data loaded: train={len(self.base_train_data)}, "
                       f"test={len(self.base_test_data) if self.base_test_data is not None else 0}")
            
        except Exception as e:
            logger.error(f"Failed to load base data: {e}")
            raise
    
    def _setup_evaluation_strategy(self) -> None:
        """Setup evaluation strategy - simplified to single config."""
        logger.info("Using single-phase evaluation strategy per user configuration")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current AutoGluon configuration - use main config only."""
        # Use main autogluon config directly, ignore phases
        base_config = {}
        
        # Copy all parameters - multi-phase configs removed
        for key, value in self.autogluon_config.items():
            base_config[key] = value
        
        return base_config
    
    @timed("autogluon.evaluate_features_columns", include_memory=True)
    def evaluate_features_from_columns(self, 
                                     feature_columns: List[str], 
                                     node_depth: int = 0, 
                                     iteration: int = 0) -> float:
        """
        Evaluate a feature set by selecting specific columns from DuckDB.
        
        Args:
            feature_columns: List of column names to select from train_features
            node_depth: Depth of node in MCTS tree (for adaptive config)
            iteration: Current MCTS iteration
            
        Returns:
            float: Evaluation score (MAP@3)
        """
        eval_start_time = time.time()
        self.evaluation_count += 1
        
        logger.info(f"AutoGluon evaluation #{self.evaluation_count}: {len(feature_columns)} features")
        logger.debug(f"🔍 DEBUG: evaluation_count={self.evaluation_count}, feature_columns={feature_columns}, iteration={iteration}")
        
        # Check cache first based on column names
        feature_hash = hashlib.md5(','.join(sorted(feature_columns)).encode()).hexdigest()
        if feature_hash in self.evaluation_cache:
            cached_score = self.evaluation_cache[feature_hash]
            logger.debug(f"Using cached evaluation: {cached_score:.5f}")
            return cached_score
        
        try:
            # Get current evaluation configuration
            eval_config = self.get_current_config()
            logger.debug(f"Using evaluation config: {eval_config}")
            
            # Prepare data by selecting columns from DuckDB
            # For first iteration (baseline), use original train/test tables
            # For subsequent iterations, use train_features/test_features tables with generated features
            if self.evaluation_count == 1:
                logger.info("🚀 First iteration detected: using original train/test data for baseline")
                train_data = self._prepare_autogluon_data_from_original_tables(eval_config)
            else:
                train_data = self._prepare_autogluon_data_from_columns(feature_columns, eval_config)
            
            # Create temporary model directory
            model_dir = self._create_temp_model_dir()
            logger.debug(f"Created model directory: {model_dir}")
            
            # Train and evaluate  
            score = self._train_and_evaluate(train_data, eval_config, model_dir)
            
            # Cache the result
            self.evaluation_cache[feature_hash] = score
            
            # Track evaluation time
            eval_time = time.time() - eval_start_time
            logger.info(f"AutoGluon evaluation complete: score={score:.5f}, time={eval_time:.2f}s")
            record_timing("autogluon.evaluation_complete", eval_time)
            
            return score
            
        except Exception as e:
            logger.error(f"AutoGluon evaluation failed: {e}")
            raise
        
        finally:
            # Note: Cleanup handled by main cleanup() method at end of session
            pass
    
    @timed("autogluon.evaluate_features", include_memory=True)
    def evaluate_features(self, 
                         features_df: pd.DataFrame, 
                         node_depth: int = 0, 
                         iteration: int = 0) -> float:
        """
        Evaluate a feature set using AutoGluon.
        
        Args:
            features_df: DataFrame with engineered features
            node_depth: Depth of node in MCTS tree (for adaptive config)
            iteration: Current MCTS iteration
            
        Returns:
            float: Evaluation score (MAP@3)
        """
        eval_start_time = time.time()
        self.evaluation_count += 1
        
        logger.info(f"AutoGluon evaluation #{self.evaluation_count}: {len(features_df)} rows, {len(features_df.columns)} cols")
        
        # Note: Multi-phase evaluation removed per user request
        
        # Check cache first
        feature_hash = self._hash_features(features_df)
        if feature_hash in self.evaluation_cache:
            cached_score = self.evaluation_cache[feature_hash]
            logger.debug(f"Using cached evaluation: {cached_score:.5f}")
            return cached_score
        
        try:
            # Get current evaluation configuration first
            eval_config = self.get_current_config()
            logger.debug(f"Using evaluation config: {eval_config}")
            
            # Prepare data for AutoGluon with config
            logger.debug("Preparing AutoGluon data...")
            train_data = self._prepare_autogluon_data(features_df, eval_config)
            logger.info(f"Prepared data: train={len(train_data)}")
            
            # Create temporary model directory
            model_dir = self._create_temp_model_dir()
            logger.debug(f"Created model directory: {model_dir}")
            
            # Log basic data shape info only
            logger.debug(f"Training data shape: {train_data.shape[0]} rows, {train_data.shape[1]} columns")
            
            # Train and evaluate
            logger.info("Starting AutoGluon training and evaluation...")
            score = self._train_and_evaluate(train_data, eval_config, model_dir)
            logger.info(f"AutoGluon evaluation completed: score={score:.5f}")
            
            # Cache result
            self.evaluation_cache[feature_hash] = score
            
            # Update statistics
            eval_time = time.time() - eval_start_time
            self.total_eval_time += eval_time
            
            if score > self.best_score:
                self.best_score = score
                logger.info(f"New best evaluation score: {score:.5f} ({self.target_metric})")
            
            logger.debug(f"Evaluation completed: score={score:.5f}, time={eval_time:.2f}s, "
                        f"features={len(features_df.columns)}")
            
            return score
            
        except Exception as e:
            logger.error(f"AutoGluon evaluation failed: {e}")
            return 0.0
        
        finally:
            # Note: Cleanup handled by main cleanup() method at end of session
            pass
    
    def _prepare_autogluon_data_from_original_tables(self, eval_config: Dict[str, Any] = None) -> TabularDataset:
        """
        Prepare training data from original train/test tables (for first iteration baseline).
        
        Args:
            eval_config: Optional evaluation configuration
            
        Returns:
            TabularDataset: Training dataset with original features only
        """
        if eval_config is None:
            eval_config = self.get_current_config()
        
        # Ensure we have DuckDB manager
        if not hasattr(self, 'duckdb_manager') or self.duckdb_manager is None:
            raise ValueError("DuckDB manager not available for baseline data loading")
        
        # Load original train data (all columns)
        train_query = "SELECT * FROM train"
        train_data = self.duckdb_manager.connection.execute(train_query).df()
        
        logger.info(f"📊 Loaded original dataset: {len(train_data)} rows, {len(train_data.columns)} columns")
        logger.debug(f"Train Data Columns: {len(train_data.columns)}")
        
        # Apply train_size sampling if configured
        train_size = eval_config.get('train_size')
        if train_size is not None:
            if isinstance(train_size, float) and 0 < train_size < 1:
                sample_size = int(len(train_data) * train_size)
                train_data = train_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
                logger.info(f"📉 Sampled to {len(train_data)} rows ({train_size*100:.1f}%)")
            elif isinstance(train_size, int) and train_size < len(train_data):
                train_data = train_data.sample(n=train_size, random_state=42).reset_index(drop=True)
                logger.info(f"📉 Sampled to {len(train_data)} rows")
        
        # Debug dataset analysis and dumping  
        self._debug_dataset_analysis(train_data, None)
        
        # Convert to TabularDataset
        train_dataset = TabularDataset(train_data)
        
        logger.debug(f"Prepared baseline AutoGluon data: train={len(train_dataset)}")
        
        return train_dataset
    
    def _prepare_autogluon_data_from_columns(self, feature_columns: List[str], eval_config: Dict[str, Any] = None) -> TabularDataset:
        """
        Prepare training and validation data by selecting specific columns from DuckDB.
        
        Args:
            feature_columns: List of column names to select from train_features/test_features
            eval_config: Optional evaluation configuration
            
        Returns:
            Tuple[TabularDataset, TabularDataset]: Training and validation datasets
        """
        if eval_config is None:
            eval_config = self.get_current_config()
        
        # Ensure we have DuckDB manager
        if not hasattr(self, 'duckdb_manager') or self.duckdb_manager is None:
            raise ValueError("DuckDB manager not available for column-based loading")
        
        # Get available columns from train_features to validate
        available_columns_result = self.duckdb_manager.connection.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'train_features'
        """).fetchall()
        available_columns = {col[0] for col in available_columns_result}
        
        # Always include ID and target columns
        required_columns = [self.id_column, self.target_column]
        
        # Filter feature columns to only those that exist
        valid_feature_columns = [col for col in feature_columns if col in available_columns]
        all_columns = list(set(required_columns + valid_feature_columns))
        
        # Filter again to ensure all columns exist
        all_columns = [col for col in all_columns if col in available_columns]
        
        if not all_columns:
            raise ValueError("No valid columns found for SQL query")
        
        # Create column list for SQL - escape column names properly
        column_list = ', '.join([f'"{col}"' if ' ' in col else col for col in all_columns])
        
        # Load train data with specific columns
        train_query = f"SELECT {column_list} FROM train_features"
        train_data = self.duckdb_manager.connection.execute(train_query).df()
        
        # Apply train_size sampling if configured
        train_size = eval_config.get('train_size')
        if train_size is not None and train_size != 1.0:
            if train_size >= len(train_data):
                # Use all data if train_size exceeds data size
                logger.debug(f"train_size ({train_size}) >= data size ({len(train_data)}), using all data")
            elif 0 < train_size < 1.0:
                # Fraction of data
                from sklearn.model_selection import train_test_split
                train_data, _ = train_test_split(
                    train_data, 
                    train_size=train_size, 
                    stratify=train_data[self.target_column],
                    random_state=42
                )
            elif train_size > 1:
                # Absolute number of samples
                if train_size < len(train_data):
                    from sklearn.model_selection import train_test_split
                    train_data, _ = train_test_split(
                        train_data, 
                        train_size=int(train_size), 
                        stratify=train_data[self.target_column],
                        random_state=42
                    )
        
        # Use full training data - AutoGluon handles validation split internally via holdout_frac
        train_final = train_data
        
        # Debug dataset analysis and dumping
        self._debug_dataset_analysis(train_final, None)
        
        
        # Convert to TabularDataset
        train_dataset = TabularDataset(train_final)
        
        logger.debug(f"Prepared AutoGluon data: train={len(train_dataset)}")
        
        return train_dataset
    
    def _prepare_autogluon_data(self, features_df: pd.DataFrame, eval_config: Dict[str, Any] = None) -> TabularDataset:
        """Prepare training and validation data for AutoGluon with DuckDB sampling support."""
        
        # Get current configuration if not provided
        if eval_config is None:
            eval_config = self.get_current_config()
        
        # Apply train_size sampling if configured
        train_size = eval_config.get('train_size')
        
        if train_size is not None:
            # Use DuckDB for efficient sampling if available
            if hasattr(self, 'duckdb_manager') and self.duckdb_manager is not None:
                try:
                    logger.info(f"🚀 Using DuckDB efficient sampling: train_size={train_size}")
                    train_data = self.duckdb_manager.sample_dataset(
                        file_path=self.database_path,
                        train_size=train_size,
                        stratify_column=self.target_column
                    )
                    logger.info(f"✅ DuckDB sampled: {len(train_data)} rows")
                except Exception as e:
                    logger.warning(f"DuckDB sampling failed, falling back to in-memory: {e}")
                    # Fallback to in-memory sampling
                    from .data_utils import prepare_training_data
                    train_data = self.base_train_data.copy()
                    original_size = len(train_data)
                    train_data = prepare_training_data(train_data, train_size)
                    logger.info(f"📊 Fallback sampling: {original_size} -> {len(train_data)} rows")
            else:
                # Standard in-memory sampling
                from .data_utils import prepare_training_data
                train_data = self.base_train_data.copy()
                original_size = len(train_data)
                train_data = prepare_training_data(train_data, train_size)
                logger.info(f"📊 In-memory sampling: {original_size} -> {len(train_data)} rows")
        else:
            # No sampling needed
            train_data = self.base_train_data.copy()
        
        # Merge features with base data using index-based approach
        # Reset indexes to ensure proper alignment
        train_data_indexed = train_data.reset_index(drop=True)
        features_indexed = features_df.reset_index(drop=True)
        
        # Ensure same number of rows for concatenation
        if len(features_indexed) != len(train_data_indexed):
            # Features were generated on full dataset, sample appropriately
            train_features = features_indexed.iloc[:len(train_data_indexed)].reset_index(drop=True)
        else:
            # Features match train data size, use directly
            train_features = features_indexed
        
        # Concatenate features with base data (index-based alignment)
        train_final = pd.concat([train_data_indexed, train_features], axis=1)
        
        # Remove duplicate columns (keep first occurrence)
        train_final = train_final.loc[:, ~train_final.columns.duplicated()]
        
        # Debug dataset analysis and dumping  
        self._debug_dataset_analysis(train_final, None)
        
        
        # Convert to TabularDataset
        train_dataset = TabularDataset(train_final)
        
        logger.debug(f"Prepared AutoGluon data: train={len(train_dataset)}")
        
        return train_dataset
    
    def _train_and_evaluate(self, 
                           train_data: TabularDataset,
                           eval_config: Dict[str, Any],
                           model_dir: str) -> float:
        """Train AutoGluon model and evaluate on validation set."""
        
        try:
            # Prepare ignored columns (ID column + user-defined ignore columns)
            ignored_columns = []
            if self.id_column:
                ignored_columns.append(self.id_column)
            if self.ignore_columns:
                ignored_columns.extend(self.ignore_columns)
            
            # PART 3: Log configuration for data leakage prevention
            logger.debug(f"AutoGluon configuration:")
            logger.debug(f"  Target column: {self.target_column}")
            logger.debug(f"  ID column: {self.id_column}")
            logger.debug(f"  Ignored columns: {self.ignore_columns}")
            logger.debug(f"  Total ignored for AutoGluon: {ignored_columns}")
            
            # Create predictor with ignored columns
            # Use the configured metric, or None to let AutoGluon choose
            eval_metric = self.target_metric if self.target_metric.lower() not in ['map@3', 'map_at_3', 'map3'] else None
            
            learner_kwargs = {}
            if ignored_columns:
                learner_kwargs['ignored_columns'] = ignored_columns
            
            predictor = TabularPredictor(
                label=self.target_column,
                path=model_dir,
                eval_metric=eval_metric,
                verbosity=eval_config.get('verbosity', 0),
                learner_kwargs=learner_kwargs
            )
            
            # Prepare fit parameters
            fit_params = {
                'time_limit': eval_config['time_limit'],
                'presets': eval_config['presets'],
                'holdout_frac': eval_config.get('holdout_frac', 0.2)
            }
            
            # Add bagging parameters only if explicitly set in config
            if 'num_bag_folds' in eval_config:
                fit_params['num_bag_folds'] = eval_config['num_bag_folds']
            if 'num_bag_sets' in eval_config:
                fit_params['num_bag_sets'] = eval_config['num_bag_sets']
            
            # Add model type constraints
            if 'included_model_types' in eval_config:
                fit_params['included_model_types'] = eval_config['included_model_types']
            elif 'excluded_model_types' in eval_config:
                fit_params['excluded_model_types'] = eval_config['excluded_model_types']
            
            # Add GPU configuration
            if eval_config.get('enable_gpu', False):
                # Use custom ag_args_fit from config if available
                if 'ag_args_fit' in eval_config:
                    fit_params['ag_args_fit'] = eval_config['ag_args_fit']
                else:
                    fit_params['ag_args_fit'] = {'num_gpus': 1}
            
            # Add ensemble configuration
            if 'ag_args_ensemble' in eval_config:
                fit_params['ag_args_ensemble'] = eval_config['ag_args_ensemble']
            
            # Train model
            logger.info(f"Training AutoGluon with fit_params: {fit_params}")
            predictor.fit(train_data, **fit_params)
            logger.info("AutoGluon training completed")
            
            # Use AutoGluon's built-in leaderboard to get validation score
            logger.info(f"Getting validation score for metric: {self.target_metric}")
            
            # Get leaderboard which contains validation scores
            leaderboard = predictor.leaderboard(silent=True)
            
            # Get the best model's validation score
            if len(leaderboard) > 0:
                # AutoGluon uses score_val column for validation scores
                score = leaderboard.iloc[0]['score_val']
                logger.info(f"Best model validation {self.target_metric} score: {score:.5f}")
                logger.debug(f"Best model: {leaderboard.iloc[0]['model']}")
            else:
                logger.error("No models in leaderboard!")
                score = 0.0
            
            return score
            
        except Exception as e:
            logger.error(f"AutoGluon training/evaluation failed: {e}")
            return 0.0
    
    
    def evaluate_final_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform final thorough evaluation of the best feature set.
        
        Args:
            features_df: Best discovered feature set
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        logger.info("Performing final thorough evaluation...")
        
        eval_start_time = time.time()
        final_config = self.autogluon_config['final_eval'].copy()
        
        try:
            # Prepare full training data (no validation split for final eval)
            train_ids = self.base_train_data['id'].values
            val_ids = self.validation_data['id'].values
            all_train_ids = np.concatenate([train_ids, val_ids])
            
            # Get feature columns
            feature_cols = [col for col in features_df.columns 
                           if col not in ['id', self.target_column]]
            
            # Merge full training data
            full_train_features = features_df[features_df['id'].isin(all_train_ids)].copy()
            full_train_data = pd.merge(
                pd.concat([self.base_train_data, self.validation_data])[['id', self.target_column]],
                full_train_features[['id'] + feature_cols],
                on='id',
                how='inner'
            ).drop('id', axis=1)
            
            # Create model directory
            final_model_dir = os.path.join(self.temp_base_dir, 'final_model')
            if os.path.exists(final_model_dir):
                shutil.rmtree(final_model_dir)
            
            # Train final model
            predictor = TabularPredictor(
                label=self.target_column,
                path=final_model_dir,
                eval_metric='log_loss',
                verbosity=final_config.get('verbosity', 2)
            )
            
            fit_params = {
                'time_limit': final_config['time_limit'],
                'presets': final_config['presets'],
                'holdout_frac': final_config.get('holdout_frac', 0.15),
                'num_bag_folds': final_config.get('num_bag_folds', 5),
                'num_bag_sets': final_config.get('num_bag_sets', 2),
            }
            
            predictor.fit(TabularDataset(full_train_data), **fit_params)
            
            # Get final evaluation metrics
            evaluation_results = predictor.evaluate(
                TabularDataset(full_train_data), 
                silent=False
            )
            
            # Generate test predictions if test data available
            test_predictions = None
            if self.base_test_data is not None:
                test_ids = self.base_test_data['id'].values
                test_features = features_df[features_df['id'].isin(test_ids)].copy()
                test_data_for_pred = test_features[feature_cols]
                test_predictions = predictor.predict_proba(test_data_for_pred)
            
            # Get feature importance
            try:
                feature_importance = predictor.feature_importance(full_train_data.drop(columns=[self.target_column]))
            except:
                feature_importance = None
                
            # Get model leaderboard
            try:
                leaderboard = predictor.leaderboard(silent=True)
            except:
                leaderboard = None
            
            eval_time = time.time() - eval_start_time
            
            results = {
                'evaluation_metrics': evaluation_results,
                'feature_importance': feature_importance,
                'leaderboard': leaderboard,
                'test_predictions': test_predictions,
                'evaluation_time': eval_time,
                'model_path': final_model_dir,
                'num_features': len(feature_cols),
                'training_samples': len(full_train_data)
            }
            
            logger.info(f"Final evaluation completed in {eval_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            return {'error': str(e)}
    
    def _update_evaluation_phase(self, iteration: int) -> None:
        """Update evaluation phase - removed multi-phase system."""
        pass  # Removed multi-phase evaluation per user request
    
    def _hash_features(self, features_df: pd.DataFrame) -> str:
        """Create hash of feature set for caching."""
        # Simple hash based on column names and data types
        # In practice, you might want a more sophisticated hash
        feature_cols = sorted([col for col in features_df.columns 
                              if col not in ['id', self.target_column]])
        return hash(tuple(feature_cols))
    
    def _check_recent_improvement(self, window: int = 10) -> bool:
        """Check if there has been recent improvement in scores."""
        # This is a simplified implementation
        # You'd want to track recent scores and check for improvement trends
        return len(self.evaluation_cache) > 0
    
    def _create_temp_model_dir(self) -> str:
        """Generate unique path for AutoGluon model directory (AutoGluon will create it)."""
        # Generate unique model directory name without creating it
        # AutoGluon expects to create the directory itself
        random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        model_dir_name = f'ag_model_{self.evaluation_count}_{random_chars}'
        model_dir = os.path.join(self.temp_base_dir, model_dir_name)
        self.model_dirs.append(model_dir)
        return model_dir
    
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation performance statistics."""
        avg_eval_time = self.total_eval_time / self.evaluation_count if self.evaluation_count > 0 else 0
        
        return {
            'total_evaluations': self.evaluation_count,
            'total_eval_time': self.total_eval_time,
            'avg_eval_time': avg_eval_time,
            'best_score': self.best_score,
            'cache_size': len(self.evaluation_cache),
            'cache_hit_rate': 0.0,  # Would need to track cache hits vs misses
            'duckdb_available': hasattr(self, 'duckdb_manager') and self.duckdb_manager is not None
        }
    
    def cleanup(self) -> None:
        """Cleanup all temporary resources."""
        logger.info("Cleaning up AutoGluon evaluator resources...")
        
        # Remove all temporary model directories
        for model_dir in self.model_dirs:
            try:
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup {model_dir}: {e}")
        
        # Clear cache
        self.evaluation_cache.clear()
        
        # Close DuckDB connection if available
        if hasattr(self, 'duckdb_manager') and self.duckdb_manager is not None:
            try:
                self.duckdb_manager.close()
                logger.info("Closed DuckDB connection")
            except Exception as e:
                logger.warning(f"Failed to close DuckDB connection: {e}")
        
        # Remove base temp directory if configured
        if self.resource_config.get('cleanup_temp_on_exit', True):
            try:
                if os.path.exists(self.temp_base_dir):
                    shutil.rmtree(self.temp_base_dir)
                    logger.info(f"Cleaned up temp directory: {self.temp_base_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp base dir: {e}")
    
    def _debug_dataset_analysis(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> None:
        """
        Debug analysis of dataset before AutoGluon evaluation.
        Dumps dataset to /tmp/ and analyzes feature quality.
        """
        # Generate iteration counter for file naming
        if not hasattr(self, '_debug_iteration_counter'):
            self._debug_iteration_counter = 0
        self._debug_iteration_counter += 1
        
        iteration = self._debug_iteration_counter
        
        try:
            # Get dataset name from config
            dataset_name = self.autogluon_config.get('dataset_name', 'unknown')
            
            # Dump datasets to /tmp for manual inspection
            train_path = f"/tmp/{dataset_name}-train-{iteration:04d}.csv"
            train_df.to_csv(train_path, index=False)
            
            # Only dump validation data if it exists
            if val_df is not None:
                val_path = f"/tmp/{dataset_name}-val-{iteration:04d}.csv" 
                val_df.to_csv(val_path, index=False)
                logger.info(f"🔍 Debug: Dumped datasets to {train_path} and {val_path}")
            else:
                logger.info(f"🔍 Debug: Dumped training dataset to {train_path} (no validation data)")
            
            
        except Exception as e:
            logger.warning(f"Debug dataset analysis failed: {e}")
    

    def _filter_useless_features_single(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out features that have no predictive signal before AutoGluon evaluation.
        Returns filtered training dataframe.
        """
        try:
            original_feature_count = len(train_df.columns)
            features_to_remove = set()
            
            # Identify features to remove
            for col in train_df.columns:
                if col == self.target_column:
                    continue
                    
                # Check for constant features (no variance)
                train_nunique = train_df[col].nunique()
                
                if train_nunique <= 1:
                    features_to_remove.add(col)
                    continue
                
                # Check for features with extremely low variance (numerical only)
                if pd.api.types.is_numeric_dtype(train_df[col]):
                    train_var = train_df[col].var()
                    
                    if train_var is not None and train_var < 1e-10:
                        features_to_remove.add(col)
                        continue
            
            # Remove useless features
            if features_to_remove:
                cols_to_keep = [col for col in train_df.columns if col not in features_to_remove]
                train_filtered = train_df[cols_to_keep].copy()
                
                logger.info(f"✅ Filtered useless features: {original_feature_count} -> {len(train_filtered.columns)} "
                           f"(removed {len(features_to_remove)} features)")
                
                return train_filtered
            else:
                logger.info(f"✅ No useless features detected, keeping all {original_feature_count} features")
                return train_df
                
        except Exception as e:
            logger.warning(f"Feature filtering failed: {e}, using all features")
            return train_df

    def _filter_useless_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter out features that have no predictive signal before AutoGluon evaluation.
        Returns filtered train and validation dataframes.
        """
        try:
            original_feature_count = len(train_df.columns)
            features_to_remove = set()
            
            # Identify features to remove
            for col in train_df.columns:
                if col == self.target_column:
                    continue
                    
                # Check for constant features (no variance)
                train_nunique = train_df[col].nunique()
                val_nunique = val_df[col].nunique()
                
                if train_nunique <= 1 or val_nunique <= 1:
                    features_to_remove.add(col)
                    continue
                
                # Check for features with extremely low variance (numerical only)
                if pd.api.types.is_numeric_dtype(train_df[col]):
                    train_var = train_df[col].var()
                    val_var = val_df[col].var()
                    
                    if (train_var is not None and train_var < 1e-10) or (val_var is not None and val_var < 1e-10):
                        features_to_remove.add(col)
                        continue
                
                # Check for features that are identical between train and validation
                # (might indicate data leakage or constant behavior)
                if train_nunique == 1 and val_nunique == 1:
                    train_unique = train_df[col].unique()[0]
                    val_unique = val_df[col].unique()[0]
                    if train_unique == val_unique:
                        features_to_remove.add(col)
            
            # Remove identified features
            if features_to_remove:
                logger.warning(f"🧹 Pre-filtering {len(features_to_remove)} useless features before AutoGluon:")
                for i, feat in enumerate(sorted(features_to_remove)):
                    if i < 10:  # Show first 10
                        logger.warning(f"  - {feat}")
                    elif i == 10:
                        logger.warning(f"  ... and {len(features_to_remove) - 10} more")
                
                # Remove features from both dataframes
                train_filtered = train_df.drop(columns=features_to_remove)
                val_filtered = val_df.drop(columns=features_to_remove)
                
                logger.info(f"📉 Features reduced from {original_feature_count} to {len(train_filtered.columns)} ({len(features_to_remove)} removed)")
                
                return train_filtered, val_filtered
            else:
                logger.info(f"✅ No useless features detected, keeping all {original_feature_count} features")
                return train_df, val_df
                
        except Exception as e:
            logger.error(f"Feature filtering failed: {e}")
            return train_df, val_df  # Return original dataframes on error

    def _generate_random_suffix(self) -> str:
        """Generate random suffix for unique temp directory per MCTS iteration."""
        timestamp = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
        random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"mcts_{timestamp}_{random_chars}"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()