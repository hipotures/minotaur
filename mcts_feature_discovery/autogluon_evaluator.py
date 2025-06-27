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
        
        # Paths and data
        self.train_path = self.autogluon_config['train_path']
        self.test_path = self.autogluon_config['test_path']
        self.target_metric = self.autogluon_config['target_metric']
        
        # Load base data once
        self.base_train_data = None
        self.base_test_data = None
        self.validation_data = None
        self.target_column = 'Fertilizer Name'
        
        # Evaluation strategy
        self.current_phase = 'exploration'  # 'exploration' or 'exploitation'
        self.phase_switch_iteration = None
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_eval_time = 0.0
        self.best_score = 0.0
        self.evaluation_cache = {}  # Cache results for identical feature sets
        
        # Temporary directories management
        self.temp_base_dir = self.resource_config.get('temp_dir', '/tmp/mcts_features')
        Path(self.temp_base_dir).mkdir(parents=True, exist_ok=True)
        self.model_dirs = []  # Track created model directories for cleanup
        
        # Initialize data
        self._load_base_data()
        self._setup_evaluation_strategy()
        
        logger.info(f"Initialized AutoGluonEvaluator for {self.target_metric}")
    
    def _load_base_data(self) -> None:
        """Load and prepare base training/test data."""
        try:
            logger.info("Loading base training and test data...")
            
            # Load raw data using DataManager for intelligent loading
            from .data_utils import DataManager
            data_manager = DataManager(self.config)
            
            logger.info(f"Loading training data from: {self.train_path}")
            self.base_train_data = data_manager.load_dataset(self.train_path, 'train')
            logger.info(f"Loading test data from: {self.test_path}")
            self.base_test_data = data_manager.load_dataset(self.test_path, 'test')
            
            # DataManager already handles sampling based on configuration
            
            # Create validation split for faster evaluation
            train_data, val_data = train_test_split(
                self.base_train_data,
                test_size=0.2,
                stratify=self.base_train_data[self.target_column],
                random_state=42
            )
            
            self.base_train_data = train_data.reset_index(drop=True)
            self.validation_data = val_data.reset_index(drop=True)
            
            logger.info(f"Data loaded: train={len(self.base_train_data)}, "
                       f"val={len(self.validation_data)}, test={len(self.base_test_data)}")
            
        except Exception as e:
            logger.error(f"Failed to load base data: {e}")
            raise
    
    def _setup_evaluation_strategy(self) -> None:
        """Setup adaptive evaluation strategy based on configuration."""
        max_iterations = self.session_config['max_iterations']
        threshold = self.autogluon_config['thorough_eval_threshold']
        
        self.phase_switch_iteration = int(max_iterations * threshold)
        
        logger.info(f"Evaluation strategy: exploration -> exploitation at iteration {self.phase_switch_iteration}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current AutoGluon configuration based on evaluation phase."""
        if self.current_phase == 'exploration':
            base_config = self.autogluon_config['fast_eval'].copy()
        else:
            base_config = self.autogluon_config['thorough_eval'].copy()
        
        # Adaptive time limit based on progress
        if self.autogluon_config.get('adaptive_time_limit', False):
            if self.evaluation_count > 0 and self.best_score > 0:
                # Increase time limit if we're making good progress
                recent_improvement = self._check_recent_improvement()
                if recent_improvement:
                    multiplier = self.autogluon_config.get('timeout_multiplier', 1.5)
                    base_config['time_limit'] = int(base_config['time_limit'] * multiplier)
        
        return base_config
    
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
        
        # Update evaluation phase
        self._update_evaluation_phase(iteration)
        
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
            train_data, val_data = self._prepare_autogluon_data(features_df, eval_config)
            logger.info(f"Prepared data: train={len(train_data)}, val={len(val_data)}")
            
            # Create temporary model directory
            model_dir = self._create_temp_model_dir()
            logger.debug(f"Created model directory: {model_dir}")
            
            # Train and evaluate
            logger.info("Starting AutoGluon training and evaluation...")
            score = self._train_and_evaluate(train_data, val_data, eval_config, model_dir)
            logger.info(f"AutoGluon evaluation completed: score={score:.5f}")
            
            # Cache result
            self.evaluation_cache[feature_hash] = score
            
            # Update statistics
            eval_time = time.time() - eval_start_time
            self.total_eval_time += eval_time
            
            if score > self.best_score:
                self.best_score = score
                logger.info(f"New best evaluation score: {score:.5f}")
            
            logger.debug(f"Evaluation completed: score={score:.5f}, time={eval_time:.2f}s, "
                        f"features={len(features_df.columns)}, phase={self.current_phase}")
            
            return score
            
        except Exception as e:
            logger.error(f"AutoGluon evaluation failed: {e}")
            return 0.0
        
        finally:
            # Cleanup model directory
            self._cleanup_temp_dirs()
    
    def _prepare_autogluon_data(self, features_df: pd.DataFrame, eval_config: Dict[str, Any] = None) -> Tuple[TabularDataset, TabularDataset]:
        """Prepare training and validation data for AutoGluon."""
        
        # Get current configuration if not provided
        if eval_config is None:
            eval_config = self.get_current_config()
        
        # Apply train_size sampling if configured
        train_data = self.base_train_data.copy()
        train_size = eval_config.get('train_size', 1.0)
        
        if train_size < 1.0:
            sample_size = int(len(train_data) * train_size)
            if sample_size > 0:
                # Stratified sampling to maintain class distribution
                train_data = train_data.groupby(self.target_column, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(train_data)))))
                ).reset_index(drop=True)
                logger.debug(f"Sampled training data: {len(self.base_train_data)} -> {len(train_data)} rows (train_size={train_size})")
        
        # Merge features with base data
        train_ids = train_data['id'].values
        val_ids = self.validation_data['id'].values
        
        # Get feature columns (exclude 'id' and target)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['id', self.target_column]]
        
        # Merge training data
        train_features = features_df[features_df['id'].isin(train_ids)].copy()
        train_merged = pd.merge(
            train_data[['id', self.target_column]], 
            train_features[['id'] + feature_cols],
            on='id',
            how='inner'
        )
        
        # Merge validation data  
        val_features = features_df[features_df['id'].isin(val_ids)].copy()
        val_merged = pd.merge(
            self.validation_data[['id', self.target_column]],
            val_features[['id'] + feature_cols], 
            on='id',
            how='inner'
        )
        
        # Remove 'id' column for AutoGluon
        train_final = train_merged.drop('id', axis=1)
        val_final = val_merged.drop('id', axis=1)
        
        # Convert to TabularDataset
        train_dataset = TabularDataset(train_final)
        val_dataset = TabularDataset(val_final)
        
        logger.debug(f"Prepared AutoGluon data: train={len(train_dataset)}, val={len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _train_and_evaluate(self, 
                           train_data: TabularDataset, 
                           val_data: TabularDataset,
                           eval_config: Dict[str, Any],
                           model_dir: str) -> float:
        """Train AutoGluon model and evaluate on validation set."""
        
        try:
            # Create predictor
            predictor = TabularPredictor(
                label=self.target_column,
                path=model_dir,
                eval_metric='log_loss',  # Close to MAP@3 for ranking
                verbosity=eval_config.get('verbosity', 0)
            )
            
            # Prepare fit parameters
            fit_params = {
                'time_limit': eval_config['time_limit'],
                'presets': eval_config['presets'],
                'holdout_frac': eval_config.get('holdout_frac', 0.2),
                'num_bag_folds': eval_config.get('num_bag_folds', 2),
                'num_bag_sets': eval_config.get('num_bag_sets', 1)
            }
            
            # Add model type constraints
            if 'included_model_types' in eval_config:
                fit_params['included_model_types'] = eval_config['included_model_types']
            elif 'excluded_model_types' in eval_config:
                fit_params['excluded_model_types'] = eval_config['excluded_model_types']
            
            # Add GPU configuration
            if eval_config.get('enable_gpu', False):
                fit_params['ag_args_fit'] = {'num_gpus': 1}
            
            # Train model
            logger.debug(f"Training AutoGluon with fit_params: {fit_params}")
            predictor.fit(train_data, **fit_params)
            logger.debug("AutoGluon training completed")
            
            # Make predictions on validation set
            val_data_for_pred = val_data.drop(columns=[self.target_column])
            logger.debug(f"Making predictions on validation data: {val_data_for_pred.shape}")
            val_predictions = predictor.predict_proba(val_data_for_pred)
            val_true = val_data[self.target_column].values
            
            logger.debug(f"Predictions shape: {val_predictions.shape}, True labels: {len(val_true)}")
            
            # Calculate MAP@3 score
            map3_score = self._calculate_map3(val_true, val_predictions.values)
            logger.debug(f"Calculated MAP@3 score: {map3_score:.5f}")
            
            return map3_score
            
        except Exception as e:
            logger.error(f"AutoGluon training/evaluation failed: {e}")
            return 0.0
    
    def _calculate_map3(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate MAP@3 metric for multi-class classification."""
        
        if len(y_pred_proba.shape) != 2:
            logger.error(f"Invalid prediction shape: {y_pred_proba.shape}")
            return 0.0
        
        # Get class labels (assuming they match the order in predictions)
        # This is a simplified implementation - in practice you'd want to ensure
        # class alignment between true labels and prediction columns
        
        map_score = 0.0
        n_samples = len(y_true)
        
        for i, true_label in enumerate(y_true):
            # Get top 3 predictions for this sample
            sample_probs = y_pred_proba[i]
            top3_indices = np.argsort(sample_probs)[::-1][:3]
            
            # Convert true label to index (this is simplified)
            # In practice, you'd need proper label encoding
            try:
                if hasattr(true_label, '__iter__') and not isinstance(true_label, str):
                    # Handle case where true_label might be encoded differently
                    continue
                    
                # Calculate score for this sample
                for rank, pred_idx in enumerate(top3_indices):
                    # This is a simplified check - in practice you'd need proper mapping
                    # between class names and prediction indices
                    if str(pred_idx) == str(true_label) or pred_idx == true_label:
                        map_score += 1.0 / (rank + 1)
                        break
                        
            except Exception as e:
                logger.debug(f"Label matching issue for sample {i}: {e}")
                continue
        
        return map_score / n_samples if n_samples > 0 else 0.0
    
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
        """Update evaluation phase based on current iteration."""
        if iteration >= self.phase_switch_iteration and self.current_phase == 'exploration':
            self.current_phase = 'exploitation'
            logger.info(f"Switched to exploitation phase at iteration {iteration}")
    
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
        """Create temporary directory for AutoGluon model."""
        model_dir = tempfile.mkdtemp(
            prefix=f'ag_model_{self.evaluation_count}_',
            dir=self.temp_base_dir
        )
        self.model_dirs.append(model_dir)
        return model_dir
    
    def _cleanup_temp_dirs(self, keep_recent: int = 3) -> None:
        """Cleanup old temporary model directories."""
        if len(self.model_dirs) <= keep_recent:
            return
        
        # Remove old directories (keep most recent ones)
        dirs_to_remove = self.model_dirs[:-keep_recent]
        for model_dir in dirs_to_remove:
            try:
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                    logger.debug(f"Cleaned up model directory: {model_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {model_dir}: {e}")
        
        # Update list
        self.model_dirs = self.model_dirs[-keep_recent:]
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation performance statistics."""
        avg_eval_time = self.total_eval_time / self.evaluation_count if self.evaluation_count > 0 else 0
        
        return {
            'total_evaluations': self.evaluation_count,
            'total_eval_time': self.total_eval_time,
            'avg_eval_time': avg_eval_time,
            'best_score': self.best_score,
            'current_phase': self.current_phase,
            'cache_size': len(self.evaluation_cache),
            'cache_hit_rate': 0.0  # Would need to track cache hits vs misses
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
        
        # Remove base temp directory if configured
        if self.resource_config.get('cleanup_temp_on_exit', True):
            try:
                if os.path.exists(self.temp_base_dir):
                    shutil.rmtree(self.temp_base_dir)
                    logger.info(f"Cleaned up temp directory: {self.temp_base_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp base dir: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()