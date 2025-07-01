"""
Concrete Implementation of Universal Feature Generator

Integrates with existing generic and custom feature modules.
"""

import importlib
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .base_generator import UniversalFeatureGenerator, FeatureType
from .base import FeatureTimingMixin

logger = logging.getLogger(__name__)


class MinotaurFeatureGenerator(UniversalFeatureGenerator):
    """
    Concrete implementation that integrates with existing feature modules.
    
    This class bridges the new universal generator interface with the
    existing modular feature architecture.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 check_signal: bool = True,
                 lowercase_columns: bool = False):
        """
        Initialize generator with configuration.
        
        Args:
            config: Feature space configuration dict
            check_signal: Whether to check for signal during generation
            lowercase_columns: Whether to lowercase feature names
        """
        super().__init__(
            check_signal=check_signal,
            lowercase_columns=lowercase_columns,
            min_signal_ratio=config.get('min_signal_ratio', 0.01),
            signal_sample_size=config.get('signal_sample_size', 1000)
        )
        
        self.config = config
        self.feature_config = config.get('feature_space', {})
        self.autogluon_config = config.get('autogluon', {})
        
        # Feature operation configurations
        self.enabled_categories = set(self.feature_config.get('enabled_categories', []))
        self.generic_operations_config = self.feature_config.get('generic_operations', {})
        self.generic_params = self.feature_config.get('generic_params', {})
        
        # Dataset configuration
        self.target_column = self.autogluon_config.get('target_column', 'target')
        self.id_column = self.autogluon_config.get('id_column', 'id')
        self.ignore_columns = self.autogluon_config.get('ignore_columns', []) or []
        
        # Module caching
        self._loaded_modules = {}
    
    def _generate_original_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate original features (just return input columns)."""
        features = {}
        
        # Get columns to include
        forbidden = kwargs.get('forbidden_columns', [])
        if not forbidden:
            forbidden = [self.target_column] + self.ignore_columns
        
        for col in df.columns:
            if col not in forbidden:
                features[col] = df[col]
        
        return features
    
    def _generate_custom_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate custom domain-specific features using existing modules."""
        features = {}
        dataset_name = kwargs.get('dataset_name', 'unknown')
        
        try:
            # Auto-detect custom domain module based on dataset name
            dataset_name_clean = dataset_name.lower().replace('-', '_').replace(' ', '_')
            
            # Check if custom domain module exists for this dataset
            from pathlib import Path
            custom_modules_dir = Path(__file__).parent / 'custom'
            module_file = f"{dataset_name_clean}.py"
            
            module_name = None
            if (custom_modules_dir / module_file).exists():
                module_name = module_file[:-3]  # Remove .py extension
            
            if not module_name:
                self._log_debug(f"No custom domain module found for dataset '{dataset_name}', skipping custom operations")
                return features
            
            # Try to load custom module
            if module_name not in self._loaded_modules:
                try:
                    custom_module = importlib.import_module(f'.custom.{module_name}', package='src.features')
                    self._loaded_modules[module_name] = custom_module
                except ImportError:
                    logger.info(f"No custom module found for dataset '{dataset_name}'")
                    return features
            
            custom_module = self._loaded_modules.get(module_name)
            if not custom_module:
                return features
            
            # Use the custom feature operations
            if hasattr(custom_module, 'CustomFeatureOperations'):
                custom_instance = custom_module.CustomFeatureOperations()
                
                # Configure the instance
                if hasattr(custom_instance, 'configure'):
                    custom_instance.configure(
                        check_signal=self.check_signal,
                        signal_sample_size=self.signal_sample_size,
                        lowercase_features=self.lowercase_columns
                    )
                elif hasattr(custom_instance, '_check_signal'):
                    custom_instance._check_signal = self.check_signal
                    if hasattr(custom_instance, '_lowercase_features'):
                        custom_instance._lowercase_features = self.lowercase_columns
                
                # Generate all custom features
                custom_features = custom_instance.generate_all_features(df)
                
                if isinstance(custom_features, dict):
                    features.update(custom_features)
                else:
                    logger.warning("Custom features returned non-dict result")
            
        except Exception as e:
            logger.error(f"Error generating custom features for {dataset_name}: {e}")
        
        return features
    
    def _generate_generic_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate generic features using existing modules."""
        features = {}
        forbidden_columns = kwargs.get('forbidden_columns', [])
        
        try:
            # Import generic modules
            from . import generic
            
            # Helper to filter forbidden columns
            def filter_columns(columns: List[str]) -> List[str]:
                return [col for col in columns 
                       if col not in forbidden_columns 
                       and col != self.target_column 
                       and col not in self.ignore_columns]
            
            # Statistical aggregations
            if self.generic_operations_config.get('statistical_aggregations', True):
                try:
                    from .generic import statistical
                    stat_op = statistical.StatisticalFeatures()
                    
                    # Configure
                    if hasattr(stat_op, 'configure'):
                        stat_op.configure(
                            check_signal=self.check_signal,
                            signal_sample_size=self.signal_sample_size,
                            lowercase_features=self.lowercase_columns
                        )
                    else:
                        stat_op._check_signal = self.check_signal
                        if hasattr(stat_op, '_lowercase_features'):
                            stat_op._lowercase_features = self.lowercase_columns
                    
                    # Get columns for aggregation
                    groupby_cols = self.generic_params.get('groupby_columns', [])
                    aggregate_cols = self.generic_params.get('aggregate_columns', [])
                    
                    # Auto-detect if not specified
                    if not groupby_cols:
                        groupby_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:3]
                    if not aggregate_cols:
                        aggregate_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
                    
                    # Filter forbidden columns
                    groupby_cols = filter_columns(groupby_cols)
                    aggregate_cols = filter_columns(aggregate_cols)
                    
                    if groupby_cols and aggregate_cols:
                        stat_features = stat_op.generate_features(
                            df, 
                            groupby_cols=groupby_cols, 
                            agg_cols=aggregate_cols
                        )
                        features.update(stat_features)
                
                except Exception as e:
                    logger.error(f"Error generating statistical features: {e}")
            
            # Polynomial features
            if self.generic_operations_config.get('polynomial_features', True):
                try:
                    from .generic import polynomial
                    poly_op = polynomial.PolynomialFeatures()
                    
                    # Configure
                    if hasattr(poly_op, 'configure'):
                        poly_op.configure(
                            check_signal=self.check_signal,
                            signal_sample_size=self.signal_sample_size,
                            lowercase_features=self.lowercase_columns
                        )
                    else:
                        poly_op._check_signal = self.check_signal
                        if hasattr(poly_op, '_lowercase_features'):
                            poly_op._lowercase_features = self.lowercase_columns
                    
                    degree = self.generic_params.get('polynomial_degree', 2)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
                    numeric_cols = filter_columns(numeric_cols)
                    
                    if numeric_cols:
                        poly_features = poly_op.generate_features(
                            df, 
                            numeric_cols=numeric_cols, 
                            degree=degree
                        )
                        features.update(poly_features)
                
                except Exception as e:
                    logger.error(f"Error generating polynomial features: {e}")
            
            # Binning features
            if self.generic_operations_config.get('binning_features', True):
                try:
                    from .generic import binning
                    bin_op = binning.BinningFeatures()
                    
                    # Configure
                    if hasattr(bin_op, 'configure'):
                        bin_op.configure(
                            check_signal=self.check_signal,
                            signal_sample_size=self.signal_sample_size,
                            lowercase_features=self.lowercase_columns
                        )
                    else:
                        bin_op._check_signal = self.check_signal
                        if hasattr(bin_op, '_lowercase_features'):
                            bin_op._lowercase_features = self.lowercase_columns
                    
                    n_bins = self.generic_params.get('binning_bins', 5)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = filter_columns(numeric_cols)
                    
                    if numeric_cols:
                        bin_features = bin_op.generate_features(
                            df, 
                            numeric_cols=numeric_cols, 
                            n_bins=n_bins
                        )
                        features.update(bin_features)
                
                except Exception as e:
                    logger.error(f"Error generating binning features: {e}")
            
            # Ranking features
            if self.generic_operations_config.get('ranking_features', True):
                try:
                    from .generic import ranking
                    rank_op = ranking.RankingFeatures()
                    
                    # Configure
                    if hasattr(rank_op, 'configure'):
                        rank_op.configure(
                            check_signal=self.check_signal,
                            signal_sample_size=self.signal_sample_size,
                            lowercase_features=self.lowercase_columns
                        )
                    else:
                        rank_op._check_signal = self.check_signal
                        if hasattr(rank_op, '_lowercase_features'):
                            rank_op._lowercase_features = self.lowercase_columns
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = filter_columns(numeric_cols)
                    
                    if numeric_cols:
                        rank_features = rank_op.generate_features(df, numeric_cols=numeric_cols)
                        features.update(rank_features)
                
                except Exception as e:
                    logger.error(f"Error generating ranking features: {e}")
            
            # Categorical features
            if self.generic_operations_config.get('categorical_features', True):
                try:
                    from .generic import categorical
                    cat_op = categorical.CategoricalFeatures()
                    
                    # Configure
                    if hasattr(cat_op, 'configure'):
                        cat_op.configure(
                            check_signal=self.check_signal,
                            signal_sample_size=self.signal_sample_size,
                            lowercase_features=self.lowercase_columns
                        )
                    else:
                        cat_op._check_signal = self.check_signal
                        if hasattr(cat_op, '_lowercase_features'):
                            cat_op._lowercase_features = self.lowercase_columns
                    
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    cat_cols = filter_columns(cat_cols)
                    
                    # Prepare forbidden columns for categorical operation
                    current_forbidden = forbidden_columns + [self.target_column, self.id_column] + self.ignore_columns
                    current_forbidden = [col for col in current_forbidden if col is not None]  # Remove None values
                    
                    if cat_cols:
                        cat_features = cat_op.generate_features(df, categorical_cols=cat_cols, forbidden_columns=current_forbidden)
                        features.update(cat_features)
                    else:
                        # Even if no object/category columns, still run auto-detection with proper forbidden list
                        cat_features = cat_op.generate_features(df, forbidden_columns=current_forbidden)
                        features.update(cat_features)
                
                except Exception as e:
                    logger.error(f"Error generating categorical features: {e}")
            
            # Text features
            if self.generic_operations_config.get('text_features', True):
                try:
                    from .generic import text
                    text_op = text.TextFeatures()
                    
                    # Configure
                    if hasattr(text_op, 'configure'):
                        text_op.configure(
                            check_signal=self.check_signal,
                            signal_sample_size=self.signal_sample_size,
                            lowercase_features=self.lowercase_columns
                        )
                    else:
                        text_op._check_signal = self.check_signal
                        if hasattr(text_op, '_lowercase_features'):
                            text_op._lowercase_features = self.lowercase_columns
                    
                    # Find text columns (string columns with average length > 10)
                    text_cols = []
                    for col in df.select_dtypes(include=['object']).columns:
                        if col not in forbidden_columns and col != self.target_column:
                            avg_len = df[col].astype(str).str.len().mean()
                            if avg_len > 10:
                                text_cols.append(col)
                    
                    if text_cols:
                        text_features = text_op.generate_features(df, text_cols=text_cols[:5])
                        features.update(text_features)
                
                except Exception as e:
                    logger.error(f"Error generating text features: {e}")
        
        except Exception as e:
            logger.error(f"Error in generic feature generation: {e}")
        
        return features
    
    def _generate_derived_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        Generate features derived from other features.
        
        This could include:
        - Features from combinations of custom + generic
        - Second-order features
        - Domain-specific derivations
        """
        features = {}
        
        # For now, this is a placeholder for future extensions
        # Could implement things like:
        # - PCA on generated features
        # - Clustering-based features
        # - Feature interactions discovered by MCTS
        
        return features