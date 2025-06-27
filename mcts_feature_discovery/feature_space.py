"""
Feature Space Manager for MCTS Feature Discovery

Defines feature operations, lazy generation, and integration with existing
feature_engineering.py module. Manages the search space for MCTS exploration.
"""

import os
import time
import hashlib
import pickle
import logging
from typing import Dict, List, Set, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path

# Import existing feature engineering module
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.feature_engineering import load_or_create_features
from .synthetic_data import generate_synthetic_features, augment_synthetic_features
from .timing import timed, timing_context, record_timing
from .data_utils import DataManager, smart_sample, estimate_memory_usage

logger = logging.getLogger(__name__)

@dataclass
class FeatureOperation:
    """Represents a single feature engineering operation."""
    
    name: str
    category: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    computational_cost: float = 1.0
    
    # Operation metadata
    output_features: List[str] = field(default_factory=list)
    is_deterministic: bool = True
    max_execution_time: float = 60.0  # seconds
    
    # Performance tracking
    success_rate: float = 1.0
    avg_improvement: float = 0.0
    total_applications: int = 0
    
    def can_apply(self, available_features: Set[str]) -> bool:
        """Check if operation can be applied given available features."""
        return all(dep in available_features for dep in self.dependencies)
    
    def __hash__(self):
        return hash(self.name)
    
    def __str__(self):
        return f"FeatureOperation({self.name}, category={self.category})"

class FeatureSpace:
    """
    Manager for feature space exploration in MCTS.
    
    Handles lazy feature generation, operation management, and integration
    with existing feature engineering infrastructure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature space with configuration."""
        self.config = config
        self.feature_config = config['feature_space']
        self.autogluon_config = config['autogluon']
        
        # Feature operation definitions
        self.operations: Dict[str, FeatureOperation] = {}
        self.enabled_categories = set(self.feature_config['enabled_categories'])
        self.category_weights = self.feature_config['category_weights']
        
        # Lazy loading configuration
        self.lazy_loading = self.feature_config['lazy_loading']
        self.cache_features = self.feature_config['cache_features']
        self.max_cache_size_mb = self.feature_config['max_cache_size_mb']
        
        # Performance limits
        self.max_features_per_node = self.feature_config['max_features_per_node']
        self.min_improvement_threshold = self.feature_config['min_improvement_threshold']
        self.feature_timeout = self.feature_config['feature_timeout']
        
        # Caching system
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.cache_metadata: Dict[str, Dict] = {}
        self.cache_size_mb = 0.0
        
        # Base data paths
        self.train_path = self.autogluon_config['train_path']
        self.test_path = self.autogluon_config['test_path']
        
        # Initialize data manager for optimized loading
        self.data_manager = DataManager(config)
        
        # Initialize operations
        self._initialize_operations()
        
        logger.info(f"Initialized FeatureSpace with {len(self.operations)} operations")
    
    def _initialize_operations(self) -> None:
        """Initialize all available feature operations."""
        
        # Basic NPK Operations
        if 'npk_interactions' in self.enabled_categories:
            self._add_npk_operations()
        
        # Environmental Operations
        if 'environmental_stress' in self.enabled_categories:
            self._add_environmental_operations()
        
        # Agricultural Domain Operations
        if 'agricultural_domain' in self.enabled_categories:
            self._add_agricultural_operations()
        
        # Statistical Operations
        if 'statistical_aggregations' in self.enabled_categories:
            self._add_statistical_operations()
        
        # Feature Transformation Operations
        if 'feature_transformations' in self.enabled_categories:
            self._add_transformation_operations()
        
        # Feature Selection Operations
        if 'feature_selection' in self.enabled_categories:
            self._add_selection_operations()
        
        logger.info(f"Initialized {len(self.operations)} feature operations across "
                   f"{len(self.enabled_categories)} categories")
    
    def _add_npk_operations(self) -> None:
        """Add NPK interaction feature operations."""
        
        base_deps = ['Nitrogen', 'Phosphorous', 'Potassium']
        
        operations = [
            FeatureOperation(
                name='npk_basic_ratios',
                category='npk_interactions',
                description='Basic NPK ratios (NP, NK, PK)',
                dependencies=base_deps,
                computational_cost=0.1,
                output_features=['NP_ratio', 'NK_ratio', 'PK_ratio']
            ),
            FeatureOperation(
                name='npk_advanced_interactions',
                category='npk_interactions', 
                description='Advanced NPK interactions (harmony, distance, balance)',
                dependencies=base_deps,
                computational_cost=0.3,
                output_features=['npk_harmony', 'npk_distance', 'npk_sum', 'npk_product']
            ),
            FeatureOperation(
                name='npk_dominance_patterns',
                category='npk_interactions',
                description='NPK dominance and deficiency patterns',
                dependencies=base_deps + ['low_Nitrogen', 'low_Phosphorous', 'low_Potassium'],
                computational_cost=0.2,
                output_features=['n_dominant', 'p_dominant', 'k_dominant', 'npk_deficiency_pattern']
            ),
            FeatureOperation(
                name='npk_statistical_features',
                category='npk_interactions',
                description='NPK variance and coefficient of variation',
                dependencies=base_deps,
                computational_cost=0.2,
                output_features=['npk_variance', 'npk_cv', 'np_pk_ratio']
            )
        ]
        
        for op in operations:
            self.operations[op.name] = op
    
    def _add_environmental_operations(self) -> None:
        """Add environmental stress feature operations."""
        
        env_deps = ['Temperature', 'Humidity', 'Moisture']
        
        operations = [
            FeatureOperation(
                name='stress_indicators',
                category='environmental_stress',
                description='Heat and drought stress indicators',
                dependencies=env_deps,
                computational_cost=0.2,
                output_features=['heat_stress', 'drought_stress', 'water_stress']
            ),
            FeatureOperation(
                name='optimal_conditions',
                category='environmental_stress',
                description='Optimal growing condition indicators',
                dependencies=env_deps,
                computational_cost=0.2,
                output_features=['optimal_temp', 'optimal_humidity', 'optimal_moisture', 'optimal_conditions']
            ),
            FeatureOperation(
                name='environmental_interactions',
                category='environmental_stress',
                description='Temperature-humidity-moisture interactions',
                dependencies=env_deps,
                computational_cost=0.3,
                output_features=['temp_humidity_interaction', 'temp_moisture_interaction', 'humidity_moisture_interaction']
            ),
            FeatureOperation(
                name='climate_zones',
                category='environmental_stress',
                description='Climate zone indicators',
                dependencies=env_deps,
                computational_cost=0.2,
                output_features=['hot_humid', 'moisture_stress']
            )
        ]
        
        for op in operations:
            self.operations[op.name] = op
    
    def _add_agricultural_operations(self) -> None:
        """Add agricultural domain-specific operations."""
        
        operations = [
            FeatureOperation(
                name='crop_nutrient_deficits',
                category='agricultural_domain',
                description='Crop-specific nutrient deficit calculations',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium', 'Crop Type'],
                computational_cost=0.5,
                output_features=['n_deficit_crop', 'p_deficit_crop', 'k_deficit_crop', 'total_deficit']
            ),
            FeatureOperation(
                name='nutrient_adequacy_ratios',
                category='agricultural_domain',
                description='Nutrient adequacy ratios for crops',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium', 'Crop Type'],
                computational_cost=0.4,
                output_features=['n_adequacy', 'p_adequacy', 'k_adequacy']
            ),
            FeatureOperation(
                name='soil_adjustments',
                category='agricultural_domain',
                description='Soil-specific nutrient adjustments',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium', 'Soil Type'],
                computational_cost=0.4,
                output_features=['soil_nutrient_factor', 'adjusted_n', 'adjusted_p', 'adjusted_k']
            ),
            FeatureOperation(
                name='crop_soil_compatibility',
                category='agricultural_domain',
                description='Crop-soil compatibility indicators',
                dependencies=['Soil Type', 'Crop Type', 'soil_crop'],
                computational_cost=0.3,
                output_features=['crop_soil_compatibility', 'season_indicator']
            ),
            FeatureOperation(
                name='fertilizer_urgency',
                category='agricultural_domain',
                description='Fertilizer need urgency indicators',
                dependencies=['n_deficit_crop', 'p_deficit_crop', 'k_deficit_crop'],
                computational_cost=0.2,
                output_features=['needs_nitrogen', 'needs_phosphorus', 'needs_potassium', 'fertilizer_urgency']
            )
        ]
        
        for op in operations:
            self.operations[op.name] = op
    
    def _add_statistical_operations(self) -> None:
        """Add statistical aggregation operations."""
        
        operations = [
            FeatureOperation(
                name='soil_groupby_stats',
                category='statistical_aggregations',
                description='Groupby statistics for soil types',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture', 'Soil Type'],
                computational_cost=0.8,
                output_features=[f'{col}_{stat}' for col in ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture'] 
                               for stat in ['soil_mean', 'soil_std', 'soil_deviation', 'soil_zscore']]
            ),
            FeatureOperation(
                name='crop_groupby_stats',
                category='statistical_aggregations',
                description='Groupby statistics for crop types',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium', 'Crop Type'],
                computational_cost=0.6,
                output_features=[f'{col}_{stat}' for col in ['Nitrogen', 'Phosphorous', 'Potassium']
                               for stat in ['crop_mean', 'crop_std', 'crop_deviation', 'crop_zscore']]
            ),
            FeatureOperation(
                name='soilcrop_groupby_stats',
                category='statistical_aggregations',
                description='Groupby statistics for soil-crop combinations',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium', 'soil_crop'],
                computational_cost=0.7,
                output_features=[f'{col}_soilcrop_{stat}' for col in ['Nitrogen', 'Phosphorous', 'Potassium']
                               for stat in ['mean', 'deviation']]
            ),
            FeatureOperation(
                name='nutrient_rankings',
                category='statistical_aggregations',
                description='Nutrient rankings within groups',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium', 'soil_crop'],
                computational_cost=0.5,
                output_features=['nitrogen_rank', 'phosphorous_rank', 'potassium_rank']
            )
        ]
        
        for op in operations:
            self.operations[op.name] = op
    
    def _add_transformation_operations(self) -> None:
        """Add feature transformation operations."""
        
        operations = [
            FeatureOperation(
                name='numerical_binning',
                category='feature_transformations',
                description='Binning of numerical features',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture'],
                computational_cost=0.3,
                output_features=[f'{col}_Binned' for col in ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture']]
            ),
            FeatureOperation(
                name='polynomial_features',
                category='feature_transformations',
                description='Polynomial features (degree 2) for key nutrients',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium'],
                computational_cost=0.4,
                output_features=[f'{col}_squared' for col in ['Nitrogen', 'Phosphorous', 'Potassium']] + ['NP_product', 'NK_product', 'PK_product']
            ),
            FeatureOperation(
                name='log_transforms',
                category='feature_transformations',
                description='Log transformations for skewed features',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium'],
                computational_cost=0.2,
                output_features=[f'log_{col}' for col in ['Nitrogen', 'Phosphorous', 'Potassium']]
            ),
            FeatureOperation(
                name='interaction_terms',
                category='feature_transformations',
                description='Interaction terms between categorical and numerical features',
                dependencies=['Nitrogen', 'Phosphorous', 'Potassium', 'Soil Type', 'Crop Type'],
                computational_cost=0.6,
                output_features=['soil_nitrogen_interaction', 'crop_phosphorous_interaction', 'soilcrop_npk_interaction']
            )
        ]
        
        for op in operations:
            self.operations[op.name] = op
    
    def _add_selection_operations(self) -> None:
        """Add feature selection operations."""
        
        operations = [
            FeatureOperation(
                name='correlation_filter',
                category='feature_selection',
                description='Remove highly correlated features',
                dependencies=[],  # Can be applied to any feature set
                computational_cost=0.5,
                output_features=[]  # Removes features rather than adding
            ),
            FeatureOperation(
                name='low_variance_filter',
                category='feature_selection',
                description='Remove low variance features',
                dependencies=[],
                computational_cost=0.3,
                output_features=[]
            ),
            FeatureOperation(
                name='univariate_selection',
                category='feature_selection',
                description='Select features based on univariate tests',
                dependencies=[],
                computational_cost=0.7,
                output_features=[]
            )
        ]
        
        for op in operations:
            self.operations[op.name] = op
    
    def get_available_operations(self, node) -> List[str]:
        """
        Get list of operations that can be applied to the given node.
        
        Args:
            node: MCTS node with current feature state
            
        Returns:
            List[str]: Available operation names
        """
        current_features = self._get_node_features(node)
        available_ops = []
        
        for op_name, operation in self.operations.items():
            # Check if operation can be applied
            if operation.can_apply(current_features):
                # Check if already applied in this path
                if op_name not in node.applied_operations:
                    # Apply category weighting
                    weight = self.category_weights.get(operation.category, 1.0)
                    if weight > 0:
                        available_ops.append(op_name)
        
        # Sort by category weight and computational cost
        available_ops.sort(
            key=lambda op_name: (
                -self.category_weights.get(self.operations[op_name].category, 1.0),
                self.operations[op_name].computational_cost
            )
        )
        
        logger.debug(f"Found {len(available_ops)} available operations for node at depth {node.depth}")
        return available_ops
    
    @timed("feature_space.generate_features", include_memory=True)
    def generate_features_for_node(self, node) -> pd.DataFrame:
        """
        Generate features for a specific MCTS node using lazy loading.
        
        Args:
            node: MCTS node
            
        Returns:
            pd.DataFrame: Generated features
        """
        generation_start = time.time()
        
        # Check if we're in mock testing mode
        testing_config = self.config.get('testing', {})
        use_mock = testing_config.get('use_mock_evaluator', False)
        
        if use_mock:
            # Use fast synthetic data generation for testing
            return self._generate_synthetic_features_for_node(node, generation_start)
        
        # Check cache first
        if self.cache_features:
            cache_key = self._get_node_cache_key(node)
            if cache_key in self.feature_cache:
                logger.debug(f"Using cached features for node")
                return self.feature_cache[cache_key].copy()
        
        try:
            # Determine feature set type based on applied operations
            feature_set = self._determine_feature_set(node.applied_operations)
            
            # Generate features using existing infrastructure
            if len(node.applied_operations) == 0:
                # Base features only
                train_features = load_or_create_features(self.train_path, 'train', feature_set, self.data_manager)
                test_features = load_or_create_features(self.test_path, 'test', feature_set, self.data_manager)
            else:
                # Apply operations incrementally
                train_features, test_features = self._apply_operations_incrementally(
                    node.applied_operations, feature_set
                )
            
            # Combine train and test for unified feature set
            combined_features = pd.concat([train_features, test_features], ignore_index=True)
            
            # Apply feature limits
            combined_features = self._apply_feature_limits(combined_features)
            
            # Cache if enabled
            if self.cache_features:
                self._cache_features(cache_key, combined_features)
            
            generation_time = time.time() - generation_start
            node.feature_generation_time = generation_time
            
            logger.debug(f"Generated {combined_features.shape[1]} features for node in {generation_time:.2f}s")
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Feature generation failed for node: {e}")
            # Return base features as fallback
            train_base = load_or_create_features(self.train_path, 'train', 'basic', self.data_manager)
            test_base = load_or_create_features(self.test_path, 'test', 'basic', self.data_manager)
            return pd.concat([train_base, test_base], ignore_index=True)
    
    @timed("feature_space.synthetic_generation")
    def _generate_synthetic_features_for_node(self, node, generation_start: float) -> pd.DataFrame:
        """Generate synthetic features for mock testing mode."""
        logger.debug("Using synthetic feature generation for mock testing")
        
        # Get sample size from config
        testing_config = self.config.get('testing', {})
        sample_size = testing_config.get('small_dataset_size', 1000)
        
        # Generate base synthetic features
        if len(node.applied_operations) == 0:
            # Root node - generate base features
            synthetic_features = generate_synthetic_features(self.config, sample_size)
        else:
            # Apply operations to synthetic data
            synthetic_features = generate_synthetic_features(self.config, sample_size)
            
            # Apply each operation in sequence
            for operation_name in node.applied_operations:
                synthetic_features = augment_synthetic_features(synthetic_features, operation_name)
        
        # Apply feature limits
        synthetic_features = self._apply_feature_limits(synthetic_features)
        
        generation_time = time.time() - generation_start
        node.feature_generation_time = generation_time
        
        logger.debug(f"Generated {synthetic_features.shape[1]} synthetic features in {generation_time:.3f}s")
        
        return synthetic_features
    
    def _get_node_features(self, node) -> Set[str]:
        """Get current feature set for a node (for dependency checking)."""
        # This is a simplified implementation
        # In practice, you'd want to track actual feature names more precisely
        base_features = {
            'Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture',
            'Soil Type', 'Crop Type', 'soil_crop', 'low_Nitrogen', 'low_Phosphorous', 'low_Potassium'
        }
        
        # Add features from applied operations
        for op_name in node.applied_operations:
            if op_name in self.operations:
                operation = self.operations[op_name]
                base_features.update(operation.output_features)
        
        return base_features
    
    def _determine_feature_set(self, applied_operations: List[str]) -> str:
        """Determine whether to use 'basic' or 'full' feature set."""
        # Check if any operations require full features
        for op_name in applied_operations:
            if op_name in self.operations:
                operation = self.operations[op_name]
                if operation.category in ['agricultural_domain', 'statistical_aggregations']:
                    return 'full'
        
        return 'basic'
    
    def _apply_operations_incrementally(self, 
                                      operations: List[str], 
                                      base_feature_set: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply feature operations incrementally."""
        
        # Start with base features
        train_df = load_or_create_features(self.train_path, 'train', base_feature_set, self.data_manager)
        test_df = load_or_create_features(self.test_path, 'test', base_feature_set, self.data_manager)
        
        # Apply each operation in sequence
        for op_name in operations:
            if op_name in self.operations:
                operation = self.operations[op_name]
                
                # Apply operation-specific transformations
                train_df = self._apply_single_operation(train_df, operation, 'train')
                test_df = self._apply_single_operation(test_df, operation, 'test')
        
        return train_df, test_df
    
    def _apply_single_operation(self, 
                              df: pd.DataFrame, 
                              operation: FeatureOperation, 
                              data_type: str) -> pd.DataFrame:
        """Apply a single feature operation to a DataFrame."""
        
        start_time = time.time()
        
        try:
            # This is where you'd implement the actual feature generation logic
            # For now, we'll rely on the existing feature_engineering.py functions
            
            if operation.category == 'feature_transformations':
                if operation.name == 'polynomial_features':
                    df = self._add_polynomial_features(df)
                elif operation.name == 'log_transforms':
                    df = self._add_log_transforms(df)
                elif operation.name == 'interaction_terms':
                    df = self._add_interaction_terms(df)
            
            elif operation.category == 'feature_selection':
                if operation.name == 'correlation_filter':
                    df = self._apply_correlation_filter(df)
                elif operation.name == 'low_variance_filter':
                    df = self._apply_variance_filter(df)
            
            # Note: Most operations are already handled by load_or_create_features
            # This method is for additional custom operations
            
            operation_time = time.time() - start_time
            
            if operation_time > operation.max_execution_time:
                logger.warning(f"Operation {operation.name} exceeded time limit: {operation_time:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to apply operation {operation.name}: {e}")
            return df  # Return unchanged DataFrame on error
    
    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features."""
        for col in ['Nitrogen', 'Phosphorous', 'Potassium']:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
        
        if all(col in df.columns for col in ['Nitrogen', 'Phosphorous']):
            df['NP_product'] = df['Nitrogen'] * df['Phosphorous']
        if all(col in df.columns for col in ['Nitrogen', 'Potassium']):
            df['NK_product'] = df['Nitrogen'] * df['Potassium']
        if all(col in df.columns for col in ['Phosphorous', 'Potassium']):
            df['PK_product'] = df['Phosphorous'] * df['Potassium']
        
        return df
    
    def _add_log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log transformations."""
        for col in ['Nitrogen', 'Phosphorous', 'Potassium']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col])  # log1p for numerical stability
        
        return df
    
    def _add_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction terms between categorical and numerical features."""
        if 'Soil Type' in df.columns and 'Nitrogen' in df.columns:
            df['soil_nitrogen_interaction'] = df['Soil Type'].astype(str) + '_N' + df['Nitrogen'].astype(str)
        
        if 'Crop Type' in df.columns and 'Phosphorous' in df.columns:
            df['crop_phosphorous_interaction'] = df['Crop Type'].astype(str) + '_P' + df['Phosphorous'].astype(str)
        
        return df
    
    def _apply_correlation_filter(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return df
        
        corr_matrix = df[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        logger.debug(f"Correlation filter removing {len(to_drop)} features")
        return df.drop(columns=to_drop)
    
    def _apply_variance_filter(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove low variance features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        variances = df[numeric_cols].var()
        low_variance_cols = variances[variances < threshold].index
        
        logger.debug(f"Variance filter removing {len(low_variance_cols)} features")
        return df.drop(columns=low_variance_cols)
    
    def _apply_feature_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature count limits."""
        if len(df.columns) > self.max_features_per_node:
            # Keep most important features (this is simplified)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > self.max_features_per_node - 10:  # Keep some non-numeric
                # Simple feature selection based on variance
                variances = df[numeric_cols].var().sort_values(ascending=False)
                keep_cols = variances.head(self.max_features_per_node - 10).index
                
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                final_cols = list(keep_cols) + list(non_numeric_cols)
                
                df = df[final_cols]
                logger.debug(f"Applied feature limit: kept {len(final_cols)} features")
        
        return df
    
    def _get_node_cache_key(self, node) -> str:
        """Generate cache key for a node."""
        operations_str = ','.join(sorted(node.applied_operations))
        return hashlib.md5(operations_str.encode()).hexdigest()
    
    def _cache_features(self, cache_key: str, features_df: pd.DataFrame) -> None:
        """Cache features with memory management."""
        # Estimate memory usage
        memory_usage_mb = features_df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Check if we need to clear cache
        if self.cache_size_mb + memory_usage_mb > self.max_cache_size_mb:
            self._cleanup_cache()
        
        # Cache the features
        self.feature_cache[cache_key] = features_df.copy()
        self.cache_metadata[cache_key] = {
            'timestamp': time.time(),
            'memory_mb': memory_usage_mb,
            'access_count': 0
        }
        
        self.cache_size_mb += memory_usage_mb
        logger.debug(f"Cached features: {memory_usage_mb:.2f}MB, total cache: {self.cache_size_mb:.2f}MB")
    
    def _cleanup_cache(self) -> None:
        """Cleanup old cache entries based on LRU policy."""
        if not self.feature_cache:
            return
        
        # Sort by access count and timestamp (LRU)
        cache_items = [(key, meta['timestamp'], meta['access_count']) 
                      for key, meta in self.cache_metadata.items()]
        cache_items.sort(key=lambda x: (x[2], x[1]))  # Sort by access count, then timestamp
        
        # Remove oldest 50% of cache
        to_remove = cache_items[:len(cache_items)//2]
        
        for key, _, _ in to_remove:
            if key in self.feature_cache:
                memory_freed = self.cache_metadata[key]['memory_mb']
                del self.feature_cache[key]
                del self.cache_metadata[key]
                self.cache_size_mb -= memory_freed
        
        logger.debug(f"Cache cleanup: removed {len(to_remove)} entries, "
                    f"cache size: {self.cache_size_mb:.2f}MB")
    
    def update_operation_performance(self, 
                                   operation_name: str, 
                                   improvement: float, 
                                   success: bool) -> None:
        """Update performance statistics for an operation."""
        if operation_name in self.operations:
            operation = self.operations[operation_name]
            
            # Update statistics
            operation.total_applications += 1
            
            if success:
                # Running average of improvement
                old_avg = operation.avg_improvement
                old_count = operation.total_applications - 1
                operation.avg_improvement = (old_avg * old_count + improvement) / operation.total_applications
                
                # Update success rate
                success_count = old_count * operation.success_rate + 1
                operation.success_rate = success_count / operation.total_applications
            else:
                # Update success rate (no improvement)
                success_count = (operation.total_applications - 1) * operation.success_rate
                operation.success_rate = success_count / operation.total_applications
    
    def get_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all operations."""
        stats = {}
        for name, operation in self.operations.items():
            stats[name] = {
                'category': operation.category,
                'total_applications': operation.total_applications,
                'success_rate': operation.success_rate,
                'avg_improvement': operation.avg_improvement,
                'computational_cost': operation.computational_cost
            }
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.feature_cache.clear()
        self.cache_metadata.clear()
        self.cache_size_mb = 0.0
        logger.info("Feature space cleanup completed")