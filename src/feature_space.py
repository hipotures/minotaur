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
from .feature_engineering import load_or_create_features
from .timing import timed, timing_context, record_timing
from .data_utils import DataManager, smart_sample, estimate_memory_usage

logger = logging.getLogger(__name__)

@dataclass
class FeatureOperation:
    """Represents a single feature engineering operation."""
    
    name: str
    description: str
    category: str = ""
    operation_type: str = ""
    operation_subtype: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    new_features: List[str] = field(default_factory=list)
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
    
    def __init__(self, config: Dict[str, Any], duckdb_manager=None):
        """Initialize feature space with configuration."""
        self.config = config
        self.feature_config = config['feature_space']
        self.autogluon_config = config['autogluon']
        self.duckdb_manager = duckdb_manager
        
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
        
        # NEW: Feature building control parameters
        self.max_features_to_build = self.feature_config.get('max_features_to_build')
        self.max_features_per_iteration = self.feature_config.get('max_features_per_iteration')
        self.feature_build_timeout = self.feature_config.get('feature_build_timeout', 300)
        
        # Feature filtering configuration (to prevent data leakage)
        self.target_column = self.autogluon_config.get('target_column')
        self.id_column = self.autogluon_config.get('id_column')
        self.ignore_columns = self.autogluon_config.get('ignore_columns', []) or []
        self.cache_miss_limit = self.feature_config.get('cache_miss_limit', 50)
        
        # Generic operations configuration
        self.generic_operations = self.feature_config.get('generic_operations', {
            'statistical_aggregations': True,
            'polynomial_features': True,
            'binning_features': True,
            'ranking_features': True
        })
        self.generic_params = self.feature_config.get('generic_params', {
            'polynomial_degree': 2,
            'binning_bins': 5,
            'groupby_columns': [],
            'aggregate_columns': []
        })
        
        # Tracking counters
        self.features_built_count = 0
        self.iteration_feature_count = 0
        self.cache_miss_count = 0
        
        # Caching system
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.cache_metadata: Dict[str, Dict] = {}
        self.cache_size_mb = 0.0
        
        # Base data paths - support both new dataset_name system and legacy paths
        dataset_name = self.autogluon_config.get('dataset_name')
        if dataset_name:
            # New system: use cached dataset
            cache_dir = Path(config.get('project_root', '.')) / 'cache' / dataset_name
            cache_db_path = cache_dir / 'dataset.duckdb'
            
            if cache_db_path.exists():
                self.train_path = str(cache_db_path)
                self.test_path = str(cache_db_path)  # Same file contains both tables
                self.use_cached_dataset = True
                self.dataset_name = dataset_name
                logger.info(f"Using cached dataset for feature space: {dataset_name}")
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
        
        # Initialize data manager for optimized loading
        self.data_manager = DataManager(config)
        
        # Initialize operations with automatic domain support
        self._initialize_operations()
        
        logger.info(f"Initialized FeatureSpace with {len(self.operations)} operations")
    
    def _initialize_operations(self) -> None:
        """Initialize all available feature operations with automatic domain loading."""
        
        # Always load generic operations
        self._add_generic_operations()
        
        # Try to load custom domain operations based on dataset name
        if self.dataset_name:
            # Try automatic domain loading based on dataset name
            domain_module = self.dataset_name.lower()
            self._load_custom_domain_operations(domain_module)
        else:
            # Fallback to config if provided (for backward compatibility)
            custom_domain_module = self.feature_config.get('custom_domain_module')
            if custom_domain_module:
                logger.warning("Using deprecated 'custom_domain_module' config. Domain modules are now loaded automatically based on dataset name.")
                self._load_custom_domain_operations(custom_domain_module)
        
        logger.info(f"Initialized {len(self.operations)} feature operations")
    
    def _add_generic_operations(self) -> None:
        """Add truly generic operations that work with any dataset (configurable)."""
        
        # All possible generic operations
        all_operations = {
            'statistical_aggregations': FeatureOperation(
                name='statistical_aggregations',
                category='statistical_aggregations',
                description='Statistical aggregations by categorical features',
                dependencies=[],  # Works with any columns
                computational_cost=0.6,
                output_features=[]
            ),
            'polynomial_features': FeatureOperation(
                name='polynomial_features',
                category='feature_transformations',
                description=f'Polynomial features (degree {self.generic_params["polynomial_degree"]})',
                dependencies=[],  # Works with any numeric columns
                computational_cost=0.4,
                output_features=[]
            ),
            'binning_features': FeatureOperation(
                name='binning_features',
                category='feature_transformations',
                description=f'Binning of numerical features ({self.generic_params["binning_bins"]} bins)',
                dependencies=[],  # Works with any numeric columns
                computational_cost=0.3,
                output_features=[]
            ),
            'ranking_features': FeatureOperation(
                name='ranking_features',
                category='feature_transformations',
                description='Ranking features for numeric columns',
                dependencies=[],  # Works with any numeric columns
                computational_cost=0.2,
                output_features=[]
            )
        }
        
        # Add only enabled operations
        enabled_ops = []
        for op_name, operation in all_operations.items():
            if self.generic_operations.get(op_name, False):
                self.operations[operation.name] = operation
                enabled_ops.append(op_name)
        
        logger.info(f"Enabled generic operations: {enabled_ops}")
        if len(enabled_ops) < len(all_operations):
            disabled_ops = [op for op in all_operations.keys() if not self.generic_operations.get(op, False)]
            logger.info(f"Disabled generic operations: {disabled_ops}")
    
    def _load_custom_domain_operations(self, domain_name: str) -> None:
        """Dynamically load custom domain operations from domain module.
        
        Args:
            domain_name: Name of the domain (e.g., 'titanic', 'fertilizer_s5e6')
        """
        # Clean the domain name (remove 'features.custom.' prefix if present)
        clean_domain_name = domain_name.replace("features.custom.", "").replace("src.features.custom.", "")
        
        try:
            # Import the custom domain module
            import importlib
            module_path = f'src.features.custom.{clean_domain_name}'
            
            logger.info(f"Attempting to load custom domain module: {module_path}")
            module = importlib.import_module(module_path)
            
            # Get the CustomFeatureOperations class
            if not hasattr(module, 'CustomFeatureOperations'):
                logger.warning(f"No CustomFeatureOperations class found in {module_path}")
                return
            
            custom_class = module.CustomFeatureOperations
            
            # Discover all get_* methods
            method_names = [name for name in dir(custom_class) 
                          if name.startswith('get_') and callable(getattr(custom_class, name))]
            
            logger.info(f"Found {len(method_names)} custom operations in {clean_domain_name}: {method_names}")
            
            # Add each method as an operation
            for method_name in method_names:
                method = getattr(custom_class, method_name)
                
                # Extract operation name (remove 'get_' prefix)
                operation_name = method_name[4:]  # Remove 'get_'
                
                # Create operation
                operation = FeatureOperation(
                    name=operation_name,
                    category='custom_domain',
                    description=f'Custom domain operation: {operation_name}',
                    dependencies=[],  # Custom operations handle their own dependencies
                    computational_cost=0.5,
                    output_features=[]
                )
                
                self.operations[operation_name] = operation
                
            logger.info(f"✅ Successfully loaded {len(method_names)} custom domain operations from {clean_domain_name}.py")
            
        except ImportError:
            logger.info(f"No custom domain module found at src/domains/{clean_domain_name}.py - using generic operations only")
        except Exception as e:
            logger.error(f"Failed to load custom domain module {clean_domain_name}: {e}")
            logger.info("Falling back to generic operations only")
    
    def _add_npk_operations_old(self) -> None:
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
    
    def _add_environmental_operations_old(self) -> None:
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
    
    def _add_agricultural_operations_old(self) -> None:
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
    
    def _add_statistical_operations_old(self) -> None:
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
    
    def _add_transformation_operations_old(self) -> None:
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
    
    def _add_selection_operations_old(self) -> None:
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
    def get_feature_columns_for_node(self, node) -> List[str]:
        """
        Get list of feature column names for a given MCTS node.
        This uses the actual columns available in train_features table.
        
        Args:
            node: MCTS node
            
        Returns:
            List[str]: List of feature column names available in train_features
        """
        # Get all available columns from train_features table
        if hasattr(self, 'duckdb_manager') and self.duckdb_manager is not None:
            try:
                # Get all column names from train_features
                result = self.duckdb_manager.connection.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'train_features'
                """).fetchall()
                all_columns = [col[0] for col in result]
            except:
                # Fallback to getting columns from pragma_table_info
                result = self.duckdb_manager.connection.execute("""
                    SELECT name FROM pragma_table_info('train_features')
                """).fetchall()
                all_columns = [col[0] for col in result]
        else:
            # If no DuckDB connection, return base features only
            return list(node.base_features) if node.base_features else []
        
        # Start with base features
        base_features = list(node.base_features) if node.base_features else []
        selected_columns = base_features.copy()
        
        # For each applied operation, add columns that match expected patterns
        for operation_name in node.applied_operations:
            if operation_name == 'statistical_aggregations':
                # Add columns matching statistical patterns
                stat_patterns = ['_mean_by_', '_std_by_', '_dev_from_', '_count_by_', 
                               '_min_by_', '_max_by_', '_norm_by_']
                for col in all_columns:
                    if any(pattern in col for pattern in stat_patterns):
                        selected_columns.append(col)
                        
            elif operation_name == 'polynomial_features':
                # Add polynomial features
                poly_suffixes = ['_squared', '_cubed', '_sqrt', '_log']
                for col in all_columns:
                    if any(col.endswith(suffix) for suffix in poly_suffixes):
                        selected_columns.append(col)
                        
            elif operation_name == 'binning_features':
                # Add binning features
                for col in all_columns:
                    if '_binned' in col or col.endswith('_bin'):
                        selected_columns.append(col)
                        
            elif operation_name == 'ranking_features':
                # Add ranking features
                rank_patterns = ['_dense_rank', '_rank_pct', '_quartile', 
                               '_is_top10pct', '_is_bottom10pct', '_above_median']
                for col in all_columns:
                    if any(pattern in col for pattern in rank_patterns):
                        selected_columns.append(col)
                        
            elif operation_name.startswith('custom_'):
                # For custom operations, include all non-base columns that aren't from other operations
                # This is a heuristic - custom features don't follow standard patterns
                known_patterns = ['_mean_by_', '_std_by_', '_squared', '_binned', '_rank_']
                for col in all_columns:
                    if col not in base_features and not any(p in col for p in known_patterns):
                        # Likely a custom feature
                        selected_columns.append(col)
        
        # Remove duplicates and ensure we don't include forbidden columns
        selected_columns = list(set(selected_columns))
        
        # Remove forbidden columns (ID, target, etc.) - they will be added back by evaluator
        forbidden = [self.id_column, self.target_column] + self.ignore_columns
        selected_columns = [col for col in selected_columns if col not in forbidden]
        
        return selected_columns
    
    def generate_features_for_node(self, node) -> pd.DataFrame:
        """
        Generate features for a specific MCTS node using lazy loading.
        
        Args:
            node: MCTS node
            
        Returns:
            pd.DataFrame: Generated features
        """
        generation_start = time.time()
        
        # Always use real data now - no more synthetic data
        # Testing mode is controlled by config limits (train_size, max_iterations, etc.)
        
        # Check cache first
        if self.cache_features:
            cache_key = self._get_node_cache_key(node)
            if cache_key in self.feature_cache:
                logger.debug(f"Using cached features for node")
                return self.feature_cache[cache_key].copy()
        
        try:
            # Use real data generation function
            return self._generate_real_features_for_node(node, generation_start)
            
        except Exception as e:
            logger.error(f"Feature generation failed for node: {e}")
            # Return minimal base features as fallback (avoid load_or_create_features for non-fertilizer data)
            try:
                train_base = self.data_manager.load_dataset(self.train_path, 'train')
                test_base = self.data_manager.load_dataset(self.test_path, 'test') if self.test_path else pd.DataFrame()
                return pd.concat([train_base, test_base], ignore_index=True)
            except Exception as fallback_error:
                logger.error(f"Fallback loading also failed: {fallback_error}")
                # Return empty DataFrame as last resort
                return pd.DataFrame()
    
    @timed("feature_space.real_data_generation")
    def _generate_real_features_for_node(self, node, generation_start: float) -> pd.DataFrame:
        """Generate features using real data with applied operations."""
        logger.debug("Using real data feature generation")
        
        # Load base dataset (avoid feature_engineering.py for non-fertilizer data)
        train_base = self.data_manager.load_dataset(self.train_path, 'train')
        
        # Note: train_size sampling moved to AutoGluon evaluator level
        # Feature generation should work on full dataset for cache consistency
        
        # Apply each operation in sequence using domain modules
        current_features = train_base.copy()
        for operation_name in node.applied_operations:
            new_features = self._apply_domain_operation(current_features, operation_name)
            # Merge new features
            for feature_name, feature_data in new_features.items():
                current_features[feature_name] = feature_data
        
        # Apply feature limits
        current_features = self._apply_feature_limits(current_features)
        
        generation_time = time.time() - generation_start
        node.feature_generation_time = generation_time
        
        logger.debug(f"Generated {current_features.shape[1]} real features in {generation_time:.3f}s")
        
        return current_features
    
    def _apply_domain_operation(self, df: pd.DataFrame, operation_name: str) -> Dict[str, pd.Series]:
        """Apply domain-specific operation to generate new features."""
        try:
            # First try generic operations using new modular system
            from src import GenericFeatureOperations
            
            if operation_name == 'statistical_aggregations':
                features = self._apply_generic_statistical_aggregations(df)
                return self._filter_no_signal_features(features, "statistical aggregations")
            elif operation_name == 'polynomial_features':
                degree = self.generic_params['polynomial_degree']
                numeric_cols = self._filter_forbidden_columns(self._get_numeric_columns(df))
                features = GenericFeatureOperations.get_polynomial_features(df, numeric_cols, degree=degree)
                return self._filter_no_signal_features(features, "polynomial features")
            elif operation_name == 'binning_features':
                n_bins = self.generic_params['binning_bins']
                numeric_cols = self._filter_forbidden_columns(self._get_numeric_columns(df))
                features = GenericFeatureOperations.get_binning_features(df, numeric_cols, n_bins=n_bins)
                return self._filter_no_signal_features(features, "binning features")
            elif operation_name == 'ranking_features':
                numeric_cols = self._filter_forbidden_columns(self._get_numeric_columns(df))
                features = GenericFeatureOperations.get_ranking_features(df, numeric_cols)
                return self._filter_no_signal_features(features, "ranking features")
            
            # Try custom domain operations
            if self.dataset_name:
                try:
                    import importlib
                    module_path = f'src.features.custom.{self.dataset_name.lower()}'
                    module = importlib.import_module(module_path)
                    
                    if hasattr(module, 'CustomFeatureOperations'):
                        custom_class = module.CustomFeatureOperations
                        method_name = f'get_{operation_name}'
                        
                        # Create instance of the custom features class
                        custom_instance = custom_class()
                        
                        if hasattr(custom_instance, method_name):
                            method = getattr(custom_instance, method_name)
                            
                            # Check if method uses forbidden columns in source code
                            self._validate_custom_method_safety(method, method_name)
                            
                            features = method(df)
                            
                            # Filter out no-signal features
                            return self._filter_no_signal_features(features, f"custom {operation_name}")
                except ImportError:
                    pass  # No custom domain module, use generic operations
            
            logger.warning(f"Unknown operation: {operation_name}")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to apply operation {operation_name}: {e}")
            return {}
    
    def _apply_generic_statistical_aggregations(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Apply statistical aggregations to any dataset (configurable)."""
        features = {}
        
        # Get columns from config or auto-detect
        groupby_cols = self.generic_params.get('groupby_columns', [])
        agg_cols = self.generic_params.get('aggregate_columns', [])
        
        # Auto-detect if not configured
        if not groupby_cols:
            groupby_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not agg_cols:
            agg_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out forbidden columns to prevent data leakage
        groupby_cols = self._filter_forbidden_columns(groupby_cols)
        agg_cols = self._filter_forbidden_columns(agg_cols)
        
        # Use the GenericFeatureOperations class for consistency
        from src import GenericFeatureOperations
        return GenericFeatureOperations.get_statistical_aggregations(df, groupby_cols, agg_cols)
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric columns from DataFrame."""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _filter_forbidden_columns(self, columns: List[str]) -> List[str]:
        """Filter out target, ID, and ignored columns to prevent data leakage.
        
        Args:
            columns: List of column names to filter
            
        Returns:
            Filtered list with forbidden columns removed
        """
        forbidden_cols = set()
        
        # Add target column
        if self.target_column:
            forbidden_cols.add(self.target_column)
        
        # Add ID column  
        if self.id_column:
            forbidden_cols.add(self.id_column)
            
        # Add ignored columns
        if self.ignore_columns:
            forbidden_cols.update(self.ignore_columns)
        
        # Filter out forbidden columns
        filtered_cols = [col for col in columns if col not in forbidden_cols]
        
        # Log filtering info if any columns were removed
        removed_cols = set(columns) & forbidden_cols
        if removed_cols:
            logger.info(f"Filtered out forbidden columns from feature generation: {removed_cols}")
        
        return filtered_cols
    
    def has_signal(self, feature_series: pd.Series) -> bool:
        """
        Check if feature has signal (different values).
        
        Args:
            feature_series: Pandas Series to check for signal
            
        Returns:
            True if feature has signal (nunique > 1)
            False if no signal (all values identical)
        """
        try:
            # Remove NaN values and check number of unique values
            unique_count = feature_series.dropna().nunique()
            return unique_count > 1
        except Exception as e:
            logger.warning(f"Error checking feature signal: {e}")
            return False  # Conservative: treat as no signal if error
    
    def _filter_no_signal_features(self, features: Dict[str, pd.Series], operation_type: str) -> Dict[str, pd.Series]:
        """
        Filter out features with no signal (all identical values).
        
        Args:
            features: Dictionary of feature name -> pandas Series
            operation_type: Type of operation for logging (e.g., "statistical aggregations")
            
        Returns:
            Filtered dictionary with only features that have signal
        """
        if not features:
            return features
        
        filtered_features = {}
        no_signal_count = 0
        
        for feature_name, feature_series in features.items():
            if self.has_signal(feature_series):
                filtered_features[feature_name] = feature_series
                logger.debug(f"Feature '{feature_name}' has signal - keeping")
            else:
                no_signal_count += 1
                logger.info(f"Skipping no-signal feature '{feature_name}' from {operation_type} - all values identical")
        
        if no_signal_count > 0:
            logger.info(f"Filtered out {no_signal_count} no-signal features from {operation_type}, kept {len(filtered_features)}")
        
        return filtered_features
    
    def _get_node_features(self, node) -> Set[str]:
        """Get current feature set for a node (for dependency checking)."""
        # Get base features dynamically from the actual dataset
        try:
            # Load base dataset to get actual column names
            train_base = self.data_manager.load_dataset(self.train_path, 'train')
            base_features = set(train_base.columns)
        except Exception as e:
            logger.warning(f"Could not load base features dynamically, using empty set: {e}")
            base_features = set()
        
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
        train_df = self.data_manager.load_dataset(self.train_path, 'train')
        test_df = self.data_manager.load_dataset(self.test_path, 'test') if self.test_path else pd.DataFrame()
        
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
    
    # NEW: Feature building control methods
    def can_build_more_features(self) -> bool:
        """Check if more features can be built based on limits."""
        if self.max_features_to_build and self.features_built_count >= self.max_features_to_build:
            logger.warning(f"🚫 Reached feature build limit: {self.max_features_to_build}")
            return False
        
        if self.cache_miss_limit and self.cache_miss_count >= self.cache_miss_limit:
            logger.warning(f"🚫 Reached cache miss limit: {self.cache_miss_limit}")
            return False
            
        return True
    
    def can_build_more_features_this_iteration(self) -> bool:
        """Check if more features can be built in this MCTS iteration."""
        if self.max_features_per_iteration and self.iteration_feature_count >= self.max_features_per_iteration:
            logger.warning(f"🚫 Reached per-iteration feature limit: {self.max_features_per_iteration}")
            return False
        return True
    
    def reset_iteration_counter(self):
        """Reset iteration feature counter for new MCTS iteration."""
        self.iteration_feature_count = 0
        logger.debug(f"🔄 Reset iteration counter. Total built: {self.features_built_count}")
    
    def increment_feature_counters(self):
        """Increment feature building counters."""
        self.features_built_count += 1
        self.iteration_feature_count += 1
        self.cache_miss_count += 1
        logger.debug(f"📊 Features built: {self.features_built_count}, iteration: {self.iteration_feature_count}")
    
    def get_build_stats(self) -> dict:
        """Get feature building statistics."""
        return {
            'features_built_total': self.features_built_count,
            'features_built_this_iteration': self.iteration_feature_count,
            'cache_misses': self.cache_miss_count,
            'max_features_to_build': self.max_features_to_build,
            'max_features_per_iteration': self.max_features_per_iteration,
            'cache_miss_limit': self.cache_miss_limit,
            'can_build_more': self.can_build_more_features(),
            'can_build_more_this_iteration': self.can_build_more_features_this_iteration()
        }
    
    def get_operation_info(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific operation for database logging."""
        if operation_name not in self.operations:
            return None
            
        operation = self.operations[operation_name]
        return {
            'category': operation.category,
            'description': operation.description,
            'cost': operation.computational_cost,
            'code': f"# {operation.description}\n# Category: {operation.category}\n# Cost: {operation.computational_cost}"
        }
    
    def generate_all_features(self, df: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """
        Generate ALL possible features for a dataset (used during registration).
        
        This method generates all available features without checking config settings.
        It's used during dataset registration to pre-compute all possible features.
        
        Args:
            df: Input DataFrame with original data
            dataset_name: Name of the dataset (for custom domain operations)
            
        Returns:
            DataFrame with all original columns plus all generated features
        """
        logger.info(f"Generating all features for dataset: {dataset_name}")
        start_time = time.time()
        
        # Start with a copy of original data
        result_df = df.copy()
        feature_count = 0
        
        # Import generic operations module
        from src import GenericFeatureOperations
        
        # 1. Generate all generic features
        logger.info("Generating generic features...")
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Filter out forbidden columns to prevent data leakage
        numeric_cols = self._filter_forbidden_columns(numeric_cols)
        categorical_cols = self._filter_forbidden_columns(categorical_cols)
        
        # Statistical aggregations
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            try:
                stat_features = GenericFeatureOperations.get_statistical_aggregations(
                    df, categorical_cols[:5], numeric_cols[:10]  # Limit to prevent explosion
                )
                for feature_name, feature_data in stat_features.items():
                    result_df[feature_name] = feature_data
                    feature_count += 1
                logger.info(f"Added {len(stat_features)} statistical aggregation features")
            except Exception as e:
                logger.warning(f"Failed to generate statistical aggregations: {e}")
        
        # Polynomial features
        if len(numeric_cols) > 0:
            try:
                poly_features = GenericFeatureOperations.get_polynomial_features(
                    df, numeric_cols, degree=2
                )
                for feature_name, feature_data in poly_features.items():
                    result_df[feature_name] = feature_data
                    feature_count += 1
                logger.info(f"Added {len(poly_features)} polynomial features")
            except Exception as e:
                logger.warning(f"Failed to generate polynomial features: {e}")
        
        # Binning features
        if len(numeric_cols) > 0:
            try:
                bin_features = GenericFeatureOperations.get_binning_features(
                    df, numeric_cols, n_bins=5
                )
                for feature_name, feature_data in bin_features.items():
                    result_df[feature_name] = feature_data
                    feature_count += 1
                logger.info(f"Added {len(bin_features)} binning features")
            except Exception as e:
                logger.warning(f"Failed to generate binning features: {e}")
        
        # Ranking features
        if len(numeric_cols) > 0:
            try:
                rank_features = GenericFeatureOperations.get_ranking_features(df, numeric_cols)
                for feature_name, feature_data in rank_features.items():
                    result_df[feature_name] = feature_data
                    feature_count += 1
                logger.info(f"Added {len(rank_features)} ranking features")
            except Exception as e:
                logger.warning(f"Failed to generate ranking features: {e}")
        
        # 2. Generate custom domain features if available
        if dataset_name:
            logger.info(f"Generating custom domain features for: {dataset_name}")
            try:
                # Import custom domain module
                import importlib
                module_path = f'src.features.custom.{dataset_name.lower()}'
                module = importlib.import_module(module_path)
                
                if hasattr(module, 'CustomFeatureOperations'):
                    custom_class = module.CustomFeatureOperations
                    
                    # Create instance of the custom features class
                    custom_instance = custom_class()
                    
                    # Use the new architecture to generate all features with timing
                    custom_features = custom_instance.generate_all_features(df)
                    
                    if isinstance(custom_features, dict):
                        for feature_name, feature_data in custom_features.items():
                            result_df[feature_name] = feature_data
                        feature_count += len(custom_features)
                        logger.info(f"Added {len(custom_features)} custom features using new architecture")
                            
            except Exception as e:
                logger.warning(f"Failed to load custom domain module: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {feature_count} features in {elapsed_time:.2f}s")
        logger.info(f"Total columns: {len(result_df.columns)} (original: {len(df.columns)})")
        
        return result_df
    
    def generate_generic_features(self, df: pd.DataFrame, check_signal: bool = True) -> pd.DataFrame:
        """
        Generate ONLY generic features (statistical, polynomial, binning, ranking).
        
        Args:
            df: Input DataFrame with original data
            check_signal: Whether to check for signal and discard no-signal features
            
        Returns:
            DataFrame with ONLY the generated generic features (no original columns)
        """
        logger.info("🔧 Generating generic features...")
        start_time = time.time()
        
        # Start with empty result DataFrame
        result_features = {}
        
        # Import generic operations module
        from src import GenericFeatureOperations
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Filter out forbidden columns to prevent data leakage
        numeric_cols = self._filter_forbidden_columns(numeric_cols)
        categorical_cols = self._filter_forbidden_columns(categorical_cols)
        
        # Statistical aggregations
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            try:
                stat_features = GenericFeatureOperations.get_statistical_aggregations(
                    df, categorical_cols[:5], numeric_cols[:10], check_signal=check_signal  # Limit to prevent explosion
                )
                result_features.update(stat_features)
                logger.info(f"Added {len(stat_features)} statistical aggregation features")
            except Exception as e:
                logger.warning(f"Failed to generate statistical aggregations: {e}")
        
        # Polynomial features
        if len(numeric_cols) > 0:
            try:
                poly_features = GenericFeatureOperations.get_polynomial_features(
                    df, numeric_cols, degree=2, check_signal=check_signal
                )
                result_features.update(poly_features)
                logger.info(f"Added {len(poly_features)} polynomial features")
            except Exception as e:
                logger.warning(f"Failed to generate polynomial features: {e}")
        
        # Binning features
        if len(numeric_cols) > 0:
            try:
                bin_features = GenericFeatureOperations.get_binning_features(
                    df, numeric_cols, n_bins=5, check_signal=check_signal
                )
                result_features.update(bin_features)
                logger.info(f"Added {len(bin_features)} binning features")
            except Exception as e:
                logger.warning(f"Failed to generate binning features: {e}")
        
        # Ranking features
        if len(numeric_cols) > 0:
            try:
                rank_features = GenericFeatureOperations.get_ranking_features(df, numeric_cols, check_signal=check_signal)
                result_features.update(rank_features)
                logger.info(f"Added {len(rank_features)} ranking features")
            except Exception as e:
                logger.warning(f"Failed to generate ranking features: {e}")
        
        # Convert to DataFrame
        if result_features:
            result_df = pd.DataFrame(result_features)
        else:
            # Return DataFrame with dummy column to avoid DuckDB empty table errors
            result_df = pd.DataFrame({'_placeholder': [0] * len(df)}, index=df.index)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(result_df.columns)} generic features in {elapsed_time:.2f}s")
        
        return result_df
    
    def generate_custom_features(self, df: pd.DataFrame, dataset_name: str, check_signal: bool = True) -> pd.DataFrame:
        """
        Generate ONLY custom domain-specific features.
        
        Args:
            df: Input DataFrame with original data
            dataset_name: Name of the dataset (for custom domain operations)
            check_signal: Whether to check for signal and discard no-signal features
            
        Returns:
            DataFrame with ONLY the generated custom features (no original columns)
        """
        logger.info(f"🎯 Generating custom domain features for: {dataset_name}")
        start_time = time.time()
        
        # Start with empty result 
        result_features = {}
        
        # Generate custom domain features if available
        if dataset_name:
            try:
                # Import custom domain module
                import importlib
                module_path = f'src.features.custom.{dataset_name.lower()}'
                module = importlib.import_module(module_path)
                
                if hasattr(module, 'CustomFeatureOperations'):
                    custom_class = module.CustomFeatureOperations
                    
                    # Create instance of the custom features class
                    custom_instance = custom_class()
                    custom_instance._check_signal = check_signal
                    
                    # Use the new architecture to generate all features with timing
                    custom_features = custom_instance.generate_all_features(df)
                    
                    if isinstance(custom_features, dict):
                        result_features.update(custom_features)
                        logger.info(f"Generated {len(custom_features)} custom features using new architecture")
                            
            except Exception as e:
                logger.warning(f"Failed to load custom domain module: {e}")
        
        # Convert to DataFrame
        if result_features:
            result_df = pd.DataFrame(result_features)
        else:
            # Return DataFrame with dummy column to avoid DuckDB empty table errors
            result_df = pd.DataFrame({'_placeholder': [0] * len(df)}, index=df.index)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(result_df.columns)} custom features in {elapsed_time:.2f}s")
        
        return result_df
    
    def get_all_possible_feature_names(self, dataset_name: str = None) -> List[str]:
        """
        Get list of all possible feature names that could be generated.
        
        Args:
            dataset_name: Name of the dataset (for custom domain operations)
            
        Returns:
            List of all possible feature names
        """
        feature_names = []
        
        # Get sample data to determine possible features
        # This is a simplified version - in production you might want to
        # actually analyze the dataset structure
        
        # Generic feature patterns
        generic_patterns = [
            "{col}_squared",
            "{col}_cubed",
            "{col}_log",
            "{col}_sqrt",
            "{col1}_x_{col2}",
            "{col}_mean_by_{group}",
            "{col}_std_by_{group}",
            "{col}_dev_from_{group}_mean",
            "{col}_bin_5",
            "{col}_rank",
            "{col}_rank_pct"
        ]
        
        # Add patterns (this is simplified - real implementation would need actual column names)
        feature_names.extend(generic_patterns)
        
        # Add custom domain features if available
        if dataset_name:
            try:
                import importlib
                module_path = f'src.features.custom.{dataset_name.lower()}'
                module = importlib.import_module(module_path)
                
                if hasattr(module, 'CustomFeatureOperations'):
                    custom_class = module.CustomFeatureOperations
                    method_names = [name[4:] for name in dir(custom_class) 
                                  if name.startswith('get_') and callable(getattr(custom_class, name))]
                    feature_names.extend(method_names)
            except Exception:
                pass
        
        return feature_names
    
    def _validate_custom_method_safety(self, method, method_name: str) -> None:
        """
        Validate that custom feature method doesn't use forbidden columns.
        
        Args:
            method: The custom feature method to validate
            method_name: Name of the method for error reporting
            
        Raises:
            ValueError: If method uses target/ID columns (critical error)
            Warning: If method uses ignored columns (logs warning and skips)
        """
        import inspect
        
        try:
            # Get method source code
            source_code = inspect.getsource(method)
            
            # Check for forbidden columns in source code
            forbidden_target_id = set()
            if self.target_column:
                forbidden_target_id.add(self.target_column)
            if self.id_column:
                forbidden_target_id.add(self.id_column)
            
            # Check for target/ID columns in source (CRITICAL ERROR)
            for forbidden_col in forbidden_target_id:
                if f"'{forbidden_col}'" in source_code or f'"{forbidden_col}"' in source_code:
                    error_msg = f"CRITICAL: Custom feature method '{method_name}' uses forbidden column '{forbidden_col}'"
                    logger.error(error_msg)
                    raise ValueError(f"{error_msg} - Custom features cannot use target/ID columns!")
            
            # Check for ignored columns (WARNING + SKIP)
            if self.ignore_columns:
                for ignored_col in self.ignore_columns:
                    if f"'{ignored_col}'" in source_code or f'"{ignored_col}"' in source_code:
                        warning_msg = f"Custom feature method '{method_name}' uses ignored column '{ignored_col}'"
                        logger.warning(f"Skipping {warning_msg}")
                        # For ignored columns, we could raise an exception to skip the method
                        # But that might be too aggressive. For now, just log warning.
                        
        except OSError:
            # If we can't get source code (e.g., built-in methods), assume it's safe
            logger.debug(f"Could not inspect source code for {method_name} - assuming safe")
        except Exception as e:
            logger.warning(f"Failed to validate custom method {method_name}: {e}")
            # Don't block execution for validation failures