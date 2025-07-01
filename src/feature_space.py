"""
MCTS Feature Space Manager

Manages feature operations for MCTS exploration by loading from src/features/ modules.
Replaces hardcoded operations with dynamic loading from generic and custom feature modules.
"""

import time
import logging
import importlib
import threading
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureOperation:
    """Definition of a feature operation that can be applied in MCTS."""
    
    def __init__(self, name: str, category: str, description: str, 
                 dependencies: List[str], computational_cost: float, 
                 output_features: List[str]):
        self.name = name
        self.category = category
        self.description = description
        self.dependencies = dependencies
        self.computational_cost = computational_cost
        self.output_features = output_features
    
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
    
    Loads feature operations from src/features/ modules instead of hardcoded definitions.
    """
    
    def __init__(self, config: Dict[str, Any], duckdb_manager=None):
        """Initialize feature space with configuration."""
        self.config = config
        self.feature_config = config['feature_space']
        self.autogluon_config = config['autogluon']
        self.duckdb_manager = duckdb_manager
        
        # New pipeline adapter (optional)
        self._adapter = None
        self._use_new_pipeline = config.get('feature_space', {}).get('use_new_pipeline', False)
        
        # Feature operation definitions
        self.operations: Dict[str, FeatureOperation] = {}
        
        # Get dataset info for custom operations
        self.dataset_name = self.autogluon_config.get('dataset_name', 'unknown')
        
        # Configuration
        self.enabled_categories = set(self.feature_config.get('enabled_categories', []))
        self.category_weights = self.feature_config.get('category_weights', {})
        self.generic_operations_config = self.feature_config.get('generic_operations', {})
        self.generic_params = self.feature_config.get('generic_params', {})
        
        # Dataset configuration
        self.target_column = self.autogluon_config.get('target_column', 'target')
        self.id_column = self.autogluon_config.get('id_column', 'id')
        self.ignore_columns = self.autogluon_config.get('ignore_columns', []) or []
        
        # Caching
        self.cache_features = self.feature_config.get('cache_features', True)
        self.feature_cache = {}
        
        # Feature catalog cache for column names (lazy loading)
        self._feature_catalog_cache = {}
        self._cache_lock = threading.Lock()
        
        # Feature performance tracking
        self.feature_stats = {}
        
        # Load operations from modules
        self._load_feature_operations()
        
        # Load feature performance stats from exploration history if we have DuckDB access
        # Temporarily disabled due to connection conflicts between DuckDBDataManager and DuckDBConnectionManager
        # if self.duckdb_manager:
        #     self._load_feature_stats_from_history()
        
        logger.info(f"Initialized FeatureSpace with {len(self.operations)} operations from src/features/")
        
        # Initialize adapter if new pipeline is enabled
        if self._use_new_pipeline:
            try:
                from .features.space_adapter import create_adapter
                self._adapter = create_adapter(config, duckdb_manager)
                logger.info("✨ New feature pipeline enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize new pipeline adapter: {e}")
                self._use_new_pipeline = False
    
    def _load_feature_operations(self):
        """Load feature operations from src/features/ modules."""
        
        # Load generic operations
        self._load_generic_operations()
        
        # Load custom operations based on dataset
        self._load_custom_operations()
        
        logger.info(f"Loaded {len(self.operations)} total feature operations")
    
    def _load_generic_operations(self):
        """Load generic operations from src/features/generic/."""
        logger.debug(f"Loading generic operations with config: {self.generic_operations_config}")
        try:
            from .features.generic import statistical, polynomial, binning, ranking
            logger.debug("Successfully imported generic feature modules")
            
            # Statistical aggregations
            if self.generic_operations_config.get('statistical_aggregations', True):
                self.operations['statistical_aggregations'] = FeatureOperation(
                    name='statistical_aggregations',
                    category='statistical_aggregations',
                    description='Group-based statistical aggregations',
                    dependencies=[],  # Works with any columns
                    computational_cost=0.5,
                    output_features=[]
                )
            
            # Polynomial features
            if self.generic_operations_config.get('polynomial_features', True):
                self.operations['polynomial_features'] = FeatureOperation(
                    name='polynomial_features',
                    category='polynomial',
                    description=f'Polynomial features (degree {self.generic_params.get("polynomial_degree", 2)})',
                    dependencies=[],  # Works with any numeric columns
                    computational_cost=0.4,
                    output_features=[]
                )
            
            # Binning features
            if self.generic_operations_config.get('binning_features', True):
                self.operations['binning_features'] = FeatureOperation(
                    name='binning_features',
                    category='binning',
                    description=f'Binning of numerical features ({self.generic_params.get("binning_bins", 5)} bins)',
                    dependencies=[],  # Works with any numeric columns
                    computational_cost=0.3,
                    output_features=[]
                )
            
            # Ranking features
            if self.generic_operations_config.get('ranking_features', True):
                self.operations['ranking_features'] = FeatureOperation(
                    name='ranking_features',
                    category='ranking',
                    description='Ranking features for numeric columns',
                    dependencies=[],  # Works with any numeric columns
                    computational_cost=0.2,
                    output_features=[]
                )
            
            logger.info(f"Loaded {len([op for op in self.operations.values() if 'transformations' in op.category or 'aggregations' in op.category])} generic operations")
            
        except ImportError as e:
            logger.warning(f"Could not load generic operations: {e}")
            import traceback
            logger.warning(f"Import traceback: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Error loading generic operations: {e}")
            import traceback
            logger.error(f"Error traceback: {traceback.format_exc()}")
    
    def _load_custom_operations(self):
        """Load custom operations based on dataset name."""
        logger.debug(f"Loading custom operations for dataset: {self.dataset_name}")
        if not self.dataset_name or self.dataset_name == 'unknown':
            logger.info("No dataset name specified, skipping custom operations")
            return
        
        try:
            # Auto-detect custom domain module based on dataset name
            dataset_name_clean = self.dataset_name.lower().replace('-', '_').replace(' ', '_')
            
            # Check if custom domain module exists for this dataset
            from pathlib import Path
            custom_modules_dir = Path(__file__).parent / 'features' / 'custom'
            module_file = f"{dataset_name_clean}.py"
            
            module_name = None
            logger.debug(f"Looking for custom module: {custom_modules_dir / module_file}")
            if (custom_modules_dir / module_file).exists():
                module_name = module_file[:-3]  # Remove .py extension
                logger.debug(f"Found custom module: {module_name}")
            
            if not module_name:
                logger.info(f"No custom domain module found for dataset '{self.dataset_name}', skipping custom operations")
                return
            
            # Import custom module
            custom_module = importlib.import_module(f'.features.custom.{module_name}', package='src')
            
            # Get CustomFeatureOperations class
            if hasattr(custom_module, 'CustomFeatureOperations'):
                custom_ops = custom_module.CustomFeatureOperations()
                
                # Get all available operations from the custom class
                for op_name, op_method in custom_ops._operation_registry.items():
                    # Create FeatureOperation for each custom operation
                    self.operations[op_name] = FeatureOperation(
                        name=op_name,
                        category='custom_domain',
                        description=f'Custom {self.dataset_name} operation: {op_name}',
                        dependencies=[],  # Custom operations handle their own dependencies
                        computational_cost=0.5,
                        output_features=[]
                    )
                
                logger.info(f"Loaded {len(custom_ops._operation_registry)} custom operations for {self.dataset_name}")
            else:
                logger.warning(f"No CustomFeatureOperations class found in {module_name}")
                
        except ImportError:
            logger.info(f"No custom module found for dataset '{self.dataset_name}' - using generic operations only")
        except Exception as e:
            logger.error(f"Error loading custom operations for {self.dataset_name}: {e}")
    
    def get_available_operations(self, node) -> List[str]:
        """
        Get list of operations that can be applied to the given node.
        Alias for get_available_operations_for_node for compatibility.
        
        Args:
            node: MCTS node
            
        Returns:
            List[str]: Available operation names
        """
        return self.get_available_operations_for_node(node)
    
    def get_available_operations_for_node(self, node) -> List[str]:
        """
        Get list of operations that can be applied to the given node.
        
        Args:
            node: MCTS node
            
        Returns:
            List[str]: Available operation names
        """
        if not hasattr(node, 'base_features') or not node.base_features:
            # For root node, use all available columns from database
            current_features = self._get_available_columns_from_db()
        else:
            current_features = set(node.base_features)
            
            # Add features from applied operations
            for op_name in getattr(node, 'applied_operations', []):
                current_features.update(self._get_operation_output_columns(op_name))
        
        available_ops = []
        
        logger.debug(f"Total operations loaded: {len(self.operations)}")
        logger.debug(f"Current features for node: {current_features}")
        
        for op_name, operation in self.operations.items():
            logger.debug(f"Checking operation {op_name} (category: {operation.category})")
            # Check if operation can be applied
            if operation.can_apply(current_features):
                logger.debug(f"  ✓ Can apply {op_name}")
                # Check if already applied in this path
                if op_name not in getattr(node, 'applied_operations', []):
                    logger.debug(f"  ✓ Not already applied {op_name}")
                    # Apply category filtering
                    if not self.enabled_categories or operation.category in self.enabled_categories:
                        logger.debug(f"  ✓ Category filter passed {op_name}")
                        # Apply category weighting
                        weight = self.category_weights.get(operation.category, 1.0)
                        if weight > 0:
                            logger.debug(f"  ✓ Added to available: {op_name}")
                            available_ops.append(op_name)
                        else:
                            logger.debug(f"  ✗ Weight is 0: {op_name}")
                    else:
                        logger.debug(f"  ✗ Category filter failed: {op_name}")
                else:
                    logger.debug(f"  ✗ Already applied: {op_name}")
            else:
                logger.debug(f"  ✗ Cannot apply {op_name} (dependencies: {operation.dependencies})")
        
        # Sort by category weight and computational cost
        available_ops.sort(
            key=lambda op_name: (
                -self.category_weights.get(self.operations[op_name].category, 1.0),
                self.operations[op_name].computational_cost
            )
        )
        
        logger.debug(f"Found {len(available_ops)} available operations for node at depth {getattr(node, 'depth', 0)}")
        return available_ops
    
    def _get_feature_columns_cached(self, operation_name: str, is_custom: bool = False) -> List[str]:
        """
        Get feature columns for an operation using lazy cached lookup.
        
        This method implements a thread-safe cache to avoid redundant database queries.
        The cache is populated on first access and reused for subsequent calls.
        
        Args:
            operation_name: Name of the operation to get features for
            is_custom: Whether this is a custom domain operation
            
        Returns:
            List of feature column names for the operation
        """
        # Generate cache key based on operation type
        if is_custom:
            cache_key = f"custom:{operation_name}"
        else:
            # Normalize operation name for cache key
            cache_key = f"generic:{operation_name.lower().replace(' ', '_')}"
        
        # Check cache first (no lock needed for read)
        if cache_key in self._feature_catalog_cache:
            return self._feature_catalog_cache[cache_key].copy()
        
        # Cache miss - need to query database
        with self._cache_lock:
            # Double-check pattern - another thread might have populated while we waited
            if cache_key in self._feature_catalog_cache:
                return self._feature_catalog_cache[cache_key].copy()
            
            # No database connection - return empty list
            if not hasattr(self, 'duckdb_manager') or self.duckdb_manager is None:
                logger.warning(f"No DuckDB connection for cached lookup of '{operation_name}'")
                self._feature_catalog_cache[cache_key] = []
                return []
            
            try:
                # Execute appropriate query based on operation type
                if is_custom:
                    query = """
                        SELECT DISTINCT feature_name 
                        FROM feature_catalog 
                        WHERE operation_name = ?
                    """
                    result = self.duckdb_manager.connection.execute(query, [operation_name]).fetchall()
                else:
                    query = """
                        SELECT DISTINCT feature_name 
                        FROM feature_catalog 
                        WHERE LOWER(REPLACE(operation_name, ' ', '_')) = LOWER(?)
                    """
                    result = self.duckdb_manager.connection.execute(query, [operation_name]).fetchall()
                
                # Extract feature names from result
                feature_names = [row[0] for row in result]
                
                # Filter out forbidden columns
                forbidden = {self.id_column, self.target_column} | set(self.ignore_columns or [])
                feature_names = [col for col in feature_names if col not in forbidden]
                
                # Store in cache
                self._feature_catalog_cache[cache_key] = feature_names
                
                logger.debug(f"Cached {len(feature_names)} features for operation '{operation_name}' (key: {cache_key})")
                return feature_names.copy()
                
            except Exception as e:
                logger.error(f"Failed to query feature catalog for '{operation_name}': {e}")
                # Cache empty result to avoid repeated failures
                self._feature_catalog_cache[cache_key] = []
                return []
    
    def get_feature_columns_for_node(self, node) -> List[str]:
        """
        Get list of feature column names for a given MCTS node.
        Uses dynamic database lookup to resolve operation features.
        
        CRITICAL FIX: Uses operation_that_created_this instead of applied_operations
        to ensure each node gets different feature sets for proper MCTS evaluation.
        """
        # Get the current operation for this node (FIX THE CORE BUG)
        current_operation = getattr(node, 'operation_that_created_this', None)
        
        # For root node, return base features
        if current_operation is None or current_operation == 'root':
            base_features = list(getattr(node, 'base_features', []))
            logger.debug(f"Root node - returning {len(base_features)} base features")
            return base_features
        
        # Check if this is a custom domain operation
        is_custom_op = current_operation in self.operations and self.operations[current_operation].category == 'custom_domain'
        
        # Use cached lookup for feature columns
        operation_features = self._get_feature_columns_cached(current_operation, is_custom=is_custom_op)
        
        if operation_features:
            logger.debug(f"Node operation '{current_operation}' - found {len(operation_features)} features from cache")
            return operation_features
        else:
            logger.warning(f"No features found for operation '{current_operation}' - falling back to pattern matching")
            # Fallback to pattern-based detection if no database entries
            return self._get_features_by_pattern_fallback(current_operation, node)
    
    def _get_features_by_pattern_fallback(self, operation_name: str, node) -> List[str]:
        """
        Fallback method using dynamic pattern detection from OPERATION_METADATA.
        Used when database lookup fails or returns no results.
        """
        # Import the metadata for dynamic pattern lookup
        try:
            from src.features.generic import detect_operation_from_feature_name, get_operation_metadata
            
            # Get all available columns from database or node
            all_columns = self._get_all_available_columns()
            if not all_columns:
                logger.warning("No columns available for pattern matching")
                return list(getattr(node, 'base_features', []))
            
            # Get operation metadata for pattern matching
            metadata = get_operation_metadata(operation_name)
            if metadata and 'output_patterns' in metadata:
                patterns = metadata['output_patterns']
                logger.debug(f"Using {len(patterns)} patterns for operation '{operation_name}': {patterns}")
            else:
                # Fallback to legacy hardcoded patterns
                patterns = self._get_legacy_patterns_for_operation(operation_name)
                logger.debug(f"Using legacy patterns for operation '{operation_name}': {patterns}")
            
            # Find columns matching the operation's patterns
            selected_columns = []
            for col in all_columns:
                col_lower = col.lower()
                if any(pattern.lower() in col_lower for pattern in patterns):
                    selected_columns.append(col)
            
            # Remove forbidden columns
            forbidden = [self.id_column, self.target_column] + self.ignore_columns
            selected_columns = [col for col in selected_columns if col not in forbidden]
            
            logger.debug(f"Pattern matching for '{operation_name}' found {len(selected_columns)} features")
            return selected_columns
            
        except Exception as e:
            logger.error(f"Pattern fallback failed for operation '{operation_name}': {e}")
            # Final fallback to base features
            return list(getattr(node, 'base_features', []))
    
    def _get_all_available_columns(self) -> List[str]:
        """Get all available columns from the database."""
        if hasattr(self, 'duckdb_manager') and self.duckdb_manager is not None:
            try:
                # Get all column names from train_features
                result = self.duckdb_manager.connection.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'train_features'
                """).fetchall()
                return [col[0] for col in result]
            except:
                try:
                    # Fallback to getting columns from pragma_table_info
                    result = self.duckdb_manager.connection.execute("""
                        SELECT name FROM pragma_table_info('train_features')
                    """).fetchall()
                    return [col[0] for col in result]
                except Exception as e:
                    logger.debug(f"Could not get columns from database: {e}")
        
        return []
    
    def _get_legacy_patterns_for_operation(self, operation_name: str) -> List[str]:
        """
        Legacy hardcoded patterns as final fallback.
        This should be removed once all operations are properly registered in database.
        """
        legacy_patterns = {
            'statistical_aggregations': ['_mean_by_', '_std_by_', '_dev_from_', '_count_by_', '_min_by_', '_max_by_', '_norm_by_'],
            'polynomial_features': ['_squared', '_cubed', '_sqrt', '_log', '_exp'],
            'binning_features': ['_bin_', '_binned', '_qbin_', '_quantile_'],
            'ranking_features': ['_rank', '_percentile', '_quartile', '_decile'],
            'interaction_features': ['_interaction_', '_cross_', '_ratio_'],
            'categorical_features': ['_encoded', '_frequency', '_target_mean'],
            'temporal_features': ['_year', '_month', '_day', '_hour', '_dayofweek'],
            'text_features': ['_length', '_word_count', '_char_count', '_upper_ratio'],
        }
        
        return legacy_patterns.get(operation_name, [operation_name.replace('_', '')])  # Default: use operation name itself
    
    def _get_available_columns_from_db(self) -> Set[str]:
        """Get available columns from the database."""
        if hasattr(self, 'duckdb_manager') and self.duckdb_manager is not None:
            try:
                result = self.duckdb_manager.connection.execute("""
                    SELECT name FROM pragma_table_info('train_features')
                """).fetchall()
                return {col[0] for col in result}
            except Exception as e:
                logger.debug(f"Could not get table columns from DuckDB: {e}")
        
        # Fallback - return empty set
        return set()
    
    def _get_operation_output_columns(self, operation_name: str) -> Set[str]:
        """
        Get expected output columns for an operation using dynamic pattern matching.
        Replaces hardcoded patterns with dynamic lookup from OPERATION_METADATA.
        """
        all_columns = self._get_available_columns_from_db()
        
        try:
            from src.features.generic import get_operation_metadata
            
            # Get operation metadata for pattern matching
            metadata = get_operation_metadata(operation_name)
            if metadata and 'output_patterns' in metadata:
                patterns = metadata['output_patterns']
            else:
                # Fallback to legacy patterns
                patterns = self._get_legacy_patterns_for_operation(operation_name)
            
            # Find columns matching the operation's patterns
            matching_columns = set()
            for col in all_columns:
                col_lower = col.lower()
                if any(pattern.lower() in col_lower for pattern in patterns):
                    matching_columns.add(col)
            
            return matching_columns
            
        except Exception as e:
            logger.debug(f"Dynamic pattern matching failed for operation '{operation_name}': {e}")
            # Return empty set if all fails
            return set()
    
    def generate_features_for_node(self, node) -> pd.DataFrame:
        """
        Generate features for a specific MCTS node using lazy loading.
        
        This method now primarily uses pre-built features from the database
        rather than generating new ones on the fly.
        """
        # Get feature columns for this node
        feature_columns = self.get_feature_columns_for_node(node)
        
        if not feature_columns:
            logger.warning("No feature columns found for node")
            return pd.DataFrame()
        
        # Load features from database
        if hasattr(self, 'duckdb_manager') and self.duckdb_manager is not None:
            try:
                # Create SQL query to select only needed columns
                column_list = ', '.join([f'"{col}"' for col in feature_columns])
                query = f"SELECT {column_list} FROM train_features"
                
                features_df = self.duckdb_manager.connection.execute(query).df()
                logger.debug(f"Loaded {len(features_df)} rows with {len(features_df.columns)} features for node")
                
                return features_df
                
            except Exception as e:
                logger.error(f"Failed to load features from database: {e}")
                return pd.DataFrame()
        else:
            logger.warning("No DuckDB manager available")
            return pd.DataFrame()
    
    def _load_feature_stats_from_history(self) -> None:
        """Load feature performance statistics from exploration history."""
        try:
            # Get dataset name for filtering
            dataset_name = self.dataset_name
            if not dataset_name or dataset_name == 'unknown':
                logger.debug("No dataset name available, skipping feature stats loading")
                return
            
            # Get target metric for filtering
            target_metric = self.autogluon_config.get('target_metric', 'unknown')
            
            # Query exploration history for this dataset + target metric combination
            query = """
            SELECT eh.features_after, eh.score, 
                   LAG(eh.score) OVER (PARTITION BY eh.session_id ORDER BY eh.iteration) as parent_score
            FROM exploration_history eh
            JOIN sessions s ON eh.session_id = s.session_id  
            WHERE JSON_EXTRACT(s.config_snapshot, '$.autogluon.dataset_name') = ?
            AND JSON_EXTRACT(s.config_snapshot, '$.autogluon.target_metric') = ?
            AND eh.features_after IS NOT NULL
            AND eh.score IS NOT NULL
            ORDER BY eh.session_id, eh.iteration
            """
            
            result = self.duckdb_manager.connection.execute(query, [dataset_name, target_metric]).fetchall()
            
            features_analyzed = 0
            improvements_found = 0
            
            for row in result:
                features_after = row[0] if row[0] else []
                current_score = row[1]
                parent_score = row[2]
                
                # Skip if no parent score (root nodes)
                if parent_score is None:
                    continue
                
                improvement = current_score - parent_score
                
                # Process each feature in this combination
                if isinstance(features_after, list):
                    for feature_name in features_after:
                        if feature_name not in self.feature_stats:
                            self.feature_stats[feature_name] = {
                                'total_uses': 0,
                                'success_count': 0,
                                'total_improvement': 0.0,
                                'category': self._get_feature_category(feature_name)
                            }
                        
                        stats = self.feature_stats[feature_name]
                        stats['total_uses'] += 1
                        
                        if improvement > 0:
                            stats['success_count'] += 1
                            improvements_found += 1
                        
                        stats['total_improvement'] += improvement
                        features_analyzed += 1
            
            logger.info(f"Loaded feature stats from history: {len(self.feature_stats)} features, "
                       f"{features_analyzed} feature uses, {improvements_found} improvements "
                       f"(dataset: {dataset_name}, metric: {target_metric})")
                       
        except Exception as e:
            logger.warning(f"Failed to load feature stats from history: {e}")
    
    def track_feature_performance(self, feature_columns: List[str], improvement: float) -> None:
        """Track performance of individual features."""
        for feature_name in feature_columns:
            if feature_name not in self.feature_stats:
                self.feature_stats[feature_name] = {
                    'total_uses': 0,
                    'success_count': 0,
                    'total_improvement': 0.0,
                    'category': self._get_feature_category(feature_name)
                }
            
            stats = self.feature_stats[feature_name]
            stats['total_uses'] += 1
            
            if improvement > 0:
                stats['success_count'] += 1
            
            stats['total_improvement'] += improvement
    
    def _get_feature_category(self, feature_name: str) -> str:
        """
        Determine feature category from name patterns using dynamic pattern matching.
        Replaces hardcoded patterns with dynamic lookup from OPERATION_METADATA.
        """
        try:
            from src.features.generic import detect_operation_from_feature_name, get_operation_metadata
            
            # Try dynamic detection first
            operation_name = detect_operation_from_feature_name(feature_name)
            if operation_name:
                metadata = get_operation_metadata(operation_name)
                if metadata and 'category' in metadata:
                    return metadata['category']
            
            # Fallback to legacy pattern matching
            feature_lower = feature_name.lower()
            
            if any(suffix in feature_lower for suffix in ['_squared', '_cubed', '_log', '_sqrt', '_reciprocal', '_x_']):
                return 'polynomial'
            elif any(pattern in feature_lower for pattern in ['_mean_by_', '_std_by_', '_count_by_', '_min_by_', '_max_by_']):
                return 'statistical_aggregations'  
            elif any(pattern in feature_lower for pattern in ['_binned', '_bin', '_qbin_']):
                return 'binning'
            elif any(pattern in feature_lower for pattern in ['_rank', '_percentile', '_quartile', '_decile']):
                return 'ranking'
            else:
                return 'custom_domain'
                
        except Exception as e:
            logger.debug(f"Feature category detection failed for '{feature_name}': {e}")
            return 'unknown'
    
    def get_feature_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for individual features."""
        stats = {}
        for feature_name, raw_stats in self.feature_stats.items():
            total_uses = raw_stats['total_uses']
            if total_uses > 0:
                stats[feature_name] = {
                    'total_uses': total_uses,
                    'success_rate': raw_stats['success_count'] / total_uses,
                    'avg_improvement': raw_stats['total_improvement'] / total_uses,
                    'category': raw_stats['category']
                }
        return stats
    
    def get_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get operation statistics (redirects to feature stats for compatibility)."""
        return self.get_feature_stats()
    
    def _filter_forbidden_columns(self, columns: List[str]) -> List[str]:
        """Filter out target, ID and ignore columns."""
        forbidden = [self.target_column, self.id_column] + self.ignore_columns
        return [col for col in columns if col not in forbidden]
    
    def generate_generic_features(self, df: pd.DataFrame, check_signal: bool = True, target_column: str = None, id_column: str = None, auto_register: bool = True, origin: str = 'generic') -> pd.DataFrame:
        """
        Generate generic features using new modular architecture.
        Used by dataset registration process.
        """
        # Use new pipeline if enabled and available
        if self._use_new_pipeline and self._adapter:
            return self._adapter.generate_generic_features_new(df, check_signal)
        
        logger.info("🔧 Generating generic features using modular architecture...")
        start_time = time.time()
        
        result_features = {}
        
        try:
            # Load and use generic operations from src/features/generic/
            from .features.generic import statistical, polynomial, binning, ranking, categorical, temporal, text
            
            # Statistical aggregations
            if self.generic_operations_config.get('statistical_aggregations', True):
                stat_op = statistical.StatisticalFeatures()
                stat_op._check_signal = check_signal
                groupby_cols = self.generic_params.get('groupby_columns', [])
                aggregate_cols = self.generic_params.get('aggregate_columns', [])
                
                # Auto-detect if not specified
                if not groupby_cols:
                    groupby_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:3]
                if not aggregate_cols:
                    aggregate_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
                
                # Filter out forbidden columns (target, ID, ignore columns)
                groupby_cols = self._filter_forbidden_columns(groupby_cols)
                aggregate_cols = self._filter_forbidden_columns(aggregate_cols)
                
                stat_features = stat_op.generate_features(df, groupby_cols=groupby_cols, agg_cols=aggregate_cols, auto_register=auto_register, origin=origin, dataset_db_path=self.config.get('dataset_db_path'))
                result_features.update(stat_features)
                logger.info(f"Added {len(stat_features)} statistical features")
            
            # Polynomial features
            if self.generic_operations_config.get('polynomial_features', True):
                poly_op = polynomial.PolynomialFeatures()
                poly_op._check_signal = check_signal
                degree = self.generic_params.get('polynomial_degree', 2)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
                
                # Filter out forbidden columns
                numeric_cols = self._filter_forbidden_columns(numeric_cols)
                
                poly_features = poly_op.generate_features(df, numeric_cols=numeric_cols, degree=degree, auto_register=auto_register, origin=origin, dataset_db_path=self.config.get('dataset_db_path'))
                result_features.update(poly_features)
                logger.info(f"Added {len(poly_features)} polynomial features")
            
            # Binning features
            if self.generic_operations_config.get('binning_features', True):
                bin_op = binning.BinningFeatures()
                bin_op._check_signal = check_signal
                n_bins = self.generic_params.get('binning_bins', 5)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Filter out forbidden columns
                numeric_cols = self._filter_forbidden_columns(numeric_cols)
                
                bin_features = bin_op.generate_features(df, numeric_cols=numeric_cols, n_bins=n_bins, auto_register=auto_register, origin=origin, dataset_db_path=self.config.get('dataset_db_path'))
                result_features.update(bin_features)
                logger.info(f"Added {len(bin_features)} binning features")
            
            # Ranking features
            if self.generic_operations_config.get('ranking_features', True):
                rank_op = ranking.RankingFeatures()
                rank_op._check_signal = check_signal
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Filter out forbidden columns
                numeric_cols = self._filter_forbidden_columns(numeric_cols)
                
                rank_features = rank_op.generate_features(df, numeric_cols=numeric_cols, auto_register=auto_register, origin=origin, dataset_db_path=self.config.get('dataset_db_path'))
                result_features.update(rank_features)
                logger.info(f"Added {len(rank_features)} ranking features")
            
            # Categorical features
            if self.generic_operations_config.get('categorical_features', True):
                cat_op = categorical.CategoricalFeatures()
                cat_op._check_signal = check_signal
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                categorical_cols = self._filter_forbidden_columns(categorical_cols)
                
                # Prepare real forbidden columns for categorical operation
                real_forbidden = []
                if target_column:
                    real_forbidden.append(target_column)
                if id_column:
                    real_forbidden.append(id_column)
                real_forbidden.extend(self.ignore_columns)
                
                if categorical_cols:
                    cat_features = cat_op.generate_features(df, categorical_cols=categorical_cols, forbidden_columns=real_forbidden, auto_register=auto_register, origin=origin, dataset_db_path=self.config.get('dataset_db_path'))
                    result_features.update(cat_features)
                    logger.info(f"Added {len(cat_features)} categorical features")
                else:
                    # Even if no object/category columns, still run auto-detection with proper forbidden list
                    cat_features = cat_op.generate_features(df, forbidden_columns=real_forbidden, auto_register=auto_register, origin=origin, dataset_db_path=self.config.get('dataset_db_path'))
                    result_features.update(cat_features)
                    logger.info(f"Added {len(cat_features)} categorical features (auto-detected)")
            
            # Temporal features
            if self.generic_operations_config.get('temporal_features', True):
                temp_op = temporal.TemporalFeatures()
                temp_op._check_signal = check_signal
                datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                datetime_cols = self._filter_forbidden_columns(datetime_cols)
                
                if datetime_cols:
                    temp_features = temp_op.generate_features(df, datetime_cols=datetime_cols, auto_register=auto_register, origin=origin, dataset_db_path=self.config.get('dataset_db_path'))
                    result_features.update(temp_features)
                    logger.info(f"Added {len(temp_features)} temporal features")
            
            # Text features  
            if self.generic_operations_config.get('text_features', True):
                text_op = text.TextFeatures()
                text_op._check_signal = check_signal
                text_cols = []
                
                # Find text columns (object columns that might contain text)
                for col in df.select_dtypes(include=['object']).columns:
                    if col not in [self.target_column, self.id_column] + self.ignore_columns:
                        # Simple heuristic: if average string length > 10, consider it text
                        if df[col].str.len().mean() > 10:
                            text_cols.append(col)
                
                if text_cols:
                    text_features = text_op.generate_features(df, text_cols=text_cols, auto_register=auto_register, origin=origin, dataset_db_path=self.config.get('dataset_db_path'))
                    result_features.update(text_features)
                    logger.info(f"Added {len(text_features)} text features")
                
        except Exception as e:
            logger.error(f"Error generating generic features: {e}")
        
        # Convert to DataFrame
        if result_features:
            result_df = pd.DataFrame(result_features)
        else:
            # Return DataFrame with dummy column to avoid DuckDB empty table errors
            result_df = pd.DataFrame({'_placeholder': [0] * len(df)}, index=df.index)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(result_df.columns)} generic features in {elapsed_time:.2f}s")
        
        return result_df
    
    def generate_custom_features(self, df: pd.DataFrame, dataset_name: str, check_signal: bool = True, auto_register: bool = True, origin: str = 'custom') -> pd.DataFrame:
        """
        Generate custom domain-specific features using new modular architecture.
        Used by dataset registration process.
        """
        # Use new pipeline if enabled and available
        if self._use_new_pipeline and self._adapter:
            return self._adapter.generate_custom_features_new(df, dataset_name, check_signal)
        
        logger.info(f"🎯 Generating custom domain features for: {dataset_name}")
        start_time = time.time()
        
        result_features = {}
        
        try:
            # Auto-detect custom domain module based on dataset name
            dataset_name_clean = dataset_name.lower().replace('-', '_').replace(' ', '_')
            
            # Check if custom domain module exists for this dataset
            from pathlib import Path
            custom_modules_dir = Path(__file__).parent / 'features' / 'custom'
            module_file = f"{dataset_name_clean}.py"
            
            module_name = None
            if (custom_modules_dir / module_file).exists():
                module_name = module_file[:-3]  # Remove .py extension
            
            if not module_name:
                logger.info(f"No custom domain module found for dataset '{dataset_name}', skipping custom operations")
                return pd.DataFrame()
            
            # Import custom module
            custom_module = importlib.import_module(f'.features.custom.{module_name}', package='src')
            
            if hasattr(custom_module, 'CustomFeatureOperations'):
                custom_instance = custom_module.CustomFeatureOperations()
                custom_instance._check_signal = check_signal
                
                # Generate all custom features
                custom_features = custom_instance.generate_all_features(df, auto_register=auto_register, origin=origin, dataset_db_path=self.config.get('dataset_db_path'))
                
                if isinstance(custom_features, dict):
                    result_features.update(custom_features)
                    logger.info(f"Generated {len(custom_features)} custom features using new architecture")
                else:
                    logger.warning("Custom features returned non-dict result")
            else:
                logger.warning(f"No CustomFeatureOperations class found in {module_name}")
                
        except ImportError:
            logger.info(f"No custom module found for dataset '{dataset_name}' - using generic operations only")
        except Exception as e:
            logger.error(f"Error generating custom features for {dataset_name}: {e}")
        
        # Convert to DataFrame
        if result_features:
            result_df = pd.DataFrame(result_features)
        else:
            # Return DataFrame with dummy column to avoid DuckDB empty table errors
            result_df = pd.DataFrame({'_placeholder': [0] * len(df)}, index=df.index)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(result_df.columns)} custom features in {elapsed_time:.2f}s")
        
        return result_df
    
    def clear_feature_catalog_cache(self) -> None:
        """
        Clear the feature catalog cache.
        
        Use this method when the feature catalog has been updated and you need
        to force a refresh of cached column names.
        """
        with self._cache_lock:
            cache_size = len(self._feature_catalog_cache)
            self._feature_catalog_cache.clear()
            logger.info(f"Cleared feature catalog cache ({cache_size} entries)")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'feature_cache'):
            self.feature_cache.clear()
        if hasattr(self, '_feature_catalog_cache'):
            self._feature_catalog_cache.clear()
        logger.debug("Feature space cleanup completed")
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get feature metadata if using new pipeline."""
        if self._use_new_pipeline and self._adapter:
            return self._adapter.get_feature_metadata()
        return {}
    
    def generate_all_features_pipeline(self, df: pd.DataFrame, dataset_name: str, 
                                     target_column: Optional[str] = None,
                                     id_column: Optional[str] = None) -> pd.DataFrame:
        """Generate all features using new pipeline (if enabled)."""
        if self._use_new_pipeline and self._adapter:
            return self._adapter.generate_all_features_pipeline(
                df, dataset_name, target_column, id_column
            )
        else:
            raise ValueError("New pipeline not enabled. Set use_new_pipeline: true in config")
    
    def _get_node_cache_key(self, node) -> str:
        """Generate cache key for node."""
        if hasattr(node, 'applied_operations'):
            operations = sorted(node.applied_operations)
            return f"node_{'_'.join(operations)}"
        return "node_empty"