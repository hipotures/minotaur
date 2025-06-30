"""
MCTS Feature Space Manager

Manages feature operations for MCTS exploration by loading from src/features/ modules.
Replaces hardcoded operations with dynamic loading from generic and custom feature modules.
"""

import time
import logging
import importlib
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
        try:
            from .features.generic import statistical, polynomial, binning, ranking
            
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
                    category='feature_transformations',
                    description=f'Polynomial features (degree {self.generic_params.get("polynomial_degree", 2)})',
                    dependencies=[],  # Works with any numeric columns
                    computational_cost=0.4,
                    output_features=[]
                )
            
            # Binning features
            if self.generic_operations_config.get('binning_features', True):
                self.operations['binning_features'] = FeatureOperation(
                    name='binning_features',
                    category='feature_transformations',
                    description=f'Binning of numerical features ({self.generic_params.get("binning_bins", 5)} bins)',
                    dependencies=[],  # Works with any numeric columns
                    computational_cost=0.3,
                    output_features=[]
                )
            
            # Ranking features
            if self.generic_operations_config.get('ranking_features', True):
                self.operations['ranking_features'] = FeatureOperation(
                    name='ranking_features',
                    category='feature_transformations',
                    description='Ranking features for numeric columns',
                    dependencies=[],  # Works with any numeric columns
                    computational_cost=0.2,
                    output_features=[]
                )
            
            logger.info(f"Loaded {len([op for op in self.operations.values() if 'transformations' in op.category or 'aggregations' in op.category])} generic operations")
            
        except ImportError as e:
            logger.warning(f"Could not load generic operations: {e}")
        except Exception as e:
            logger.error(f"Error loading generic operations: {e}")
    
    def _load_custom_operations(self):
        """Load custom operations based on dataset name."""
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
            if (custom_modules_dir / module_file).exists():
                module_name = module_file[:-3]  # Remove .py extension
            
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
        
        for op_name, operation in self.operations.items():
            # Check if operation can be applied
            if operation.can_apply(current_features):
                # Check if already applied in this path
                if op_name not in getattr(node, 'applied_operations', []):
                    # Apply category filtering
                    if not self.enabled_categories or operation.category in self.enabled_categories:
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
        
        logger.debug(f"Found {len(available_ops)} available operations for node at depth {getattr(node, 'depth', 0)}")
        return available_ops
    
    def get_feature_columns_for_node(self, node) -> List[str]:
        """
        Get list of feature column names for a given MCTS node.
        Uses actual columns available in train_features table.
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
            return list(getattr(node, 'base_features', [])) if hasattr(node, 'base_features') else []
        
        # Start with base features
        base_features = list(getattr(node, 'base_features', [])) if hasattr(node, 'base_features') else []
        selected_columns = base_features.copy()
        
        # For each applied operation, add columns that match expected patterns
        for operation_name in getattr(node, 'applied_operations', []):
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
                    if '_binned' in col or col.endswith('_bin') or '_qbin_' in col:
                        selected_columns.append(col)
                        
            elif operation_name == 'ranking_features':
                # Add ranking features
                rank_patterns = ['_rank', '_percentile', '_quartile', '_decile']
                for col in all_columns:
                    if any(pattern in col for pattern in rank_patterns):
                        selected_columns.append(col)
            
            else:
                # For custom operations, add columns that might be generated
                # This is a heuristic - we add columns that contain the operation name
                for col in all_columns:
                    if operation_name.replace('_', '') in col.replace('_', '').lower():
                        selected_columns.append(col)
        
        # Remove duplicates and ensure we don't include forbidden columns
        selected_columns = list(set(selected_columns))
        
        # Remove forbidden columns (ID, target, etc.) - they will be added back by evaluator
        forbidden = [self.id_column, self.target_column] + self.ignore_columns
        selected_columns = [col for col in selected_columns if col not in forbidden]
        
        return selected_columns
    
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
        """Get expected output columns for an operation (heuristic)."""
        # This is a simplified heuristic - in practice, we'd need to know
        # what columns each operation actually produces
        all_columns = self._get_available_columns_from_db()
        
        if operation_name == 'statistical_aggregations':
            return {col for col in all_columns if any(pattern in col for pattern in ['_mean_by_', '_std_by_'])}
        elif operation_name == 'polynomial_features':
            return {col for col in all_columns if any(col.endswith(suffix) for suffix in ['_squared', '_log'])}
        elif operation_name == 'binning_features':
            return {col for col in all_columns if '_binned' in col or '_qbin_' in col}
        elif operation_name == 'ranking_features':
            return {col for col in all_columns if any(pattern in col for pattern in ['_rank', '_percentile'])}
        else:
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
        """Determine feature category from name patterns."""
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
    
    def generate_generic_features(self, df: pd.DataFrame, check_signal: bool = True) -> pd.DataFrame:
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
                
                stat_features = stat_op.generate_features(df, groupby_cols=groupby_cols, agg_cols=aggregate_cols)
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
                
                poly_features = poly_op.generate_features(df, numeric_cols=numeric_cols, degree=degree)
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
                
                bin_features = bin_op.generate_features(df, numeric_cols=numeric_cols, n_bins=n_bins)
                result_features.update(bin_features)
                logger.info(f"Added {len(bin_features)} binning features")
            
            # Ranking features
            if self.generic_operations_config.get('ranking_features', True):
                rank_op = ranking.RankingFeatures()
                rank_op._check_signal = check_signal
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Filter out forbidden columns
                numeric_cols = self._filter_forbidden_columns(numeric_cols)
                
                rank_features = rank_op.generate_features(df, numeric_cols=numeric_cols)
                result_features.update(rank_features)
                logger.info(f"Added {len(rank_features)} ranking features")
            
            # Categorical features
            if self.generic_operations_config.get('categorical_features', True):
                cat_op = categorical.CategoricalFeatures()
                cat_op._check_signal = check_signal
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                categorical_cols = self._filter_forbidden_columns(categorical_cols)
                
                if categorical_cols:
                    cat_features = cat_op.generate_features(df, categorical_cols=categorical_cols)
                    result_features.update(cat_features)
                    logger.info(f"Added {len(cat_features)} categorical features")
            
            # Temporal features
            if self.generic_operations_config.get('temporal_features', True):
                temp_op = temporal.TemporalFeatures()
                temp_op._check_signal = check_signal
                datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                datetime_cols = self._filter_forbidden_columns(datetime_cols)
                
                if datetime_cols:
                    temp_features = temp_op.generate_features(df, datetime_cols=datetime_cols)
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
                    text_features = text_op.generate_features(df, text_cols=text_cols)
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
    
    def generate_custom_features(self, df: pd.DataFrame, dataset_name: str, check_signal: bool = True) -> pd.DataFrame:
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
                custom_features = custom_instance.generate_all_features(df)
                
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
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'feature_cache'):
            self.feature_cache.clear()
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