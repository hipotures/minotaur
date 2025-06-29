"""
MCTS-Driven Automated Feature Engineering System

This package implements Monte Carlo Tree Search for automated feature discovery
in the fertilizer prediction domain, using AutoGluon for fast evaluation.

Components:
- mcts_engine: Core MCTS algorithm implementation
- feature_space: Feature operation definitions and lazy generation
- autogluon_evaluator: Fast AutoGluon evaluation wrapper
- discovery_db: SQLite database interface for logging
- llm_generator: LLM-assisted feature generation
- analytics: Dashboard and reporting tools
"""

__version__ = "1.0.0"
__author__ = "MCTS Feature Discovery System"

from .discovery_db import FeatureDiscoveryDB
from .mcts_engine import MCTSEngine, FeatureNode
from .feature_space import FeatureSpace, FeatureOperation
try:
    from .autogluon_evaluator import AutoGluonEvaluator
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    AutoGluonEvaluator = None

from .timing import TimingCollector, initialize_timing, get_timing_collector, timed, timing_context, performance_monitor
from .analytics import AnalyticsGenerator, generate_quick_report
from .data_utils import DataManager, prepare_training_data
from .feature_cache import FeatureCacheManager
# Import from new feature system
from .features.custom.kaggle_s5e6 import CustomFeatureOperations as FertilizerS5E6Operations

# Create backward compatibility wrapper for GenericFeatureOperations
class GenericFeatureOperations:
    """Backward compatibility wrapper for generic feature operations."""
    
    @staticmethod
    def get_statistical_aggregations(df, groupby_cols, agg_cols):
        """Statistical aggregations by categorical features."""
        from .features.generic.statistical import get_statistical_aggregations
        return get_statistical_aggregations(df, groupby_cols, agg_cols)
    
    @staticmethod
    def get_polynomial_features(df, numeric_cols, degree=2):
        """Polynomial features for numeric columns."""
        from .features.generic.polynomial import get_polynomial_features
        return get_polynomial_features(df, numeric_cols, degree)
    
    @staticmethod
    def get_binning_features(df, numeric_cols, n_bins=5):
        """Binning features for numeric columns."""
        from .features.generic.binning import get_binning_features
        return get_binning_features(df, numeric_cols, n_bins)
    
    @staticmethod
    def get_ranking_features(df, numeric_cols):
        """Ranking features for numeric columns."""
        from .features.generic.ranking import get_ranking_features
        return get_ranking_features(df, numeric_cols)
from .logging_utils import setup_session_logging, set_session_context, clear_session_context
from .dataset_manager import DatasetManager

__all__ = [
    'FeatureDiscoveryDB',
    'MCTSEngine', 
    'FeatureNode',
    'FeatureSpace',
    'FeatureOperation',
    'AutoGluonEvaluator',
    'TimingCollector',
    'initialize_timing',
    'get_timing_collector',
    'timed',
    'timing_context', 
    'performance_monitor',
    'AnalyticsGenerator',
    'generate_quick_report',
    'DataManager',
    'prepare_training_data',
    'FeatureCacheManager',
    'GenericFeatureOperations',
    'FertilizerS5E6Operations',
    'setup_session_logging',
    'set_session_context',
    'clear_session_context',
    'DatasetManager',
    'AUTOGLUON_AVAILABLE'
]