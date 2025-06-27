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

from .mock_evaluator import MockAutoGluonEvaluator
from .timing import TimingCollector, initialize_timing, get_timing_collector, timed, timing_context, performance_monitor
from .analytics import AnalyticsGenerator, generate_quick_report
from .data_utils import DataManager

__all__ = [
    'FeatureDiscoveryDB',
    'MCTSEngine', 
    'FeatureNode',
    'FeatureSpace',
    'FeatureOperation',
    'AutoGluonEvaluator',
    'MockAutoGluonEvaluator',
    'TimingCollector',
    'initialize_timing',
    'get_timing_collector',
    'timed',
    'timing_context', 
    'performance_monitor',
    'AnalyticsGenerator',
    'generate_quick_report',
    'DataManager',
    'AUTOGLUON_AVAILABLE'
]