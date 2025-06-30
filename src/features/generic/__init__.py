"""
Generic Feature Operations

Universal feature operations that work across all domains.
Provides dynamic discovery and metadata for feature operations.
"""

from typing import Dict, List, Any, Optional
import logging

from .statistical import StatisticalFeatures
from .polynomial import PolynomialFeatures
from .binning import BinningFeatures
from .ranking import RankingFeatures
from .temporal import TemporalFeatures
from .text import TextFeatures
from .categorical import CategoricalFeatures

logger = logging.getLogger(__name__)

# Operation metadata with categories and output patterns
OPERATION_METADATA = {
    'statistical_aggregations': {
        'class': StatisticalFeatures,
        'category': 'statistical',
        'description': 'Statistical aggregations by categorical features',
        'output_patterns': ['_mean_by_', '_std_by_', '_dev_from_', '_count_by_', '_min_by_', '_max_by_', '_norm_by_'],
        'is_generic': True,
        'requires_categorical': True,
        'requires_numeric': True,
    },
    'polynomial_features': {
        'class': PolynomialFeatures,
        'category': 'polynomial',
        'description': 'Polynomial feature transformations',
        'output_patterns': ['_squared', '_cubed', '_sqrt', '_log', '_exp'],
        'is_generic': True,
        'requires_categorical': False,
        'requires_numeric': True,
    },
    'binning_features': {
        'class': BinningFeatures,
        'category': 'binning',
        'description': 'Quantile-based binning transformations',
        'output_patterns': ['_bin_', '_quantile_'],
        'is_generic': True,
        'requires_categorical': False,
        'requires_numeric': True,
    },
    'ranking_features': {
        'class': RankingFeatures,
        'category': 'ranking',
        'description': 'Rank-based transformations',
        'output_patterns': ['_rank', '_percentile_'],
        'is_generic': True,
        'requires_categorical': False,
        'requires_numeric': True,
    },
    'temporal_features': {
        'class': TemporalFeatures,
        'category': 'temporal',
        'description': 'Time-based feature engineering',
        'output_patterns': ['_year', '_month', '_day', '_hour', '_dayofweek'],
        'is_generic': True,
        'requires_categorical': False,
        'requires_numeric': False,
    },
    'text_features': {
        'class': TextFeatures,
        'category': 'text',
        'description': 'Text processing and NLP features',
        'output_patterns': ['_length', '_word_count', '_char_count', '_upper_ratio'],
        'is_generic': True,
        'requires_categorical': False,
        'requires_numeric': False,
    },
    'categorical_features': {
        'class': CategoricalFeatures,
        'category': 'categorical',
        'description': 'Categorical encoding and transformations',
        'output_patterns': ['_encoded', '_frequency', '_target_mean'],
        'is_generic': True,
        'requires_categorical': True,
        'requires_numeric': False,
    },
}

# Legacy compatibility: Collect all generic operations
GENERIC_OPERATIONS = {
    op_name: metadata['class'] 
    for op_name, metadata in OPERATION_METADATA.items()
}


def get_operation_metadata(operation_name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific operation.
    
    Args:
        operation_name: Name of the operation
        
    Returns:
        Operation metadata dictionary or None if not found
    """
    return OPERATION_METADATA.get(operation_name)


def get_operations_by_category(category: str) -> List[str]:
    """
    Get all operation names in a specific category.
    
    Args:
        category: Category name (e.g., 'statistical', 'polynomial')
        
    Returns:
        List of operation names in the category
    """
    return [
        op_name for op_name, metadata in OPERATION_METADATA.items()
        if metadata['category'] == category
    ]


def detect_operation_from_feature_name(feature_name: str) -> Optional[str]:
    """
    Detect which operation likely generated a feature based on naming patterns.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Operation name or None if no pattern matches
    """
    feature_lower = feature_name.lower()
    
    # Check each operation's output patterns
    for op_name, metadata in OPERATION_METADATA.items():
        for pattern in metadata['output_patterns']:
            if pattern in feature_lower:
                return op_name
    
    return None


def get_applicable_operations(has_categorical: bool = True, has_numeric: bool = True) -> List[str]:
    """
    Get operations that can be applied given the data characteristics.
    
    Args:
        has_categorical: Whether dataset has categorical columns
        has_numeric: Whether dataset has numeric columns
        
    Returns:
        List of applicable operation names
    """
    applicable = []
    
    for op_name, metadata in OPERATION_METADATA.items():
        # Check if operation requirements are met
        if metadata.get('requires_categorical', False) and not has_categorical:
            continue
        if metadata.get('requires_numeric', False) and not has_numeric:
            continue
        
        applicable.append(op_name)
    
    return applicable


def create_operation_instance(operation_name: str, **kwargs):
    """
    Create an instance of a feature operation.
    
    Args:
        operation_name: Name of the operation
        **kwargs: Arguments to pass to the operation constructor
        
    Returns:
        Operation instance or None if operation not found
    """
    metadata = get_operation_metadata(operation_name)
    if not metadata:
        logger.error(f"Unknown operation: {operation_name}")
        return None
    
    operation_class = metadata['class']
    try:
        return operation_class(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create operation {operation_name}: {e}")
        return None


def validate_operation_registry():
    """
    Validate that all operations in the registry are properly configured.
    
    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []
    
    for op_name, metadata in OPERATION_METADATA.items():
        # Check required fields
        required_fields = ['class', 'category', 'description', 'output_patterns', 'is_generic']
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Operation {op_name} missing required field: {field}")
        
        # Check class exists and is callable
        if 'class' in metadata:
            op_class = metadata['class']
            if not callable(op_class):
                errors.append(f"Operation {op_name} class is not callable")
        
        # Check output patterns is a list
        if 'output_patterns' in metadata:
            patterns = metadata['output_patterns']
            if not isinstance(patterns, list):
                errors.append(f"Operation {op_name} output_patterns must be a list")
    
    return errors


__all__ = [
    'StatisticalFeatures',
    'PolynomialFeatures',
    'BinningFeatures',
    'RankingFeatures',
    'TemporalFeatures',
    'TextFeatures',
    'CategoricalFeatures',
    'GENERIC_OPERATIONS',
    'OPERATION_METADATA',
    'get_operation_metadata',
    'get_operations_by_category',
    'detect_operation_from_feature_name',
    'get_applicable_operations',
    'create_operation_instance',
    'validate_operation_registry'
]