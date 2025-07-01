"""
Base Class for Custom Feature Operations

Provides common functionality for domain-specific feature engineering.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..base import CustomFeatureOperation

logger = logging.getLogger(__name__)


class BaseDomainFeatures(CustomFeatureOperation):
    """Base class for domain-specific feature operations."""
    
    def __init__(self, domain_name: str):
        """Initialize with domain name."""
        super().__init__(domain_name)
        self._operation_registry = {}
        self._register_operations()
    
    def _register_operations(self):
        """
        Register all available operations for this domain.
        Should be overridden by subclasses.
        """
        pass
    
    def get_available_operations(self) -> List[str]:
        """Get list of available operations for this domain."""
        return list(self._operation_registry.keys())
    
    def generate_all_features(self, df: pd.DataFrame, auto_register: bool = True, origin: str = 'custom', **kwargs) -> Dict[str, pd.Series]:
        """
        Generate all available features for this domain.
        
        Args:
            df: Input dataframe
            auto_register: Whether to auto-register features (default True, set False for test data)
            origin: Feature origin type ('train', 'generic', 'custom') for catalog registration
            **kwargs: Additional parameters to pass to individual operations
            
        Returns:
            Dictionary of all generated features
        """
        all_features = {}
        
        # Remove auto_register, origin, and dataset_db_path from kwargs before passing to individual operations
        # Keep dataset_db_path for auto-registration
        operation_kwargs = kwargs.copy()
        operation_kwargs.pop('auto_register', None)
        operation_kwargs.pop('origin', None)
        dataset_db_path = operation_kwargs.pop('dataset_db_path', None)
        
        for operation_name, operation_func in self._operation_registry.items():
            try:
                logger.info(f"Generating {operation_name} features for {self.domain_name}")
                features = operation_func(df, **operation_kwargs)
                all_features.update(features)
            except Exception as e:
                logger.error(f"Error generating {operation_name} features: {e}")
                continue
        
        self.log_timing_summary(f"{self.domain_name} - All Features")
        
        # Handle auto-registration if enabled and features were generated
        logger.debug(f"Auto-registration check: enabled={self._auto_registration_enabled}, features={len(all_features)}, auto_register={auto_register}")
        if self._auto_registration_enabled and all_features and auto_register:
            logger.debug(f"Attempting auto-registration for {len(all_features)} custom features")
            self._auto_register_custom_operation_metadata(all_features, origin=origin, dataset_db_path=dataset_db_path, **operation_kwargs)
        
        return all_features
    
    def generate_specific_features(self, df: pd.DataFrame, 
                                  operation_name: str, 
                                  **kwargs) -> Dict[str, pd.Series]:
        """
        Generate features for a specific operation.
        
        Args:
            df: Input dataframe
            operation_name: Name of the operation to execute
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of generated features
        """
        if operation_name not in self._operation_registry:
            logger.error(f"Unknown operation: {operation_name}")
            return {}
        
        try:
            features = self._operation_registry[operation_name](df, **kwargs)
            self.log_timing_summary(f"{self.domain_name} - {operation_name}")
            return features
        except Exception as e:
            logger.error(f"Error generating {operation_name} features: {e}")
            return {}
    
    def get_operation_name(self) -> str:
        """Return the name of this feature operation."""
        return f"{self.domain_name} Custom Features"
    
    def _generate_features_impl(self, df: pd.DataFrame, auto_register: bool = True, origin: str = 'custom', **kwargs) -> Dict[str, pd.Series]:
        """Implementation that generates all features by default."""
        return self.generate_all_features(df, auto_register=auto_register, origin=origin, **kwargs)
    
    # Utility methods for common domain operations
    
    def _create_interaction_feature(self, df: pd.DataFrame, 
                                   col1: str, 
                                   col2: str, 
                                   operation: str = 'multiply',
                                   features_dict: dict = None) -> Optional[pd.Series]:
        """Create interaction between two columns."""
        if col1 not in df.columns or col2 not in df.columns:
            return None
        
        feature_name = f"{col1}_{operation}_{col2}"
        
        with self._time_feature(feature_name, features_dict):
            if operation == 'multiply':
                result = df[col1] * df[col2]
            elif operation == 'add':
                result = df[col1] + df[col2]
            elif operation == 'subtract':
                result = df[col1] - df[col2]
            elif operation == 'divide':
                result = self._safe_divide(df[col1], df[col2])
            else:
                logger.warning(f"Unknown operation: {operation}")
                return None
            
            if features_dict is not None:
                features_dict[feature_name] = result
            return result
    
    def _create_ratio_feature(self, df: pd.DataFrame, 
                             numerator_col: str, 
                             denominator_col: str, 
                             name_suffix: str = "ratio",
                             features_dict: dict = None) -> Optional[pd.Series]:
        """Create ratio between two columns."""
        if numerator_col not in df.columns or denominator_col not in df.columns:
            return None
        
        feature_name = f"{numerator_col}_per_{denominator_col}"
        if name_suffix != "ratio":
            feature_name = f"{numerator_col}_{name_suffix}_{denominator_col}"
        
        with self._time_feature(feature_name, features_dict):
            result = self._safe_divide(df[numerator_col], df[denominator_col])
            if features_dict is not None:
                features_dict[feature_name] = result
            return result
    
    def _create_boolean_feature(self, df: pd.DataFrame, 
                               condition: pd.Series, 
                               feature_name: str,
                               features_dict: dict = None) -> pd.Series:
        """Create boolean feature from condition."""
        with self._time_feature(feature_name, features_dict):
            result = condition.astype(int)
            if features_dict is not None:
                features_dict[feature_name] = result
            return result
    
    def _extract_from_text(self, df: pd.DataFrame, 
                          text_col: str, 
                          pattern: str, 
                          feature_name: str,
                          extract_group: int = 0,
                          features_dict: dict = None) -> Optional[pd.Series]:
        """Extract pattern from text column."""
        if text_col not in df.columns:
            return None
        
        with self._time_feature(feature_name, features_dict):
            result = df[text_col].str.extract(pattern, expand=False)
            if features_dict is not None:
                features_dict[feature_name] = result
            return result