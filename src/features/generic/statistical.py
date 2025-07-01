"""
Statistical Feature Operations

Statistical aggregations by categorical features including:
- Group-based means, standard deviations
- Deviations from group statistics
- Count and frequency encodings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from ..base import GenericFeatureOperation

logger = logging.getLogger(__name__)


class StatisticalFeatures(GenericFeatureOperation):
    """Generate statistical aggregation features."""
    
    def get_operation_name(self) -> str:
        return "Statistical Aggregations"
    
    def _generate_features_impl(self, df: pd.DataFrame, 
                               groupby_cols: Optional[List[str]] = None,
                               agg_cols: Optional[List[str]] = None,
                               **kwargs) -> Dict[str, pd.Series]:
        """
        Generate statistical aggregation features.
        
        Args:
            df: Input dataframe
            groupby_cols: Categorical columns to group by (auto-detected if None)
            agg_cols: Numeric columns to aggregate (auto-detected if None)
            
        Returns:
            Dictionary of feature name -> pandas Series
        """
        features = {}
        
        # Auto-detect columns if not provided
        if groupby_cols is None:
            groupby_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Filter out forbidden columns during auto-detection
            groupby_cols = self._filter_forbidden_columns(groupby_cols)
        if agg_cols is None:
            agg_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Filter out forbidden columns during auto-detection
            agg_cols = self._filter_forbidden_columns(agg_cols)
        
        # Generate features for each groupby/agg column combination
        for group_col in groupby_cols:
            if group_col not in df.columns:
                logger.warning(f"Groupby column '{group_col}' not found in dataframe")
                continue
                
            for agg_col in agg_cols:
                if agg_col not in df.columns:
                    logger.warning(f"Aggregation column '{agg_col}' not found in dataframe")
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[agg_col]):
                    logger.warning(f"Column '{agg_col}' is not numeric, skipping aggregation")
                    continue
                
                try:
                    # Generate features for this combination
                    combo_features = self._generate_group_features(df, group_col, agg_col)
                    features.update(combo_features)
                    
                except Exception as e:
                    logger.error(f"Error in statistical aggregation for {agg_col} by {group_col}: {e}")
                    continue
        
        return features
    
    def _generate_group_features(self, df: pd.DataFrame, 
                                group_col: str, 
                                agg_col: str) -> Dict[str, pd.Series]:
        """Generate all statistical features for a single group/agg combination."""
        features = {}
        
        # Clean the aggregation column first
        clean_col = self._clean_numeric_data(df[agg_col])
        
        # Create a temporary dataframe with cleaned data
        temp_df = df[[group_col]].copy()
        temp_df['clean_value'] = clean_col
        
        # Mean aggregation
        feature_name = f'{agg_col.lower()}_mean_by_{group_col.lower()}'.lower()
        with self._time_feature(feature_name, features):
            group_mean = temp_df.groupby(group_col)['clean_value'].transform('mean')
            features[feature_name] = group_mean
        
        # Standard deviation with safeguards
        feature_name = f'{agg_col.lower()}_std_by_{group_col.lower()}'.lower()
        with self._time_feature(feature_name, features):
            group_std = temp_df.groupby(group_col)['clean_value'].transform(
                lambda x: x.std() if len(x) > 1 else 0
            )
            features[feature_name] = group_std.fillna(0)
        
        # Deviation from group mean
        feature_name = f'{agg_col.lower()}_dev_from_{group_col}_mean'.lower()
        with self._time_feature(feature_name, features):
            deviation = clean_col - group_mean
            features[feature_name] = self._clean_numeric_data(deviation, fill_value=0)
        
        # Count encoding (number of samples in each group)
        feature_name = f'{agg_col.lower()}_count_by_{group_col.lower()}'.lower()
        with self._time_feature(feature_name, features):
            group_counts = temp_df.groupby(group_col)['clean_value'].transform('count')
            features[feature_name] = group_counts
        
        # Min/Max within groups
        feature_name = f'{agg_col.lower()}_min_by_{group_col.lower()}'.lower()
        with self._time_feature(feature_name, features):
            group_min = temp_df.groupby(group_col)['clean_value'].transform('min')
            features[feature_name] = group_min
        
        feature_name = f'{agg_col.lower()}_max_by_{group_col.lower()}'.lower()
        with self._time_feature(feature_name, features):
            group_max = temp_df.groupby(group_col)['clean_value'].transform('max')
            features[feature_name] = group_max
        
        # Normalized position within group (0 to 1)
        feature_name = f'{agg_col.lower()}_norm_by_{group_col.lower()}'.lower()
        with self._time_feature(feature_name, features):
            # Avoid division by zero
            range_val = group_max - group_min
            normalized = self._safe_divide(clean_col - group_min, range_val)
            features[feature_name] = normalized.clip(0, 1)
        
        return features
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input has both categorical and numeric columns."""
        if not super().validate_input(df):
            return False
        
        # Check for at least one categorical and one numeric column
        has_categorical = len(df.select_dtypes(include=['object', 'category']).columns) > 0
        has_numeric = len(df.select_dtypes(include=[np.number]).columns) > 0
        
        if not has_categorical:
            logger.warning("No categorical columns found for grouping")
            return False
        
        if not has_numeric:
            logger.warning("No numeric columns found for aggregation")
            return False
        
        return True


# Convenience function for backward compatibility
def get_statistical_aggregations(df: pd.DataFrame, 
                                groupby_cols: List[str], 
                                agg_cols: List[str],
                                check_signal: bool = True) -> Dict[str, pd.Series]:
    """
    Legacy function for statistical aggregations.
    
    Args:
        df: Input dataframe
        groupby_cols: Categorical columns to group by
        agg_cols: Numeric columns to aggregate
        check_signal: Whether to check for signal and discard no-signal features
        
    Returns:
        Dictionary of feature name -> pandas Series
    """
    generator = StatisticalFeatures()
    generator._check_signal = check_signal
    return generator.generate_features(df, groupby_cols=groupby_cols, agg_cols=agg_cols)