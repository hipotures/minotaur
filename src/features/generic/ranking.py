"""
Ranking Feature Operations

Ranking and percentile features including:
- Rank within dataset
- Percentile ranks
- Normalized ranks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from ..base import GenericFeatureOperation

logger = logging.getLogger(__name__)


class RankingFeatures(GenericFeatureOperation):
    """Generate ranking and percentile features."""
    
    def get_operation_name(self) -> str:
        return "Ranking Features"
    
    def _generate_features_impl(self, df: pd.DataFrame,
                               numeric_cols: Optional[List[str]] = None,
                               method: str = 'average',
                               **kwargs) -> Dict[str, pd.Series]:
        """
        Generate ranking features for numeric columns.
        
        Args:
            df: Input dataframe
            numeric_cols: Numeric columns to rank (auto-detected if None)
            method: Ranking method ('average', 'min', 'max', 'first', 'dense')
            
        Returns:
            Dictionary of feature name -> pandas Series
        """
        features = {}
        
        # Auto-detect numeric columns if not provided
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Filter out forbidden columns during auto-detection
            numeric_cols = self._filter_forbidden_columns(numeric_cols)
        else:
            # Validate provided columns
            numeric_cols = [col for col in numeric_cols 
                           if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols:
            logger.warning("No numeric columns found for ranking")
            return features
        
        # Generate ranking features for each column
        for col in numeric_cols:
            try:
                col_features = self._generate_column_ranks(df[col], col, method)
                features.update(col_features)
            except Exception as e:
                logger.error(f"Error creating ranking features for column '{col}': {e}")
                continue
        
        return features
    
    def _generate_column_ranks(self, series: pd.Series, 
                              col_name: str, 
                              method: str) -> Dict[str, pd.Series]:
        """Generate ranking features for a single column."""
        features = {}
        
        # Clean the data
        clean_data = self._clean_numeric_data(series)
        
        # Check if we have any valid data to rank
        if clean_data.isna().all() or clean_data.nunique() == 1:
            logger.warning(f"Column '{col_name}' has no variation for ranking")
            # Return constant features
            features[f'{col_name.lower()}_rank'.lower()] = pd.Series(1, index=series.index)
            features[f'{col_name.lower()}_percentile'.lower()] = pd.Series(0.5, index=series.index)
            return features
        
        # Rank within entire dataset
        feature_name = f'{col_name.lower()}_rank'.lower()
        with self._time_feature(feature_name, features):
            rank_values = clean_data.rank(method=method, na_option='bottom')
            features[feature_name] = rank_values
        
        # Dense rank (no gaps between ranks)
        feature_name = f'{col_name.lower()}_dense_rank'.lower()
        with self._time_feature(feature_name, features):
            dense_rank_values = clean_data.rank(method='dense', na_option='bottom')
            features[feature_name] = dense_rank_values
        
        # Percentile rank (0 to 1)
        feature_name = f'{col_name.lower()}_percentile'.lower()
        with self._time_feature(feature_name, features):
            percentile_values = clean_data.rank(pct=True, na_option='bottom')
            features[feature_name] = percentile_values.clip(0, 1)
        
        # Normalized rank (0 to 1, but based on rank not percentile)
        feature_name = f'{col_name.lower()}_norm_rank'.lower()
        with self._time_feature(feature_name, features):
            max_rank = rank_values.max()
            if max_rank > 1:
                norm_rank = (rank_values - 1) / (max_rank - 1)
            else:
                norm_rank = pd.Series(0.5, index=series.index)
            features[feature_name] = norm_rank.clip(0, 1)
        
        # Quantile groups (divide into quartiles, deciles, etc.)
        feature_name = f'{col_name.lower()}_quartile'.lower()
        with self._time_feature(feature_name, features):
            quartile_values = pd.qcut(clean_data, q=4, labels=False, duplicates='drop')
            features[feature_name] = quartile_values.fillna(-1).astype('category')
        
        feature_name = f'{col_name.lower()}_decile'.lower()
        with self._time_feature(feature_name, features):
            # Only create deciles if we have enough unique values
            if clean_data.nunique() >= 10:
                decile_values = pd.qcut(clean_data, q=10, labels=False, duplicates='drop')
                features[feature_name] = decile_values.fillna(-1).astype('category')
        
        # Top/Bottom indicators
        feature_name = f'{col_name.lower()}_is_top10pct'.lower()
        with self._time_feature(feature_name, features):
            features[feature_name] = (percentile_values >= 0.9).astype(int)
        
        feature_name = f'{col_name.lower()}_is_bottom10pct'.lower()
        with self._time_feature(feature_name, features):
            features[feature_name] = (percentile_values <= 0.1).astype(int)
        
        # Relative position features
        feature_name = f'{col_name.lower()}_above_median'.lower()
        with self._time_feature(feature_name, features):
            median_val = clean_data.median()
            features[feature_name] = (clean_data > median_val).astype(int)
        
        feature_name = f'{col_name.lower()}_zscore_rank'.lower()
        with self._time_feature(feature_name, features):
            # Z-score normalized rank
            mean_val = clean_data.mean()
            std_val = clean_data.std()
            if std_val > 0:
                zscore = (clean_data - mean_val) / std_val
                # Convert z-score to rank-like value (0 to 1)
                # Using sigmoid-like transformation
                zscore_rank = 1 / (1 + np.exp(-zscore / 2))
                features[feature_name] = zscore_rank
            else:
                features[feature_name] = pd.Series(0.5, index=series.index)
        
        return features
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input has numeric columns."""
        if not super().validate_input(df):
            return False
        
        # Check for at least one numeric column
        has_numeric = len(df.select_dtypes(include=[np.number]).columns) > 0
        
        if not has_numeric:
            logger.warning("No numeric columns found for ranking")
            return False
        
        return True


# Convenience function for backward compatibility
def get_ranking_features(df: pd.DataFrame, 
                        numeric_cols: List[str],
                        check_signal: bool = True) -> Dict[str, pd.Series]:
    """
    Legacy function for ranking features.
    
    Args:
        df: Input dataframe
        numeric_cols: Numeric columns to rank
        check_signal: Whether to check for signal and discard no-signal features
        
    Returns:
        Dictionary of feature name -> pandas Series
    """
    generator = RankingFeatures()
    generator._check_signal = check_signal
    return generator.generate_features(df, numeric_cols=numeric_cols)