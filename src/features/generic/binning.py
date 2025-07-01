"""
Binning Feature Operations

Discretization and binning features including:
- Quantile-based binning
- Equal-width binning
- Custom binning strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from ..base import GenericFeatureOperation

logger = logging.getLogger(__name__)


class BinningFeatures(GenericFeatureOperation):
    """Generate binning and discretization features."""
    
    def _init_parameters(self):
        """Initialize default parameters."""
        self.default_n_bins = 5
        self.min_unique_values = 2  # Minimum unique values to create bins
    
    def get_operation_name(self) -> str:
        return "binning_features"
    
    def _generate_features_impl(self, df: pd.DataFrame,
                               numeric_cols: Optional[List[str]] = None,
                               n_bins: int = None,
                               strategy: str = 'quantile',
                               **kwargs) -> Dict[str, pd.Series]:
        """
        Generate binning features for numeric columns.
        
        Args:
            df: Input dataframe
            numeric_cols: Numeric columns to bin (auto-detected if None)
            n_bins: Number of bins (default: 5)
            strategy: Binning strategy ('quantile', 'uniform', or 'kmeans')
            
        Returns:
            Dictionary of feature name -> pandas Series
        """
        features = {}
        
        if n_bins is None:
            n_bins = self.default_n_bins
        
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
            logger.warning("No numeric columns found for binning")
            return features
        
        # Generate binning features for each column
        for col in numeric_cols:
            try:
                col_features = self._generate_column_bins(df[col], col, n_bins, strategy)
                features.update(col_features)
            except Exception as e:
                logger.error(f"Error creating binning features for column '{col}': {e}")
                continue
        
        return features
    
    def _generate_column_bins(self, series: pd.Series, 
                             col_name: str, 
                             n_bins: int,
                             strategy: str) -> Dict[str, pd.Series]:
        """Generate binning features for a single column."""
        features = {}
        
        # Clean the data first
        clean_data = self._clean_numeric_data(series)
        
        # Check if we have enough unique values for binning
        unique_values = clean_data.nunique()
        if unique_values < self.min_unique_values:
            logger.warning(f"Column '{col_name}' has only {unique_values} unique values, skipping binning")
            return features
        
        # Adjust bins if necessary
        n_bins_adjusted = min(n_bins, unique_values)
        if n_bins_adjusted < n_bins:
            logger.debug(f"Adjusted bins for '{col_name}' from {n_bins} to {n_bins_adjusted}")
        
        # Convert to float32 for compatibility if needed
        if clean_data.dtype == np.float16:
            clean_data = clean_data.astype(np.float32)
        
        # Generate bins based on strategy
        if strategy == 'quantile':
            feature_name = f'{col_name.lower()}_qbin_{n_bins}'.lower()
            with self._time_feature(feature_name, features):
                binned = self._quantile_binning(clean_data, n_bins_adjusted)
                features[feature_name] = binned
        
        elif strategy == 'uniform':
            feature_name = f'{col_name.lower()}_ubin_{n_bins}'.lower()
            with self._time_feature(feature_name, features):
                binned = self._uniform_binning(clean_data, n_bins_adjusted)
                features[feature_name] = binned
        
        else:
            # Default to quantile binning
            feature_name = f'{col_name.lower()}_bin_{n_bins}'.lower()
            with self._time_feature(feature_name, features):
                binned = self._quantile_binning(clean_data, n_bins_adjusted)
                features[feature_name] = binned
        
        # Also create binary features for each bin
        if n_bins_adjusted <= 10:  # Limit to avoid too many features
            bin_dummies = self._create_bin_dummies(binned, col_name, n_bins)
            for dummy_name, dummy_series in bin_dummies.items():
                with self._time_feature(dummy_name, features):
                    features[dummy_name] = dummy_series
        
        return features
    
    def _quantile_binning(self, data: pd.Series, n_bins: int) -> pd.Series:
        """Perform quantile-based binning."""
        try:
            # Try quantile-based binning first
            binned = pd.qcut(data, q=n_bins, labels=False, duplicates='drop')
            return binned.fillna(-1).astype('category')
            
        except Exception as quantile_error:
            logger.debug(f"Quantile binning failed: {quantile_error}, trying equal-width bins")
            
            # Fallback to equal-width binning
            return self._uniform_binning(data, n_bins)
    
    def _uniform_binning(self, data: pd.Series, n_bins: int) -> pd.Series:
        """Perform equal-width binning."""
        try:
            binned = pd.cut(data, bins=n_bins, labels=False, duplicates='drop')
            return binned.fillna(-1).astype('category')
            
        except Exception as cut_error:
            logger.error(f"Both binning methods failed: {cut_error}")
            # Ultimate fallback: create a constant feature
            return pd.Series(0, index=data.index, dtype='category')
    
    def _create_bin_dummies(self, binned: pd.Series, 
                           col_name: str, 
                           n_bins: int) -> Dict[str, pd.Series]:
        """Create binary dummy features for each bin."""
        dummies = {}
        
        # Get unique bin values (excluding -1 which represents missing)
        unique_bins = [b for b in binned.unique() if b != -1]
        
        for bin_val in unique_bins:
            dummy_name = f'{col_name.lower()}_is_bin_{int(bin_val)}'.lower()
            dummies[dummy_name] = (binned == bin_val).astype(int)
        
        return dummies
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input has numeric columns."""
        if not super().validate_input(df):
            return False
        
        # Check for at least one numeric column
        has_numeric = len(df.select_dtypes(include=[np.number]).columns) > 0
        
        if not has_numeric:
            logger.warning("No numeric columns found for binning")
            return False
        
        return True


# Convenience function for backward compatibility
def get_binning_features(df: pd.DataFrame, 
                        numeric_cols: List[str], 
                        n_bins: int = 5,
                        check_signal: bool = True) -> Dict[str, pd.Series]:
    """
    Legacy function for binning features.
    
    Args:
        df: Input dataframe
        numeric_cols: Numeric columns to bin
        n_bins: Number of bins
        check_signal: Whether to check for signal and discard no-signal features
        
    Returns:
        Dictionary of feature name -> pandas Series
    """
    generator = BinningFeatures()
    generator._check_signal = check_signal
    return generator.generate_features(df, numeric_cols=numeric_cols, n_bins=n_bins)