"""
Polynomial Feature Operations

Polynomial transformations and interactions including:
- Squared, cubed features
- Square root and logarithm transformations
- Multiplicative interactions between features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from ..base import GenericFeatureOperation

logger = logging.getLogger(__name__)


class PolynomialFeatures(GenericFeatureOperation):
    """Generate polynomial and mathematical transformation features."""
    
    def _init_parameters(self):
        """Initialize default parameters."""
        self.default_degree = 2
        self.max_interaction_features = 3  # Limit interactions to first N features
    
    def get_operation_name(self) -> str:
        return "polynomial_features"
    
    def _generate_features_impl(self, df: pd.DataFrame,
                               numeric_cols: Optional[List[str]] = None,
                               degree: int = None,
                               include_interactions: bool = True,
                               **kwargs) -> Dict[str, pd.Series]:
        """
        Generate polynomial features for numeric columns.
        
        Args:
            df: Input dataframe
            numeric_cols: Numeric columns to transform (auto-detected if None)
            degree: Polynomial degree (default: 2)
            include_interactions: Whether to include interaction terms
            
        Returns:
            Dictionary of feature name -> pandas Series
        """
        features = {}
        
        if degree is None:
            degree = self.default_degree
        
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
            logger.warning("No numeric columns found for polynomial features")
            return features
        
        # Generate polynomial features for each column
        for col in numeric_cols:
            try:
                col_features = self._generate_column_polynomials(df[col], col, degree)
                features.update(col_features)
            except Exception as e:
                logger.error(f"Error creating polynomial features for column '{col}': {e}")
                continue
        
        # Generate interaction features if requested
        if include_interactions and len(numeric_cols) >= 2:
            try:
                interaction_features = self._generate_interactions(df, numeric_cols)
                features.update(interaction_features)
            except Exception as e:
                logger.error(f"Error creating interaction features: {e}")
        
        return features
    
    def _generate_column_polynomials(self, series: pd.Series, 
                                    col_name: str, 
                                    degree: int) -> Dict[str, pd.Series]:
        """Generate polynomial transformations for a single column."""
        features = {}
        
        # Clean the data
        clean_data = self._clean_numeric_data(series)
        
        # Squared features with overflow protection
        if degree >= 2:
            feature_name = f'{col_name.lower()}_squared'.lower()
            with self._time_feature(feature_name, features):
                # Scale down values before squaring to prevent overflow
                scaled_data = clean_data / 1000 if clean_data.abs().max() > 1000 else clean_data
                squared = scaled_data ** 2
                features[feature_name] = self._clean_numeric_data(squared)
        
        # Cubic features (selective and with stronger scaling)
        if degree >= 3:
            feature_name = f'{col_name.lower()}_cubed'.lower()
            with self._time_feature(feature_name, features):
                scaled_data = clean_data / 10000 if clean_data.abs().max() > 100 else clean_data
                cubed = scaled_data ** 3
                features[feature_name] = self._clean_numeric_data(cubed)
        
        # Log features (for positive values)
        positive_mask = clean_data > 0
        if positive_mask.any():
            feature_name = f'{col_name.lower()}_log'.lower()
            with self._time_feature(feature_name, features):
                log_values = pd.Series(index=clean_data.index, dtype=float)
                log_values[positive_mask] = np.log1p(clean_data[positive_mask])
                log_values[~positive_mask] = 0
                features[feature_name] = self._clean_numeric_data(log_values, fill_value=0)
        
        # Square root features (for non-negative values) - with NaN handling fix
        non_negative_mask = clean_data.dropna() >= 0
        if non_negative_mask.any():
            feature_name = f'{col_name.lower()}_sqrt'.lower()
            with self._time_feature(feature_name, features):
                sqrt_values = pd.Series(index=clean_data.index, dtype=float)
                # Only apply sqrt where data is non-negative and not NaN
                valid_mask = (clean_data >= 0) & clean_data.notna()
                sqrt_values[valid_mask] = np.sqrt(clean_data[valid_mask])
                sqrt_values[~valid_mask] = np.nan
                features[feature_name] = self._clean_numeric_data(sqrt_values, fill_value=0)
        
        # Reciprocal features (1/x) for non-zero values
        non_zero_mask = clean_data.abs() > 1e-10
        if non_zero_mask.any():
            feature_name = f'{col_name.lower()}_reciprocal'.lower()
            with self._time_feature(feature_name, features):
                reciprocal_values = pd.Series(index=clean_data.index, dtype=float)
                reciprocal_values[non_zero_mask] = 1.0 / clean_data[non_zero_mask]
                reciprocal_values[~non_zero_mask] = 0
                features[feature_name] = self._clean_numeric_data(reciprocal_values)
        
        return features
    
    def _generate_interactions(self, df: pd.DataFrame, 
                              numeric_cols: List[str]) -> Dict[str, pd.Series]:
        """Generate interaction terms between numeric columns."""
        features = {}
        
        # Limit to first N columns to avoid explosion
        cols_to_use = numeric_cols[:self.max_interaction_features]
        
        for i, col1 in enumerate(cols_to_use):
            for col2 in cols_to_use[i+1:]:
                feature_name = f'{col1}_x_{col2}'.lower()
                with self._time_feature(feature_name, features):
                    try:
                        clean_col1 = self._clean_numeric_data(df[col1])
                        clean_col2 = self._clean_numeric_data(df[col2])
                        
                        # Scale values if needed to prevent overflow
                        max_val = max(clean_col1.abs().max(), clean_col2.abs().max())
                        if max_val > 1000:
                            scale_factor = np.sqrt(max_val / 1000)
                            clean_col1 = clean_col1 / scale_factor
                            clean_col2 = clean_col2 / scale_factor
                        
                        interaction = clean_col1 * clean_col2
                        features[feature_name] = self._clean_numeric_data(interaction)
                        
                    except Exception as e:
                        logger.error(f"Error creating interaction term {col1} x {col2}: {e}")
                        continue
        
        return features
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input has numeric columns."""
        if not super().validate_input(df):
            return False
        
        # Check for at least one numeric column
        has_numeric = len(df.select_dtypes(include=[np.number]).columns) > 0
        
        if not has_numeric:
            logger.warning("No numeric columns found for polynomial features")
            return False
        
        return True


# Convenience function for backward compatibility
def get_polynomial_features(df: pd.DataFrame, 
                           numeric_cols: List[str], 
                           degree: int = 2,
                           check_signal: bool = True) -> Dict[str, pd.Series]:
    """
    Legacy function for polynomial features.
    
    Args:
        df: Input dataframe
        numeric_cols: Numeric columns to transform
        degree: Polynomial degree
        check_signal: Whether to check for signal and discard no-signal features
        
    Returns:
        Dictionary of feature name -> pandas Series
    """
    generator = PolynomialFeatures()
    generator._check_signal = check_signal
    return generator.generate_features(df, numeric_cols=numeric_cols, degree=degree)