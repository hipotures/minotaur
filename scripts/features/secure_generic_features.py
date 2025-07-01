"""
Generic Feature Operations

Universal feature operations that work across domains:
- Statistical aggregations
- Polynomial transformations
- Binning and ranking features
- Feature selection operations

Enhanced with comprehensive data validation and error handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class GenericFeatureOperations:
    """Generic feature operations applicable across domains with robust error handling."""
    
    # Constants for data validation
    MAX_VALUE = 1e15
    MIN_VALUE = -1e15
    EPSILON = 1e-10
    
    @staticmethod
    def _clean_numeric_data(series: pd.Series, 
                          fill_value: Optional[float] = None,
                          clip_values: bool = True,
                          handle_inf: bool = True) -> pd.Series:
        """
        Clean numeric data by handling NaN, infinities, and extreme values.
        
        Args:
            series: Input pandas Series
            fill_value: Value to use for filling NaN (if None, uses median)
            clip_values: Whether to clip extreme values
            handle_inf: Whether to replace infinities
            
        Returns:
            Cleaned pandas Series
        """
        if series.empty or series.isna().all():
            return pd.Series(0, index=series.index)
        
        # Create a copy to avoid modifying original data
        cleaned = series.copy()
        
        # Handle infinities
        if handle_inf:
            cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        if fill_value is not None:
            cleaned = cleaned.fillna(fill_value)
        else:
            # Use median for filling, but check if there are any finite values
            finite_values = cleaned[np.isfinite(cleaned)]
            if len(finite_values) > 0:
                cleaned = cleaned.fillna(finite_values.median())
            else:
                cleaned = cleaned.fillna(0)
        
        # Clip extreme values to prevent overflow
        if clip_values:
            cleaned = cleaned.clip(lower=GenericFeatureOperations.MIN_VALUE, 
                                 upper=GenericFeatureOperations.MAX_VALUE)
        
        return cleaned
    
    @staticmethod
    def _safe_divide(numerator: Union[pd.Series, np.ndarray], 
                    denominator: Union[pd.Series, np.ndarray, float]) -> pd.Series:
        """Safely divide two arrays/series handling division by zero."""
        if isinstance(denominator, (int, float)):
            if abs(denominator) < GenericFeatureOperations.EPSILON:
                return pd.Series(0, index=numerator.index if hasattr(numerator, 'index') else None)
        
        # Convert to numpy arrays for calculation
        num_array = np.array(numerator)
        den_array = np.array(denominator)
        
        # Replace zeros and very small values in denominator
        den_array = np.where(np.abs(den_array) < GenericFeatureOperations.EPSILON, 
                           GenericFeatureOperations.EPSILON, den_array)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            result = num_array / den_array
            result = np.where(~np.isfinite(result), 0, result)
        
        if hasattr(numerator, 'index'):
            return pd.Series(result, index=numerator.index)
        return pd.Series(result)
    
    @staticmethod
    def get_statistical_aggregations(df: pd.DataFrame, 
                                   groupby_cols: List[str], 
                                   agg_cols: List[str]) -> Dict[str, pd.Series]:
        """Statistical aggregations by categorical features with robust error handling."""
        features = {}
        
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
                    # Clean the aggregation column first
                    clean_col = GenericFeatureOperations._clean_numeric_data(df[agg_col])
                    
                    # Create a temporary dataframe with cleaned data
                    temp_df = df[[group_col]].copy()
                    temp_df['clean_value'] = clean_col
                    
                    # Mean aggregation
                    group_mean = temp_df.groupby(group_col)['clean_value'].transform('mean')
                    features[f'{agg_col}_mean_by_{group_col}'] = group_mean
                    
                    # Standard deviation with safeguards
                    group_std = temp_df.groupby(group_col)['clean_value'].transform(
                        lambda x: x.std() if len(x) > 1 else 0
                    )
                    features[f'{agg_col}_std_by_{group_col}'] = group_std.fillna(0)
                    
                    # Deviation from group mean (safe subtraction)
                    deviation = clean_col - group_mean
                    features[f'{agg_col}_dev_from_{group_col}_mean'] = GenericFeatureOperations._clean_numeric_data(
                        deviation, fill_value=0
                    )
                    
                except Exception as e:
                    logger.error(f"Error in statistical aggregation for {agg_col} by {group_col}: {e}")
                    continue
        
        return features
    
    @staticmethod
    def get_polynomial_features(df: pd.DataFrame, 
                               numeric_cols: List[str], 
                               degree: int = 2) -> Dict[str, pd.Series]:
        """Polynomial features for numeric columns with overflow protection."""
        features = {}
        
        # Validate and filter numeric columns
        valid_numeric_cols = []
        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                valid_numeric_cols.append(col)
            else:
                logger.warning(f"Column '{col}' is not numeric or not found, skipping")
        
        for col in valid_numeric_cols:
            try:
                # Clean the column data
                clean_data = GenericFeatureOperations._clean_numeric_data(df[col])
                
                # Squared features with overflow protection
                if degree >= 2:
                    # Scale down values before squaring to prevent overflow
                    scaled_data = clean_data / 1000 if clean_data.abs().max() > 1000 else clean_data
                    squared = scaled_data ** 2
                    features[f'{col}_squared'] = GenericFeatureOperations._clean_numeric_data(squared)
                
                # Cubic features (selective and with stronger scaling)
                if degree >= 3 and len(valid_numeric_cols) <= 5:
                    scaled_data = clean_data / 10000 if clean_data.abs().max() > 100 else clean_data
                    cubed = scaled_data ** 3
                    features[f'{col}_cubed'] = GenericFeatureOperations._clean_numeric_data(cubed)
                
                # Log features (for positive values)
                positive_mask = clean_data > 0
                if positive_mask.any():
                    log_values = pd.Series(index=clean_data.index, dtype=float)
                    log_values[positive_mask] = np.log1p(clean_data[positive_mask])
                    log_values[~positive_mask] = 0
                    features[f'{col}_log'] = GenericFeatureOperations._clean_numeric_data(log_values, fill_value=0)
                
                # Square root features (for non-negative values)
                non_negative_mask = clean_data >= 0
                if non_negative_mask.any():
                    sqrt_values = pd.Series(index=clean_data.index, dtype=float)
                    sqrt_values[non_negative_mask] = np.sqrt(clean_data[non_negative_mask])
                    sqrt_values[~non_negative_mask] = 0
                    features[f'{col}_sqrt'] = GenericFeatureOperations._clean_numeric_data(sqrt_values, fill_value=0)
                
            except Exception as e:
                logger.error(f"Error creating polynomial features for column '{col}': {e}")
                continue
        
        # Interaction terms with overflow protection
        if len(valid_numeric_cols) >= 2:
            for i, col1 in enumerate(valid_numeric_cols[:3]):
                for col2 in valid_numeric_cols[i+1:4]:
                    try:
                        clean_col1 = GenericFeatureOperations._clean_numeric_data(df[col1])
                        clean_col2 = GenericFeatureOperations._clean_numeric_data(df[col2])
                        
                        # Scale values if needed to prevent overflow
                        max_val = max(clean_col1.abs().max(), clean_col2.abs().max())
                        if max_val > 1000:
                            scale_factor = np.sqrt(max_val / 1000)
                            clean_col1 = clean_col1 / scale_factor
                            clean_col2 = clean_col2 / scale_factor
                        
                        interaction = clean_col1 * clean_col2
                        features[f'{col1}_x_{col2}'] = GenericFeatureOperations._clean_numeric_data(interaction)
                        
                    except Exception as e:
                        logger.error(f"Error creating interaction term {col1} x {col2}: {e}")
                        continue
        
        return features
    
    @staticmethod
    def get_binning_features(df: pd.DataFrame, 
                           numeric_cols: List[str], 
                           n_bins: int = 5) -> Dict[str, pd.Series]:
        """Binning features for numeric columns with comprehensive error handling."""
        features = {}
        
        for col in numeric_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in dataframe")
                continue
                
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column '{col}' is not numeric, skipping binning")
                continue
            
            try:
                # Clean the data first
                clean_data = GenericFeatureOperations._clean_numeric_data(df[col])
                
                # Check if we have enough unique values for binning
                unique_values = clean_data.nunique()
                if unique_values < n_bins:
                    logger.warning(f"Column '{col}' has only {unique_values} unique values, adjusting bins")
                    n_bins_adjusted = min(n_bins, unique_values)
                else:
                    n_bins_adjusted = n_bins
                
                if n_bins_adjusted < 2:
                    logger.warning(f"Cannot create bins for column '{col}' with less than 2 unique values")
                    features[f'{col}_bin_{n_bins}'] = pd.Series(0, index=df.index, dtype='category')
                    continue
                
                # Convert to float32 for compatibility
                if clean_data.dtype == np.float16:
                    clean_data = clean_data.astype(np.float32)
                
                # Try quantile-based binning first
                try:
                    binned = pd.qcut(clean_data, q=n_bins_adjusted, labels=False, duplicates='drop')
                    features[f'{col}_bin_{n_bins}'] = binned.fillna(-1).astype('category')
                    
                except Exception as quantile_error:
                    logger.warning(f"Quantile binning failed for '{col}': {quantile_error}, trying equal-width bins")
                    
                    # Fallback to equal-width binning
                    try:
                        binned = pd.cut(clean_data, bins=n_bins_adjusted, labels=False, duplicates='drop')
                        features[f'{col}_bin_{n_bins}'] = binned.fillna(-1).astype('category')
                        
                    except Exception as cut_error:
                        logger.error(f"Both binning methods failed for '{col}': {cut_error}")
                        # Ultimate fallback: create a constant feature
                        features[f'{col}_bin_{n_bins}'] = pd.Series(0, index=df.index, dtype='category')
                
            except Exception as e:
                logger.error(f"Unexpected error in binning for column '{col}': {e}")
                features[f'{col}_bin_{n_bins}'] = pd.Series(0, index=df.index, dtype='category')
        
        return features
    
    @staticmethod
    def get_ranking_features(df: pd.DataFrame, 
                           numeric_cols: List[str]) -> Dict[str, pd.Series]:
        """Ranking features for numeric columns with proper handling of ties and missing values."""
        features = {}
        
        for col in numeric_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in dataframe")
                continue
                
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column '{col}' is not numeric, skipping ranking")
                continue
            
            try:
                # Clean the data
                clean_data = GenericFeatureOperations._clean_numeric_data(df[col])
                
                # Check if we have any valid data to rank
                if clean_data.isna().all() or clean_data.nunique() == 1:
                    logger.warning(f"Column '{col}' has no variation for ranking")
                    features[f'{col}_rank'] = pd.Series(1, index=df.index)
                    features[f'{col}_percentile'] = pd.Series(0.5, index=df.index)
                    continue
                
                # Rank within entire dataset (handling NaN properly)
                rank_values = clean_data.rank(method='dense', na_option='bottom')
                features[f'{col}_rank'] = rank_values
                
                # Percentile rank (0 to 1)
                percentile_values = clean_data.rank(pct=True, na_option='bottom')
                features[f'{col}_percentile'] = percentile_values.clip(0, 1)
                
            except Exception as e:
                logger.error(f"Error creating ranking features for column '{col}': {e}")
                # Fallback: create constant features
                features[f'{col}_rank'] = pd.Series(1, index=df.index)
                features[f'{col}_percentile'] = pd.Series(0.5, index=df.index)
        
        return features
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate dataframe and return statistics about potential issues.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': [],
            'columns_with_nan': {},
            'columns_with_inf': {},
            'columns_with_zeros': {},
            'columns_with_negative': {},
            'columns_with_extreme_values': {}
        }
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                validation_results['numeric_columns'].append(col)
                
                # Check for NaN
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    validation_results['columns_with_nan'][col] = nan_count
                
                # Check for infinities
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    validation_results['columns_with_inf'][col] = inf_count
                
                # Check for zeros
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    validation_results['columns_with_zeros'][col] = zero_count
                
                # Check for negative values
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    validation_results['columns_with_negative'][col] = negative_count
                
                # Check for extreme values
                if not df[col].isna().all():
                    col_min = df[col].min()
                    col_max = df[col].max()
                    if col_min < GenericFeatureOperations.MIN_VALUE or col_max > GenericFeatureOperations.MAX_VALUE:
                        validation_results['columns_with_extreme_values'][col] = {
                            'min': col_min,
                            'max': col_max
                        }
        
        return validation_results