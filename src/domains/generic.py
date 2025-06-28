"""
Generic Feature Operations

Universal feature operations that work across domains:
- Statistical aggregations
- Polynomial transformations
- Binning and ranking features
- Feature selection operations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class GenericFeatureOperations:
    """Generic feature operations applicable across domains."""
    
    
    @staticmethod
    def get_statistical_aggregations(df: pd.DataFrame, groupby_cols: List[str], agg_cols: List[str]) -> Dict[str, pd.Series]:
        """Statistical aggregations by categorical features."""
        features = {}
        
        for group_col in groupby_cols:
            if group_col in df.columns:
                for agg_col in agg_cols:
                    if agg_col in df.columns and pd.api.types.is_numeric_dtype(df[agg_col]):
                        # Mean aggregation
                        group_mean = df.groupby(group_col)[agg_col].transform('mean')
                        features[f'{agg_col}_mean_by_{group_col}'] = group_mean
                        
                        # Standard deviation
                        group_std = df.groupby(group_col)[agg_col].transform('std')
                        features[f'{agg_col}_std_by_{group_col}'] = group_std.fillna(0)
                        
                        # Deviation from group mean
                        features[f'{agg_col}_dev_from_{group_col}_mean'] = df[agg_col] - group_mean
                        
        return features
    
    @staticmethod
    def get_polynomial_features(df: pd.DataFrame, numeric_cols: List[str], degree: int = 2) -> Dict[str, pd.Series]:
        """Polynomial features for numeric columns."""
        features = {}
        
        numeric_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        for col in numeric_cols:
            # Squared features
            if degree >= 2:
                features[f'{col}_squared'] = df[col] ** 2
            
            # Cubic features (selective)
            if degree >= 3 and len(numeric_cols) <= 5:  # Limit cubic features
                features[f'{col}_cubed'] = df[col] ** 3
                
            # Log features (for positive values)
            if (df[col] > 0).all():
                features[f'{col}_log'] = np.log1p(df[col])
                
            # Square root features
            if (df[col] >= 0).all():
                features[f'{col}_sqrt'] = np.sqrt(df[col])
        
        # Interaction terms (limited to avoid explosion)
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:3]):  # Limit to first 3 columns
                for col2 in numeric_cols[i+1:4]:  # And next 3
                    features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return features
    
    @staticmethod
    def get_binning_features(df: pd.DataFrame, numeric_cols: List[str], n_bins: int = 5) -> Dict[str, pd.Series]:
        """Binning features for numeric columns."""
        features = {}
        
        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Quantile-based binning with type conversion for compatibility
                try:
                    # Convert float16 to float32 for pandas compatibility
                    col_data = df[col]
                    if col_data.dtype == np.float16:
                        col_data = col_data.astype(np.float32)
                    
                    binned = pd.cut(col_data, bins=n_bins, labels=False, duplicates='drop')
                    features[f'{col}_bin_{n_bins}'] = binned.fillna(-1).astype('category')
                except Exception as e:
                    logger.warning(f"Binning failed for column {col}: {e}")
                    # Fallback: create simple quartile bins
                    try:
                        col_data = df[col].astype(np.float32)
                        features[f'{col}_bin_{n_bins}'] = pd.qcut(col_data, q=n_bins, labels=False, duplicates='drop').fillna(-1)
                    except Exception:
                        # Ultimate fallback: skip this column
                        logger.warning(f"Skipping binning for problematic column: {col}")
                        continue
        
        return features
    
    @staticmethod
    def get_ranking_features(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, pd.Series]:
        """Ranking features for numeric columns."""
        features = {}
        
        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Rank within entire dataset
                features[f'{col}_rank'] = df[col].rank(method='dense')
                
                # Percentile rank
                features[f'{col}_percentile'] = df[col].rank(pct=True)
                
        return features
    
    
    
    
