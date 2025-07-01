"""
Temporal Feature Operations

Time-based feature engineering including:
- Date/time component extraction
- Cyclical encoding for periodic features
- Lag features for time series
- Rolling window statistics
- Time-based aggregations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

from ..base import GenericFeatureOperation

logger = logging.getLogger(__name__)


class TemporalFeatures(GenericFeatureOperation):
    """Temporal and time-series feature operations."""
    
    def __init__(self):
        """Initialize temporal feature operations."""
        super().__init__()
    
    def get_operation_name(self) -> str:
        """Return operation name."""
        return "temporal_features"
    
    def _generate_features_impl(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate temporal features based on parameters."""
        features = {}
        
        # Get parameters
        datetime_columns = kwargs.get('datetime_columns', [])
        numeric_columns = kwargs.get('numeric_columns', [])
        lags = kwargs.get('lags', [1, 2, 3])
        windows = kwargs.get('windows', [3, 7])
        
        # Extract datetime features
        if datetime_columns:
            dt_features = self.extract_datetime_features(df, datetime_columns)
            features.update(dt_features)
        
        # Create lag features
        if numeric_columns and lags:
            lag_features = self.create_lag_features(df, numeric_columns, lags)
            features.update(lag_features)
        
        # Create rolling features
        if numeric_columns and windows:
            rolling_features = self.create_rolling_features(df, numeric_columns, windows)
            features.update(rolling_features)
        
        return features
    
    def extract_datetime_features(self, df: pd.DataFrame, 
                                 datetime_columns: List[str]) -> Dict[str, pd.Series]:
        """Extract comprehensive datetime features."""
        features = {}
        
        for col in datetime_columns:
            if col not in df.columns:
                continue
            
            try:
                # Convert to datetime if not already
                with self._time_feature(f'{col.lower()}_conversion'):
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        dt_series = pd.to_datetime(df[col], errors='coerce')
                    else:
                        dt_series = df[col]
                
                # Basic components
                with self._time_feature(f'{col.lower()}_basic_components'):
                    features[f'{col.lower()}_year'] = dt_series.dt.year
                    features[f'{col.lower()}_month'] = dt_series.dt.month
                    features[f'{col.lower()}_day'] = dt_series.dt.day
                    features[f'{col.lower()}_dayofweek'] = dt_series.dt.dayofweek
                    features[f'{col.lower()}_dayofyear'] = dt_series.dt.dayofyear
                    features[f'{col.lower()}_weekofyear'] = dt_series.dt.isocalendar().week.astype('Int64')
                    features[f'{col.lower()}_quarter'] = dt_series.dt.quarter
                    features[f'{col.lower()}_hour'] = dt_series.dt.hour
                    features[f'{col.lower()}_minute'] = dt_series.dt.minute
                
                # Cyclical encoding for periodic features
                with self._time_feature(f'{col.lower()}_cyclical_encoding'):
                    features[f'{col.lower()}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                    features[f'{col.lower()}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                    features[f'{col.lower()}_day_sin'] = np.sin(2 * np.pi * dt_series.dt.day / 31)
                    features[f'{col.lower()}_day_cos'] = np.cos(2 * np.pi * dt_series.dt.day / 31)
                    features[f'{col.lower()}_dayofweek_sin'] = np.sin(2 * np.pi * dt_series.dt.dayofweek / 7)
                    features[f'{col.lower()}_dayofweek_cos'] = np.cos(2 * np.pi * dt_series.dt.dayofweek / 7)
                    features[f'{col.lower()}_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
                    features[f'{col.lower()}_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
                
                # Boolean features
                with self._time_feature(f'{col.lower()}_boolean_features'):
                    features[f'{col.lower()}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
                    features[f'{col.lower()}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
                    features[f'{col.lower()}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
                    features[f'{col.lower()}_is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
                    features[f'{col.lower()}_is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
                    features[f'{col.lower()}_is_year_start'] = dt_series.dt.is_year_start.astype(int)
                    features[f'{col.lower()}_is_year_end'] = dt_series.dt.is_year_end.astype(int)
                
                # Time since epoch (useful for trend)
                with self._time_feature(f'{col.lower()}_timestamp'):
                    features[f'{col.lower()}_timestamp'] = dt_series.astype(np.int64) // 10**9
                
                # Days since reference date
                if not dt_series.isna().all():
                    with self._time_feature(f'{col.lower()}_days_since'):
                        reference_date = pd.Timestamp.now()
                        features[f'{col.lower()}_days_since_today'] = (reference_date - dt_series).dt.days
                        features[f'{col.lower()}_is_future'] = (dt_series > reference_date).astype(int)
                
            except Exception as e:
                logger.error(f"Error extracting datetime features from {col}: {e}")
        
        return features
    
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str], 
                           lags: List[int] = [1, 2, 3]) -> Dict[str, pd.Series]:
        """Create lagged features for time series."""
        features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                with self._time_feature(f'{col.lower()}_lag_{lag}'):
                    features[f'{col.lower()}_lag_{lag}'] = df[col].shift(lag)
        
        return features
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               columns: List[str], 
                               windows: List[int] = [3, 7]) -> Dict[str, pd.Series]:
        """Create rolling statistics features."""
        features = {}
        
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            for window in windows:
                # Rolling mean
                with self._time_feature(f'{col.lower()}_rolling_mean_{window}'):
                    features[f'{col.lower()}_rolling_mean_{window}'] = df[col].rolling(
                        window=window, min_periods=1
                    ).mean()
                
                # Rolling std
                with self._time_feature(f'{col.lower()}_rolling_std_{window}'):
                    features[f'{col.lower()}_rolling_std_{window}'] = df[col].rolling(
                        window=window, min_periods=1
                    ).std().fillna(0)
                
                # Rolling min/max
                with self._time_feature(f'{col.lower()}_rolling_min_max_{window}'):
                    features[f'{col.lower()}_rolling_min_{window}'] = df[col].rolling(
                        window=window, min_periods=1
                    ).min()
                    features[f'{col.lower()}_rolling_max_{window}'] = df[col].rolling(
                        window=window, min_periods=1
                    ).max()
        
        return features
    
    def create_expanding_features(self, df: pd.DataFrame, 
                                 columns: List[str]) -> Dict[str, pd.Series]:
        """Create expanding window features."""
        features = {}
        
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Expanding mean (cumulative average)
            with self._time_feature(f'{col.lower()}_expanding_mean'):
                features[f'{col.lower()}_expanding_mean'] = df[col].expanding(min_periods=1).mean()
            
            # Expanding std
            with self._time_feature(f'{col.lower()}_expanding_std'):
                features[f'{col.lower()}_expanding_std'] = df[col].expanding(min_periods=1).std().fillna(0)
            
            # Cumulative sum
            with self._time_feature(f'{col.lower()}_cumsum'):
                features[f'{col.lower()}_cumsum'] = df[col].cumsum()
        
        return features


# Standalone functions for backward compatibility
def get_temporal_features(df: pd.DataFrame, 
                         datetime_columns: List[str] = [],
                         numeric_columns: List[str] = [],
                         lags: List[int] = [1, 2, 3],
                         windows: List[int] = [3, 7]) -> Dict[str, pd.Series]:
    """
    Get temporal features from dataframe.
    
    Args:
        df: Input dataframe
        datetime_columns: Columns to extract datetime features from
        numeric_columns: Columns to create lag/rolling features for
        lags: Lag periods for time series features
        windows: Window sizes for rolling statistics
        
    Returns:
        Dictionary of temporal features
    """
    temporal_op = TemporalFeatures()
    return temporal_op.generate_features(
        df, 
        datetime_columns=datetime_columns,
        numeric_columns=numeric_columns,
        lags=lags,
        windows=windows
    )