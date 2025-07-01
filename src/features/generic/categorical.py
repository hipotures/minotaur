"""
Categorical Feature Operations

Advanced categorical encoding and transformation including:
- Frequency and count encoding
- Target encoding with smoothing
- One-hot encoding for low cardinality
- Binary encoding for high cardinality
- Ordinal encoding with custom ordering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

from ..base import GenericFeatureOperation

logger = logging.getLogger(__name__)


class CategoricalFeatures(GenericFeatureOperation):
    """Categorical feature encoding and transformation operations."""
    
    def __init__(self):
        """Initialize categorical feature operations."""
        super().__init__()
    
    def get_operation_name(self) -> str:
        """Return operation name."""
        return "categorical_features"
    
    def _generate_features_impl(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate categorical features based on parameters."""
        features = {}
        
        # Get parameters
        categorical_columns = kwargs.get('categorical_columns', [])
        target = kwargs.get('target', None)
        max_cardinality_onehot = kwargs.get('max_cardinality_onehot', 10)
        smoothing = kwargs.get('smoothing', 1.0)
        forbidden_columns = kwargs.get('forbidden_columns', [])
        
        if not categorical_columns:
            # Auto-detect categorical columns
            categorical_columns = self._detect_categorical_columns(df)
            # Filter out forbidden columns using provided list (not static patterns)
            if forbidden_columns:
                categorical_columns = [col for col in categorical_columns if col not in forbidden_columns]
            else:
                # Fallback to static patterns if no forbidden columns provided
                categorical_columns = self._filter_forbidden_columns(categorical_columns)
        
        if not categorical_columns:
            return features
        
        # Frequency encoding (always safe to do)
        freq_features = self.get_frequency_encoding(df, categorical_columns)
        features.update(freq_features)
        
        # Target encoding (if target provided)
        if target is not None:
            target_features = self.get_target_encoding(df, categorical_columns, target, smoothing)
            features.update(target_features)
        
        # One-hot encoding for low cardinality
        low_card_cols = [col for col in categorical_columns 
                        if col in df.columns and df[col].nunique() <= max_cardinality_onehot]
        if low_card_cols:
            onehot_features = self.get_onehot_encoding(df, low_card_cols)
            features.update(onehot_features)
        
        # Label encoding
        label_features = self.get_label_encoding(df, categorical_columns)
        features.update(label_features)
        
        return features
    
    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect categorical columns."""
        categorical_cols = []
        
        for col in df.columns:
            # Check if column is object or categorical dtype
            if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                categorical_cols.append(col)
            # Check if numeric column has low cardinality (likely categorical)
            elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < 20:
                # Additional check: are values integers?
                if df[col].dropna().apply(lambda x: x == int(x)).all():
                    categorical_cols.append(col)
        
        return categorical_cols
    
    def get_frequency_encoding(self, df: pd.DataFrame, 
                              categorical_columns: List[str]) -> Dict[str, pd.Series]:
        """Encode categories by their frequency."""
        features = {}
        
        for col in categorical_columns:
            if col not in df.columns:
                continue
            
            with self._time_feature(f'{col.lower()}_frequency_encoding'):
                # Calculate frequencies
                freq_map = df[col].value_counts(normalize=True).to_dict()
                features[f'{col.lower()}_frequency'] = df[col].map(freq_map).fillna(0)
                
                # Also add count encoding
                count_map = df[col].value_counts().to_dict()
                features[f'{col.lower()}_count'] = df[col].map(count_map).fillna(0)
                
                # Add rank encoding (by frequency)
                rank_map = pd.Series(range(len(freq_map)), 
                                   index=pd.Index(freq_map.keys())).to_dict()
                features[f'{col.lower()}_frequency_rank'] = df[col].map(rank_map).fillna(-1)
        
        return features
    
    def get_target_encoding(self, df: pd.DataFrame, 
                           categorical_columns: List[str], 
                           target: pd.Series,
                           smoothing: float = 1.0) -> Dict[str, pd.Series]:
        """Target encoding with smoothing to prevent overfitting."""
        features = {}
        
        if not pd.api.types.is_numeric_dtype(target):
            logger.warning("Target encoding requires numeric target variable")
            return features
        
        global_mean = target.mean()
        
        for col in categorical_columns:
            if col not in df.columns:
                continue
            
            with self._time_feature(f'{col.lower()}_target_encoding'):
                # Calculate category statistics
                temp_df = pd.DataFrame({col: df[col], 'target': target})
                category_stats = temp_df.groupby(col)['target'].agg(['mean', 'count'])
                
                # Apply smoothing
                smoothed_mean = (
                    (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
                    (category_stats['count'] + smoothing)
                )
                
                # Map to original data
                features[f'{col.lower()}_target_encoded'] = df[col].map(smoothed_mean).fillna(global_mean)
                
                # Also add target variance encoding
                category_var = temp_df.groupby(col)['target'].var()
                features[f'{col.lower()}_target_variance'] = df[col].map(category_var).fillna(0)
        
        return features
    
    def get_onehot_encoding(self, df: pd.DataFrame, 
                           categorical_columns: List[str]) -> Dict[str, pd.Series]:
        """One-hot encoding for low cardinality categorical variables."""
        features = {}
        
        for col in categorical_columns:
            if col not in df.columns:
                continue
            
            with self._time_feature(f'{col.lower()}_onehot_encoding'):
                # Get unique values
                unique_values = df[col].dropna().unique()
                
                # Create binary features for each unique value
                for value in unique_values:
                    # Clean the value name for use in feature name
                    clean_value = str(value).replace(' ', '_').replace('-', '_')
                    feature_name = f'{col.lower()}_is_{clean_value}'
                    # Handle NaN values in one-hot encoding
                    one_hot_series = (df[col] == value)
                    features[feature_name] = one_hot_series.fillna(False).astype(int)
        
        return features
    
    def get_label_encoding(self, df: pd.DataFrame, 
                          categorical_columns: List[str]) -> Dict[str, pd.Series]:
        """Simple label encoding for categorical variables."""
        features = {}
        
        for col in categorical_columns:
            if col not in df.columns:
                continue
            
            with self._time_feature(f'{col.lower()}_label_encoding'):
                # Create mapping from unique values to integers
                unique_values = df[col].dropna().unique()
                value_to_int = {val: i for i, val in enumerate(sorted(unique_values))}
                
                # Add special value for missing
                value_to_int[np.nan] = -1
                
                # Apply mapping with robust NA handling
                try:
                    mapped_series = df[col].map(value_to_int).fillna(-1)
                    # Convert to int with explicit error handling for different pandas versions
                    features[f'{col.lower()}_label_encoded'] = pd.to_numeric(mapped_series, errors='coerce').fillna(-1).astype(int)
                except Exception as e:
                    logger.warning(f"Label encoding failed for {col}: {e}, skipping...")
                    continue
        
        return features
    
    def get_binary_encoding(self, df: pd.DataFrame, 
                           categorical_columns: List[str]) -> Dict[str, pd.Series]:
        """Binary encoding for high cardinality categorical variables."""
        features = {}
        
        for col in categorical_columns:
            if col not in df.columns:
                continue
            
            with self._time_feature(f'{col.lower()}_binary_encoding'):
                # Get unique values and create label encoding first
                unique_values = df[col].dropna().unique()
                value_to_int = {val: i for i, val in enumerate(unique_values)}
                
                # Robust label encoding with error handling
                try:
                    mapped_series = df[col].map(value_to_int).fillna(-1)
                    label_encoded = pd.to_numeric(mapped_series, errors='coerce').fillna(-1).astype(int)
                except Exception as e:
                    logger.warning(f"Binary encoding failed for {col}: {e}, skipping...")
                    continue
                
                # Determine number of bits needed
                max_val = len(unique_values)
                n_bits = int(np.ceil(np.log2(max_val + 1)))  # +1 for missing value code
                
                # Create binary features
                for i in range(n_bits):
                    features[f'{col.lower()}_binary_bit_{i}'] = ((label_encoded >> i) & 1).astype(int)
        
        return features
    
    def get_combination_features(self, df: pd.DataFrame,
                               column_pairs: List[tuple]) -> Dict[str, pd.Series]:
        """Create combination features from pairs of categorical columns."""
        features = {}
        
        for col1, col2 in column_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            with self._time_feature(f'{col1}_{col2}_combination'):
                # Create combination feature
                features[f'{col1}_{col2}_combination'] = (
                    df[col1].astype(str) + '_' + df[col2].astype(str)
                )
                
                # Also create frequency encoding of combination
                combo_freq = features[f'{col1}_{col2}_combination'].value_counts(normalize=True)
                features[f'{col1}_{col2}_combination_frequency'] = (
                    features[f'{col1}_{col2}_combination'].map(combo_freq).fillna(0)
                )
        
        return features


# Standalone functions for backward compatibility
def get_categorical_features(df: pd.DataFrame, 
                           categorical_columns: List[str] = [],
                           target: Optional[pd.Series] = None,
                           max_cardinality_onehot: int = 10,
                           smoothing: float = 1.0) -> Dict[str, pd.Series]:
    """
    Get categorical features from dataframe.
    
    Args:
        df: Input dataframe
        categorical_columns: Columns to encode (auto-detect if empty)
        target: Target variable for target encoding
        max_cardinality_onehot: Maximum unique values for one-hot encoding
        smoothing: Smoothing parameter for target encoding
        
    Returns:
        Dictionary of categorical features
    """
    categorical_op = CategoricalFeatures()
    return categorical_op.generate_features(
        df, 
        categorical_columns=categorical_columns,
        target=target,
        max_cardinality_onehot=max_cardinality_onehot,
        smoothing=smoothing
    )