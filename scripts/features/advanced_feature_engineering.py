"""
Advanced Feature Engineering Module

Comprehensive feature engineering operations for ML:
- Missing value imputation strategies
- Text feature extraction
- Date/time feature engineering
- Categorical encoding techniques
- Outlier detection and handling
- Feature selection methods
- Cyclical features
- Lag features for time series
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class MissingValueImputer:
    """Advanced strategies for handling missing values."""
    
    @staticmethod
    def smart_impute(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Smart imputation based on data characteristics.
        
        Strategies:
        - 'auto': Chooses best method per column
        - 'median': For numeric columns
        - 'mode': For categorical columns
        - 'forward_fill': For time series
        - 'interpolate': For numeric sequences
        - 'knn': K-nearest neighbors (requires scikit-learn)
        """
        df_imputed = df.copy()
        
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
                
            if strategy == 'auto':
                # Numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check if it might be a time series (monotonic index)
                    if df.index.is_monotonic_increasing and df[col].notna().sum() > 10:
                        df_imputed[col] = df[col].interpolate(method='linear', limit_direction='both')
                    else:
                        # Use median for general numeric
                        df_imputed[col] = df[col].fillna(df[col].median())
                        
                # Categorical columns
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    # Use mode
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df_imputed[col] = df[col].fillna(mode_val[0])
                    else:
                        df_imputed[col] = df[col].fillna('missing')
                        
                # Boolean columns
                elif pd.api.types.is_bool_dtype(df[col]):
                    # Use mode for boolean
                    df_imputed[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else False)
                    
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df_imputed[col] = df[col].fillna(df[col].median())
                    
            elif strategy == 'mode':
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df_imputed[col] = df[col].fillna(mode_val[0])
                    
            elif strategy == 'forward_fill':
                df_imputed[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
            elif strategy == 'interpolate':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df_imputed[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        return df_imputed
    
    @staticmethod
    def create_missing_indicators(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, pd.Series]:
        """Create binary indicators for missing values."""
        features = {}
        cols_to_check = columns if columns else df.columns
        
        for col in cols_to_check:
            if col in df.columns and df[col].isna().sum() > 0:
                features[f'{col}_was_missing'] = df[col].isna().astype(int)
                
        return features
    
    @staticmethod
    def group_based_imputation(df: pd.DataFrame, 
                              target_col: str, 
                              group_cols: List[str]) -> pd.Series:
        """Impute values based on group statistics."""
        result = df[target_col].copy()
        
        for group_col in group_cols:
            if group_col in df.columns:
                # Calculate group means/modes
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    group_values = df.groupby(group_col)[target_col].transform('median')
                else:
                    group_values = df.groupby(group_col)[target_col].transform(
                        lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
                    )
                
                # Fill missing values where group value exists
                mask = result.isna() & group_values.notna()
                result[mask] = group_values[mask]
        
        return result


class TextFeatureExtractor:
    """Extract features from text data."""
    
    @staticmethod
    def get_basic_text_features(df: pd.DataFrame, text_columns: List[str]) -> Dict[str, pd.Series]:
        """Extract basic text statistics."""
        features = {}
        
        for col in text_columns:
            if col not in df.columns:
                continue
                
            # Convert to string and handle NaN
            text_series = df[col].fillna('').astype(str)
            
            # Length features
            features[f'{col}_length'] = text_series.str.len()
            features[f'{col}_word_count'] = text_series.str.split().str.len()
            
            # Character type counts
            features[f'{col}_digit_count'] = text_series.str.count(r'\d')
            features[f'{col}_upper_count'] = text_series.str.count(r'[A-Z]')
            features[f'{col}_lower_count'] = text_series.str.count(r'[a-z]')
            features[f'{col}_space_count'] = text_series.str.count(r'\s')
            features[f'{col}_punctuation_count'] = text_series.str.count(r'[^\w\s]')
            
            # Ratios
            length_safe = features[f'{col}_length'].replace(0, 1)  # Avoid division by zero
            features[f'{col}_digit_ratio'] = features[f'{col}_digit_count'] / length_safe
            features[f'{col}_upper_ratio'] = features[f'{col}_upper_count'] / length_safe
            features[f'{col}_punctuation_ratio'] = features[f'{col}_punctuation_count'] / length_safe
            
            # Special patterns
            features[f'{col}_has_email'] = text_series.str.contains(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', na=False).astype(int)
            features[f'{col}_has_url'] = text_series.str.contains(r'https?://|www\.', na=False).astype(int)
            features[f'{col}_has_phone'] = text_series.str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', na=False).astype(int)
            
            # Sentiment indicators (simple)
            features[f'{col}_exclamation_count'] = text_series.str.count('!')
            features[f'{col}_question_count'] = text_series.str.count(r'\?')
            features[f'{col}_starts_with_capital'] = text_series.str.match(r'^[A-Z]', na=False).astype(int)
            
        return features
    
    @staticmethod
    def get_tfidf_features(texts: pd.Series, 
                          max_features: int = 100, 
                          ngram_range: Tuple[int, int] = (1, 2)) -> pd.DataFrame:
        """Extract TF-IDF features from text."""
        try:
            # Handle missing values
            texts_clean = texts.fillna('').astype(str)
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                max_df=0.95,
                min_df=2
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(texts_clean)
            
            # Convert to DataFrame
            feature_names = [f'tfidf_{word}' for word in vectorizer.get_feature_names_out()]
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=feature_names,
                index=texts.index
            )
            
            return tfidf_df
            
        except Exception as e:
            logger.error(f"Error extracting TF-IDF features: {e}")
            return pd.DataFrame(index=texts.index)
    
    @staticmethod
    def get_word_embedding_stats(texts: pd.Series, 
                                embedding_dict: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, pd.Series]:
        """
        Get statistical features from word embeddings.
        If no embedding_dict provided, uses simple character-based features.
        """
        features = {}
        
        if embedding_dict is None:
            # Fallback: character-based embedding simulation
            texts_clean = texts.fillna('').astype(str)
            
            # Average character code (simple embedding proxy)
            features['text_avg_char_code'] = texts_clean.apply(
                lambda x: np.mean([ord(c) for c in x]) if len(x) > 0 else 0
            )
            features['text_std_char_code'] = texts_clean.apply(
                lambda x: np.std([ord(c) for c in x]) if len(x) > 1 else 0
            )
        else:
            # Real embedding features would go here
            pass
            
        return features


class DateTimeFeatureExtractor:
    """Extract features from datetime columns."""
    
    @staticmethod
    def extract_datetime_features(df: pd.DataFrame, datetime_columns: List[str]) -> Dict[str, pd.Series]:
        """Extract comprehensive datetime features."""
        features = {}
        
        for col in datetime_columns:
            if col not in df.columns:
                continue
                
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    dt_series = pd.to_datetime(df[col], errors='coerce')
                else:
                    dt_series = df[col]
                
                # Basic components
                features[f'{col}_year'] = dt_series.dt.year
                features[f'{col}_month'] = dt_series.dt.month
                features[f'{col}_day'] = dt_series.dt.day
                features[f'{col}_dayofweek'] = dt_series.dt.dayofweek
                features[f'{col}_dayofyear'] = dt_series.dt.dayofyear
                features[f'{col}_weekofyear'] = dt_series.dt.isocalendar().week
                features[f'{col}_quarter'] = dt_series.dt.quarter
                features[f'{col}_hour'] = dt_series.dt.hour
                features[f'{col}_minute'] = dt_series.dt.minute
                
                # Cyclical encoding for periodic features
                features[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                features[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                features[f'{col}_day_sin'] = np.sin(2 * np.pi * dt_series.dt.day / 31)
                features[f'{col}_day_cos'] = np.cos(2 * np.pi * dt_series.dt.day / 31)
                features[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * dt_series.dt.dayofweek / 7)
                features[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * dt_series.dt.dayofweek / 7)
                features[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
                features[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
                
                # Boolean features
                features[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
                features[f'{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
                features[f'{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
                features[f'{col}_is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
                features[f'{col}_is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
                features[f'{col}_is_year_start'] = dt_series.dt.is_year_start.astype(int)
                features[f'{col}_is_year_end'] = dt_series.dt.is_year_end.astype(int)
                
                # Time since epoch (useful for trend)
                features[f'{col}_timestamp'] = dt_series.astype(np.int64) // 10**9
                
                # If we have a reference date (e.g., today), calculate days since
                if not dt_series.isna().all():
                    reference_date = pd.Timestamp.now()
                    features[f'{col}_days_since_today'] = (reference_date - dt_series).dt.days
                    features[f'{col}_is_future'] = (dt_series > reference_date).astype(int)
                
            except Exception as e:
                logger.error(f"Error extracting datetime features from {col}: {e}")
                
        return features


class CategoricalEncoder:
    """Advanced categorical encoding techniques."""
    
    @staticmethod
    def get_frequency_encoding(df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, pd.Series]:
        """Encode categories by their frequency."""
        features = {}
        
        for col in categorical_columns:
            if col in df.columns:
                # Calculate frequencies
                freq_map = df[col].value_counts(normalize=True).to_dict()
                features[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
                
                # Also add count encoding
                count_map = df[col].value_counts().to_dict()
                features[f'{col}_count'] = df[col].map(count_map).fillna(0)
                
        return features
    
    @staticmethod
    def get_target_encoding(df: pd.DataFrame, 
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
                
            # Calculate category statistics
            temp_df = pd.DataFrame({col: df[col], 'target': target})
            category_stats = temp_df.groupby(col)['target'].agg(['mean', 'count'])
            
            # Apply smoothing
            smoothed_mean = (
                (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
                (category_stats['count'] + smoothing)
            )
            
            # Map to original data
            features[f'{col}_target_encoded'] = df[col].map(smoothed_mean).fillna(global_mean)
            
        return features
    
    @staticmethod
    def get_label_encoding(df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, pd.Series]:
        """Simple label encoding for ordinal categories."""
        features = {}
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                # Handle unseen categories
                features[f'{col}_label_encoded'] = pd.Series(
                    le.fit_transform(df[col].fillna('missing').astype(str)),
                    index=df.index
                )
                
        return features


class OutlierHandler:
    """Methods for detecting and handling outliers."""
    
    @staticmethod
    def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    @staticmethod
    def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method."""
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.Series(False, index=series.index)
            
        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold
    
    @staticmethod
    def cap_outliers(series: pd.Series, method: str = 'iqr', **kwargs) -> pd.Series:
        """Cap outliers to specified bounds."""
        if method == 'iqr':
            multiplier = kwargs.get('multiplier', 1.5)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
        elif method == 'percentile':
            lower_pct = kwargs.get('lower_pct', 0.01)
            upper_pct = kwargs.get('upper_pct', 0.99)
            
            lower_bound = series.quantile(lower_pct)
            upper_bound = series.quantile(upper_pct)
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return series.clip(lower=lower_bound, upper=upper_bound)


class TimeSeriesFeatures:
    """Features specific to time series data."""
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, 
                           columns: List[str], 
                           lags: List[int] = [1, 2, 3, 7]) -> Dict[str, pd.Series]:
        """Create lagged features for time series."""
        features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                features[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
        return features
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, 
                               columns: List[str], 
                               windows: List[int] = [3, 7, 14]) -> Dict[str, pd.Series]:
        """Create rolling statistics features."""
        features = {}
        
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            for window in windows:
                # Rolling mean
                features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                features[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                
                # Rolling min/max
                features[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                features[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                
        return features
    
    @staticmethod
    def create_expanding_features(df: pd.DataFrame, columns: List[str]) -> Dict[str, pd.Series]:
        """Create expanding window features."""
        features = {}
        
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Expanding mean (cumulative average)
            features[f'{col}_expanding_mean'] = df[col].expanding(min_periods=1).mean()
            
            # Expanding std
            features[f'{col}_expanding_std'] = df[col].expanding(min_periods=1).std().fillna(0)
            
            # Cumulative sum
            features[f'{col}_cumsum'] = df[col].cumsum()
            
        return features


class FeatureInteractions:
    """Create interaction features between columns."""
    
    @staticmethod
    def create_arithmetic_interactions(df: pd.DataFrame, 
                                     col_pairs: List[Tuple[str, str]]) -> Dict[str, pd.Series]:
        """Create arithmetic interactions between column pairs."""
        features = {}
        
        for col1, col2 in col_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
                
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                # Addition
                features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                
                # Subtraction
                features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                
                # Multiplication (already in polynomial features, but included for completeness)
                features[f'{col1}_times_{col2}'] = df[col1] * df[col2]
                
                # Division (safe)
                denominator = df[col2].replace(0, np.nan)
                features[f'{col1}_div_{col2}'] = (df[col1] / denominator).fillna(0)
                
                # Ratio features
                features[f'{col1}_ratio_sum'] = df[col1] / (df[col1] + df[col2] + 1e-8)
                
        return features
    
    @staticmethod
    def create_categorical_combinations(df: pd.DataFrame, 
                                      col_pairs: List[Tuple[str, str]]) -> Dict[str, pd.Series]:
        """Create combinations of categorical features."""
        features = {}
        
        for col1, col2 in col_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
                
            # Combine categories
            features[f'{col1}_X_{col2}'] = (
                df[col1].astype(str) + '_' + df[col2].astype(str)
            ).astype('category')
            
        return features


class FeatureSelector:
    """Methods for selecting important features."""
    
    @staticmethod
    def select_low_variance_features(df: pd.DataFrame, 
                                   threshold: float = 0.01) -> List[str]:
        """Remove features with very low variance."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        selected_features = []
        for col in numeric_cols:
            if df[col].var() > threshold:
                selected_features.append(col)
                
        return selected_features
    
    @staticmethod
    def select_uncorrelated_features(df: pd.DataFrame, 
                                   threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Select upper triangle
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > threshold)]
        
        # Return features to keep
        return [col for col in numeric_df.columns if col not in to_drop]


def create_all_features(df: pd.DataFrame, 
                       config: Dict[str, Any]) -> pd.DataFrame:
    """
    Main function to create all features based on configuration.
    
    Example config:
    {
        'numeric_columns': ['age', 'income'],
        'categorical_columns': ['city', 'category'],
        'text_columns': ['description'],
        'datetime_columns': ['created_at'],
        'target_column': 'target',
        'create_polynomial': True,
        'create_interactions': True,
        'handle_outliers': True
    }
    """
    all_features = {}
    
    # Start with original dataframe
    result_df = df.copy()
    
    # Handle missing values first
    if config.get('handle_missing', True):
        imputer = MissingValueImputer()
        missing_indicators = imputer.create_missing_indicators(df)
        all_features.update(missing_indicators)
        
        # Impute the dataframe
        result_df = imputer.smart_impute(result_df)
    
    # Extract features based on column types
    numeric_cols = config.get('numeric_columns', [])
    categorical_cols = config.get('categorical_columns', [])
    text_cols = config.get('text_columns', [])
    datetime_cols = config.get('datetime_columns', [])
    
    # Numeric features
    if numeric_cols:
        # Basic polynomial features
        if config.get('create_polynomial', True):
            from generic import GenericFeatureOperations
            poly_features = GenericFeatureOperations.get_polynomial_features(
                result_df, numeric_cols, degree=2
            )
            all_features.update(poly_features)
        
        # Outlier indicators
        if config.get('handle_outliers', True):
            outlier_handler = OutlierHandler()
            for col in numeric_cols:
                if col in result_df.columns:
                    outliers = outlier_handler.detect_outliers_iqr(result_df[col])
                    all_features[f'{col}_is_outlier'] = outliers.astype(int)
    
    # Categorical features
    if categorical_cols:
        encoder = CategoricalEncoder()
        
        # Frequency encoding
        freq_features = encoder.get_frequency_encoding(result_df, categorical_cols)
        all_features.update(freq_features)
        
        # Target encoding if target provided
        if 'target_column' in config and config['target_column'] in result_df.columns:
            target_features = encoder.get_target_encoding(
                result_df, categorical_cols, 
                result_df[config['target_column']]
            )
            all_features.update(target_features)
    
    # Text features
    if text_cols:
        text_extractor = TextFeatureExtractor()
        text_features = text_extractor.get_basic_text_features(result_df, text_cols)
        all_features.update(text_features)
    
    # Datetime features
    if datetime_cols:
        dt_extractor = DateTimeFeatureExtractor()
        dt_features = dt_extractor.extract_datetime_features(result_df, datetime_cols)
        all_features.update(dt_features)
    
    # Interaction features
    if config.get('create_interactions', False):
        interactions = FeatureInteractions()
        
        # Numeric interactions
        if len(numeric_cols) >= 2:
            numeric_pairs = [(numeric_cols[i], numeric_cols[j]) 
                           for i in range(len(numeric_cols)) 
                           for j in range(i+1, min(i+2, len(numeric_cols)))]
            arithmetic_features = interactions.create_arithmetic_interactions(
                result_df, numeric_pairs[:5]  # Limit interactions
            )
            all_features.update(arithmetic_features)
    
    # Time series features if specified
    if config.get('is_time_series', False) and numeric_cols:
        ts_features = TimeSeriesFeatures()
        
        # Lag features
        lag_features = ts_features.create_lag_features(
            result_df, numeric_cols[:3], lags=[1, 2, 3]
        )
        all_features.update(lag_features)
        
        # Rolling features
        rolling_features = ts_features.create_rolling_features(
            result_df, numeric_cols[:3], windows=[3, 7]
        )
        all_features.update(rolling_features)
    
    # Combine all features
    feature_df = pd.DataFrame(all_features, index=result_df.index)
    final_df = pd.concat([result_df, feature_df], axis=1)
    
    # Feature selection if requested
    if config.get('select_features', False):
        selector = FeatureSelector()
        
        # Remove low variance features
        selected_cols = selector.select_low_variance_features(final_df)
        
        # Keep all non-numeric columns and selected numeric columns
        non_numeric = final_df.select_dtypes(exclude=[np.number]).columns.tolist()
        final_cols = non_numeric + selected_cols
        final_df = final_df[final_cols]
    
    return final_df