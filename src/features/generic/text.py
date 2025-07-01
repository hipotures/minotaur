"""
Text Feature Operations

Natural language processing and text mining features including:
- Basic text statistics (length, word count, etc.)
- Character type analysis
- Pattern detection (emails, URLs, phone numbers)
- Text complexity metrics
- Simple sentiment indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
import logging

from ..base import GenericFeatureOperation

logger = logging.getLogger(__name__)


class TextFeatures(GenericFeatureOperation):
    """Text analysis and NLP feature operations."""
    
    def __init__(self):
        """Initialize text feature operations."""
        super().__init__()
    
    def get_operation_name(self) -> str:
        """Return operation name."""
        return "Text Features"
    
    def _generate_features_impl(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate text features based on parameters."""
        features = {}
        
        # Get parameters
        text_columns = kwargs.get('text_columns', [])
        extract_patterns = kwargs.get('extract_patterns', True)
        extract_complexity = kwargs.get('extract_complexity', True)
        
        if not text_columns:
            # Auto-detect text columns
            text_columns = [col for col in df.columns 
                           if pd.api.types.is_string_dtype(df[col]) or 
                           pd.api.types.is_object_dtype(df[col])]
        
        # Extract basic text features
        if text_columns:
            basic_features = self.get_basic_text_features(df, text_columns)
            features.update(basic_features)
            
            if extract_patterns:
                pattern_features = self.get_pattern_features(df, text_columns)
                features.update(pattern_features)
            
            if extract_complexity:
                complexity_features = self.get_complexity_features(df, text_columns)
                features.update(complexity_features)
        
        return features
    
    def get_basic_text_features(self, df: pd.DataFrame, 
                               text_columns: List[str]) -> Dict[str, pd.Series]:
        """Extract basic text statistics."""
        features = {}
        
        for col in text_columns:
            if col not in df.columns:
                continue
            
            # Convert to string and handle NaN
            col_lower = col.lower()
            with self._time_feature(f'{col_lower}_preprocessing'):
                text_series = df[col].fillna('').astype(str)
            
            # Length features
            with self._time_feature(f'{col_lower}_length_features'):
                features[f'{col_lower}_length'] = text_series.str.len()
                features[f'{col_lower}_word_count'] = text_series.str.split().str.len()
                features[f'{col_lower}_avg_word_length'] = text_series.apply(
                    lambda x: np.mean([len(word) for word in x.split()]) if x else 0
                )
            
            # Character type counts - log each feature individually
            with self._time_feature(f'{col_lower}_digit_count'):
                features[f'{col_lower}_digit_count'] = text_series.str.count(r'\d')
            
            with self._time_feature(f'{col_lower}_upper_count'):
                features[f'{col_lower}_upper_count'] = text_series.str.count(r'[A-Z]')
            
            with self._time_feature(f'{col_lower}_lower_count'):
                features[f'{col_lower}_lower_count'] = text_series.str.count(r'[a-z]')
            
            with self._time_feature(f'{col_lower}_space_count'):
                features[f'{col_lower}_space_count'] = text_series.str.count(r'\s')
            
            with self._time_feature(f'{col_lower}_punctuation_count'):
                features[f'{col_lower}_punctuation_count'] = text_series.str.count(r'[^\w\s]')
            
            with self._time_feature(f'{col_lower}_special_char_count'):
                features[f'{col_lower}_special_char_count'] = text_series.str.count(r'[^a-zA-Z0-9\s]')
            
            # Ratios - log each feature individually
            length_safe = features[f'{col_lower}_length'].replace(0, 1)  # Avoid division by zero
            
            with self._time_feature(f'{col_lower}_digit_ratio'):
                features[f'{col_lower}_digit_ratio'] = features[f'{col_lower}_digit_count'] / length_safe
            
            with self._time_feature(f'{col_lower}_upper_ratio'):
                features[f'{col_lower}_upper_ratio'] = features[f'{col_lower}_upper_count'] / length_safe
            
            with self._time_feature(f'{col_lower}_lower_ratio'):
                features[f'{col_lower}_lower_ratio'] = features[f'{col_lower}_lower_count'] / length_safe
            
            with self._time_feature(f'{col_lower}_punctuation_ratio'):
                features[f'{col_lower}_punctuation_ratio'] = features[f'{col_lower}_punctuation_count'] / length_safe
            
            with self._time_feature(f'{col_lower}_space_ratio'):
                features[f'{col_lower}_space_ratio'] = features[f'{col_lower}_space_count'] / length_safe
            
            # Text properties - log each feature individually
            with self._time_feature(f'{col_lower}_is_empty'):
                features[f'{col_lower}_is_empty'] = (text_series == '').astype(int)
            
            with self._time_feature(f'{col_lower}_is_numeric'):
                features[f'{col_lower}_is_numeric'] = text_series.str.match(r'^\d+$', na=False).astype(int)
            
            with self._time_feature(f'{col_lower}_is_alpha'):
                features[f'{col_lower}_is_alpha'] = text_series.str.isalpha().astype(int)
            
            with self._time_feature(f'{col_lower}_is_alphanumeric'):
                features[f'{col_lower}_is_alphanumeric'] = text_series.str.isalnum().astype(int)
            
            with self._time_feature(f'{col_lower}_starts_with_capital'):
                features[f'{col_lower}_starts_with_capital'] = text_series.str.match(r'^[A-Z]', na=False).astype(int)
            
            with self._time_feature(f'{col_lower}_is_all_caps'):
                features[f'{col_lower}_is_all_caps'] = (text_series.str.isupper() & (features[f'{col_lower}_upper_count'] > 0)).astype(int)
            
            with self._time_feature(f'{col_lower}_is_all_lower'):
                features[f'{col_lower}_is_all_lower'] = (text_series.str.islower() & (features[f'{col_lower}_lower_count'] > 0)).astype(int)
        
        return features
    
    def get_pattern_features(self, df: pd.DataFrame, 
                           text_columns: List[str]) -> Dict[str, pd.Series]:
        """Extract pattern-based features from text."""
        features = {}
        
        for col in text_columns:
            if col not in df.columns:
                continue
            
            text_series = df[col].fillna('').astype(str)
            col_lower = col.lower()
            
            # Pattern features - log each feature individually
            with self._time_feature(f'{col_lower}_has_email'):
                features[f'{col_lower}_has_email'] = text_series.str.contains(
                    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 
                    na=False, flags=re.IGNORECASE
                ).astype(int)
            
            with self._time_feature(f'{col_lower}_has_url'):
                features[f'{col_lower}_has_url'] = text_series.str.contains(
                    r'https?://|www\.', 
                    na=False, flags=re.IGNORECASE
                ).astype(int)
            
            with self._time_feature(f'{col_lower}_has_phone'):
                features[f'{col_lower}_has_phone'] = text_series.str.contains(
                    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 
                    na=False
                ).astype(int)
            
            with self._time_feature(f'{col_lower}_has_date'):
                features[f'{col_lower}_has_date'] = text_series.str.contains(
                    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', 
                    na=False
                ).astype(int)
            
            with self._time_feature(f'{col_lower}_has_time'):
                features[f'{col_lower}_has_time'] = text_series.str.contains(
                    r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b', 
                    na=False
                ).astype(int)
            
            with self._time_feature(f'{col_lower}_has_currency'):
                features[f'{col_lower}_has_currency'] = text_series.str.contains(
                    r'[$£€¥₹]\s*\d+|\d+\s*(?:USD|EUR|GBP|INR)', 
                    na=False, flags=re.IGNORECASE
                ).astype(int)
            
            with self._time_feature(f'{col_lower}_hashtag_count'):
                features[f'{col_lower}_hashtag_count'] = text_series.str.count(r'#\w+')
            
            with self._time_feature(f'{col_lower}_mention_count'):
                features[f'{col_lower}_mention_count'] = text_series.str.count(r'@\w+')
        
        return features
    
    def get_complexity_features(self, df: pd.DataFrame, 
                              text_columns: List[str]) -> Dict[str, pd.Series]:
        """Extract text complexity and sentiment indicators."""
        features = {}
        
        for col in text_columns:
            if col not in df.columns:
                continue
            
            text_series = df[col].fillna('').astype(str)
            col_lower = col.lower()
            
            # Complexity features - log each feature individually
            with self._time_feature(f'{col_lower}_sentence_count'):
                features[f'{col_lower}_sentence_count'] = text_series.str.count(r'[.!?]+')
            
            with self._time_feature(f'{col_lower}_question_count'):
                features[f'{col_lower}_question_count'] = text_series.str.count(r'\?')
            
            with self._time_feature(f'{col_lower}_exclamation_count'):
                features[f'{col_lower}_exclamation_count'] = text_series.str.count('!')
            
            with self._time_feature(f'{col_lower}_has_question'):
                features[f'{col_lower}_has_question'] = (features[f'{col_lower}_question_count'] > 0).astype(int)
            
            with self._time_feature(f'{col_lower}_parentheses_count'):
                features[f'{col_lower}_parentheses_count'] = text_series.str.count(r'[()]')
            
            with self._time_feature(f'{col_lower}_quote_count'):
                features[f'{col_lower}_quote_count'] = text_series.str.count(r'["\']')
            
            with self._time_feature(f'{col_lower}_unique_word_count'):
                features[f'{col_lower}_unique_word_count'] = text_series.apply(
                    lambda x: len(set(x.lower().split())) if x else 0
                )
            
            # Calculate word count for derived features
            word_count = text_series.str.split().str.len()
            word_count_safe = word_count.replace(0, 1)
            
            with self._time_feature(f'{col_lower}_vocabulary_richness'):
                features[f'{col_lower}_vocabulary_richness'] = (
                    features[f'{col_lower}_unique_word_count'] / word_count_safe
                )
            
            with self._time_feature(f'{col_lower}_avg_sentence_length'):
                sentence_count_safe = features[f'{col_lower}_sentence_count'].replace(0, 1)
                features[f'{col_lower}_avg_sentence_length'] = (
                    word_count / sentence_count_safe
                )
            
            with self._time_feature(f'{col_lower}_char_entropy'):
                features[f'{col_lower}_char_entropy'] = text_series.apply(self._calculate_entropy)
        
        return features
    
    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of characters in text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate probabilities and entropy
        text_length = len(text)
        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                probability = count / text_length
                entropy -= probability * np.log2(probability)
        
        return entropy


# Standalone functions for backward compatibility
def get_text_features(df: pd.DataFrame, 
                     text_columns: List[str] = [],
                     extract_patterns: bool = True,
                     extract_complexity: bool = True) -> Dict[str, pd.Series]:
    """
    Get text features from dataframe.
    
    Args:
        df: Input dataframe
        text_columns: Columns to extract text features from (auto-detect if empty)
        extract_patterns: Whether to extract pattern-based features
        extract_complexity: Whether to extract complexity features
        
    Returns:
        Dictionary of text features
    """
    text_op = TextFeatures()
    return text_op.generate_features(
        df, 
        text_columns=text_columns,
        extract_patterns=extract_patterns,
        extract_complexity=extract_complexity
    )