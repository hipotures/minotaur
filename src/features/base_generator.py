"""
Universal Feature Generator Base Class

Provides a unified interface for all feature generation with built-in:
- Signal detection at generation time
- Standardized timing and logging
- Feature metadata tracking
- Support for original, custom, and generic features
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features in the system."""
    ORIGINAL = "original"      # Raw features from CSV/dataset
    CUSTOM = "custom"          # Domain-specific engineered features
    GENERIC = "generic"        # Domain-agnostic engineered features
    DERIVED = "derived"        # Features derived from other features


@dataclass
class FeatureMetadata:
    """Metadata for a generated feature."""
    name: str
    feature_type: FeatureType
    category: str
    generation_time: float
    has_signal: bool
    source_columns: List[str] = field(default_factory=list)
    operation: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)


class UniversalFeatureGenerator(ABC):
    """
    Base class for all feature generation with unified interface.
    
    Key features:
    - Signal detection during generation (not post-hoc)
    - Standardized logging format
    - Feature metadata tracking
    - Performance optimization
    - Optional lowercase normalization
    """
    
    def __init__(self, 
                 check_signal: bool = True,
                 lowercase_columns: bool = False,
                 min_signal_ratio: float = 0.01,
                 signal_sample_size: int = 1000):
        """
        Initialize feature generator.
        
        Args:
            check_signal: Whether to check for signal during generation
            lowercase_columns: Whether to lowercase all feature names
            min_signal_ratio: Minimum ratio of unique values to consider signal
            signal_sample_size: Sample size for signal detection on large datasets
        """
        self.check_signal = check_signal
        self.lowercase_columns = lowercase_columns
        self.min_signal_ratio = min_signal_ratio
        self.signal_sample_size = signal_sample_size
        
        # Feature tracking
        self._generated_features: Dict[str, FeatureMetadata] = {}
        self._discarded_features: Dict[str, FeatureMetadata] = {}
        self._total_generation_time: float = 0.0
        
        # Performance tracking
        self._signal_check_cache: Dict[str, bool] = {}
    
    def generate_features(self, 
                         df: pd.DataFrame, 
                         feature_type: FeatureType,
                         **kwargs) -> Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]:
        """
        Main entry point for feature generation.
        
        Args:
            df: Input dataframe
            feature_type: Type of features to generate
            **kwargs: Additional parameters for specific generators
            
        Returns:
            Tuple of (features_df, metadata_dict)
        """
        start_time = time.time()
        
        # Reset tracking for this generation run
        self._reset_tracking()
        
        # Generate features based on type
        if feature_type == FeatureType.ORIGINAL:
            features_dict = self._generate_original_features(df, **kwargs)
        elif feature_type == FeatureType.CUSTOM:
            features_dict = self._generate_custom_features(df, **kwargs)
        elif feature_type == FeatureType.GENERIC:
            features_dict = self._generate_generic_features(df, **kwargs)
        elif feature_type == FeatureType.DERIVED:
            features_dict = self._generate_derived_features(df, **kwargs)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Process generated features
        processed_features = self._process_features(features_dict, df, feature_type)
        
        # Convert to DataFrame
        if processed_features:
            features_df = pd.DataFrame(processed_features)
        else:
            # Return DataFrame with placeholder column to avoid DuckDB errors
            features_df = pd.DataFrame({'_no_features_placeholder': [0] * len(df.index)}, index=df.index)
        
        # Log summary
        self._log_generation_summary(feature_type, time.time() - start_time)
        
        return features_df, self._generated_features.copy()
    
    def _process_features(self, 
                         features_dict: Dict[str, pd.Series],
                         original_df: pd.DataFrame,
                         feature_type: FeatureType) -> Dict[str, pd.Series]:
        """Process features with signal detection and metadata tracking."""
        processed = {}
        
        for feature_name, feature_series in features_dict.items():
            start_time = time.time()
            
            # Normalize name if requested
            if self.lowercase_columns:
                feature_name = feature_name.lower()
            
            # Check signal
            has_signal = True
            if self.check_signal:
                has_signal = self._check_feature_signal(feature_series)
            
            generation_time = time.time() - start_time
            
            # Create metadata
            metadata = FeatureMetadata(
                name=feature_name,
                feature_type=feature_type,
                category=self._infer_category(feature_name),
                generation_time=generation_time,
                has_signal=has_signal,
                source_columns=self._infer_source_columns(feature_name, original_df.columns),
                statistics=self._calculate_feature_statistics(feature_series)
            )
            
            # Log feature generation
            if has_signal:
                logger.debug(f"Generated feature '{feature_name}' in {generation_time:.3f}s")
                processed[feature_name] = feature_series
                self._generated_features[feature_name] = metadata
            else:
                logger.debug(f"Generated feature '{feature_name}' in {generation_time:.3f}s [no signal, discarded]")
                self._discarded_features[feature_name] = metadata
            
            self._total_generation_time += generation_time
        
        return processed
    
    def _check_feature_signal(self, feature_series: pd.Series) -> bool:
        """
        Check if feature has signal using optimized early-exit strategy.
        
        Uses caching and sampling for performance on large datasets.
        """
        # Convert to hashable representation for caching
        cache_key = str(feature_series.name) if hasattr(feature_series, 'name') else None
        if cache_key and cache_key in self._signal_check_cache:
            return self._signal_check_cache[cache_key]
        
        try:
            # For small series, check directly
            series_len = len(feature_series)
            if series_len < self.signal_sample_size * 2:
                unique_count = feature_series.nunique()
                has_signal = unique_count > 1 and (unique_count / series_len) >= self.min_signal_ratio
            else:
                # For large series, use sampling
                sample = feature_series.sample(min(self.signal_sample_size, series_len))
                unique_count = sample.nunique()
                
                # If sample has signal, full series likely has signal
                if unique_count > 1:
                    has_signal = True
                else:
                    # Double-check with another sample
                    sample2 = feature_series.sample(min(self.signal_sample_size, series_len))
                    has_signal = sample2.nunique() > 1
            
            # Cache result
            if cache_key:
                self._signal_check_cache[cache_key] = has_signal
            
            return has_signal
            
        except Exception as e:
            logger.warning(f"Error checking signal: {e}")
            return True  # Assume signal on error
    
    def _calculate_feature_statistics(self, feature_series: pd.Series) -> Dict[str, float]:
        """Calculate basic statistics for feature metadata."""
        try:
            stats = {}
            
            if feature_series.dtype in ['int64', 'float64', 'float32', 'int32']:
                stats['mean'] = float(feature_series.mean())
                stats['std'] = float(feature_series.std())
                stats['min'] = float(feature_series.min())
                stats['max'] = float(feature_series.max())
                stats['null_ratio'] = float(feature_series.isna().sum() / len(feature_series))
            else:
                stats['unique_count'] = int(feature_series.nunique())
                stats['null_ratio'] = float(feature_series.isna().sum() / len(feature_series))
                stats['mode_frequency'] = float(feature_series.value_counts().iloc[0] / len(feature_series)) if len(feature_series) > 0 else 0.0
            
            return stats
        except Exception:
            return {}
    
    def _infer_category(self, feature_name: str) -> str:
        """Infer feature category from name patterns."""
        feature_lower = feature_name.lower()
        
        # Statistical patterns
        if any(p in feature_lower for p in ['_mean_', '_std_', '_count_', '_min_', '_max_', '_sum_']):
            return 'statistical'
        # Polynomial patterns
        elif any(p in feature_lower for p in ['_squared', '_cubed', '_log', '_sqrt', '_x_']):
            return 'polynomial'
        # Binning patterns
        elif any(p in feature_lower for p in ['_bin', '_qbin', '_bucket']):
            return 'binning'
        # Ranking patterns
        elif any(p in feature_lower for p in ['_rank', '_percentile', '_quartile']):
            return 'ranking'
        # Text patterns
        elif any(p in feature_lower for p in ['_length', '_count', '_has_', '_entropy']):
            return 'text'
        # Categorical patterns
        elif any(p in feature_lower for p in ['_encoded', '_onehot', '_frequency']):
            return 'categorical'
        else:
            return 'unknown'
    
    def _infer_source_columns(self, feature_name: str, original_columns: List[str]) -> List[str]:
        """Infer source columns used to create this feature."""
        source_cols = []
        
        # Check if any original column name appears in feature name
        for col in original_columns:
            if col.lower() in feature_name.lower():
                source_cols.append(col)
        
        return source_cols
    
    def _reset_tracking(self):
        """Reset tracking for new generation run."""
        self._generated_features.clear()
        self._discarded_features.clear()
        self._total_generation_time = 0.0
        self._signal_check_cache.clear()
    
    def _log_generation_summary(self, feature_type: FeatureType, total_time: float):
        """Log summary of feature generation."""
        generated_count = len(self._generated_features)
        discarded_count = len(self._discarded_features)
        total_count = generated_count + discarded_count
        
        if total_count > 0:
            avg_time = self._total_generation_time / total_count
            logger.info(
                f"{feature_type.value.capitalize()} Features: "
                f"Generated {generated_count} features ({discarded_count} discarded) "
                f"in {total_time:.3f}s (avg: {avg_time:.3f}s/feature)"
            )
    
    def get_feature_metadata(self) -> Dict[str, FeatureMetadata]:
        """Get metadata for all generated features."""
        return self._generated_features.copy()
    
    def get_discarded_metadata(self) -> Dict[str, FeatureMetadata]:
        """Get metadata for discarded features."""
        return self._discarded_features.copy()
    
    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    def _generate_original_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate original features (from raw data)."""
        pass
    
    @abstractmethod
    def _generate_custom_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate custom domain-specific features."""
        pass
    
    @abstractmethod
    def _generate_generic_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate generic domain-agnostic features."""
        pass
    
    @abstractmethod
    def _generate_derived_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate features derived from other features."""
        pass


class FeatureGeneratorConfig:
    """Configuration for feature generation."""
    
    def __init__(self,
                 check_signal: bool = True,
                 lowercase_columns: bool = False,
                 min_signal_ratio: float = 0.01,
                 signal_sample_size: int = 1000,
                 max_features_per_type: Optional[int] = None,
                 parallel_generation: bool = False,
                 cache_features: bool = True,
                 forbidden_columns: Optional[List[str]] = None):
        """
        Initialize configuration.
        
        Args:
            check_signal: Check for signal during generation
            lowercase_columns: Lowercase all feature names
            min_signal_ratio: Minimum unique value ratio for signal
            signal_sample_size: Sample size for signal detection
            max_features_per_type: Maximum features per type
            parallel_generation: Enable parallel feature generation
            cache_features: Cache generated features
            forbidden_columns: Columns to exclude from feature generation
        """
        self.check_signal = check_signal
        self.lowercase_columns = lowercase_columns
        self.min_signal_ratio = min_signal_ratio
        self.signal_sample_size = signal_sample_size
        self.max_features_per_type = max_features_per_type
        self.parallel_generation = parallel_generation
        self.cache_features = cache_features
        self.forbidden_columns = forbidden_columns or []
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureGeneratorConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames})