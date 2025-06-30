"""
Base Classes for Feature Engineering

Provides abstract base classes and mixins for all feature operations.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FeatureTimingMixin:
    """Mixin for adding timing capabilities to feature operations."""
    
    def __init__(self):
        self._feature_timings = {}
        self._class_total_time = 0.0
        self._feature_count = 0
        self._check_signal = True  # Default to checking signal
        self._signal_sample_size = 1000  # Sample size for large datasets
        self._signal_cache = {}  # Cache signal check results
        self._lowercase_features = False  # Option to lowercase feature names
    
    @contextmanager
    def _time_feature(self, feature_name: str, features_dict: dict = None):
        """Context manager for timing individual feature generation with signal checking."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            # Apply lowercase if enabled
            if self._lowercase_features:
                original_name = feature_name
                feature_name = feature_name.lower()
                if features_dict is not None and original_name in features_dict and original_name != feature_name:
                    features_dict[feature_name] = features_dict.pop(original_name)
            
            self._feature_timings[feature_name] = duration
            self._class_total_time += duration
            self._feature_count += 1
            
            # Check signal if features_dict provided and feature was added and signal checking is enabled
            if features_dict is not None and feature_name in features_dict and self._check_signal:
                if self._check_feature_signal(features_dict[feature_name]):
                    logger.debug(f"Generated feature '{feature_name}' in {duration:.3f}s")
                else:
                    logger.debug(f"Generated feature '{feature_name}' in {duration:.3f}s [no signal, discarded]")
                    # Remove the feature from dictionary
                    del features_dict[feature_name]
            else:
                # No signal checking - just log timing
                logger.debug(f"Generated feature '{feature_name}' in {duration:.3f}s")
    
    def _check_feature_signal(self, feature_series: pd.Series) -> bool:
        """Check if feature has signal (different values) using early exit strategy with caching."""
        try:
            # Create cache key from series characteristics
            cache_key = None
            if hasattr(feature_series, 'name') and feature_series.name:
                cache_key = f"{feature_series.name}_{len(feature_series)}_{feature_series.dtype}"
                if cache_key in self._signal_cache:
                    return self._signal_cache[cache_key]
            
            # Quick check for empty or all-null series
            non_null_count = feature_series.count()
            if non_null_count == 0:
                result = False
            elif non_null_count == 1:
                result = False
            else:
                # For small series, check directly
                if len(feature_series) <= self._signal_sample_size * 2:
                    unique_count = feature_series.nunique()
                    result = unique_count > 1
                else:
                    # For large series, use sampling strategy
                    sample_size = min(self._signal_sample_size, len(feature_series))
                    sample = feature_series.dropna().sample(n=sample_size, random_state=42)
                    
                    # If sample has multiple values, feature has signal
                    if sample.nunique() > 1:
                        result = True
                    else:
                        # Double-check with different sample
                        sample2 = feature_series.dropna().sample(n=sample_size, random_state=123)
                        result = sample2.nunique() > 1
            
            # Cache the result
            if cache_key:
                self._signal_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.debug(f"Error in signal check: {e}")
            return True  # Conservative: assume signal on error
    
    def _add_feature_with_signal_check(self, features_dict: dict, feature_name: str, feature_series, operation_name: str = "unknown"):
        """Add feature to dictionary only if it has signal."""
        import pandas as pd
        
        if not isinstance(feature_series, pd.Series):
            # Convert to Series if needed
            feature_series = pd.Series(feature_series)
        
        if self._check_feature_signal(feature_series):
            features_dict[feature_name] = feature_series
            logger.debug(f"Added feature '{feature_name}' (has signal)")
        else:
            logger.debug(f"Generated feature '{feature_name}' [no signal, discarded]")
            # Don't add to features_dict
    
    def log_timing_summary(self, operation_name: str):
        """Log timing summary at INFO level."""
        if self._feature_count > 0:
            avg_time = self._class_total_time / self._feature_count
            logger.info(
                f"{operation_name}: Generated {self._feature_count} features "
                f"in {self._class_total_time:.3f}s (avg: {avg_time:.3f}s/feature)"
            )
    
    def reset_timings(self):
        """Reset timing statistics."""
        self._feature_timings.clear()
        self._class_total_time = 0.0
        self._feature_count = 0
        self._signal_cache.clear()
    
    def configure(self, 
                  check_signal: Optional[bool] = None,
                  signal_sample_size: Optional[int] = None,
                  lowercase_features: Optional[bool] = None):
        """Configure mixin options."""
        if check_signal is not None:
            self._check_signal = check_signal
        if signal_sample_size is not None:
            self._signal_sample_size = signal_sample_size
        if lowercase_features is not None:
            self._lowercase_features = lowercase_features


class AbstractFeatureOperation(ABC):
    """Abstract base class for all feature operations."""
    
    @abstractmethod
    def generate_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        Generate features from the input dataframe.
        
        Args:
            df: Input dataframe
            **kwargs: Additional parameters specific to the operation
            
        Returns:
            Dictionary mapping feature names to pandas Series
        """
        pass
    
    @abstractmethod
    def get_operation_name(self) -> str:
        """Return the name of this feature operation."""
        pass
    
    @abstractmethod
    def validate_input(self, df: pd.DataFrame) -> bool:
        """
        Validate that the input dataframe is suitable for this operation.
        
        Returns:
            True if valid, False otherwise
        """
        pass


class GenericFeatureOperation(AbstractFeatureOperation, FeatureTimingMixin):
    """Base class for generic feature operations."""
    
    def __init__(self):
        super().__init__()
        FeatureTimingMixin.__init__(self)
        self._init_parameters()
        self._auto_registration_enabled = True  # Enable auto-registration by default
    
    def _init_parameters(self):
        """Initialize operation-specific parameters."""
        pass
    
    def generate_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate features with timing and auto-registration."""
        self.reset_timings()
        
        if not self.validate_input(df):
            logger.warning(f"{self.get_operation_name()}: Invalid input dataframe")
            return {}
        
        try:
            features = self._generate_features_impl(df, **kwargs)
            self.log_timing_summary(self.get_operation_name())
            
            # Auto-register operation and features if enabled
            if self._auto_registration_enabled and features:
                self._auto_register_operation_metadata(features, **kwargs)
            
            return features
        except Exception as e:
            logger.error(f"Error in {self.get_operation_name()}: {e}")
            return {}
    
    def _auto_register_operation_metadata(self, features: Dict[str, pd.Series], **kwargs):
        """
        Automatically register operation metadata in the database.
        
        Args:
            features: Generated features dictionary
            **kwargs: Additional parameters passed to generate_features
        """
        try:
            # Try to use global database connection if available
            import duckdb
            from src.project_root import PROJECT_ROOT
            import os
            
            # Connect to the main database directly
            db_path = os.path.join(PROJECT_ROOT, 'data', 'minotaur.duckdb')
            if not os.path.exists(db_path):
                logger.debug("Database not found, skipping auto-registration")
                return
            
            # Get operation details
            operation_name = self.get_operation_name()
            
            # Get operation metadata from generic registry
            from src.features.generic import get_operation_metadata
            
            # Map human-readable names to metadata keys
            operation_key = self._map_operation_name_to_key(operation_name)
            metadata = get_operation_metadata(operation_key)
            
            if not metadata:
                logger.debug(f"No metadata found for operation '{operation_key}' (from '{operation_name}'), skipping auto-registration")
                return
            
            # Connect directly to database for auto-registration
            with duckdb.connect(db_path) as conn:
                # Register or update operation category
                conn.execute("""
                    INSERT INTO operation_categories (operation_name, category, description, is_generic, output_patterns)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (operation_name) DO UPDATE SET
                        category = EXCLUDED.category,
                        description = EXCLUDED.description,
                        is_generic = EXCLUDED.is_generic,
                        output_patterns = EXCLUDED.output_patterns
                """, [
                    operation_name,
                    metadata.get('category', 'unknown'),
                    metadata.get('description', f'Generic {operation_name} operation'),
                    True,  # is_generic = True
                    metadata.get('output_patterns', [])
                ])
                
                # Register individual features
                for feature_name, feature_series in features.items():
                    # Detect operation from feature name for validation
                    detected_op = self._detect_operation_from_feature_name(feature_name, metadata)
                    
                    conn.execute("""
                        INSERT INTO feature_catalog (feature_name, feature_category, python_code, operation_name, description)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (feature_name) DO UPDATE SET
                            operation_name = EXCLUDED.operation_name,
                            feature_category = EXCLUDED.feature_category,
                            description = EXCLUDED.description
                    """, [
                        feature_name,
                        metadata.get('category', 'unknown'),
                        f"Generated by {operation_name}",  # python_code placeholder
                        operation_name,
                        f"Feature generated by {operation_name} operation"
                    ])
                
                logger.debug(f"Auto-registered {len(features)} features for operation '{operation_name}'")
                
        except Exception as e:
            logger.debug(f"Auto-registration failed for operation '{self.get_operation_name()}': {e}")
            # Don't raise the error - auto-registration is optional
    
    def _detect_operation_from_feature_name(self, feature_name: str, metadata: Dict[str, Any]) -> str:
        """Validate that feature name matches operation patterns."""
        patterns = metadata.get('output_patterns', [])
        feature_lower = feature_name.lower()
        
        for pattern in patterns:
            if pattern.lower() in feature_lower:
                return self.get_operation_name()
        
        return self.get_operation_name()  # Default to current operation
    
    def _map_operation_name_to_key(self, operation_name: str) -> str:
        """
        Map human-readable operation names to metadata keys.
        
        Args:
            operation_name: Human-readable operation name (e.g., "Statistical Aggregations")
            
        Returns:
            Metadata key (e.g., "statistical_aggregations")
        """
        # Create mapping from human-readable names to metadata keys
        name_mapping = {
            "Statistical Aggregations": "statistical_aggregations",
            "Polynomial Features": "polynomial_features",
            "Binning Features": "binning_features", 
            "Ranking Features": "ranking_features",
            "Temporal Features": "temporal_features",
            "Text Features": "text_features",
            "Categorical Features": "categorical_features",
            "Interaction Features": "interaction_features",
        }
        
        # Try exact match first
        if operation_name in name_mapping:
            return name_mapping[operation_name]
        
        # Try converting to lowercase with underscores
        converted = operation_name.lower().replace(' ', '_').replace('-', '_')
        
        # Check if converted version exists in metadata
        from src.features.generic import OPERATION_METADATA
        if converted in OPERATION_METADATA:
            return converted
        
        # Return original name as fallback
        return operation_name
    
    @abstractmethod
    def _generate_features_impl(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Implementation of feature generation (to be overridden by subclasses)."""
        pass
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Basic validation for generic operations."""
        if df is None or df.empty:
            return False
        return True
    
    def _filter_forbidden_columns(self, columns: List[str]) -> List[str]:
        """Filter out common forbidden columns (target, ID, etc.)."""
        # Common forbidden column patterns
        forbidden_patterns = [
            'target',    # Generic target
            'label',     # Generic label
            'price',     # Common target
            'id',        # ID columns
            'row_id',    # Common ID pattern
            'sample_id', # Common ID pattern
        ]
        
        filtered_columns = []
        for col in columns:
            col_lower = col.lower()
            # Check if column matches any forbidden pattern
            is_forbidden = any(pattern in col_lower for pattern in forbidden_patterns)
            if not is_forbidden:
                filtered_columns.append(col)
            else:
                logger.debug(f"Filtered out forbidden column: {col}")
        
        return filtered_columns
    
    @staticmethod
    def _clean_numeric_data(series: pd.Series, 
                          fill_value: Optional[float] = None,
                          clip_values: bool = True) -> pd.Series:
        """
        Clean numeric data by handling NaN, infinities, and extreme values.
        
        Args:
            series: Input pandas Series
            fill_value: Value to use for filling NaN (if None, uses median)
            clip_values: Whether to clip extreme values
            
        Returns:
            Cleaned pandas Series
        """
        MAX_VALUE = 1e15
        MIN_VALUE = -1e15
        
        if series.empty or series.isna().all():
            return pd.Series(0, index=series.index)
        
        # Create a copy to avoid modifying original data
        cleaned = series.copy()
        
        # Handle infinities
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
            cleaned = cleaned.clip(lower=MIN_VALUE, upper=MAX_VALUE)
        
        return cleaned
    
    @staticmethod
    def _safe_divide(numerator: Union[pd.Series, np.ndarray], 
                    denominator: Union[pd.Series, np.ndarray, float]) -> pd.Series:
        """Safely divide two arrays/series handling division by zero."""
        EPSILON = 1e-10
        
        if isinstance(denominator, (int, float)):
            if abs(denominator) < EPSILON:
                return pd.Series(0, index=numerator.index if hasattr(numerator, 'index') else None)
        
        # Convert to numpy arrays for calculation
        num_array = np.array(numerator)
        den_array = np.array(denominator)
        
        # Replace zeros and very small values in denominator
        den_array = np.where(np.abs(den_array) < EPSILON, EPSILON, den_array)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            result = num_array / den_array
            result = np.where(~np.isfinite(result), 0, result)
        
        if hasattr(numerator, 'index'):
            return pd.Series(result, index=numerator.index)
        return pd.Series(result)


class CustomFeatureOperation(AbstractFeatureOperation, FeatureTimingMixin):
    """Base class for custom domain-specific feature operations."""
    
    def __init__(self, domain_name: str):
        super().__init__()
        FeatureTimingMixin.__init__(self)
        self.domain_name = domain_name
        self._auto_registration_enabled = True  # Enable auto-registration by default
    
    def generate_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate features with timing and auto-registration."""
        self.reset_timings()
        
        if not self.validate_input(df):
            logger.warning(f"{self.get_operation_name()}: Invalid input for domain {self.domain_name}")
            return {}
        
        try:
            features = self._generate_features_impl(df, **kwargs)
            self.log_timing_summary(f"{self.domain_name} - {self.get_operation_name()}")
            
            # Auto-register operation and features if enabled
            if self._auto_registration_enabled and features:
                self._auto_register_custom_operation_metadata(features, **kwargs)
            
            return features
        except Exception as e:
            logger.error(f"Error in {self.domain_name} {self.get_operation_name()}: {e}")
            return {}
    
    def _auto_register_custom_operation_metadata(self, features: Dict[str, pd.Series], **kwargs):
        """
        Automatically register custom operation metadata in the database.
        
        Args:
            features: Generated features dictionary
            **kwargs: Additional parameters passed to generate_features
        """
        try:
            # Connect to database directly for auto-registration
            import duckdb
            from src.project_root import PROJECT_ROOT
            import os
            
            # Connect to the main database
            db_path = os.path.join(PROJECT_ROOT, 'data', 'minotaur.duckdb')
            if not os.path.exists(db_path):
                logger.debug("Database not found, skipping auto-registration")
                return
            
            # Get operation details
            operation_name = self.get_operation_name()
            
            # Infer output patterns from generated features
            output_patterns = self._infer_output_patterns(features)
            
            # Connect directly to database for auto-registration
            with duckdb.connect(db_path) as conn:
                # Register or update operation category
                conn.execute("""
                    INSERT INTO operation_categories (operation_name, category, description, dataset_name, is_generic, output_patterns)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (operation_name) DO UPDATE SET
                        category = EXCLUDED.category,
                        description = EXCLUDED.description,
                        dataset_name = EXCLUDED.dataset_name,
                        is_generic = EXCLUDED.is_generic,
                        output_patterns = EXCLUDED.output_patterns
                """, [
                    operation_name,
                    'custom_domain',  # category for custom operations
                    f'Custom {operation_name} operation for {self.domain_name}',
                    self.domain_name,  # dataset_name
                    False,  # is_generic = False
                    output_patterns
                ])
                
                # Register individual features
                for feature_name, feature_series in features.items():
                    conn.execute("""
                        INSERT INTO feature_catalog (feature_name, feature_category, python_code, operation_name, description)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (feature_name) DO UPDATE SET
                            operation_name = EXCLUDED.operation_name,
                            feature_category = EXCLUDED.feature_category,
                            description = EXCLUDED.description
                    """, [
                        feature_name,
                        'custom_domain',
                        f"Generated by {operation_name} for {self.domain_name}",  # python_code placeholder
                        operation_name,
                        f"Custom feature for {self.domain_name} dataset"
                    ])
                
                logger.debug(f"Auto-registered {len(features)} custom features for operation '{operation_name}' (domain: {self.domain_name})")
                
        except Exception as e:
            logger.debug(f"Auto-registration failed for custom operation '{self.get_operation_name()}': {e}")
            # Don't raise the error - auto-registration is optional
    
    def _infer_output_patterns(self, features: Dict[str, pd.Series]) -> List[str]:
        """
        Infer output patterns from generated feature names.
        
        Args:
            features: Generated features dictionary
            
        Returns:
            List of common patterns found in feature names
        """
        if not features:
            return []
        
        # Extract common patterns from feature names
        patterns = set()
        feature_names = list(features.keys())
        
        # Look for common suffixes and infixes
        for feature_name in feature_names:
            parts = feature_name.lower().split('_')
            
            # Look for patterns like '_something_'
            for i in range(len(parts) - 1):
                if len(parts[i]) > 2:  # Ignore very short parts
                    patterns.add(f'_{parts[i]}_')
            
            # Look for suffixes
            if len(parts) > 1 and len(parts[-1]) > 2:
                patterns.add(f'_{parts[-1]}')
        
        # Limit to most common patterns (max 10)
        return list(patterns)[:10]
    
    @abstractmethod
    def _generate_features_impl(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Implementation of feature generation (to be overridden by subclasses)."""
        pass
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Basic validation that can be extended by subclasses."""
        if df is None or df.empty:
            return False
        return True
    
    @staticmethod
    def _safe_divide(numerator: Union[pd.Series, np.ndarray], 
                    denominator: Union[pd.Series, np.ndarray, float]) -> pd.Series:
        """Safely divide two arrays/series handling division by zero."""
        EPSILON = 1e-10
        
        if isinstance(denominator, (int, float)):
            if abs(denominator) < EPSILON:
                return pd.Series(0, index=numerator.index if hasattr(numerator, 'index') else None)
        
        # Convert to numpy arrays for calculation
        num_array = np.array(numerator)
        den_array = np.array(denominator)
        
        # Replace zeros and very small values in denominator
        den_array = np.where(np.abs(den_array) < EPSILON, EPSILON, den_array)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            result = num_array / den_array
            result = np.where(~np.isfinite(result), 0, result)
        
        if hasattr(numerator, 'index'):
            return pd.Series(result, index=numerator.index)
        return pd.Series(result)