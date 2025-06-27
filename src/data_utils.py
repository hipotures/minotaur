"""
Data Utilities for Fast Loading and Caching

Provides optimized data loading with parquet support, memory management,
and intelligent caching for the MCTS feature discovery system.
"""

import os
import time
import pickle
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

from .timing import timed, timing_context, record_timing

logger = logging.getLogger(__name__)

class DataManager:
    """Optimized data manager with parquet support and caching."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data manager."""
        self.config = config
        self.cache_dir = Path(config.get('resources', {}).get('temp_dir', '/tmp/mcts_features'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.enable_cache = config.get('feature_space', {}).get('cache_features', True)
        self.max_cache_size_mb = config.get('feature_space', {}).get('max_cache_size_mb', 2048)
        self.prefer_parquet = config.get('data', {}).get('prefer_parquet', True)
        
        # Memory tracking
        self.cache_registry: Dict[str, Dict[str, Any]] = {}
        self.current_cache_size_mb = 0.0
        
        logger.info(f"Initialized DataManager with cache at: {self.cache_dir}")
    
    @timed("data.load_dataset", include_memory=True)
    def load_dataset(self, file_path: str, data_type: str = 'auto', use_cache: bool = True) -> pd.DataFrame:
        """
        Load dataset with automatic format detection, caching, and smart sampling.
        
        Args:
            file_path: Path to data file
            data_type: Type hint ('train', 'test', 'auto')
            use_cache: Whether to use cached version if available
            
        Returns:
            pd.DataFrame: Loaded dataset (potentially sampled)
        """
        file_path = Path(file_path)
        
        # Get data configuration
        data_config = self.config.get('data', {})
        testing_config = self.config.get('testing', {})
        
        # Determine if we should use small dataset
        use_small_dataset = (
            data_config.get('use_small_dataset', False) or 
            testing_config.get('use_small_dataset', False)
        )
        
        small_dataset_size = (
            data_config.get('small_dataset_size', 5000) or 
            testing_config.get('small_dataset_size', 5000)
        )
        
        # Create cache key that includes sampling parameters
        cache_suffix = f"_small_{small_dataset_size}" if use_small_dataset else ""
        cache_key = self._get_cache_key(str(file_path), data_type + cache_suffix)
        
        # Check cache first
        if use_cache and self.enable_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Loaded {data_type} data from cache: {file_path.name}")
                return cached_data
        
        # Determine optimal loading strategy
        parquet_path = file_path.with_suffix('.parquet')
        
        if self.prefer_parquet and parquet_path.exists():
            # Load from parquet (faster)
            logger.debug(f"Loading from parquet: {parquet_path}")
            df = self._load_parquet(parquet_path)
        elif file_path.suffix.lower() == '.parquet':
            # Direct parquet file
            df = self._load_parquet(file_path)
        elif file_path.suffix.lower() == '.csv':
            # Load CSV and optionally convert to parquet
            logger.debug(f"Loading from CSV: {file_path}")
            df = self._load_csv(file_path)
            
            # Save as parquet for faster future loading
            if self.prefer_parquet and data_config.get('auto_convert_csv', True):
                self._save_parquet(df, parquet_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Apply smart sampling if enabled
        if use_small_dataset and len(df) > small_dataset_size:
            logger.info(f"Applying smart sampling: {len(df)} -> {small_dataset_size} rows")
            
            # Determine stratification column
            stratify_col = data_config.get('stratify_column')
            if stratify_col and stratify_col not in df.columns:
                stratify_col = None
                
            df = smart_sample(df, small_dataset_size, stratify_col)
            logger.info(f"Dataset sampled to {len(df)} rows")
        
        # Apply memory optimization if enabled
        if data_config.get('dtype_optimization', True):
            df = self._optimize_dtypes(df)
        
        # Check memory limit
        memory_limit_mb = data_config.get('memory_limit_mb', 500)
        current_memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        if current_memory_mb > memory_limit_mb:
            logger.warning(f"Dataset exceeds memory limit ({current_memory_mb:.1f}MB > {memory_limit_mb}MB)")
            
            # Suggest sampling
            recommend = recommend_sampling_strategy(df, memory_limit_mb)
            if recommend['sample_needed']:
                logger.info(f"Auto-sampling to meet memory limit...")
                df = smart_sample(df, recommend['recommended_sample_size'], 
                                data_config.get('stratify_column'))
                logger.info(f"Auto-sampled to {len(df)} rows ({recommend['estimated_final_size_mb']:.1f}MB)")
        
        # Cache the processed data
        if use_cache and self.enable_cache:
            self._save_to_cache(cache_key, df, str(file_path))
        
        logger.info(f"Loaded {data_type} dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    @timed("data.load_csv")
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV with optimized settings."""
        # Optimize dtypes for memory efficiency
        with timing_context("data.csv_read"):
            df = pd.read_csv(file_path, low_memory=False)
        
        with timing_context("data.optimize_dtypes"):
            df = self._optimize_dtypes(df)
        
        return df
    
    @timed("data.load_parquet")
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load parquet file."""
        return pd.read_parquet(file_path)
    
    @timed("data.save_parquet")
    def _save_parquet(self, df: pd.DataFrame, file_path: Path) -> None:
        """Save DataFrame as parquet."""
        try:
            df.to_parquet(file_path, compression='snappy', index=False)
            logger.debug(f"Saved parquet file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to save parquet file {file_path}: {e}")
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency."""
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if reasonable cardinality
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    try:
                        df[col] = df[col].astype('category')
                    except:
                        pass
                        
            elif 'int' in str(col_type):
                # Downcast integers
                df[col] = pd.to_numeric(df[col], downcast='integer')
                
            elif 'float' in str(col_type):
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        
        if memory_reduction > 5:  # Only log if significant reduction
            logger.debug(f"Memory optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB ({memory_reduction:.1f}% reduction)")
        
        return df
    
    def _get_cache_key(self, file_path: str, data_type: str) -> str:
        """Generate cache key for file."""
        # Include file modification time in key for invalidation
        try:
            mtime = os.path.getmtime(file_path)
            content = f"{file_path}:{data_type}:{mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except:
            # Fallback if file doesn't exist
            content = f"{file_path}:{data_type}"
            return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache."""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with timing_context("data.cache_load"):
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Update cache registry
                if cache_key in self.cache_registry:
                    self.cache_registry[cache_key]['last_accessed'] = time.time()
                
                return cached_data
                
        except Exception as e:
            logger.warning(f"Failed to load cached data {cache_key}: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except:
                pass
            return None
    
    def _save_to_cache(self, cache_key: str, df: pd.DataFrame, source_path: str) -> None:
        """Save DataFrame to cache."""
        if not self.enable_cache:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Check cache size limits
            df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            if df_size_mb > self.max_cache_size_mb * 0.3:  # Don't cache if > 30% of limit
                logger.debug(f"Skipping cache for large dataset: {df_size_mb:.1f}MB")
                return
            
            # Cleanup cache if needed
            if self.current_cache_size_mb + df_size_mb > self.max_cache_size_mb:
                self._cleanup_cache()
            
            with timing_context("data.cache_save"):
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update registry
            self.cache_registry[cache_key] = {
                'file_path': cache_path,
                'source_path': source_path,
                'size_mb': df_size_mb,
                'created_at': time.time(),
                'last_accessed': time.time()
            }
            
            self.current_cache_size_mb += df_size_mb
            logger.debug(f"Cached dataset: {cache_key} ({df_size_mb:.1f}MB)")
            
        except Exception as e:
            logger.warning(f"Failed to save to cache {cache_key}: {e}")
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache files to free space."""
        if not self.cache_registry:
            return
        
        # Sort by last accessed time (oldest first)
        items_by_age = sorted(
            self.cache_registry.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest items until we're under 70% of limit
        target_size = self.max_cache_size_mb * 0.7
        current_size = self.current_cache_size_mb
        
        for cache_key, metadata in items_by_age:
            if current_size <= target_size:
                break
            
            try:
                # Remove file
                file_path = metadata['file_path']
                if file_path.exists():
                    file_path.unlink()
                
                # Update tracking
                current_size -= metadata['size_mb']
                del self.cache_registry[cache_key]
                
                logger.debug(f"Removed cached file: {cache_key}")
                
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_key}: {e}")
        
        self.current_cache_size_mb = current_size
        logger.info(f"Cache cleanup completed. Size: {current_size:.1f}MB")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_cached_files': len(self.cache_registry),
            'total_size_mb': self.current_cache_size_mb,
            'max_size_mb': self.max_cache_size_mb,
            'usage_percent': (self.current_cache_size_mb / self.max_cache_size_mb) * 100,
            'cache_dir': str(self.cache_dir),
            'files': list(self.cache_registry.keys())
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        for cache_key, metadata in self.cache_registry.items():
            try:
                file_path = metadata['file_path']
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_key}: {e}")
        
        self.cache_registry.clear()
        self.current_cache_size_mb = 0.0
        logger.info("Cache cleared")
    
    def preload_datasets(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """Preload multiple datasets for faster access."""
        logger.info(f"Preloading {len(file_paths)} datasets...")
        
        datasets = {}
        for file_path in file_paths:
            try:
                dataset_name = Path(file_path).stem
                datasets[dataset_name] = self.load_dataset(file_path, dataset_name)
            except Exception as e:
                logger.error(f"Failed to preload {file_path}: {e}")
        
        logger.info(f"Preloaded {len(datasets)} datasets successfully")
        return datasets

# Utility functions

def estimate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """Estimate memory usage of DataFrame."""
    memory_usage = df.memory_usage(deep=True)
    
    return {
        'total_mb': memory_usage.sum() / 1024 / 1024,
        'per_column_mb': {
            col: memory_usage[col] / 1024 / 1024 
            for col in memory_usage.index
        },
        'rows': len(df),
        'columns': len(df.columns)
    }

def recommend_sampling_strategy(df: pd.DataFrame, target_size_mb: float = 100.0) -> Dict[str, Any]:
    """Recommend sampling strategy to reduce memory usage."""
    current_memory = estimate_memory_usage(df)
    current_mb = current_memory['total_mb']
    
    if current_mb <= target_size_mb:
        return {
            'sample_needed': False,
            'current_size_mb': current_mb,
            'target_size_mb': target_size_mb
        }
    
    # Calculate required sampling ratio
    sample_ratio = target_size_mb / current_mb
    sample_size = int(len(df) * sample_ratio)
    
    return {
        'sample_needed': True,
        'current_size_mb': current_mb,
        'target_size_mb': target_size_mb,
        'sample_ratio': sample_ratio,
        'recommended_sample_size': sample_size,
        'estimated_final_size_mb': current_mb * sample_ratio
    }

@timed("data.smart_sample")
def smart_sample(df: pd.DataFrame, target_rows: int, stratify_column: str = None) -> pd.DataFrame:
    """Intelligent sampling with optional stratification."""
    if len(df) <= target_rows:
        return df
    
    if stratify_column and stratify_column in df.columns:
        # Stratified sampling
        try:
            return df.groupby(stratify_column, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(target_rows * len(x) / len(df)))))
            ).reset_index(drop=True)
        except:
            # Fallback to simple sampling
            pass
    
    # Simple random sampling
    return df.sample(n=target_rows, random_state=42).reset_index(drop=True)