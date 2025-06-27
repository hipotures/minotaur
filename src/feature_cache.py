"""
MD5-based Feature Cache Manager

Provides optimized feature caching with MD5-based dataset identification.
Caches features per dataset to avoid recomputation across sessions.
"""

import hashlib
import logging
from pathlib import Path
from typing import Union
import pandas as pd
import time

logger = logging.getLogger(__name__)

class FeatureCacheManager:
    """MD5-based feature cache manager for efficient feature storage and retrieval."""
    
    def __init__(self, data_path: str):
        """Initialize cache manager with data path hash."""
        self.data_path = data_path.rstrip('/')
        self.path_hash = hashlib.md5(self.data_path.encode()).hexdigest()
        self.cache_dir = Path(f"data/{self.path_hash}")
        self.features_dir = self.cache_dir / "features"
        
        logger.info(f"🔗 Cache manager initialized for: {self.data_path}")
        logger.info(f"📁 Cache directory: {self.cache_dir}")
        
    def get_dataset_cache_dir(self) -> Path:
        """Get the cache directory for this dataset."""
        return self.cache_dir
        
    def ensure_base_datasets(self, train_path: str, test_path: str):
        """Sprawdz/wygeneruj podstawowe pliki parquet."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Train parquet
        train_parquet = self.cache_dir / "train.parquet"
        if not train_parquet.exists():
            logger.info(f"💾 Converting {train_path} to parquet...")
            df = pd.read_csv(train_path)
            df.to_parquet(train_parquet, index=False)
            logger.info(f"✅ Created train.parquet cache ({len(df)} rows)")
        else:
            logger.info(f"📂 Using cached train.parquet")
            
        # Test parquet  
        test_parquet = self.cache_dir / "test.parquet"
        if not test_parquet.exists():
            logger.info(f"💾 Converting {test_path} to parquet...")
            df = pd.read_csv(test_path)
            df.to_parquet(test_parquet, index=False)
            logger.info(f"✅ Created test.parquet cache ({len(df)} rows)")
        else:
            logger.info(f"📂 Using cached test.parquet")
            
    def get_feature_path(self, feature_name: str, data_type: str) -> Path:
        """Get path for a specific feature file."""
        return self.features_dir / data_type / f"{feature_name}.parquet"
        
    def is_feature_cached(self, feature_name: str, data_type: str) -> bool:
        """Check if feature is already cached."""
        return self.get_feature_path(feature_name, data_type).exists()
        
    def load_feature(self, feature_name: str, data_type: str) -> pd.Series:
        """Load cached feature."""
        path = self.get_feature_path(feature_name, data_type)
        logger.debug(f"📂 Loading cached feature: {feature_name} ({data_type})")
        return pd.read_parquet(path).iloc[:, 0]
        
    def save_feature(self, feature_name: str, data_type: str, feature_data: pd.Series):
        """Save feature to cache."""
        path = self.get_feature_path(feature_name, data_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({feature_name: feature_data})
        df.to_parquet(path, index=False)
        logger.info(f"💾 Cached new feature: {feature_name} ({data_type})")
        
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if not self.features_dir.exists():
            return {"total_features": 0, "cache_size_mb": 0}
            
        total_features = 0
        total_size = 0
        
        for data_type_dir in self.features_dir.iterdir():
            if data_type_dir.is_dir():
                for feature_file in data_type_dir.glob("*.parquet"):
                    total_features += 1
                    total_size += feature_file.stat().st_size
                    
        return {
            "total_features": total_features,
            "cache_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
    
    def build_feature_if_missing(self, feature_name: str, data_type: str, build_func):
        """Build feature only if not in cache, with detailed logging."""
        if self.is_feature_cached(feature_name, data_type):
            logger.info(f"📂 CACHE HIT: {feature_name} ({data_type})")
            return self.load_feature(feature_name, data_type)
        else:
            logger.info(f"🔨 CACHE MISS: Building {feature_name} ({data_type})")
            start_time = time.time()
            feature_data = build_func()
            build_time = time.time() - start_time
            self.save_feature(feature_name, data_type, feature_data)
            logger.info(f"✅ BUILT & CACHED: {feature_name} ({data_type}) in {build_time:.2f}s")
            return feature_data
    
    def batch_build_features(self, feature_definitions: list):
        """Build all missing features in batch at startup."""
        missing_features = []
        total_features = len(feature_definitions)
        
        # Check which features are missing
        for feature_def in feature_definitions:
            if not self.is_feature_cached(feature_def['name'], feature_def['data_type']):
                missing_features.append(feature_def)
                
        missing_count = len(missing_features)
        
        if missing_count > 0:
            logger.info(f"🏗️ Building {missing_count}/{total_features} missing features...")
            
            # Build each missing feature with progress tracking
            for i, feature_def in enumerate(missing_features, 1):
                logger.info(f"📊 Progress: {i}/{missing_count} - Building {feature_def['name']}")
                self.build_feature_if_missing(
                    feature_def['name'], 
                    feature_def['data_type'], 
                    feature_def['build_func']
                )
            
            logger.info(f"✅ Completed building {missing_count} features")
        else:
            logger.info(f"✅ All {total_features} features cached - no building needed")
    
    def get_missing_features(self, feature_definitions: list) -> list:
        """Get list of missing features that need to be built."""
        missing = []
        for feature_def in feature_definitions:
            if not self.is_feature_cached(feature_def['name'], feature_def['data_type']):
                missing.append(feature_def)
        return missing
    
    def clear_cache(self, data_type: str = None):
        """Clear cache for specific data type or all cache."""
        if data_type:
            data_type_dir = self.features_dir / data_type
            if data_type_dir.exists():
                import shutil
                shutil.rmtree(data_type_dir)
                logger.info(f"🗑️ Cleared {data_type} cache")
        else:
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                logger.info(f"🗑️ Cleared entire cache: {self.cache_dir}")
    
    def validate_cache_integrity(self) -> dict:
        """Validate cache file integrity."""
        validation_results = {"valid": 0, "corrupted": 0, "missing": 0}
        
        if not self.features_dir.exists():
            return validation_results
            
        for data_type_dir in self.features_dir.iterdir():
            if data_type_dir.is_dir():
                for feature_file in data_type_dir.glob("*.parquet"):
                    try:
                        # Try to read the parquet file
                        pd.read_parquet(feature_file)
                        validation_results["valid"] += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Corrupted cache file: {feature_file} - {e}")
                        validation_results["corrupted"] += 1
                        
        return validation_results