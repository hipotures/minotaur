"""
Feature Space Adapter

Adapts the new UniversalFeatureGenerator and Pipeline to work with
the existing FeatureSpace interface for backward compatibility.
"""

import time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

from .generator import MinotaurFeatureGenerator
from .pipeline import FeaturePipelineManager, PipelineConfig
from .base_generator import FeatureGeneratorConfig, FeatureType

logger = logging.getLogger(__name__)


class FeatureSpaceAdapter:
    """
    Adapter that allows the existing FeatureSpace to use the new 
    generator and pipeline architecture while maintaining compatibility.
    """
    
    def __init__(self, config: Dict[str, Any], duckdb_manager=None):
        """Initialize adapter with configuration."""
        self.config = config
        self.duckdb_manager = duckdb_manager
        
        # Create generator configuration
        feature_config = config.get('feature_space', {})
        self.generator_config = FeatureGeneratorConfig(
            check_signal=feature_config.get('check_signal', True),
            lowercase_columns=feature_config.get('lowercase_features', False),
            min_signal_ratio=feature_config.get('min_signal_ratio', 0.01),
            signal_sample_size=feature_config.get('signal_sample_size', 1000),
            cache_features=feature_config.get('cache_features', True),
            forbidden_columns=config.get('autogluon', {}).get('ignore_columns', [])
        )
        
        # Create pipeline configuration
        self.pipeline_config = PipelineConfig(
            apply_generic_to_custom=feature_config.get('apply_generic_to_custom', True),
            cache_intermediate_results=feature_config.get('cache_intermediate', False),
            parallel_stages=feature_config.get('parallel_generation', False)
        )
        
        # Initialize generator and pipeline
        self.generator = MinotaurFeatureGenerator(
            config=config,
            check_signal=self.generator_config.check_signal,
            lowercase_columns=self.generator_config.lowercase_columns
        )
        
        self.pipeline = FeaturePipelineManager(
            generator=self.generator,
            config=self.generator_config,
            duckdb_connection=duckdb_manager.connection if duckdb_manager else None
        )
        
        # Cache for feature metadata
        self._feature_metadata_cache = {}
    
    def generate_generic_features_new(self, df: pd.DataFrame, check_signal: bool = True) -> pd.DataFrame:
        """
        Generate generic features using new pipeline (called by FeatureSpace).
        
        This method is called by the existing FeatureSpace.generate_generic_features
        method to use the new architecture.
        """
        logger.info("🔧 Generating generic features using new pipeline architecture...")
        
        # Configure signal checking
        self.generator.check_signal = check_signal
        
        # Generate only generic features
        features_df, metadata = self.generator.generate_features(
            df,
            FeatureType.GENERIC,
            forbidden_columns=self.generator_config.forbidden_columns
        )
        
        # Cache metadata
        self._feature_metadata_cache.update(metadata)
        
        return features_df
    
    def generate_custom_features_new(self, df: pd.DataFrame, dataset_name: str, check_signal: bool = True) -> pd.DataFrame:
        """
        Generate custom features using new pipeline (called by FeatureSpace).
        
        This method is called by the existing FeatureSpace.generate_custom_features
        method to use the new architecture.
        """
        logger.info(f"🎯 Generating custom domain features using new pipeline for: {dataset_name}")
        
        # Configure signal checking
        self.generator.check_signal = check_signal
        
        # Generate only custom features
        features_df, metadata = self.generator.generate_features(
            df,
            FeatureType.CUSTOM,
            dataset_name=dataset_name
        )
        
        # Cache metadata
        self._feature_metadata_cache.update(metadata)
        
        return features_df
    
    def generate_all_features_pipeline(self, 
                                     df: pd.DataFrame,
                                     dataset_name: str,
                                     target_column: Optional[str] = None,
                                     id_column: Optional[str] = None) -> pd.DataFrame:
        """
        Generate all features using the full pipeline.
        
        This can be used by dataset_importer.py instead of calling
        generic and custom separately.
        """
        logger.info("🚀 Generating all features using new pipeline architecture...")
        
        # Check cache first
        cached_result = self.pipeline.load_cached_features(dataset_name)
        if cached_result:
            features_df, metadata = cached_result
            self._feature_metadata_cache = metadata
            logger.info(f"Loaded {len(features_df.columns)} features from cache")
            return features_df
        
        # Generate using pipeline
        features_df, metadata = self.pipeline.generate_all_features(
            df,
            dataset_name=dataset_name,
            target_column=target_column,
            id_column=id_column
        )
        
        # Cache metadata
        self._feature_metadata_cache = metadata
        
        return features_df
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get cached feature metadata."""
        return self._feature_metadata_cache.copy()
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature generation statistics."""
        total_features = len(self._feature_metadata_cache)
        features_with_signal = sum(1 for meta in self._feature_metadata_cache.values() if meta.has_signal)
        
        # Group by type
        type_counts = {}
        category_counts = {}
        
        for meta in self._feature_metadata_cache.values():
            # Count by type
            type_name = meta.feature_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            # Count by category
            category_counts[meta.category] = category_counts.get(meta.category, 0) + 1
        
        # Calculate average generation time
        total_time = sum(meta.generation_time for meta in self._feature_metadata_cache.values())
        avg_time = total_time / total_features if total_features > 0 else 0
        
        return {
            'total_features': total_features,
            'features_with_signal': features_with_signal,
            'features_discarded': total_features - features_with_signal,
            'type_breakdown': type_counts,
            'category_breakdown': category_counts,
            'total_generation_time': total_time,
            'average_generation_time': avg_time
        }
    
    def save_feature_metadata_to_db(self, dataset_name: str):
        """Save feature metadata to database."""
        if not self.duckdb_manager or not self._feature_metadata_cache:
            return
        
        try:
            conn = self.duckdb_manager.connection
            
            # Prepare records for insertion
            records = []
            for feature_name, metadata in self._feature_metadata_cache.items():
                record = (
                    feature_name,
                    dataset_name,
                    metadata.feature_type.value,
                    metadata.category,
                    metadata.generation_time,
                    metadata.has_signal,
                    str(metadata.source_columns),  # JSON as string
                    metadata.operation or '',
                    str(metadata.parameters),       # JSON as string
                    str(metadata.statistics)        # JSON as string
                )
                records.append(record)
            
            # Insert into feature_metadata table
            conn.executemany("""
                INSERT OR REPLACE INTO feature_metadata 
                (feature_name, dataset_name, feature_type, category, 
                 generation_time, has_signal, source_columns, operation, 
                 parameters, statistics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            logger.info(f"Saved {len(records)} feature metadata records to database")
            
        except Exception as e:
            logger.error(f"Failed to save feature metadata: {e}")


def create_adapter(config: Dict[str, Any], duckdb_manager=None) -> FeatureSpaceAdapter:
    """Factory function to create feature space adapter."""
    return FeatureSpaceAdapter(config, duckdb_manager)