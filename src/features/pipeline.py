"""
Feature Pipeline Manager

Orchestrates feature generation with proper ordering and dependencies.
Manages the flow: original → custom → generic (applied to both original and custom).
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

from .base_generator import (
    UniversalFeatureGenerator, 
    FeatureType, 
    FeatureMetadata,
    FeatureGeneratorConfig
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Represents a stage in the feature generation pipeline."""
    name: str
    feature_type: FeatureType
    input_features: Set[str] = field(default_factory=set)
    output_features: Set[str] = field(default_factory=set)
    execution_time: float = 0.0
    metadata: Dict[str, FeatureMetadata] = field(default_factory=dict)


class FeaturePipelineManager:
    """
    Manages the feature generation pipeline with proper ordering and caching.
    
    Key features:
    - Orchestrates generation order: original → custom → generic
    - Applies generic operations to custom features
    - Caches features in DuckDB for reuse
    - Tracks feature lineage and dependencies
    - Progressive generation with early stopping
    """
    
    def __init__(self, 
                 generator: UniversalFeatureGenerator,
                 config: FeatureGeneratorConfig,
                 duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initialize pipeline manager.
        
        Args:
            generator: Universal feature generator instance
            config: Feature generation configuration
            duckdb_connection: Optional DuckDB connection for caching
        """
        self.generator = generator
        self.config = config
        self.duckdb_connection = duckdb_connection
        
        # Pipeline stages
        self.stages: List[PipelineStage] = []
        self.current_stage: Optional[PipelineStage] = None
        
        # Feature tracking
        self.all_features: Dict[str, FeatureMetadata] = {}
        self.feature_lineage: Dict[str, List[str]] = {}  # feature -> source features
        
        # Caching
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.cache_enabled = config.cache_features and duckdb_connection is not None
    
    def generate_all_features(self, 
                            df: pd.DataFrame,
                            dataset_name: str,
                            target_column: Optional[str] = None,
                            id_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]:
        """
        Generate all features following the pipeline order.
        
        Args:
            df: Input dataframe
            dataset_name: Name of the dataset for custom features
            target_column: Target column to exclude
            id_column: ID column to preserve
            
        Returns:
            Tuple of (features_df, metadata_dict)
        """
        start_time = time.time()
        logger.info(f"Starting feature pipeline for dataset: {dataset_name}")
        
        # Reset pipeline state
        self._reset_pipeline()
        
        # Stage 1: Original features (raw data)
        original_df = self._generate_original_features(df, target_column, id_column)
        
        # Stage 2: Custom features (domain-specific)
        custom_df = self._generate_custom_features(df, dataset_name)
        
        # Stage 3: Generic features on original
        generic_original_df = self._generate_generic_features(
            df, 
            feature_set_name="generic_original",
            apply_to_custom=False
        )
        
        # Stage 4: Generic features on custom (if enabled)
        generic_custom_df = None
        if self.config.apply_generic_to_custom:
            # Combine original + custom for generic operations
            combined_df = pd.concat([df, custom_df], axis=1)
            generic_custom_df = self._generate_generic_features(
                combined_df,
                feature_set_name="generic_custom",
                apply_to_custom=True
            )
        
        # Combine all features
        final_df = self._combine_features([
            original_df,
            custom_df,
            generic_original_df,
            generic_custom_df
        ])
        
        # Cache if enabled
        if self.cache_enabled:
            self._cache_features(final_df, dataset_name)
        
        # Log pipeline summary
        self._log_pipeline_summary(time.time() - start_time)
        
        return final_df, self.all_features.copy()
    
    def _generate_original_features(self, 
                                  df: pd.DataFrame,
                                  target_column: Optional[str],
                                  id_column: Optional[str]) -> pd.DataFrame:
        """Generate original features (raw data columns)."""
        stage = PipelineStage(
            name="original",
            feature_type=FeatureType.ORIGINAL
        )
        self.current_stage = stage
        stage_start = time.time()
        
        # Select columns (exclude target if specified)
        columns_to_keep = [col for col in df.columns if col != target_column]
        
        # Ensure ID column is first if specified
        if id_column and id_column in columns_to_keep:
            columns_to_keep.remove(id_column)
            columns_to_keep = [id_column] + columns_to_keep
        
        original_df = df[columns_to_keep].copy()
        
        # Track as original features
        for col in columns_to_keep:
            metadata = FeatureMetadata(
                name=col,
                feature_type=FeatureType.ORIGINAL,
                category="original",
                generation_time=0.0,
                has_signal=True,  # Assume original features have signal
                source_columns=[col]
            )
            self.all_features[col] = metadata
            stage.output_features.add(col)
        
        stage.execution_time = time.time() - stage_start
        self.stages.append(stage)
        
        logger.info(f"Original features: {len(columns_to_keep)} columns in {stage.execution_time:.3f}s")
        return original_df
    
    def _generate_custom_features(self, 
                                df: pd.DataFrame,
                                dataset_name: str) -> pd.DataFrame:
        """Generate custom domain-specific features."""
        stage = PipelineStage(
            name="custom",
            feature_type=FeatureType.CUSTOM,
            input_features=set(df.columns)
        )
        self.current_stage = stage
        stage_start = time.time()
        
        # Generate custom features
        custom_df, metadata = self.generator.generate_features(
            df,
            FeatureType.CUSTOM,
            dataset_name=dataset_name
        )
        
        # Update tracking
        stage.metadata = metadata
        stage.output_features = set(custom_df.columns)
        self.all_features.update(metadata)
        
        # Track lineage
        for feature_name, feature_meta in metadata.items():
            self.feature_lineage[feature_name] = feature_meta.source_columns
        
        stage.execution_time = time.time() - stage_start
        self.stages.append(stage)
        
        logger.info(f"Custom features: {len(custom_df.columns)} features in {stage.execution_time:.3f}s")
        return custom_df
    
    def _generate_generic_features(self, 
                                 df: pd.DataFrame,
                                 feature_set_name: str,
                                 apply_to_custom: bool) -> pd.DataFrame:
        """Generate generic features."""
        stage = PipelineStage(
            name=feature_set_name,
            feature_type=FeatureType.GENERIC,
            input_features=set(df.columns)
        )
        self.current_stage = stage
        stage_start = time.time()
        
        # Configure which columns to use
        if apply_to_custom:
            # Use both original and custom features
            forbidden_columns = self.config.forbidden_columns
        else:
            # Use only original features
            custom_features = {name for name, meta in self.all_features.items() 
                             if meta.feature_type == FeatureType.CUSTOM}
            forbidden_columns = self.config.forbidden_columns + list(custom_features)
        
        # Generate generic features
        generic_df, metadata = self.generator.generate_features(
            df,
            FeatureType.GENERIC,
            forbidden_columns=forbidden_columns
        )
        
        # Update tracking
        stage.metadata = metadata
        stage.output_features = set(generic_df.columns)
        self.all_features.update(metadata)
        
        # Track lineage
        for feature_name, feature_meta in metadata.items():
            self.feature_lineage[feature_name] = feature_meta.source_columns
        
        stage.execution_time = time.time() - stage_start
        self.stages.append(stage)
        
        logger.info(f"{feature_set_name}: {len(generic_df.columns)} features in {stage.execution_time:.3f}s")
        return generic_df
    
    def _combine_features(self, feature_dfs: List[Optional[pd.DataFrame]]) -> pd.DataFrame:
        """Combine multiple feature DataFrames, handling duplicates."""
        # Filter out None values
        valid_dfs = [df for df in feature_dfs if df is not None and not df.empty]
        
        if not valid_dfs:
            raise ValueError("No valid feature DataFrames to combine")
        
        # Start with first DataFrame
        combined = valid_dfs[0].copy()
        seen_columns = set(combined.columns)
        
        # Add other DataFrames, avoiding duplicates
        for df in valid_dfs[1:]:
            new_columns = [col for col in df.columns if col not in seen_columns]
            if new_columns:
                combined = pd.concat([combined, df[new_columns]], axis=1)
                seen_columns.update(new_columns)
        
        return combined
    
    def _cache_features(self, features_df: pd.DataFrame, dataset_name: str):
        """Cache features in DuckDB for reuse."""
        if not self.duckdb_connection:
            return
        
        try:
            table_name = f"feature_cache_{dataset_name}"
            
            # Drop existing cache
            self.duckdb_connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Register DataFrame and create table
            self.duckdb_connection.register('features_df', features_df)
            self.duckdb_connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM features_df")
            self.duckdb_connection.unregister('features_df')
            
            # Also cache metadata
            metadata_table = f"feature_metadata_{dataset_name}"
            self._cache_metadata(metadata_table)
            
            logger.info(f"Cached {len(features_df.columns)} features in DuckDB")
            
        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")
    
    def _cache_metadata(self, table_name: str):
        """Cache feature metadata in DuckDB."""
        if not self.duckdb_connection:
            return
        
        try:
            # Convert metadata to DataFrame
            metadata_records = []
            for feature_name, metadata in self.all_features.items():
                record = {
                    'feature_name': feature_name,
                    'feature_type': metadata.feature_type.value,
                    'category': metadata.category,
                    'generation_time': metadata.generation_time,
                    'has_signal': metadata.has_signal,
                    'source_columns': ','.join(metadata.source_columns),
                    'operation': metadata.operation or '',
                    'parameters': str(metadata.parameters),
                    'statistics': str(metadata.statistics)
                }
                metadata_records.append(record)
            
            if metadata_records:
                metadata_df = pd.DataFrame(metadata_records)
                
                # Create table
                self.duckdb_connection.execute(f"DROP TABLE IF EXISTS {table_name}")
                self.duckdb_connection.register('metadata_df', metadata_df)
                self.duckdb_connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM metadata_df")
                self.duckdb_connection.unregister('metadata_df')
                
        except Exception as e:
            logger.warning(f"Failed to cache metadata: {e}")
    
    def load_cached_features(self, dataset_name: str) -> Optional[Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]]:
        """Load cached features if available."""
        if not self.cache_enabled:
            return None
        
        try:
            table_name = f"feature_cache_{dataset_name}"
            
            # Check if cache exists
            result = self.duckdb_connection.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            ).fetchone()
            
            if not result:
                return None
            
            # Load features
            features_df = self.duckdb_connection.execute(f"SELECT * FROM {table_name}").df()
            
            # Load metadata
            metadata_table = f"feature_metadata_{dataset_name}"
            metadata_df = self.duckdb_connection.execute(f"SELECT * FROM {metadata_table}").df()
            
            # Reconstruct metadata
            metadata = {}
            for _, row in metadata_df.iterrows():
                metadata[row['feature_name']] = FeatureMetadata(
                    name=row['feature_name'],
                    feature_type=FeatureType(row['feature_type']),
                    category=row['category'],
                    generation_time=row['generation_time'],
                    has_signal=row['has_signal'],
                    source_columns=row['source_columns'].split(',') if row['source_columns'] else []
                )
            
            logger.info(f"Loaded {len(features_df.columns)} cached features")
            return features_df, metadata
            
        except Exception as e:
            logger.debug(f"Failed to load cached features: {e}")
            return None
    
    def get_feature_dependencies(self, feature_name: str) -> Dict[str, Any]:
        """Get dependencies and lineage for a feature."""
        if feature_name not in self.all_features:
            return {}
        
        metadata = self.all_features[feature_name]
        
        # Get direct dependencies
        direct_deps = self.feature_lineage.get(feature_name, [])
        
        # Get recursive dependencies
        all_deps = set()
        to_process = list(direct_deps)
        while to_process:
            dep = to_process.pop(0)
            if dep not in all_deps:
                all_deps.add(dep)
                to_process.extend(self.feature_lineage.get(dep, []))
        
        return {
            'feature_name': feature_name,
            'feature_type': metadata.feature_type.value,
            'direct_dependencies': direct_deps,
            'all_dependencies': list(all_deps),
            'dependency_depth': len(all_deps),
            'generation_time': metadata.generation_time
        }
    
    def _reset_pipeline(self):
        """Reset pipeline state for new run."""
        self.stages.clear()
        self.current_stage = None
        self.all_features.clear()
        self.feature_lineage.clear()
        self.feature_cache.clear()
    
    def _log_pipeline_summary(self, total_time: float):
        """Log summary of pipeline execution."""
        total_features = len(self.all_features)
        features_with_signal = sum(1 for meta in self.all_features.values() if meta.has_signal)
        discarded_features = total_features - features_with_signal
        
        logger.info("=" * 60)
        logger.info("Feature Pipeline Summary:")
        logger.info(f"Total features generated: {total_features}")
        logger.info(f"Features with signal: {features_with_signal}")
        logger.info(f"Features discarded: {discarded_features}")
        logger.info(f"Total execution time: {total_time:.3f}s")
        logger.info("")
        
        # Stage summary
        logger.info("Stage Summary:")
        for stage in self.stages:
            logger.info(
                f"  {stage.name}: {len(stage.output_features)} features "
                f"in {stage.execution_time:.3f}s"
            )
        
        # Feature type breakdown
        type_counts = {}
        for meta in self.all_features.values():
            type_counts[meta.feature_type.value] = type_counts.get(meta.feature_type.value, 0) + 1
        
        logger.info("")
        logger.info("Feature Type Breakdown:")
        for feature_type, count in type_counts.items():
            logger.info(f"  {feature_type}: {count} features")
        
        logger.info("=" * 60)


class PipelineConfig:
    """Configuration for feature pipeline."""
    
    def __init__(self,
                 apply_generic_to_custom: bool = True,
                 cache_intermediate_results: bool = False,
                 parallel_stages: bool = False,
                 max_features_per_stage: Optional[int] = None,
                 early_stopping_threshold: float = 0.01):
        """
        Initialize pipeline configuration.
        
        Args:
            apply_generic_to_custom: Apply generic operations to custom features
            cache_intermediate_results: Cache results after each stage
            parallel_stages: Run independent stages in parallel
            max_features_per_stage: Maximum features per stage
            early_stopping_threshold: Stop if improvement < threshold
        """
        self.apply_generic_to_custom = apply_generic_to_custom
        self.cache_intermediate_results = cache_intermediate_results
        self.parallel_stages = parallel_stages
        self.max_features_per_stage = max_features_per_stage
        self.early_stopping_threshold = early_stopping_threshold