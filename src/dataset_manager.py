#!/usr/bin/env python3
"""
Dataset Manager - Centralized dataset access and validation

Provides interface to registered datasets and replaces direct file path access
in the MCTS feature discovery system.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatasetManager:
    """Centralized manager for registered datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize dataset manager with configuration."""
        self.config = config
        self.base_data_dir = Path("data")
        self._dataset_cache = {}
        
    def get_dataset_from_config(self) -> Dict[str, Any]:
        """Get dataset information from configuration.
        
        Returns:
            Dict with dataset metadata including paths and info
        """
        autogluon_config = self.config.get('autogluon', {})
        
        # Check if using new dataset name system
        dataset_name = autogluon_config.get('dataset_name')
        if dataset_name:
            return self._get_registered_dataset(dataset_name)
        
        # Fallback to legacy path-based system
        train_path = autogluon_config.get('train_path')
        test_path = autogluon_config.get('test_path')
        
        if not train_path:
            raise ValueError("No dataset_name or train_path specified in configuration")
        
        logger.warning("⚠️ Using legacy path-based dataset access")
        logger.warning("💡 Consider registering this dataset: scripts/duckdb_manager.py datasets --register --help")
        
        return {
            'dataset_name': 'legacy_dataset',
            'train_path': train_path,
            'test_path': test_path,
            'target_column': autogluon_config.get('target_column', 'target'),
            'id_column': autogluon_config.get('id_column'),
            'is_registered': False,
            'duckdb_path': None
        }
    
    def _get_registered_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a registered dataset."""
        
        # Check cache first
        if dataset_name in self._dataset_cache:
            return self._dataset_cache[dataset_name]
        
        # Load from database registration
        try:
            from .discovery_db import FeatureDiscoveryDB
            
            # Create temporary DB connection to lookup dataset
            temp_db = FeatureDiscoveryDB(self.config)
            
            # Use the new database service API
            dataset_repo = temp_db.db_service.dataset_repo
            result = dataset_repo.get_by_name(dataset_name)
            
            if not result:
                raise ValueError(f"Dataset '{dataset_name}' not found or inactive")
            
            # Build dataset info from Pydantic model
            dataset_info = {
                'dataset_id': result.dataset_id,
                'dataset_name': result.dataset_name,
                'train_path': result.train_path,
                'test_path': result.test_path,
                'target_column': result.target_column,
                'id_column': result.id_column,
                'train_records': result.train_records,
                'train_columns': result.train_columns,
                'test_records': result.test_records,
                'test_columns': result.test_columns,
                'competition_name': result.competition_name,
                'description': result.description,
                'is_registered': True,
                'duckdb_path': Path("cache") / result.dataset_name / "dataset.duckdb"
            }
            
            # Cache the result
            self._dataset_cache[dataset_name] = dataset_info
            
            logger.info(f"✅ Retrieved registered dataset: {result.dataset_name}")
            if result.train_records is not None:
                logger.info(f"   📊 Train: {result.train_records:,} records, {result.train_columns} columns")
            else:
                logger.info(f"   📊 Train: records not counted yet")
            if result.test_records:
                logger.info(f"   🧪 Test: {result.test_records:,} records, {result.test_columns} columns")
            
            return dataset_info
                
        except Exception as e:
            logger.error(f"Failed to retrieve dataset '{dataset_name}': {e}")
            raise ValueError(f"Dataset '{dataset_name}' is not properly registered")
    
    def validate_dataset_registration(self, dataset_name: str) -> bool:
        """Validate that a dataset is properly registered."""
        try:
            dataset_info = self._get_registered_dataset(dataset_name)
            
            # Check if source files exist
            if dataset_info['train_path'] and not Path(dataset_info['train_path']).exists():
                logger.error(f"❌ Train file not found: {dataset_info['train_path']}")
                return False
            
            if dataset_info['test_path'] and not Path(dataset_info['test_path']).exists():
                logger.error(f"❌ Test file not found: {dataset_info['test_path']}")
                return False
            
            # Check if DuckDB file exists (for registered datasets)
            if dataset_info['is_registered'] and dataset_info['duckdb_path']:
                if not dataset_info['duckdb_path'].exists():
                    logger.error(f"❌ DuckDB file not found: {dataset_info['duckdb_path']}")
                    logger.error("💡 Re-register dataset to recreate DuckDB file")
                    return False
            
            logger.info(f"✅ Dataset '{dataset_name}' validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dataset validation failed: {e}")
            return False
    
    def get_dataset_connection(self, dataset_name: str):
        """Get DuckDB connection to dataset (for registered datasets)."""
        dataset_info = self._get_registered_dataset(dataset_name)
        
        if not dataset_info['is_registered']:
            raise ValueError(f"Dataset '{dataset_name}' is not registered - cannot provide DuckDB connection")
        
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB not available - cannot connect to dataset")
        
        duckdb_path = dataset_info['duckdb_path']
        if not duckdb_path.exists():
            raise FileNotFoundError(f"DuckDB file not found: {duckdb_path}")
        
        logger.debug(f"Connecting to dataset DuckDB: {duckdb_path}")
        return duckdb.connect(str(duckdb_path))
    
    def load_dataset_files(self, dataset_name: str, 
                          sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load dataset files as pandas DataFrames.
        
        Args:
            dataset_name: Name of registered dataset
            sample_size: Optional sample size for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        dataset_info = self._get_registered_dataset(dataset_name)
        
        # Load train data
        train_path = dataset_info['train_path']
        if not train_path or not Path(train_path).exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        
        logger.info(f"📊 Loading train data from: {train_path}")
        train_df = pd.read_csv(train_path)
        
        # Apply sampling if requested
        if sample_size and len(train_df) > sample_size:
            logger.info(f"🎲 Sampling {sample_size:,} rows from {len(train_df):,} total")
            train_df = train_df.sample(n=sample_size, random_state=42)
        
        # Load test data (optional)
        test_df = None
        test_path = dataset_info.get('test_path')
        if test_path and Path(test_path).exists():
            logger.info(f"🧪 Loading test data from: {test_path}")
            test_df = pd.read_csv(test_path)
        
        logger.info(f"✅ Loaded dataset: train={len(train_df):,} rows")
        if test_df is not None:
            logger.info(f"   Test: {len(test_df):,} rows")
        
        return train_df, test_df
    
    def get_target_column(self, dataset_name: str) -> str:
        """Get target column name for dataset."""
        dataset_info = self._get_registered_dataset(dataset_name)
        target_col = dataset_info.get('target_column')
        
        if not target_col:
            raise ValueError(f"No target column specified for dataset '{dataset_name}'")
        
        return target_col
    
    def get_id_column(self, dataset_name: str) -> Optional[str]:
        """Get ID column name for dataset (if specified)."""
        dataset_info = self._get_registered_dataset(dataset_name)
        return dataset_info.get('id_column')
    
    def list_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all available datasets."""
        try:
            from .discovery_db import FeatureDiscoveryDB
            
            temp_db = FeatureDiscoveryDB(self.config)
            
            # Use the new database service API
            dataset_repo = temp_db.db_service.dataset_repo
            results = dataset_repo.list_all()
            
            datasets = {}
            for dataset in results:
                datasets[dataset.dataset_name] = {
                    'competition_name': dataset.competition_name,
                    'train_records': dataset.train_records,
                    'test_records': dataset.test_records,
                    'target_column': dataset.target_column,
                    'is_active': dataset.is_active
                }
            
            return datasets
                
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return {}
    
    def update_dataset_usage(self, dataset_name: str) -> None:
        """Update last_used timestamp for dataset."""
        try:
            from .discovery_db import FeatureDiscoveryDB
            
            temp_db = FeatureDiscoveryDB(self.config)
            
            # Use the new database service API - first find dataset by name
            dataset_repo = temp_db.db_service.dataset_repo
            dataset = dataset_repo.find_by_name(dataset_name)
            
            if dataset:
                dataset_repo.mark_dataset_used(dataset.dataset_id)
                logger.debug(f"Updated usage timestamp for dataset: {dataset_name}")
            else:
                logger.warning(f"Dataset '{dataset_name}' not found for usage update")
            
        except Exception as e:
            logger.warning(f"Failed to update dataset usage: {e}")