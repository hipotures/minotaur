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
    
    def __init__(self, config: Dict[str, Any], db_service=None):
        """Initialize dataset manager with configuration."""
        self.config = config
        self.base_data_dir = Path("data")
        self._dataset_cache = {}
        self.db_service = db_service  # Reuse existing database connection
        
    def get_dataset_from_config(self) -> Dict[str, Any]:
        """Get dataset information from configuration.
        
        Returns:
            Dict with dataset metadata including paths and info
        """
        autogluon_config = self.config.get('autogluon', {})
        
        # Check if using new dataset name system
        dataset_name = autogluon_config.get('dataset_name')
        if not dataset_name:
            raise ValueError("Configuration error: 'autogluon.dataset_name' must be specified.")

        return self._get_registered_dataset(dataset_name)
    
    def _get_registered_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a registered dataset."""
        
        # Check cache first
        if dataset_name in self._dataset_cache:
            return self._dataset_cache[dataset_name]
        
        # Load from database registration
        try:
            # Query dataset information directly
            query = """
            SELECT name, dataset_path, train_path, test_path, 
                   target_column, id_column, description, is_active
            FROM datasets 
            WHERE name = :dataset_name AND is_active = true
            """
            
            # Use db_service if available, otherwise create temporary connection
            if self.db_service:
                results = self.db_service.execute_query(query, {'dataset_name': dataset_name})
            else:
                # Create temporary read-only connection
                db_path = self.config.get('database', {}).get('path', 'data/minotaur.duckdb')
                if not DUCKDB_AVAILABLE:
                    raise ImportError("DuckDB not available for dataset access")
                conn = duckdb.connect(db_path, read_only=True)
                cursor = conn.execute(query, {'dataset_name': dataset_name})
                results = cursor.fetchall()
                conn.close()
                
                # Convert to dict format
                if results:
                    columns = ['name', 'dataset_path', 'train_path', 'test_path', 
                              'target_column', 'id_column', 'description', 'is_active']
                    results = [dict(zip(columns, row)) for row in results]
            
            if not results:
                raise ValueError(f"Dataset '{dataset_name}' not found or inactive")
            
            result = results[0]
            
            # Build dataset info from database result
            dataset_info = {
                'dataset_id': None,  # Not used in new system
                'dataset_name': result['name'],
                'dataset_path': result['dataset_path'],
                'train_path': result['train_path'],
                'test_path': result['test_path'],
                'target_column': result['target_column'],
                'id_column': result['id_column'],
                'train_records': None,  # Will be loaded on demand
                'train_columns': None,
                'test_records': None,
                'test_columns': None,
                'competition_name': None,
                'description': result['description'],
                'is_registered': True,
                'duckdb_path': Path("cache") / dataset_name / "dataset.duckdb"
            }
            
            # Cache the result
            self._dataset_cache[dataset_name] = dataset_info
            
            logger.info(f"✅ Retrieved registered dataset: {dataset_name}")
            logger.info(f"   📂 Path: {result['dataset_path']}")
            if result['train_path']:
                logger.info(f"   📊 Train: {result['train_path']}")
            if result['test_path']:
                logger.info(f"   🧪 Test: {result['test_path']}")
            
            return dataset_info
                
        except Exception as e:
            logger.info(f"Dataset '{dataset_name}' not available: {e}")
            # Preserve original error message if it's already about dataset not found
            if "not found or inactive" in str(e):
                raise e
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
            query = """
            SELECT name, dataset_path, target_column, description, is_active
            FROM datasets 
            WHERE is_active = true
            ORDER BY name
            """
            
            if self.db_service:
                results = self.db_service.execute_query(query)
            else:
                # Create temporary read-only connection
                db_path = self.config.get('database', {}).get('path', 'data/minotaur.duckdb')
                if not DUCKDB_AVAILABLE:
                    raise ImportError("DuckDB not available for dataset access")
                conn = duckdb.connect(db_path, read_only=True)
                cursor = conn.execute(query)
                results = cursor.fetchall()
                conn.close()
                
                # Convert to dict format
                if results:
                    columns = ['name', 'dataset_path', 'target_column', 'description', 'is_active']
                    results = [dict(zip(columns, row)) for row in results]
            
            datasets = {}
            for row in results:
                datasets[row['name']] = {
                    'competition_name': None,  # Not stored in new schema
                    'train_records': None,
                    'test_records': None,
                    'target_column': row['target_column'],
                    'is_active': row['is_active']
                }
            
            return datasets
                
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return {}
    
    def update_dataset_usage(self, dataset_name: str) -> None:
        """Update last_used timestamp for dataset."""
        try:
            # For now, just log the usage - the new schema doesn't have a last_used field
            logger.debug(f"Dataset usage recorded: {dataset_name}")
            
            # TODO: If needed, add a last_used column to datasets table:
            # ALTER TABLE datasets ADD COLUMN last_used TIMESTAMP;
            # UPDATE datasets SET last_used = CURRENT_TIMESTAMP WHERE name = :dataset_name
            
        except Exception as e:
            logger.warning(f"Failed to update dataset usage: {e}")