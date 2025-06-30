"""
Automatic registration of original dataset features (train origin).

This module handles automatic registration of original dataset columns 
as features with origin='train' in the feature catalog.
"""

import logging
import pandas as pd
from typing import List, Optional

logger = logging.getLogger(__name__)


def register_train_features(dataset_name: str, train_path: str, target_column: str, id_column: str):
    """
    Register original dataset columns as train features in the catalog.
    
    Args:
        dataset_name: Name of the dataset
        train_path: Path to training CSV file
        target_column: Target column to exclude
        id_column: ID column to exclude
    """
    try:
        # Import here to avoid circular imports
        import duckdb
        from src.project_root import PROJECT_ROOT
        import os
        
        # Connect to database
        db_path = os.path.join(PROJECT_ROOT, 'data', 'minotaur.duckdb')
        if not os.path.exists(db_path):
            logger.warning("Database not found, skipping train features registration")
            return
        
        # Read CSV header to get column names
        df_header = pd.read_csv(train_path, nrows=0)  # Read only header
        all_columns = df_header.columns.tolist()
        
        # Filter out target and ID columns
        train_features = []
        for col in all_columns:
            if col.lower() != target_column.lower() and col.lower() != id_column.lower():
                train_features.append(col)
        
        logger.info(f"Registering {len(train_features)} train features for dataset '{dataset_name}'")
        
        # Connect and register features
        with duckdb.connect(db_path) as conn:
            for feature_name in train_features:
                # Convert to lowercase for consistency
                feature_name_lower = feature_name.lower()
                
                conn.execute("""
                    INSERT INTO feature_catalog (feature_name, feature_category, python_code, operation_name, description, origin)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (feature_name) DO UPDATE SET
                        feature_category = EXCLUDED.feature_category,
                        python_code = EXCLUDED.python_code,
                        operation_name = EXCLUDED.operation_name,
                        description = EXCLUDED.description,
                        origin = EXCLUDED.origin
                """, [
                    feature_name_lower,
                    'train',
                    'OriginalDatasetColumn',
                    'Original Dataset Features',
                    f"Original column '{feature_name}' from {dataset_name} dataset",
                    'train'
                ])
        
        logger.info(f"Successfully registered {len(train_features)} train features")
        
    except Exception as e:
        logger.error(f"Failed to register train features: {e}")


def get_original_column_mapping(train_path: str) -> dict:
    """
    Get mapping of original column names to lowercase versions.
    
    Args:
        train_path: Path to training CSV file
        
    Returns:
        Dictionary mapping original -> lowercase column names
    """
    try:
        df_header = pd.read_csv(train_path, nrows=0)
        original_columns = df_header.columns.tolist()
        
        mapping = {}
        for col in original_columns:
            mapping[col] = col.lower()
        
        return mapping
        
    except Exception as e:
        logger.error(f"Failed to create column mapping: {e}")
        return {}