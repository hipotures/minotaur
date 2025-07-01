#!/usr/bin/env python3
"""
Migration script to fix feature_catalog operation names after the refactoring.

This script updates operation names in feature_catalog tables to match the new
lowercase naming convention used by the feature generation classes.

Usage:
    python scripts/fix_feature_catalog_names.py [dataset_name]
    
    If dataset_name is provided, only that dataset will be updated.
    Otherwise, all registered datasets will be updated.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging
import duckdb
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping of old names to new names
OPERATION_NAME_MAPPING = {
    'Statistical Aggregations': 'statistical_aggregations',
    'Polynomial Features': 'polynomial_features',
    'Binning Features': 'binning_features',
    'Ranking Features': 'ranking_features',
    'Categorical Features': 'categorical_features',
    'Text Features': 'text_features',
    'Temporal Features': 'temporal_features'
}

def update_feature_catalog(db_path: str) -> int:
    """Update operation names in a feature_catalog table."""
    updates_made = 0
    
    try:
        with duckdb.connect(db_path) as conn:
            # Check if feature_catalog exists
            tables = conn.execute("SHOW TABLES").fetchall()
            if not any('feature_catalog' in str(t) for t in tables):
                logger.info(f"No feature_catalog table found in {db_path}")
                return 0
            
            # Update each operation name
            for old_name, new_name in OPERATION_NAME_MAPPING.items():
                result = conn.execute("""
                    UPDATE feature_catalog 
                    SET operation_name = ? 
                    WHERE operation_name = ?
                """, [new_name, old_name])
                
                # Get number of affected rows
                affected = conn.execute("SELECT COUNT(*) FROM feature_catalog WHERE operation_name = ?", [new_name]).fetchone()[0]
                if affected > 0:
                    logger.info(f"Updated {affected} rows: '{old_name}' -> '{new_name}'")
                    updates_made += affected
            
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error updating {db_path}: {e}")
        return 0
    
    return updates_made

def main():
    """Main function to update feature catalogs."""
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Get project root
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / 'cache'
    
    if not cache_dir.exists():
        logger.error("No cache directory found")
        return 1
    
    total_updates = 0
    
    if dataset_name:
        # Update specific dataset
        dataset_db = cache_dir / dataset_name / 'dataset.duckdb'
        if dataset_db.exists():
            logger.info(f"Updating feature catalog for dataset: {dataset_name}")
            updates = update_feature_catalog(str(dataset_db))
            total_updates += updates
        else:
            logger.error(f"Dataset not found: {dataset_name}")
            return 1
    else:
        # Update all datasets
        logger.info("Updating feature catalogs for all datasets...")
        for dataset_dir in cache_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_db = dataset_dir / 'dataset.duckdb'
                if dataset_db.exists():
                    logger.info(f"\nUpdating dataset: {dataset_dir.name}")
                    updates = update_feature_catalog(str(dataset_db))
                    total_updates += updates
    
    logger.info(f"\nTotal updates made: {total_updates}")
    
    if total_updates > 0:
        logger.info("\n✅ Feature catalog names updated successfully!")
        logger.info("Note: The FeatureSpace query has been updated to handle both old and new names,")
        logger.info("so your system should work correctly even without this migration.")
    else:
        logger.info("\n✅ No updates needed - feature catalogs are already up to date!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())