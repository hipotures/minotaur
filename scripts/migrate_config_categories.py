#!/usr/bin/env python3
"""
Config Migration Script - Update hardcoded categories to dynamic ones

This script updates YAML configuration files to use dynamic categories
from the operation metadata instead of hardcoded category lists.
"""

import os
import yaml
import sys
from pathlib import Path

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.generic import OPERATION_METADATA, get_operations_by_category


def get_available_categories():
    """Get all available categories from operation metadata."""
    categories = set()
    for metadata in OPERATION_METADATA.values():
        categories.add(metadata['category'])
    return sorted(list(categories))


def get_default_category_weights():
    """Generate default category weights based on operation metadata."""
    categories = get_available_categories()
    
    # Default weights based on general usefulness
    default_weights = {
        'statistical': 1.0,    # Basic statistical operations
        'polynomial': 0.8,     # Polynomial transformations
        'binning': 1.0,        # Quantile binning
        'ranking': 0.9,        # Rank transformations
        'temporal': 0.7,       # Time-based (if applicable)
        'text': 0.6,          # Text processing (if applicable)
        'categorical': 1.1,    # Categorical encoding
        'custom_domain': 2.0,  # Domain-specific (highest priority)
    }
    
    # Return weights for all available categories
    return {cat: default_weights.get(cat, 1.0) for cat in categories}


def migrate_config_file(config_path: Path):
    """Migrate a single configuration file."""
    print(f"Migrating: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dynamic categories
    available_categories = get_available_categories()
    default_weights = get_default_category_weights()
    
    # Update feature_space section
    if 'feature_space' not in config:
        config['feature_space'] = {}
    
    feature_space = config['feature_space']
    
    # Replace enabled_categories with dynamic discovery
    if 'enabled_categories' in feature_space:
        old_categories = feature_space['enabled_categories']
        print(f"  Old enabled_categories: {old_categories}")
        
        # Replace with comment and dynamic discovery setting
        del feature_space['enabled_categories']
        feature_space['# NOTE'] = 'Categories are now discovered dynamically from operation metadata'
        feature_space['use_dynamic_categories'] = True
        
        # Add optional category filter for advanced users
        feature_space['category_filter'] = {
            'include': available_categories,  # Include all by default
            'exclude': []                     # Exclude none by default
        }
        
        print(f"  New available categories: {available_categories}")
    
    # Update category_weights to use dynamic categories
    if 'category_weights' in feature_space:
        old_weights = feature_space['category_weights']
        print(f"  Old category_weights: {old_weights}")
        
        # Keep any existing weights that map to valid categories
        new_weights = {}
        for cat in available_categories:
            if cat in old_weights:
                new_weights[cat] = old_weights[cat]
            else:
                new_weights[cat] = default_weights[cat]
        
        feature_space['category_weights'] = new_weights
        print(f"  New category_weights: {new_weights}")
    else:
        # Add default weights if none exist
        feature_space['category_weights'] = default_weights
        print(f"  Added default category_weights: {default_weights}")
    
    # Add metadata reference for documentation
    if '# DYNAMIC_CATEGORIES_INFO' not in feature_space:
        feature_space['# DYNAMIC_CATEGORIES_INFO'] = 'Categories and weights are generated from src/features/generic/OPERATION_METADATA'
    
    # Save updated config
    backup_path = config_path.with_suffix('.yaml.backup')
    print(f"  Creating backup: {backup_path}")
    config_path.rename(backup_path)
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"  ✅ Migrated: {config_path}")


def main():
    """Main migration function."""
    print("🔄 Starting YAML config migration to dynamic categories...")
    
    # Find all config files
    config_dir = Path(__file__).parent.parent / 'config'
    config_files = list(config_dir.glob('*.yaml')) + list(config_dir.glob('*.yml'))
    
    # Filter out CI and output files
    config_files = [f for f in config_files if not any(skip in str(f) for skip in ['.github', 'outputs', 'ci.yml'])]
    
    print(f"Found {len(config_files)} config files to migrate:")
    for cf in config_files:
        print(f"  - {cf.name}")
    
    print(f"\n📊 Available dynamic categories: {get_available_categories()}")
    print(f"📊 Default category weights: {get_default_category_weights()}")
    
    # Auto-proceed in non-interactive mode
    print(f"\n🚀 Auto-proceeding with migration...")
    
    # Migrate each file
    print("\n🚀 Starting migration...")
    migrated_count = 0
    
    for config_file in config_files:
        try:
            migrate_config_file(config_file)
            migrated_count += 1
        except Exception as e:
            print(f"❌ Error migrating {config_file}: {e}")
    
    print(f"\n✅ Migration complete! Migrated {migrated_count}/{len(config_files)} files")
    print("\n📝 Migration summary:")
    print("  - enabled_categories replaced with use_dynamic_categories: true")
    print("  - category_weights updated with dynamic categories")
    print("  - category_filter added for advanced filtering")
    print("  - Backup files created with .yaml.backup extension")
    print("\n🔍 Review the migrated files and remove backup files when satisfied")


if __name__ == '__main__':
    main()