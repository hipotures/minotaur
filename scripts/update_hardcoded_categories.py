#!/usr/bin/env python3
"""
Update Hardcoded Categories Script

This script finds and updates remaining hardcoded category references
in source code files to use dynamic categories from operation metadata.
"""

import os
import re
import sys
from pathlib import Path

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from features.generic import OPERATION_METADATA
except ImportError:
    print("❌ Could not import OPERATION_METADATA. Make sure src/ is in the path.")
    sys.exit(1)


def get_dynamic_category_mapping():
    """Create mapping from old hardcoded categories to new dynamic ones."""
    # Mapping from old hardcoded category names to new dynamic ones
    category_mapping = {
        # Old agricultural/domain specific categories -> custom_domain
        'npk_interactions': 'custom_domain',
        'environmental_stress': 'custom_domain', 
        'agricultural_domain': 'custom_domain',
        'feature_selection': 'custom_domain',
        'feature_transformations': 'polynomial',  # Best match for transformations
        
        # Old operation names -> new categories
        'statistical_aggregations': 'statistical',
        'polynomial_features': 'polynomial',
        'binning_features': 'binning',
        'ranking_features': 'ranking',
        'temporal_features': 'temporal',
        'text_features': 'text',
        'categorical_features': 'categorical',
        'interaction_features': 'custom_domain',  # Domain-specific interactions
        
        # Custom domain stays as custom_domain
        'custom_domain': 'custom_domain',
    }
    
    return category_mapping


def update_python_file(file_path: Path):
    """Update hardcoded categories in a Python file."""
    print(f"Updating Python file: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    category_mapping = get_dynamic_category_mapping()
    
    # Pattern 1: 'category': 'hardcoded_category'
    def replace_category_assignment(match):
        old_category = match.group(1)
        if old_category in category_mapping:
            new_category = category_mapping[old_category]
            print(f"  Replaced 'category': '{old_category}' -> '{new_category}'")
            return f"'category': '{new_category}'"
        return match.group(0)
    
    content = re.sub(r"'category':\s*'([^']+)'", replace_category_assignment, content)
    
    # Pattern 2: # Category: hardcoded_category (in comments)
    def replace_category_comment(match):
        old_category = match.group(1)
        if old_category in category_mapping:
            new_category = category_mapping[old_category]
            print(f"  Replaced comment '# Category: {old_category}' -> '{new_category}'")
            return f"# Category: {new_category}"
        return match.group(0)
    
    content = re.sub(r"# Category:\s*([^\n]+)", replace_category_comment, content)
    
    # Pattern 3: FeatureCategory.HARDCODED_NAME
    def replace_feature_category_enum(match):
        old_category = match.group(1).lower()
        if old_category in category_mapping:
            new_category = category_mapping[old_category].upper()
            print(f"  Replaced FeatureCategory.{match.group(1)} -> {new_category}")
            return f"FeatureCategory.{new_category}"
        return match.group(0)
    
    content = re.sub(r"FeatureCategory\.([A-Z_]+)", replace_feature_category_enum, content)
    
    # Save if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Updated: {file_path}")
        return True
    else:
        print(f"  ⏭️  No changes needed: {file_path}")
        return False


def update_sql_file(file_path: Path):
    """Update hardcoded categories in SQL migration files."""
    print(f"Updating SQL file: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    category_mapping = get_dynamic_category_mapping()
    
    # Pattern: INSERT VALUES with hardcoded categories
    # This is complex for SQL - for now, add a comment suggesting manual review
    if "006_dynamic_feature_categories.sql" in str(file_path):
        if "-- NOTE: This migration uses hardcoded categories" not in content:
            content = "-- NOTE: This migration uses hardcoded categories and should be reviewed\n" + \
                     "-- Consider using dynamic categories from OPERATION_METADATA\n" + content
            print(f"  Added warning comment about hardcoded categories")
    
    # Save if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Updated: {file_path}")
        return True
    else:
        print(f"  ⏭️  No changes needed: {file_path}")
        return False


def update_markdown_file(file_path: Path):
    """Update hardcoded categories in Markdown documentation."""
    print(f"Updating Markdown file: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Get current dynamic categories for examples
    available_categories = sorted(set(metadata['category'] for metadata in OPERATION_METADATA.values()))
    category_list = ', '.join(available_categories)
    
    # Replace static category lists with dynamic examples
    if "statistical, polynomial, binning, ranking, temporal, text, categorical" in content:
        content = content.replace(
            "statistical, polynomial, binning, ranking, temporal, text, categorical",
            f"{category_list} (dynamically discovered)"
        )
        print(f"  Replaced static category list with dynamic list")
    
    # Add note about dynamic discovery if talking about categories
    if "enabled_categories:" in content and "# NOTE: Categories are now dynamic" not in content:
        content = content.replace(
            "enabled_categories:",
            "# NOTE: Categories are now dynamic - see OPERATION_METADATA\nenabled_categories:"
        )
        print(f"  Added dynamic categories note")
    
    # Save if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Updated: {file_path}")
        return True
    else:
        print(f"  ⏭️  No changes needed: {file_path}")
        return False


def main():
    """Main update function."""
    print("🔄 Starting hardcoded category reference updates...")
    
    # List of files to update based on the search results
    files_to_update = [
        # Python source files
        "src/feature_space.py",
        "src/features/base.py", 
        "src/db/repositories/feature_repository.py",
        "scripts/tests/titanic_best_features_test.py",
        "test_dynamic_system.py",
        
        # SQL migration files
        "src/db/migrations/006_dynamic_feature_categories.sql",
        
        # Documentation files
        "docs/features/FEATURES_OVERVIEW.md",
        "docs/mcts/MCTS_OPERATIONS.md",
    ]
    
    project_root = Path(__file__).parent.parent
    updated_count = 0
    
    print(f"\n📊 Available dynamic categories: {sorted(set(metadata['category'] for metadata in OPERATION_METADATA.values()))}")
    print(f"📊 Category mapping: {get_dynamic_category_mapping()}")
    
    print(f"\n🚀 Processing {len(files_to_update)} files...")
    
    for file_path_str in files_to_update:
        file_path = project_root / file_path_str
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        try:
            if file_path.suffix == '.py':
                if update_python_file(file_path):
                    updated_count += 1
            elif file_path.suffix == '.sql':
                if update_sql_file(file_path):
                    updated_count += 1
            elif file_path.suffix == '.md':
                if update_markdown_file(file_path):
                    updated_count += 1
            else:
                print(f"⏭️  Skipping unsupported file type: {file_path}")
                
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")
    
    print(f"\n✅ Update complete! Updated {updated_count} files")
    print("\n📝 Update summary:")
    print("  - Replaced hardcoded category names with dynamic equivalents")
    print("  - Updated comments and documentation references")
    print("  - Added warnings for files that need manual review")
    print("\n🔍 Review the updated files and test the system to ensure everything works correctly")


if __name__ == '__main__':
    main()