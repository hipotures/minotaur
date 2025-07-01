# Feature Catalog Migration Guide

## Overview

As of 2025-07-01, the feature operation naming convention has been standardized to use lowercase names with underscores. This change improves consistency between configuration files and database entries.

## Changes Made

### 1. Generic Feature Operations
All generic feature operations now return lowercase names:
- `"Statistical Aggregations"` → `"statistical_aggregations"`
- `"Polynomial Features"` → `"polynomial_features"`
- `"Binning Features"` → `"binning_features"`
- `"Ranking Features"` → `"ranking_features"`

### 2. FeatureSpace Query Updates
The FeatureSpace query logic has been updated to:
- Handle case-insensitive matching
- Convert spaces to underscores for comparison
- Properly route custom domain operations to their catalog entries

### 3. Custom Features
Custom features continue to be registered under `"{domain} Custom Features"` (e.g., "titanic Custom Features"), but the query logic now correctly handles this.

## Migration Instructions

### Option 1: Automatic Migration (Recommended)
Run the migration script to update existing feature_catalog entries:

```bash
# Update all datasets
python scripts/fix_feature_catalog_names.py

# Update specific dataset
python scripts/fix_feature_catalog_names.py titanic
```

### Option 2: Re-register Dataset
Re-register your dataset to regenerate the feature_catalog with new names:

```bash
python scripts/duckdb_manager.py datasets --register --dataset-name titanic --auto
```

### Option 3: No Action Required
The updated FeatureSpace query handles both old and new naming conventions, so your system will continue to work without migration. However, migration is recommended for consistency.

## Verification

To verify the operation names in your feature catalog:

```bash
bin/duckdb cache/[dataset_name]/dataset.duckdb -c "SELECT DISTINCT operation_name FROM feature_catalog;"
```

## Impact

- **Backward Compatible**: The system works with both old and new operation names
- **Performance**: No performance impact
- **Future Features**: New features will use the lowercase naming convention