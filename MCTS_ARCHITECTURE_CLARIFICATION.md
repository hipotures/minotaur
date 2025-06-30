# MCTS Architecture - How It Actually Works Now (2025-06-29)

## Executive Summary

The current MCTS system has evolved significantly from traditional feature generation during search to a **pre-built feature selection approach**. Here's how it actually works:

## Core Architecture Flow

### 1. Dataset Registration (Pre-MCTS Phase)
**Location**: Dataset manager (`manager.py datasets --register`)

**What happens**:
- All possible features are pre-built and cached in DuckDB tables during dataset registration
- Features are built on 100% of the training data for consistency
- Creates separate tables: `train_features`, `test_features`, `train_generic`, `train_custom`, etc.
- No-signal features (constant values) are filtered out during registration
- Column synchronization between train/test is enforced

**Key files**:
- `src/dataset_importer.py` - Handles feature pre-building
- `src/feature_space.py` - `generate_all_features()`, `generate_generic_features()`, `generate_custom_features()`
- `src/manager/modules/datasets/register.py` - Registration command

### 2. MCTS Search (Feature Selection Phase)
**Location**: Main MCTS runner (`mcts.py`)

**What happens**:
- MCTS operates on pre-built features, NOT generating new ones during search
- Each node represents a subset of available feature columns
- Node operations select which pre-existing feature columns to include
- AutoGluon evaluation uses SQL SELECT to load only chosen columns from DuckDB

**Key files**:
- `src/mcts_engine.py` - Tree search and node expansion
- `src/feature_space.py` - `get_feature_columns_for_node()` returns column names
- `src/autogluon_evaluator.py` - Loads features via SQL SELECT

## Detailed Component Analysis

### Feature Space Manager (`src/feature_space.py`)

**Current role**:
- Maintains operation definitions (statistical, polynomial, binning, ranking, custom domain)
- Maps MCTS operations to column name patterns in DuckDB tables
- Returns lists of column names, NOT DataFrames with new features

**Key methods**:
```python
get_feature_columns_for_node(node) -> List[str]  # Returns column names
get_available_operations(node) -> List[str]      # Available operations
```

**NOT used during search**:
```python
generate_features_for_node(node) -> pd.DataFrame  # Legacy method
```

### MCTS Engine (`src/mcts_engine.py`)

**Node representation**:
- Each `FeatureNode` represents a feature selection state
- `applied_operations` tracks which feature categories are included
- No actual feature generation during tree traversal

**Search process**:
1. **Selection**: Navigate tree using UCB1 to find promising feature combinations
2. **Expansion**: Add new operation (feature category) to include more columns
3. **Evaluation**: Use AutoGluon with SQL SELECT on chosen columns
4. **Backpropagation**: Update node scores based on AutoGluon results

### Dataset Manager (`src/dataset_manager.py`)

**Responsibilities**:
- Provides interface to registered datasets
- Returns DuckDB connection for column-based data access
- Validates dataset registration and feature availability

**Key insight**: Uses `dataset_name` from config to locate pre-built DuckDB files in `cache/` directory.

### AutoGluon Evaluator

**Current approach**:
- Receives list of feature column names from MCTS
- Executes SQL SELECT to load only those columns from DuckDB
- Applies sampling (train_size) at evaluation time, not feature generation time
- Returns MAP@3 score for fertilizer competition

## Feature Building During Registration

### Generic Features (`src/features/generic/`)
- Statistical aggregations (mean/std by groups)
- Polynomial features (squares, products)
- Binning (quantile-based bins)
- Ranking (percentiles, quartiles)

### Custom Domain Features (`src/features/custom/kaggle_s5e6.py`)
- NPK nutrient ratios and interactions
- Environmental stress indicators
- Agricultural compatibility scores
- Fertilizer urgency calculations

### Pre-building Process
1. Load original CSV/Parquet files
2. Generate ALL possible generic features using `GenericFeatureOperations`
3. Generate ALL possible custom features using domain-specific classes
4. Save to separate DuckDB tables (`train_generic`, `train_custom`, etc.)
5. Combine into final `train_features` and `test_features` tables
6. Filter out no-signal columns
7. Ensure train/test column synchronization

## Key Architectural Decisions

### Why Pre-build Features?
1. **Consistency**: Same features available across all MCTS runs
2. **Performance**: No I/O bottleneck during tree search
3. **Reproducibility**: Deterministic feature sets
4. **Memory efficiency**: Load only needed columns via SQL
5. **Speed**: DuckDB column-based access is very fast

### Why Column Selection vs. Generation?
1. **Search space control**: Finite, well-defined feature space
2. **AutoGluon optimization**: Can leverage column-based sampling
3. **Caching efficiency**: No need to cache generated features
4. **Debugging**: Easy to inspect what features are being used

### Current Limitations
1. **Feature space size**: Limited to pre-built features
2. **Dynamic operations**: Cannot create new feature combinations during search
3. **Memory usage**: All features stored in DuckDB (but only needed columns loaded)

## Configuration Impact

### Key config sections:
```yaml
autogluon:
  dataset_name: 'playground-series-s5e6-2025'  # Must be registered dataset
  target_metric: 'MAP@3'

feature_space:
  enabled_categories:
    - 'statistical_aggregations'    # Selects generic statistical columns
    - 'agricultural_domain'         # Selects custom domain columns
  max_features_per_node: 300        # Limits column count per evaluation
```

### Dataset registration:
```bash
# Pre-builds ALL features and stores in DuckDB
python manager.py datasets --register \
  --dataset-name playground-series-s5e6-2025 \
  --auto \
  --dataset-path /mnt/ml/competitions/2025/playground-series-s5e6/
```

## Performance Characteristics

### Fast Operations (Current Architecture)
- Column selection from DuckDB: ~milliseconds
- AutoGluon evaluation on selected columns: ~5-30 seconds
- MCTS tree traversal: ~milliseconds per node

### Slow Operations (Legacy Architecture - NOT USED)
- Feature generation during search: ~30-300 seconds
- I/O for loading full datasets: ~seconds
- Memory management for large feature sets: ~seconds

## Current Search Strategy

### Node States
- **Root**: Base columns only (Nitrogen, Phosphorous, etc.)
- **Depth 1**: Base + one operation (e.g., + statistical_aggregations)
- **Depth 2**: Base + two operations (e.g., + polynomial_features)
- **Depth N**: Base + N operations

### Column Selection Logic
```python
# Example for statistical_aggregations operation
if operation_name == 'statistical_aggregations':
    stat_patterns = ['_mean_by_', '_std_by_', '_dev_from_', '_count_by_']
    for col in all_columns:
        if any(pattern in col for pattern in stat_patterns):
            selected_columns.append(col)
```

### Evaluation Process
1. MCTS selects feature operation combination
2. `get_feature_columns_for_node()` returns matching column names
3. AutoGluon executes: `SELECT {columns} FROM train_features TABLESAMPLE({train_size})`
4. AutoGluon trains model and returns MAP@3 score
5. MCTS updates tree with score

## Future Enhancement Opportunities

1. **Dynamic feature combinations**: Allow operations on selected column subsets
2. **Hierarchical operations**: Chain operations (e.g., polynomial of aggregations)
3. **Feature interaction discovery**: Cross-operation combinations
4. **Online feature generation**: Selective generation of promising features
5. **Multi-dataset learning**: Transfer knowledge between registered datasets

This architecture provides a solid foundation for rapid feature space exploration while maintaining the benefits of MCTS-driven search strategy.