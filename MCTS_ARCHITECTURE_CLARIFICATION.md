<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30
Compatible with commit: e7adb806e391a36f70f03c1a5b1d02926dd95023
Changes: Added MCTS node ID tracking, database persistence, validation framework
-->

# MCTS Architecture - How It Actually Works Now (2025-06-30)

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
- `node_id` auto-assigned using class-level counter for unique identification
- `parent` references maintain tree structure for backpropagation
- No actual feature generation during tree traversal

**Node ID Management** (Added 2025-06-30):
- Each `FeatureNode` has auto-assigned `node_id` using class-level counter
- Database tracking with `mcts_node_id` column in `exploration_history`
- Parent-child relationships tracked via `parent_node_id`
- Proper tree structure persistence for session resumption and analysis

**Key methods**:
```python
FeatureNode.__post_init__()  # Auto-assigns node_id = ++_node_counter
FeatureNode.add_child(child)  # Maintains parent-child relationships
log_exploration_step(..., mcts_node_id=node.node_id)  # Database logging
```

**Search process**:
1. **Selection**: Navigate tree using UCB1 to find promising feature combinations
2. **Expansion**: Add new operation (feature category) to include more columns
3. **Evaluation**: Use AutoGluon with SQL SELECT on chosen columns
4. **Backpropagation**: Update node scores and visit counts through parent chain

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

## MCTS Validation Framework (Added 2025-06-30)

**Location**: `scripts/mcts/`

**Validation scripts**:
- `validate_mcts.py` - Comprehensive MCTS implementation validation
- `visualize_tree.py` - ASCII tree structure visualization  
- `analyze_session.py` - Deep session analysis and metrics
- `monitor_live.py` - Real-time MCTS monitoring

**Validation tests**:
1. **Node ID Assignment** - ✅ Working (validates sequential node ID assignment)
2. **Parent-Child Relationships** - ✅ Working (validates tree structure integrity)
3. **Visit Count Accumulation** - ❌ Needs improvement (backpropagation issue)
4. **Feature Evolution** - ✅ Working (validates feature changes per operation)
5. **Tree Growth** - ✅ Working (validates iteration progression)
6. **MCTS Logging** - ✅ Working (validates dedicated MCTS logger functionality)

**Usage**:
```bash
# Validate latest MCTS session
python scripts/mcts/validate_mcts.py

# Visualize tree structure
python scripts/mcts/visualize_tree.py SESSION_ID

# Analyze session performance
python scripts/mcts/analyze_session.py SESSION_ID
```

## Database Persistence Layer

**Key tables**:
- `exploration_history` - Main MCTS exploration tracking with node IDs
- `mcts_tree_nodes` - Future dedicated tree structure storage (migration available)
- `sessions` - Session metadata and statistics

**Critical fields added**:
- `mcts_node_id` - Internal MCTS node identifier for tree reconstruction
- `parent_node_id` - Parent's MCTS node ID for tree relationships
- `node_visits` - Visit count for UCB1 calculations

## Known Issues (2025-06-30)

1. **Missing Iteration 0**: Root evaluation (baseline) not being logged to database
   - MCTS shows "iteration 0" in logs but exploration_history starts from iteration 1
   - Root node evaluation happening but not persisted properly
   - Affects completeness of tree analysis

2. **Visit Count Accumulation**: Nodes not getting multiple visits during backpropagation
   - All nodes show `node_visits = 1` in database
   - Indicates backpropagation may not be updating visit counts properly
   - UCB1 calculations may be affected

## Future Enhancement Opportunities

1. **Dynamic feature combinations**: Allow operations on selected column subsets
2. **Hierarchical operations**: Chain operations (e.g., polynomial of aggregations)
3. **Feature interaction discovery**: Cross-operation combinations
4. **Online feature generation**: Selective generation of promising features
5. **Multi-dataset learning**: Transfer knowledge between registered datasets
6. **Session resumption**: Complete tree restoration from database state
7. **Advanced tree analysis**: Exploit validation framework for optimization insights

This architecture provides a solid foundation for rapid feature space exploration while maintaining the benefits of MCTS-driven search strategy, now enhanced with comprehensive tree persistence and validation capabilities.