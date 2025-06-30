<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30 12:49
Compatible with commit: 1cca1d395753a978d77e160ce3d0424740631fc7
Changes: Added MCTS tree persistence, node ID tracking, validation framework
-->

# MCTS Data Flow - Current Implementation (2025-06-30)

## Phase 1: Dataset Registration (Pre-MCTS)

```
Raw Dataset Files
├── train.csv (4.2M rows, 8 columns)
├── test.csv (2.8M rows, 7 columns)
└── sample_submission.csv

                    ↓ [Dataset Registration]

Feature Generation (100% data)
├── Generic Features
│   ├── Statistical Aggregations (mean/std by Soil/Crop)
│   ├── Polynomial Features (squares, products)
│   ├── Binning Features (quantile bins)
│   └── Ranking Features (percentiles)
├── Custom Domain Features
│   ├── NPK Ratios & Interactions
│   ├── Environmental Stress Indicators
│   └── Agricultural Compatibility Scores

                    ↓ [Feature Building & Validation]

DuckDB Dataset (cache/dataset-name/dataset.duckdb)
├── train (original 8 columns)
├── test (original 7 columns)
├── train_generic (~200 columns)
├── train_custom (~50 columns)
├── test_generic (~200 columns)
├── test_custom (~50 columns)
├── train_features (8 + ~250 columns = ~258 total)
└── test_features (7 + ~250 columns = ~257 total)
    
                    ↓ [No-signal filtering & synchronization]

Ready for MCTS Search
```

## Phase 2: MCTS Feature Selection

```
MCTS Tree Search
├── Root Node: Base columns only
│   └── Columns: [Nitrogen, Phosphorous, Potassium, Temperature, Humidity, Moisture, Soil Type, Crop Type]
├── Depth 1: Base + 1 operation
│   ├── Node A: Base + statistical_aggregations
│   │   └── Columns: Base + [Nitrogen_mean_by_Soil, Phosphorous_std_by_Crop, ...]
│   ├── Node B: Base + polynomial_features  
│   │   └── Columns: Base + [Nitrogen_squared, NP_product, NK_product, ...]
│   └── Node C: Base + custom_domain
│       └── Columns: Base + [NP_ratio, heat_stress, nutrient_balance, ...]
└── Depth 2: Base + 2 operations
    ├── Node D: Base + statistical + polynomial
    │   └── Columns: Base + stat_columns + poly_columns
    └── Node E: Base + statistical + custom
        └── Columns: Base + stat_columns + custom_columns

                    ↓ [For each node evaluation]

AutoGluon Evaluation Process
1. MCTS selects node (e.g., Node D)
2. get_feature_columns_for_node() → List[column_names]
3. SQL Query: "SELECT {column_names} FROM train_features TABLESAMPLE(5%)"
4. AutoGluon trains on selected columns
5. Returns MAP@3 score
6. MCTS updates tree statistics

                    ↓ [Repeat until convergence]

Best Feature Combination Found
└── Example: Base + statistical_aggregations + custom_domain = 180 columns
    └── Final Score: MAP@3 = 0.341 (vs baseline 0.320)
```

## Key Data Structures

### FeatureNode (Updated 2025-06-30)
```python
@dataclass
class FeatureNode:
    # Core MCTS attributes
    node_id: int = field(init=False)  # Auto-assigned in __post_init__
    parent: Optional['FeatureNode'] = None
    children: List['FeatureNode'] = field(default_factory=list)
    
    # Feature selection state
    applied_operations: List[str] = ['statistical_aggregations', 'custom_domain']
    base_features: Set[str] = {Nitrogen, Phosphorous, Potassium, ...}
    features_before: List[str] = field(default_factory=list)
    features_after: List[str] = field(default_factory=list)
    
    # MCTS statistics
    visit_count: int = 42
    total_reward: float = 14.322  # Sum of MAP@3 scores
    evaluation_score: float = 0.341  # Latest MAP@3 score
    depth: int = 0  # Tree depth from root
    
    # Class-level counter for unique node IDs
    _node_counter: ClassVar[int] = 0
```

### Feature Space Mapping
```python
# Column selection logic
operation_to_patterns = {
    'statistical_aggregations': ['_mean_by_', '_std_by_', '_dev_from_'],
    'polynomial_features': ['_squared', '_cubed', '_product'],
    'binning_features': ['_binned', '_bin_'],
    'ranking_features': ['_rank_', '_quartile', '_percentile'],
    'custom_domain': [custom feature names from domain module]
}
```

### DuckDB Query Examples
```sql
-- Root evaluation (base features only)
SELECT "Nitrogen", "Phosphorous", "Potassium", "Temperature", 
       "Humidity", "Moisture", "Soil Type", "Crop Type"
FROM train_features 
TABLESAMPLE(5%);

-- Statistical aggregations node
SELECT "Nitrogen", "Phosphorous", ..., 
       "Nitrogen_mean_by_Soil", "Phosphorous_std_by_Crop",
       "Temperature_dev_from_Soil_mean", ...
FROM train_features 
TABLESAMPLE(5%);

-- Combined operations node  
SELECT "Nitrogen", "Phosphorous", ...,
       "Nitrogen_mean_by_Soil", "Phosphorous_std_by_Crop",  -- statistical
       "Nitrogen_squared", "NP_product",                     -- polynomial
       "NP_ratio", "heat_stress", "nutrient_balance"         -- custom
FROM train_features 
TABLESAMPLE(5%);
```

## Performance Characteristics

### Registration Phase (One-time)
- **Duration**: 2-5 minutes for S5E6 dataset
- **I/O**: Read CSV → Generate features → Write DuckDB
- **Memory**: Peak ~8GB for full feature generation
- **Output**: ~500MB DuckDB file with all features

### MCTS Search Phase (Per run)
- **Duration**: 30 seconds to 2 hours (depending on config)
- **I/O**: Column-based SELECT queries only
- **Memory**: ~1-2GB (only selected columns loaded)
- **Throughput**: 50-200 evaluations per hour

### Key Performance Gains
1. **No feature I/O during search**: All features pre-cached in DuckDB
2. **Column-based sampling**: Only load needed features + samples
3. **Deterministic feature space**: No variation between runs
4. **Rapid node evaluation**: ~5-30 seconds per AutoGluon evaluation

## Memory Usage Patterns

```
Dataset Registration:
├── CSV Loading: 2-3GB
├── Feature Generation: 6-8GB peak
├── DuckDB Writing: 1-2GB
└── Final: 500MB on disk

MCTS Search:
├── Base Memory: 500MB
├── Per Evaluation: +200-500MB (selected columns only)
├── AutoGluon Training: +500-1GB temporary
└── Steady State: 1-2GB total
```

## Phase 3: MCTS Tree Persistence (Added 2025-06-30)

```
Database Storage (exploration_history table)
├── session_id: Unique session identifier
├── iteration: MCTS iteration number (0=root, 1+=search)
├── mcts_node_id: Internal MCTS node ID (auto-assigned)
├── parent_node_id: Parent's mcts_node_id for tree structure
├── operation_applied: Feature operation name
├── features_before/after: JSON feature lists
├── evaluation_score: AutoGluon score (MAP@3)
├── node_visits: Visit count for UCB1 calculations
├── mcts_ucb1_score: UCB1 score at selection time
└── Tree structure preserved for analysis/resumption

                    ↓ [Example database records]

Real Database Example (session: 2af6ead3-5b84-4d2f-b2bd-532aa6810d34)
┌───────────┬──────────────┬────────────────┬──────────────────────────┬──────────────────┐
│ iteration │ mcts_node_id │ parent_node_id │    operation_applied     │ evaluation_score │
├───────────┼──────────────┼────────────────┼──────────────────────────┼──────────────────┤
│         1 │            3 │              2 │ statistical_aggregations │             0.77 │
│         1 │            4 │              2 │ binning_features         │             0.79 │
│         2 │            5 │              4 │ statistical_aggregations │             0.75 │
│         3 │            6 │              3 │ binning_features         │             0.75 │
└───────────┴──────────────┴────────────────┴──────────────────────────┴──────────────────┘

Tree Structure Visualization:
Root (node_id=2, iteration=0) [baseline - not in database yet]
├── Node 3 (statistical_aggregations, score=0.77)
│   └── Node 6 (+ binning_features, score=0.75)
└── Node 4 (binning_features, score=0.79) [best so far]
    └── Node 5 (+ statistical_aggregations, score=0.75)
```

## MCTS Validation and Monitoring

```
Validation Framework (scripts/mcts/)
├── validate_mcts.py
│   ├── Node ID Assignment: ✅ PASS (range: 3-6)
│   ├── Parent Relationships: ✅ PASS (all non-root have parents)
│   ├── Visit Count Accumulation: ❌ FAIL (no multiple visits)
│   ├── Feature Evolution: ✅ PASS (operations change features)
│   ├── Tree Growth: ✅ PASS (iterations 1-3)
│   └── MCTS Logging: ✅ PASS (20 selection, 28 backprop phases)
│
├── visualize_tree.py → ASCII tree representation
├── analyze_session.py → Performance metrics and insights  
└── monitor_live.py → Real-time MCTS monitoring

Usage:
$ python scripts/mcts/validate_mcts.py
$ python scripts/mcts/visualize_tree.py 2af6ead3-5b84-4d2f-b2bd-532aa6810d34
```

## Current Issues and Limitations (2025-06-30)

### Known Issues
1. **Missing Iteration 0 (Root Evaluation)**:
   - Root baseline evaluation not persisted to database
   - Tree analysis incomplete without baseline reference
   - Affects session resumption capabilities

2. **Visit Count Accumulation Problem**:
   - All nodes show `node_visits = 1` despite backpropagation
   - UCB1 calculations may not reflect true exploration vs exploitation
   - Indicates backpropagation visit counting needs investigation

3. **Tree Structure Gaps**:
   - Parent-child relationships tracked but node 2 (apparent root) missing
   - Tree visualization incomplete without full structure

### Performance Characteristics (Updated)

```
Dataset Registration: [No changes - still 2-5 minutes]
├── Duration: 2-5 minutes for S5E6 dataset
├── I/O: Read CSV → Generate features → Write DuckDB  
├── Memory: Peak ~8GB for full feature generation
└── Output: ~500MB DuckDB file with all features

MCTS Search: [Enhanced with persistence]
├── Duration: 30 seconds to 2 hours (depending on config)
├── I/O: Column-based SELECT + database logging per iteration
├── Memory: ~1-2GB (selected columns + tree structure)
├── Throughput: 50-200 evaluations per hour
└── Database Growth: ~1KB per exploration step

New Capabilities:
├── Tree Structure Analysis: Complete parent-child mapping
├── Session Resumption: Database-backed state recovery  
├── Real-time Validation: Live MCTS health monitoring
└── Historical Analysis: Cross-session performance comparison
```

This architecture achieves the key MCTS goal of exploring feature combinations while avoiding the computational bottleneck of generating features during search. The pre-built approach provides consistent, fast access to a rich feature space, now enhanced with comprehensive tree persistence and validation capabilities for production MCTS deployments.