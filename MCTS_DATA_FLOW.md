# MCTS Data Flow - Current Implementation

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

### FeatureNode
```python
@dataclass
class FeatureNode:
    applied_operations: List[str] = ['statistical_aggregations', 'custom_domain']
    base_features: Set[str] = {Nitrogen, Phosphorous, Potassium, ...}
    visit_count: int = 42
    total_reward: float = 14.322  # Sum of MAP@3 scores
    evaluation_score: float = 0.341  # Latest MAP@3 score
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

This architecture achieves the key MCTS goal of exploring feature combinations while avoiding the computational bottleneck of generating features during search. The pre-built approach provides consistent, fast access to a rich feature space.