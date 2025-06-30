<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30 13:35
Compatible with commit: 02d53a5
Changes: Refactored to focus only on data flow, removed duplicated content moved to other docs
-->

# MCTS Data Flow - Phase-by-Phase Analysis

## 📊 Phase 1: Dataset Registration (Pre-MCTS)

### Raw Data Processing Flow
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

### Feature Generation Examples

**Statistical Aggregations**:
```sql
-- Input: Original columns [Nitrogen, Phosphorous, Potassium, Soil Type, Crop Type]
-- Output: Aggregated features

SELECT 
    Soil_Type,
    AVG(Nitrogen) as Nitrogen_mean_by_Soil_Type,
    STDDEV(Nitrogen) as Nitrogen_std_by_Soil_Type,
    AVG(Phosphorous) as Phosphorous_mean_by_Soil_Type,
    STDDEV(Phosphorous) as Phosphorous_std_by_Soil_Type
FROM train_data 
GROUP BY Soil_Type;
```

**Custom Domain Features**:
```python
# NPK ratios and agricultural indicators
df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
df['PK_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
df['nutrient_balance'] = (df['Nitrogen'] + df['Phosphorous'] + df['Potassium']) / 3
df['moisture_stress'] = np.where(df['Moisture'] < 40, 1, 0)
```

## 🌳 Phase 2: MCTS Feature Selection

### Tree Search Data Flow
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

### Node Evaluation Data Pipeline
```
Node Selection (UCB1)
        ↓
Feature Column Mapping
        ↓
DuckDB Query Execution
        ↓
DataFrame Preparation
        ↓
AutoGluon Training
        ↓
Performance Evaluation
        ↓
Tree Statistics Update
        ↓
Database Logging
```

## 🗂️ Key Data Structures

### FeatureNode (Core Data Structure)
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
    visit_count: int = 0
    total_reward: float = 0.0
    
    # Evaluation results
    evaluation_score: Optional[float] = None
    evaluation_time: Optional[float] = None
    operation_that_created_this: Optional[str] = None
    depth: int = 0
```

### Feature Space Mapping
```python
# Column mapping for different operations
feature_mapping = {
    'base': ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type'],
    'statistical_aggregations': ['Nitrogen_mean_by_Soil_Type', 'Phosphorous_std_by_Crop_Type', ...],
    'polynomial_features': ['Nitrogen_squared', 'NP_product', 'NK_product', ...],
    'binning_features': ['Nitrogen_bin_1', 'Phosphorous_bin_2', ...],
    'custom_domain': ['NP_ratio', 'PK_ratio', 'nutrient_balance', 'moisture_stress', ...]
}
```

### DuckDB Query Examples

**Feature Column Selection**:
```sql
-- Root node evaluation (base features only)
SELECT Nitrogen, Phosphorous, Potassium, Temperature, Humidity, Moisture, 
       Soil_Type, Crop_Type, Fertilizer_Name
FROM train_features 
TABLESAMPLE(5%);

-- Node with statistical aggregations
SELECT Nitrogen, Phosphorous, Potassium, Temperature, Humidity, Moisture,
       Soil_Type, Crop_Type, Fertilizer_Name,
       Nitrogen_mean_by_Soil_Type, Phosphorous_std_by_Crop_Type,
       Potassium_max_by_Soil_Type, Temperature_min_by_Crop_Type
FROM train_features 
TABLESAMPLE(5%);

-- Node with statistical + custom domain features  
SELECT Nitrogen, Phosphorous, Potassium, Temperature, Humidity, Moisture,
       Soil_Type, Crop_Type, Fertilizer_Name,
       Nitrogen_mean_by_Soil_Type, Phosphorous_std_by_Crop_Type,
       NP_ratio, PK_ratio, NK_ratio, nutrient_balance, moisture_stress
FROM train_features 
TABLESAMPLE(5%);
```

## 🔍 Phase 3: MCTS Tree Persistence

### Database Logging Flow
```
Node Evaluation Complete
        ↓
Extract Node Metadata
        ↓
Serialize Feature Lists (JSON)
        ↓
Log to exploration_history Table
        ↓
Update Session Statistics
        ↓
Session-Specific Log File
```

### Real Database Examples

**Session Creation**:
```sql
INSERT INTO sessions (session_id, session_name, config_snapshot, dataset_hash)
VALUES ('2af6ead3-5b84-4d2f-b2bd-532aa6810d34', 
        'session_20250630_132513',
        '{"mcts": {"exploration_weight": 1.4}, ...}',
        'sha256:abc123...');
```

**Exploration Step Logging**:
```sql
INSERT INTO exploration_history (
    session_id, iteration, mcts_node_id, parent_node_id,
    operation_applied, features_before, features_after,
    evaluation_score, target_metric, evaluation_time,
    node_visits, memory_usage_mb
) VALUES (
    '2af6ead3-5b84-4d2f-b2bd-532aa6810d34',
    1,  -- First MCTS iteration
    3,  -- Node ID
    2,  -- Parent node ID (root = 2)
    'statistical_aggregations',
    '["Nitrogen", "Phosphorous", "Potassium", ...]',  -- JSON array
    '["Nitrogen", "Phosphorous", ..., "Nitrogen_mean_by_Soil_Type", ...]',
    0.341,  -- MAP@3 score
    'MAP@3',
    45.2,   -- 45.2 seconds evaluation time
    1,      -- First visit to this node
    1248.5  -- Memory usage in MB
);
```

### Tree Reconstruction Query
```sql
-- Reconstruct full MCTS tree for a session
WITH RECURSIVE mcts_tree AS (
    -- Root nodes (iteration 0)
    SELECT session_id, mcts_node_id, parent_node_id, operation_applied,
           evaluation_score, 0 as level, 
           CAST(mcts_node_id AS VARCHAR) as path
    FROM exploration_history 
    WHERE session_id = '2af6ead3-5b84-4d2f-b2bd-532aa6810d34' 
      AND iteration = 0
    
    UNION ALL
    
    -- Child nodes
    SELECT e.session_id, e.mcts_node_id, e.parent_node_id, e.operation_applied,
           e.evaluation_score, t.level + 1,
           t.path || '->' || CAST(e.mcts_node_id AS VARCHAR)
    FROM exploration_history e
    JOIN mcts_tree t ON e.parent_node_id = t.mcts_node_id
    WHERE e.session_id = '2af6ead3-5b84-4d2f-b2bd-532aa6810d34'
)
SELECT * FROM mcts_tree ORDER BY level, path;
```

## 📈 Memory Usage Patterns

### Data Loading Patterns
```
Dataset Registration (One-time):
├── Peak Memory: ~8GB (full feature generation)
├── Steady State: ~500MB (DuckDB file)
└── Duration: 2-5 minutes

MCTS Search (Per iteration):
├── Peak Memory: ~2GB (AutoGluon training)
├── Per-iteration: ~100-500MB (selected features only)
└── Duration: 30 sec - 2 min per iteration
```

### Feature Selection Memory Impact
```python
# Memory efficient: Only load selected columns
selected_columns = ['Nitrogen', 'NP_ratio', 'Nitrogen_mean_by_Soil_Type']  # 3 columns
df = conn.execute(f"SELECT {','.join(selected_columns)} FROM train_features").df()
# Memory usage: ~50MB for 100K rows x 3 columns

# vs. Memory intensive: Load all features
df = conn.execute("SELECT * FROM train_features").df()  
# Memory usage: ~2GB for 100K rows x 250+ columns
```

## 🔄 Data Synchronization

### Train/Test Consistency
```python
# Ensure column synchronization during registration
train_columns = set(train_df.columns)
test_columns = set(test_df.columns)

# Remove columns that exist in only one dataset
common_columns = train_columns.intersection(test_columns)
train_features = train_df[common_columns]
test_features = test_df[common_columns]

# Log synchronization results
logger.info(f"Train-only columns removed: {train_columns - common_columns}")
logger.info(f"Test-only columns removed: {test_columns - common_columns}")
logger.info(f"Final synchronized columns: {len(common_columns)}")
```

### No-Signal Feature Filtering
```python
# Remove features with no predictive signal
def filter_no_signal_features(df):
    features_to_remove = []
    
    for column in df.columns:
        if df[column].nunique() <= 1:  # Constant or single value
            features_to_remove.append(column)
        elif df[column].isna().sum() / len(df) > 0.95:  # >95% missing
            features_to_remove.append(column)
    
    return df.drop(columns=features_to_remove), features_to_remove
```

---

*For validation and testing details, see [MCTS_VALIDATION.md](MCTS_VALIDATION.md)*  
*For performance and configuration, see [MCTS_OPERATIONS.md](MCTS_OPERATIONS.md)*