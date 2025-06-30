<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30 13:35
Compatible with commit: 02d53a5
Changes: Created new implementation document with technical details and component analysis
-->

# MCTS Implementation - Technical Details

## 🔧 Core Components

### Feature Space Manager (`src/feature_space.py`)

**Primary responsibilities**:
- Maintains operation definitions (statistical, polynomial, binning, ranking, custom domain)
- Manages feature column selection during MCTS search
- Interfaces with pre-built feature tables in DuckDB

**Key methods**:
```python
FeatureSpace.__init__()  # Initialize with config and dataset manager
get_feature_columns_for_node(node)  # Returns column names for SQL SELECT
get_available_operations(node)  # Returns valid operations for expansion
apply_operation_to_node(node, operation)  # Creates child node with operation
```

**Current role**:
- **NOT generating features** during search (pre-built approach)
- **Column selection logic** for existing features
- **Operation definitions** for node expansion

### MCTS Engine (`src/mcts_engine.py`)

**Algorithm implementation**:
- **UCB1 Selection**: `ucb1_score = avg_reward + C * sqrt(ln(parent_visits) / node_visits)`
- **Tree Expansion**: Add new operation nodes up to configured depth
- **Backpropagation**: Update visit counts and reward totals up the tree
- **Best Path Extraction**: Return highest-scoring node path

**Key methods**:
```python
FeatureNode.__post_init__()  # Auto-assigns node_id = ++_node_counter
FeatureNode.add_child(child)  # Maintains parent-child relationships
run_search(evaluator, feature_space, db, initial_features)  # Main MCTS loop
select_node()  # UCB1-based selection
expand_node(node)  # Create children with new operations
backpropagate(node, reward)  # Update statistics up the tree
```

**Session-specific MCTS Logging** (Added 2025-06-30):
- Individual session log files: `logs/mcts/session_YYYYMMDD_HHMMSS.log`
- Only created when DEBUG logging level is enabled
- Session context preserved in log entries for cross-session analysis
- Replaces single `logs/mcts.log` file approach

**Search process**:
1. **Selection**: Navigate tree using UCB1 to find promising feature combinations
2. **Expansion**: Add new operation (feature category) to include more columns
3. **Simulation**: Evaluate node using AutoGluon on selected feature columns
4. **Backpropagation**: Update node statistics and propagate rewards to ancestors

### Dataset Manager (`src/dataset_manager.py`)

**Registration process**:
- Validates dataset structure (train.csv, test.csv, target column)
- Builds ALL possible features on 100% data during registration
- Creates DuckDB tables: `train_features`, `test_features`, `train_generic`, `train_custom`
- Filters no-signal features (constant values, single unique value)
- Enforces column synchronization between train/test

**Key methods**:
```python
DatasetManager.register_dataset()  # Main registration workflow
_generate_all_features()  # Calls feature generation
_validate_features()  # No-signal detection and filtering
_create_feature_tables()  # DuckDB table creation
```

**DuckDB Integration**:
- Uses connection pooling for thread safety
- Implements automatic schema migrations
- Provides query optimization for feature selection

### AutoGluon Evaluator (`src/autogluon_evaluator.py`)

**Evaluation strategy**:
- Loads only selected feature columns via SQL SELECT (not all features)
- Uses configurable data sampling for speed (train_size parameter)
- Supports multiple metrics: MAP@3, accuracy, F1, AUC, etc.
- GPU acceleration when available

**Key methods**:
```python
evaluate_features()  # Main evaluation method
_prepare_data()  # SQL SELECT for chosen columns only
_train_model()  # AutoGluon TabularPredictor training
_calculate_metrics()  # Target metric computation
```

**Performance optimizations**:
- Data sampling configurable per session
- Model type filtering (e.g., XGBoost only for speed)
- Time limits on training
- GPU utilization when available

## 🏗️ Feature Building Architecture

### Pre-building Process

**Location**: Dataset registration phase (`src/dataset_importer.py`)

**Generic Features** (`src/features/generic/`):
```python
# Statistical aggregations
statistical_ops = StatisticalAggregations()
features = statistical_ops.generate_features(
    df, 
    groupby_columns=['Soil Type', 'Crop Type'],
    aggregate_columns=['Nitrogen', 'Phosphorous', 'Potassium']
)

# Results in columns like:
# Nitrogen_mean_by_Soil_Type, Phosphorous_std_by_Crop_Type, etc.
```

**Custom Domain Features** (`src/features/custom/kaggle_s5e6.py`):
```python
# NPK ratios and agricultural domain logic
custom_ops = KaggleS5E6Features()
features = custom_ops.get_domain_features(df)

# Results in columns like:
# NP_ratio, PK_ratio, NK_ratio, nutrient_balance, moisture_stress, etc.
```

**Feature Categories**:
1. **Statistical Aggregations**: Mean, std, min, max by categorical groups
2. **Polynomial Features**: Squares, products, ratios of numeric columns  
3. **Binning Features**: Quantile-based discretization
4. **Ranking Features**: Percentile ranks within groups
5. **Domain Custom**: Competition-specific engineered features

## 🗃️ Database Schema & Persistence

### Core Tables

**Sessions Table**:
```sql
CREATE TABLE sessions (
    session_id VARCHAR PRIMARY KEY,
    session_name VARCHAR,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    total_iterations INTEGER DEFAULT 0,
    best_score DOUBLE DEFAULT 0.0,
    config_snapshot JSON,
    status VARCHAR DEFAULT 'active',
    dataset_hash VARCHAR
);
```

**Exploration History Table**:
```sql
CREATE TABLE exploration_history (
    id BIGINT PRIMARY KEY,
    session_id VARCHAR,
    iteration INTEGER,
    mcts_node_id INTEGER,
    parent_node_id BIGINT,
    operation_applied VARCHAR,
    features_before JSON,
    features_after JSON,
    evaluation_score DOUBLE,
    target_metric VARCHAR,
    evaluation_time DOUBLE,
    node_visits INTEGER DEFAULT 1,
    memory_usage_mb DOUBLE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Database Operations

**Connection Management**:
```python
# Thread-safe connection pooling
conn_manager = DuckDBConnectionManager(config)
with conn_manager.get_connection() as conn:
    result = conn.execute("SELECT * FROM exploration_history").df()
```

**MCTS Tree Persistence**:
- Each node evaluation logged with complete metadata
- Parent-child relationships tracked via `parent_node_id`
- Full feature lists stored as JSON for reproducibility
- Performance metrics and timing data captured

## 🎯 Node States and Operations

### FeatureNode Structure (Updated 2025-06-30)

```python
@dataclass
class FeatureNode:
    # Tree structure
    node_id: int                    # Auto-assigned sequential ID
    parent: Optional['FeatureNode'] = None
    children: List['FeatureNode'] = field(default_factory=list)
    depth: int = 0
    
    # MCTS statistics
    visit_count: int = 0
    total_reward: float = 0.0
    
    # Feature information
    feature_columns: List[str] = field(default_factory=list)
    operation_that_created_this: Optional[str] = None
    
    # Evaluation results
    evaluation_score: Optional[float] = None
    evaluation_time: Optional[float] = None
```

### Column Selection Logic

**Root Node**: 
- Contains base dataset columns (original features)
- Example: `['Nitrogen', 'Phosphorous', 'Potassium', 'Soil Type', 'Crop Type', ...]`

**Child Nodes**:
- Include parent columns PLUS operation-specific columns
- Example after `statistical_aggregations`: Parent columns + `['Nitrogen_mean_by_Soil_Type', 'Phosphorous_std_by_Crop_Type', ...]`

**Feature Selection Process**:
1. Node defines which pre-built columns to include
2. SQL SELECT query loads only those columns from DuckDB
3. AutoGluon trains on the selected subset
4. Evaluation score determines node value for UCB1

## 🔄 Key Architectural Decisions

### Why Pre-build Features?

**Problem with traditional MCTS feature generation**:
- Inconsistent results due to data sampling
- Slow feature generation (5-30 minutes per node)
- Complex dependency management
- Difficult to reproduce exact results

**Pre-building advantages**:
- **Consistency**: Same features every time (deterministic)
- **Speed**: 150-900x faster evaluation (seconds vs minutes)
- **Reliability**: No generation failures during search
- **Reproducibility**: Exact feature sets can be recreated

### Why Column Selection vs. Generation?

**Column selection approach**:
```python
# Fast: SQL SELECT with specific columns
selected_features = ['Nitrogen', 'NP_ratio', 'Nitrogen_mean_by_Soil_Type']
df = conn.execute(f"SELECT {','.join(selected_features)} FROM train_features").df()
```

**vs. Generation approach** (NOT used):
```python
# Slow: Runtime feature computation
df = apply_statistical_ops(df, groupby_cols=['Soil Type'])  # 5-30 minutes
```

### Current Limitations

1. **Fixed Feature Set**: Cannot generate truly novel features during search
2. **Memory Usage**: All features pre-built (but only selected columns loaded)
3. **Dataset Dependency**: Features tied to specific registered datasets

## 🛠️ APIs and Integration Points

### Main Entry Points

**MCTS Runner** (`mcts.py`):
```python
runner = FeatureDiscoveryRunner(config_path='config/mcts_config.yaml')
results = runner.run_discovery()
# Returns: {'session_id': '...', 'best_score': 0.85, 'iterations': 42}
```

**Dataset Registration**:
```python
from src import DatasetManager
manager = DatasetManager(config)
manager.register_dataset(
    name='my-competition',
    train_path='data/train.csv',
    test_path='data/test.csv',
    target_column='target'
)
```

**Session Analysis**:
```python
from src import AnalyticsGenerator
analytics = AnalyticsGenerator(config)
summary = analytics.generate_session_summary(session_id)
best_features = analytics.get_best_features(top_k=10)
```

### Configuration Integration

**Feature Space Configuration**:
```yaml
feature_space:
  max_features_per_node: 300
  enabled_categories:
    - 'statistical_aggregations'
    - 'polynomial_features' 
    - 'binning_features'
    - 'agricultural_domain'  # Domain-specific
```

**MCTS Configuration**:
```yaml
mcts:
  exploration_weight: 1.4        # UCB1 C parameter
  max_tree_depth: 8            # Maximum operation chaining
  expansion_threshold: 1       # Visits before expansion
  max_children_per_node: 5     # Branching factor limit
```

### Error Handling and Logging

**Comprehensive Logging**:
- Session-specific MCTS logs in DEBUG mode
- Database operation logs
- Feature generation timing logs
- AutoGluon evaluation logs

**Error Recovery**:
- Session resumption from database state
- Automatic backup creation
- Graceful handling of evaluation failures
- Memory management with automatic cleanup

## 🔗 File Organization

### Core Implementation Files
```
src/
├── mcts_engine.py              # MCTS algorithm implementation
├── feature_space.py            # Feature operation management
├── dataset_manager.py          # Dataset registration and caching
├── autogluon_evaluator.py      # ML evaluation
├── discovery_db.py             # Database interface
├── db_service.py               # Database operations
└── features/                   # Feature operation modules
    ├── base.py                 # Abstract base classes
    ├── generic/                # Domain-agnostic operations
    │   ├── statistical.py      # Statistical aggregations
    │   ├── polynomial.py       # Polynomial features
    │   ├── binning.py         # Quantile binning
    │   └── ranking.py         # Rank transformations
    └── custom/                 # Domain-specific operations
        ├── titanic.py         # Titanic dataset features
        └── kaggle_s5e6.py     # Fertilizer competition features
```

### Database Layer
```
src/db/
├── core/
│   ├── connection.py          # DuckDB connection management
│   └── base_repository.py     # Repository pattern base
├── models/                    # Pydantic data models
│   ├── session.py            # Session data structures
│   └── exploration.py        # Exploration history models
├── repositories/              # Data access layer
│   ├── session_repository.py # Session CRUD operations
│   └── exploration_repository.py # Exploration history CRUD
└── migrations/               # Database schema versions
    ├── 001_initial_schema.sql
    ├── 002_add_feature_metadata.sql
    └── 005_mcts_tree_persistence.sql
```

---

*Continue to [MCTS_DATA_FLOW.md](MCTS_DATA_FLOW.md) for data flow diagrams and examples*