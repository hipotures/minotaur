<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30 14:35
Compatible with commit: TBD
Changes: Created comprehensive integration guide for features system
-->

# Features Integration - Pipeline & Dataset Management

## 🔗 System Integration Overview

The Feature Engineering System is deeply integrated with the Minotaur ecosystem, providing seamless feature generation during dataset registration and direct integration with MCTS feature selection.

### Integration Architecture
```
Dataset Registration → Feature Generation → DuckDB Storage → MCTS Selection
        ↓                       ↓                    ↓             ↓
   Data Validation      Signal Detection      Column Mapping   Performance
   Schema Analysis      Timing Tracking       Query Optimization   Evaluation
```

## 🗄️ Dataset Registration Integration

### Automatic Feature Generation Flow

**During Dataset Registration**:
1. **Dataset Validation**: Schema analysis and data type detection
2. **Feature Generation**: All operations run on 100% of data
3. **Signal Detection**: No-signal features filtered automatically
4. **DuckDB Storage**: Features stored in separate tables
5. **Metadata Tracking**: Complete feature metadata recorded

### Registration Process with Features

```bash
# Standard registration with automatic feature generation
python manager.py datasets --register \
  --dataset-name fertilizer-competition \
  --dataset-path /data/kaggle-s5e6/ \
  --auto

# Registration process output:
# 📋 Dataset Registration Progress
# ===============================
# ✅ Dataset validation completed
# ✅ Schema analysis completed
# 🔧 Generating features...
#    ├── Generic features: 234 generated, 45 filtered (no signal)
#    ├── Custom features: 67 generated, 12 filtered (no signal)
#    └── Total: 244 features ready for MCTS
# ✅ DuckDB storage completed
# ✅ Feature metadata recorded
```

### Dataset Tables Structure

**DuckDB Tables Created**:
```sql
-- Original data
train                    -- Original training data
test                     -- Original test data

-- Generic features  
train_generic           -- Generic features applied to train
test_generic            -- Generic features applied to test

-- Custom features
train_custom            -- Custom domain features for train
test_custom             -- Custom domain features for test

-- Combined features
train_features          -- train + train_generic + train_custom
test_features           -- test + test_generic + test_custom
```

### Feature Generation Configuration

```yaml
# Dataset registration with feature control
dataset_registration:
  # Feature generation settings
  generate_features: true
  
  # Signal detection
  check_signal: true
  min_signal_ratio: 0.01
  signal_sample_size: 1000
  
  # Feature categories to generate
  enabled_categories:
    - 'statistical_aggregations'
    - 'polynomial_features'
    - 'binning_features'
    - 'ranking_features'
    - 'temporal_features'      # If datetime columns detected
    - 'text_features'          # If text columns detected  
    - 'categorical_features'
    - 'agricultural_domain'    # Domain-specific (if applicable)
  
  # Performance settings
  max_features_per_operation: 500
  parallel_generation: false
  cache_intermediate: true
```

### Column Configuration Priority Logic

**Target and ID Column Detection**:

The system uses a three-tier priority logic for determining target and ID columns during dataset registration:

**Priority Order**: `Config File → CLI Parameters → Auto-detection`

```python
# Priority Logic Implementation
target_column = config.get('target_column') or None
if CLI_target_provided:
    target_column = CLI_target_value  # CLI overrides config
    
if target_column is None:
    target_column = auto_detection()
    if target_column is None:
        ERROR + STOP  # Target is required
```

**Target Column Logic**:
1. **Config File**: Load from `config/mcts_config.yaml`
2. **CLI Override**: `--target-column Survived` overrides config
3. **Auto-detection**: If both missing, attempt auto-detection
4. **Error Handling**: **ERROR + STOP** if target cannot be determined

**ID Column Logic**:
1. **Config File**: Load from `config/mcts_config.yaml`
2. **CLI Override**: `--id-column PassengerId` overrides config  
3. **Auto-detection**: If both missing, attempt auto-detection
4. **Error Handling**: **WARNING + CONTINUE** if ID cannot be determined (optional)

**Example Configuration**:
```yaml
# config/mcts_config.yaml
autogluon:
  target_column: 'Survived'    # Base configuration
  id_column: 'PassengerId'     # Base configuration
  ignore_columns: []           # Additional columns to ignore
```

**Example CLI Override**:
```bash
# CLI parameters override config values
python manager.py datasets --register \
  --dataset-name titanic \
  --dataset-path datasets/Titanic \
  --target-column Survived \      # Overrides config
  --id-column PassengerId         # Overrides config
```

**Column Name Normalization**:
- All column names are automatically converted to lowercase for database consistency
- Original case is preserved in logs for clarity
- Feature generation respects database column names (lowercase)

## 🔄 Pipeline Types & Migration

### Legacy Pipeline (Pre-2025)

**Characteristics**:
- Post-generation signal filtering
- Limited metadata tracking
- Single-threaded execution
- Basic timing information

**Usage**:
```python
# Automatic fallback for compatibility
feature_space = FeatureSpace(config)
features = feature_space.get_all_available_features(df, dataset_name)
```

### New Pipeline (Current)

**Characteristics**:
- During-generation signal detection (~50% faster)
- Comprehensive metadata tracking
- Advanced caching and optimization
- Detailed timing and performance logging

**Enabling New Pipeline**:
```yaml
# In MCTS configuration
feature_space:
  use_new_pipeline: true
  check_signal: true
  apply_generic_to_custom: true
  cache_features: true
```

**Usage**:
```python
# Explicitly use new pipeline
feature_space = FeatureSpace(config)
features_df = feature_space.generate_all_features_pipeline(
    df, 
    dataset_name='fertilizer-s5e6',
    target_column='Fertilizer_Name',
    id_column='id'
)

# Access metadata
metadata = feature_space.get_feature_metadata()
for name, meta in metadata.items():
    print(f"{name}: {meta.feature_type.value}, signal={meta.has_signal}")
```

### Migration Strategy

**Gradual Migration Approach**:
1. **Phase 1**: Test new pipeline in development with `use_new_pipeline: true`
2. **Phase 2**: Compare results between old and new pipeline
3. **Phase 3**: Production deployment with new pipeline
4. **Phase 4**: Remove legacy pipeline support

**Backward Compatibility**:
```python
# Old imports still work with deprecation warnings
from src.domains.generic import GenericFeatureOperations  # Still works
from src.features.generic.statistical import StatisticalFeatures  # Recommended

# Old FeatureSpace methods still work
feature_space.get_all_available_features()  # Legacy method
feature_space.generate_all_features_pipeline()  # New method
```

## 🧮 MCTS Integration

### Feature Column Selection

**MCTS Node Structure**:
```python
# Each MCTS node defines which feature categories to include
node_features = {
    'base': ['Nitrogen', 'Phosphorous', 'Potassium', 'Soil_Type'],
    'operations': ['statistical_aggregations', 'custom_domain']
}

# Column mapping for SQL queries
selected_columns = feature_space.get_feature_columns_for_node(node)
# Returns: ['Nitrogen', 'Phosphorous', ..., 'Nitrogen_mean_by_Soil_Type', 'NP_ratio']
```

**SQL Query Generation**:
```sql
-- MCTS selects specific columns for evaluation
SELECT 
    Nitrogen, Phosphorous, Potassium, Soil_Type, Crop_Type,  -- Base features
    Nitrogen_mean_by_Soil_Type, Phosphorous_std_by_Crop_Type, -- Statistical
    NP_ratio, PK_ratio, nutrient_balance,                    -- Custom domain
    Fertilizer_Name                                          -- Target
FROM train_features 
TABLESAMPLE(5%)  -- Configurable sampling for speed
```

### Feature Space Configuration for MCTS

```yaml
# MCTS-specific feature configuration
feature_space:
  # Core settings
  max_features_per_node: 300
  use_new_pipeline: true
  
  # Available operations for MCTS tree expansion
  enabled_categories:
    - 'statistical_aggregations'
    - 'polynomial_features'
    - 'binning_features'
    - 'agricultural_domain'
  
  # MCTS performance optimization
  sample_for_signal_check: true
  signal_sample_size: 1000
  cache_column_mappings: true
  
  # Generic operation parameters
  generic_params:
    statistical:
      groupby_columns: ['Soil_Type', 'Crop_Type']
      aggregate_columns: ['Nitrogen', 'Phosphorous', 'Potassium']
    polynomial:
      degree: 2
      include_interactions: true
    binning:
      n_bins: 5
      strategy: 'quantile'
```

## 📊 Manager Command Integration

### Feature Analysis Commands

**Basic Feature Operations**:
```bash
# List all features with performance metrics
python manager.py features --list

# Show top performing features
python manager.py features --top 20

# Feature catalog overview
python manager.py features --catalog
```

**Advanced Analysis**:
```bash
# Detailed feature impact analysis
python manager.py features --impact "Nitrogen_mean_by_Soil_Type"

# Search features by name or category
python manager.py features --search "nitrogen"
python manager.py features --search "ratio"

# Filter by specific criteria
python manager.py features --list --category statistical_aggregations
python manager.py features --list --dataset fertilizer-s5e6
python manager.py features --list --min-impact 0.01
```

**Export and Integration**:
```bash
# Export feature data for analysis
python manager.py features --export csv
python manager.py features --export json

# Integration with other commands
python manager.py datasets --show fertilizer-s5e6  # Shows feature generation status
python manager.py sessions --details SESSION_ID    # Shows feature usage
```

### Dataset-Feature Integration Commands

```bash
# Dataset commands that interact with features
python manager.py datasets --register --dataset-name NAME --auto
python manager.py datasets --show NAME --include-features
python manager.py datasets --stats NAME --feature-breakdown
python manager.py datasets --update NAME --regenerate-features
```

## 🗃️ Database Schema Integration

### Feature Metadata Tables

**Feature Catalog** (`feature_catalog`):
```sql
CREATE TABLE feature_catalog (
    feature_name VARCHAR PRIMARY KEY,
    dataset_name VARCHAR,
    feature_type VARCHAR,  -- 'ORIGINAL', 'GENERIC', 'CUSTOM'
    category VARCHAR,      -- 'statistical_aggregations', 'custom_domain', etc.
    operation VARCHAR,     -- Specific operation name
    source_columns JSON,   -- Input columns used
    generation_time REAL,  -- Time to generate
    has_signal BOOLEAN,    -- Signal detection result
    created_at TIMESTAMP
);
```

**Feature Impact** (`feature_impact`):
```sql
CREATE TABLE feature_impact (
    feature_name VARCHAR,
    session_id VARCHAR,
    dataset_hash VARCHAR,
    impact REAL,          -- Performance improvement
    usage_count INTEGER,  -- Times used in session
    best_score REAL,      -- Best score achieved
    timestamp TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
```

**Feature Dependencies** (`feature_dependencies`):
```sql
CREATE TABLE feature_dependencies (
    feature_name VARCHAR,
    depends_on_feature VARCHAR,
    dependency_type VARCHAR,  -- 'DERIVED_FROM', 'GROUPED_BY', etc.
    strength REAL,            -- Dependency strength (0-1)
    PRIMARY KEY (feature_name, depends_on_feature)
);
```

### Query Examples

**Feature Performance Analysis**:
```sql
-- Top performing features across all sessions
SELECT 
    fc.feature_name,
    fc.category,
    COUNT(fi.session_id) as session_count,
    AVG(fi.impact) as avg_impact,
    MAX(fi.impact) as max_impact
FROM feature_catalog fc
JOIN feature_impact fi ON fc.feature_name = fi.feature_name
GROUP BY fc.feature_name, fc.category
ORDER BY avg_impact DESC
LIMIT 20;
```

**Dataset Feature Summary**:
```sql
-- Feature generation summary for dataset
SELECT 
    category,
    COUNT(*) as feature_count,
    AVG(generation_time) as avg_gen_time,
    SUM(CASE WHEN has_signal THEN 1 ELSE 0 END) as signal_count
FROM feature_catalog 
WHERE dataset_name = 'fertilizer-s5e6'
GROUP BY category
ORDER BY feature_count DESC;
```

## 🔧 Configuration Integration

### MCTS Configuration with Features

```yaml
# Complete MCTS configuration with feature integration
autogluon:
  dataset_name: 'fertilizer-s5e6'  # Must match registered dataset
  target_metric: 'MAP@3'
  train_size: 10000
  time_limit: 120

# Feature system configuration
feature_space:
  # Pipeline selection
  use_new_pipeline: true
  
  # Performance settings
  max_features_per_node: 300
  check_signal: true
  signal_sample_size: 1000
  
  # Feature categories
  enabled_categories:
    - 'statistical_aggregations'
    - 'polynomial_features'
    - 'binning_features'
    - 'agricultural_domain'
  
  # Operation parameters
  generic_params:
    statistical:
      groupby_columns: ['Soil_Type', 'Crop_Type']
      aggregate_columns: ['Nitrogen', 'Phosphorous', 'Potassium']
    polynomial:
      degree: 2
      max_interaction_features: 5
    binning:
      n_bins: 5
      strategy: 'quantile'

# MCTS algorithm
mcts:
  exploration_weight: 1.4
  max_tree_depth: 8
  max_children_per_node: 5

# Session management
session:
  max_iterations: 50
  max_runtime_hours: 2.0
```

### Dataset Registration Configuration

```yaml
# Dataset registration with feature settings
dataset_registration:
  # Basic settings
  auto_detect_schema: true
  validate_data_quality: true
  
  # Feature generation
  generate_features: true
  feature_pipeline: 'new'  # 'legacy' or 'new'
  
  # Signal detection
  check_signal: true
  min_signal_ratio: 0.01
  signal_sample_size: 1000
  
  # Performance
  parallel_generation: false
  max_memory_gb: 8
  
  # Feature categories (auto-detected + specified)
  enabled_categories:
    - 'statistical_aggregations'
    - 'polynomial_features'
    - 'binning_features'
    - 'ranking_features'
    - 'categorical_features'
    # Domain-specific categories auto-detected:
    # - 'agricultural_domain' (if NPK columns found)
    # - 'temporal_features' (if datetime columns found)
    # - 'text_features' (if text columns found)
```

## 🚀 Performance Integration

### Caching Strategy

**DuckDB Column Storage**:
```
# Feature data is stored in columnar format for fast selection
train_features/
├── Nitrogen (column)
├── Phosphorous (column)
├── Nitrogen_mean_by_Soil_Type (column)
├── NP_ratio (column)
└── ... (250+ columns)

# MCTS queries only load required columns
SELECT Nitrogen, NP_ratio, Nitrogen_mean_by_Soil_Type FROM train_features
# Only 3 columns loaded instead of 250+
```

**Metadata Caching**:
```python
# Feature metadata cached in memory for fast access
feature_space._metadata_cache = {
    'Nitrogen_mean_by_Soil_Type': FeatureMetadata(
        feature_type=FeatureType.GENERIC,
        category='statistical_aggregations',
        has_signal=True,
        generation_time=0.045
    )
}
```

### Memory Management

**Lazy Loading Strategy**:
```python
# Features generated once during registration, loaded on-demand
class FeatureSpace:
    def get_feature_columns_for_node(self, node):
        # Only returns column names, no data loading
        return self._build_column_list(node.applied_operations)
    
    def load_features_for_evaluation(self, columns, sample_size=None):
        # SQL query loads only specified columns
        query = f"SELECT {','.join(columns)} FROM train_features"
        if sample_size:
            query += f" TABLESAMPLE({sample_size})"
        return self.conn.execute(query).df()
```

### Integration Performance Metrics

| Integration Point | Cold Start | Warm Cache | Notes |
|------------------|------------|-------------|-------|
| Dataset Registration | 5-15 min | N/A | One-time feature generation |
| MCTS Column Selection | 1-5ms | <1ms | Metadata lookup only |
| Feature Loading | 100-500ms | 50-200ms | SQL query execution |
| Manager Commands | 200-1000ms | 100-500ms | Database queries |

## 🔄 Migration Examples

### From Legacy to New Pipeline

**Step 1: Test Compatibility**
```python
# Test both pipelines with same data
config_legacy = {'feature_space': {'use_new_pipeline': False}}
config_new = {'feature_space': {'use_new_pipeline': True}}

# Compare results
legacy_features = FeatureSpace(config_legacy).get_all_available_features(df)
new_features = FeatureSpace(config_new).generate_all_features_pipeline(df)

# Check feature overlap
common_features = set(legacy_features.columns) & set(new_features.columns)
print(f"Common features: {len(common_features)}")
```

**Step 2: Update Configuration**
```yaml
# Gradual migration configuration
feature_space:
  use_new_pipeline: true          # Enable new pipeline
  fallback_to_legacy: true        # Fallback on errors
  compare_with_legacy: false      # Disable comparison in production
  
  # New pipeline specific settings
  check_signal: true
  signal_sample_size: 1000
  cache_features: true
```

**Step 3: Monitor Performance**
```bash
# Monitor feature generation during registration
python manager.py datasets --register --dataset-name test-migration --auto

# Check feature catalog
python manager.py features --catalog

# Validate MCTS integration
python mcts.py --config config/mcts_config_new_pipeline.yaml
```

## 🔗 External System Integration

### Kaggle Competition Workflow

```bash
# Complete competition workflow
# 1. Register competition dataset
python manager.py datasets --register \
  --dataset-name kaggle-s5e6-2025 \
  --dataset-path /data/playground-series-s5e6/ \
  --auto

# 2. Run MCTS feature discovery
python mcts.py --config config/mcts_config_s5e6_production.yaml

# 3. Analyze best features
python manager.py features --top 20 --dataset kaggle-s5e6-2025

# 4. Export features for submission
python manager.py features --export csv --dataset kaggle-s5e6-2025
```

### AutoGluon Integration

```python
# Direct integration with AutoGluon training
from src.autogluon_evaluator import AutoGluonEvaluator
from src.feature_space import FeatureSpace

# Load features for AutoGluon
evaluator = AutoGluonEvaluator(config)
feature_space = FeatureSpace(config)

# Get features for specific MCTS node
selected_columns = feature_space.get_feature_columns_for_node(mcts_node)
train_df = feature_space.load_features_for_evaluation(selected_columns)

# Train AutoGluon model
score = evaluator.evaluate_features(train_df, target_column='Fertilizer_Name')
```

---

*For development guides, see [FEATURES_DEVELOPMENT.md](FEATURES_DEVELOPMENT.md)*  
*For performance optimization, see [FEATURES_PERFORMANCE.md](FEATURES_PERFORMANCE.md)*  
*For operation details, see [FEATURES_OPERATIONS.md](FEATURES_OPERATIONS.md)*