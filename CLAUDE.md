# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview - Minotaur

This is an **independent MCTS-driven feature discovery system** for agricultural fertilizer prediction:

**Primary Focus**: Automated feature engineering using Monte Carlo Tree Search (MCTS) with AutoGluon evaluation for Kaggle competition "Predicting Optimal Fertilizers" (Playground Series S5E6).

**Data Source**: Read-only access to `/mnt/ml/competitions/2025/playground-series-s5e6/` with automatic local parquet caching for 4.2x faster loading.

**Database Architecture**: Advanced DuckDB-based system with repository pattern, connection pooling, and automatic migrations.

## High-Level Architecture

### Core Design Patterns
- **Layered Architecture**: CLI → Service Layer → Repository Layer → Database Layer
- **Repository Pattern**: Clean separation between data access and business logic
- **Service Pattern**: Complex operation orchestration across repositories
- **Strategy Pattern**: Pluggable feature operations and evaluation strategies
- **Connection Pooling**: Thread-safe DuckDB connection management
- **Lazy Loading**: On-demand feature generation with caching

### Key Components and Their Interactions

1. **MCTS Engine** (`src/mcts_engine.py`)
   - Implements UCB1 selection algorithm for node exploration
   - Tree expansion with configurable depth limits
   - Backpropagation of rewards through the tree
   - Memory management with automatic node pruning

2. **Feature Space** (`src/feature_space.py`)
   - Manages available feature operations
   - Integrates with `src/features/` modular system
   - Lazy feature generation with caching
   - Validates features to prevent data leakage

3. **AutoGluon Evaluator** (`src/autogluon_evaluator.py`)
   - Fast ML model evaluation using TabularPredictor
   - MAP@3 metric calculation for multi-class problems
   - Integrates with DuckDB for efficient data sampling

4. **Database System**
   - **DuckDB**: Primary analytical database (no SQLite)
   - **Connection Manager** (`src/db/core/connection.py`): Connection pooling
   - **Migration System**: Version-controlled schema updates
   - **Repositories**: SessionRepository, ExplorationRepository, FeatureRepository, DatasetRepository

5. **Feature Engineering Architecture**
   ```
   src/features/
   ├── base.py                    # Abstract base classes and timing mixin
   ├── generic/                   # Domain-agnostic operations
   │   ├── statistical.py         # Statistical aggregations
   │   ├── polynomial.py          # Polynomial features
   │   ├── binning.py            # Quantile binning
   │   └── ranking.py            # Rank transformations
   └── custom/                    # Domain-specific operations
       ├── titanic.py            # Titanic dataset features
       └── kaggle_s5e6.py        # Fertilizer competition features
   ```

### Data Flow
1. **Dataset Registration**: CSV/Parquet → DuckDB cache → Feature generation
2. **MCTS Loop**: Selection (UCB1) → Expansion → Simulation (AutoGluon) → Backpropagation
3. **Feature Pipeline**: Load from cache → Apply operations → Validate → Store results
4. **Evaluation**: Prepare features → Train AutoGluon → Calculate MAP@3 → Update tree

## Key Commands

### Dataset Management (Required First Step)
```bash
# Register a new dataset with auto-detection
python manager.py datasets --register \
  --dataset-name playground-series-s5e6-2025 \
  --auto \
  --dataset-path /mnt/ml/competitions/2025/playground-series-s5e6/

# List all registered datasets
python manager.py datasets --list

# Show dataset details
python manager.py datasets --show playground-series-s5e6-2025

# Remove a dataset
python manager.py datasets --remove playground-series-s5e6-2025
```

### MCTS Feature Discovery Operations
```bash
# Ultra-fast testing with 100 samples (30 seconds) - uses mock evaluator
python mcts.py --config config/mcts_config_s5e6_fast_test.yaml

# Fast real evaluation with 5% data (2-5 minutes)
python mcts.py --config config/mcts_config_s5e6_fast_real.yaml

# Production feature discovery with 80% data (hours)
python mcts.py --config config/mcts_config_s5e6_production.yaml

# Titanic domain testing (30-60 seconds)
python mcts.py --config config/mcts_config_titanic_test.yaml

# Session management
python mcts.py --list-sessions
python mcts.py --list-sessions 10              # Show last 10 sessions
python mcts.py --resume                         # Resume last session
python mcts.py --resume SESSION_ID              # Resume specific session

# Configuration validation
python mcts.py --config config/mcts_config_s5e6_fast_real.yaml --validate-config
```

### Database Management
```bash
# Show all available management modules
python manager.py

# Session analysis
python manager.py sessions --list
python manager.py sessions --details SESSION_ID
python manager.py sessions --compare session1 session2
python manager.py sessions --export SESSION_ID --format json

# Feature analysis
python manager.py features --top 10
python manager.py features --search npk
python manager.py features --category agricultural_domain
python manager.py features --performance

# Analytics and reporting
python manager.py analytics --summary
python manager.py analytics --best-features
python manager.py analytics --operation-stats

# System maintenance
python manager.py backup --create
python manager.py backup --list
python manager.py backup --restore backup_name

# System validation
python manager.py selfcheck --run
python manager.py selfcheck --fix

# Data verification
python manager.py verification --verify-latest
python manager.py verification --check-integrity
```

### Environment Setup and Dependencies
```bash
# Initialize UV environment (Python 3.12)
uv init
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies with UV
uv add -r requirements.txt

# Run example models (generates Kaggle submissions)
python scripts/examples/001_fertilizer_prediction_gpu.py      # Baseline LightGBM (MAP@3: 0.32065)
python scripts/examples/003_fertilizer_prediction_xgboost_gpu.py # XGBoost GPU (MAP@3: 0.33311)
python scripts/examples/005_fertilizer_prediction_xgboost_optuna.py # Best performing (MAP@3: 0.33453)
python scripts/examples/009_fertilizer_prediction_autogluon_full_features.py # Latest AutoGluon
```

### Testing and Quality Assurance
```bash
# Run comprehensive test suite
pytest                                    # Full test suite with coverage
pytest tests/unit/                       # Unit tests only
pytest tests/integration/                # Integration tests only
pytest -m "not slow"                     # Skip slow tests
pytest -k "test_mcts"                    # Run specific test patterns
pytest -k "test_feature" -v               # Verbose output for feature tests

# Generate coverage report
pytest --cov=src --cov-report=html       # HTML coverage report in htmlcov/
pytest --cov=src --cov-report=term       # Terminal coverage report

# Test specific components
pytest tests/unit/test_mcts_engine.py    # Test MCTS implementation
pytest tests/unit/test_feature_space.py  # Test feature operations
pytest tests/unit/test_db_service.py     # Test database layer

# Syntax checking
python -m py_compile src/discovery_db.py
python -m py_compile src/db_service.py
python -m py_compile src/features/base.py
```

### Direct DuckDB Access
```bash
# Access the database directly with DuckDB client
bin/duckdb data/minotaur.duckdb

# Common queries
.tables                                   # List all tables
.schema exploration_history               # Show table schema
SELECT COUNT(*) FROM sessions;            # Count sessions
SELECT * FROM datasets WHERE active = true; # List active datasets
```

## Development Workflows

### Adding New Feature Operations

1. **For Generic Features** (domain-agnostic):
   ```python
   # Create new file in src/features/generic/
   # Example: src/features/generic/my_operations.py
   from ..base import AbstractFeatureOperation, FeatureTimingMixin
   
   class MyFeatureOperation(AbstractFeatureOperation, FeatureTimingMixin):
       def generate_features(self, df, **kwargs):
           features = {}
           with self._time_feature('my_feature'):
               features['my_feature'] = df['col1'] * df['col2']
           return features
   ```

2. **For Domain-Specific Features**:
   ```python
   # Create new file in src/features/custom/
   # Example: src/features/custom/my_domain.py
   from ..base import BaseDomainFeatures
   
   class CustomFeatureOperations(BaseDomainFeatures):
       def __init__(self):
           super().__init__('my_domain')
           
       def get_domain_features(self, df, **kwargs):
           # Implement domain-specific logic
           pass
   ```

3. **Register in Feature Space**:
   - Update configuration to include new feature categories
   - Feature operations are automatically discovered via module imports

### Working with the Database

1. **Creating New Tables**: Add migration file to `src/db/migrations/`
2. **Adding New Repository**: Create in `src/db/repositories/`
3. **Service Layer Changes**: Update `src/db_service.py`
4. **Model Changes**: Update Pydantic models in `src/db/models/`

### Session Development Workflow

1. **Start with fast config** for rapid iteration
2. **Monitor progress** with `python manager.py sessions --details`
3. **Analyze features** with `python manager.py features --top`
4. **Export best features** when satisfied with results

## Configuration System

### Base + Override Pattern
- **Never modify** `config/mcts_config.yaml` (base configuration)
- Create override configs that inherit from base
- Override configs only specify changed values

### Key Configuration Sections
```yaml
# Dataset configuration (modern approach)
autogluon:
  dataset_name: 'playground-series-s5e6-2025'  # Use registered dataset
  target_metric: 'MAP@3'                       # Competition metric

# MCTS parameters
mcts:
  exploration_weight: 1.4                      # UCB1 exploration factor
  max_tree_depth: 8                           # Maximum feature operation depth
  
# Feature space control
feature_space:
  max_features_per_node: 300                  # Limit features per node
  enabled_categories:                         # Active feature types
    - 'statistical_aggregations'
    - 'agricultural_domain'
```

## Common Issues and Solutions

### Dataset Registration Fails
- Ensure the dataset path contains train.csv/train.parquet
- Check that target column is correctly detected
- Use `--force-update` to re-register existing datasets

### Out of Memory During Feature Generation
- Reduce `max_features_per_node` in config
- Use smaller data samples with `train_size` parameter
- Enable feature pruning with `min_improvement_threshold`

### Slow AutoGluon Evaluation
- Reduce `time_limit` in autogluon config
- Use fewer models in `included_model_types`
- Enable GPU with `enable_gpu: true`

### Session Resume Fails
- Check session exists with `--list-sessions`
- Ensure database hasn't been corrupted
- Use backup restoration if needed

## Performance Optimization Tips

1. **Use DuckDB's native operations** when possible
2. **Enable feature caching** for repeated operations
3. **Monitor memory usage** with system tools
4. **Use connection pooling** (automatic in new architecture)
5. **Profile slow operations** with timing logs at DEBUG level

## Logging and Debugging

### Log Levels
- **DEBUG**: Individual feature timing, detailed operations
- **INFO**: Class-level summaries, major operations
- **WARNING**: Potential issues, deprecated features
- **ERROR**: Failures requiring attention

### Enable DEBUG Logging
```yaml
# In config file
logging:
  level: 'DEBUG'
```

### View Timing Information
- Individual features logged at DEBUG level
- Class summaries logged at INFO level
- Full timing report in session outputs

## Memories

### DuckDB Interaction
- Plik klient duckdb możesz tak wywołać bin/duckdb ma prawa wykonywania

### Feature Engineering Refactoring (2025-06-29)
- Migrated from `src/domains` to `src/features` with modular structure
- Added timing capabilities with DEBUG/INFO level logging
- Implemented no-signal feature detection (filters features with nunique() <= 1)
- Created backward compatibility wrappers for smooth migration