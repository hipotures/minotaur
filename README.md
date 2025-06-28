# Minotaur - MCTS-Driven Feature Discovery System

Advanced automated feature engineering system using Monte Carlo Tree Search (MCTS) with enterprise-grade dataset management for agricultural fertilizer prediction.

> **Primary Focus**: Kaggle competition "Predicting Optimal Fertilizers" (Playground Series S5E6)  
> **Data Source**: Read-only access to `/mnt/ml/competitions/2025/playground-series-s5e6/`  
> **Architecture**: Modern DuckDB-based system with repository pattern and connection pooling

## 🚀 Quick Start

### Environment Setup
```bash
# Initialize UV environment (Python 3.12)
uv init
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies with UV
uv add -r requirements.txt
```

### First Run - Database Initialization
```bash
# Create database and initialize schema (required first step)
python run_feature_discovery.py --config config/mcts_config.yaml --new-session

# This will fail with dataset registration error, but creates the database
# Then list available datasets:
python scripts/duckdb_manager.py datasets --list
```

### Dataset Management
```bash
# Register the S5E6 dataset (required for system operation)
python scripts/duckdb_manager.py datasets --register \
  --dataset-name playground-series-s5e6-2025 \
  --auto \
  --dataset-path /mnt/ml/competitions/2025/playground-series-s5e6/

# Verify dataset registration
python scripts/duckdb_manager.py datasets --show playground-series-s5e6-2025

# List all registered datasets
python scripts/duckdb_manager.py datasets --list
```

### MCTS Feature Discovery
```bash
# Fast testing with S5E6 dataset (2-5 minutes)
python run_feature_discovery.py --config config/mcts_config_s5e6_fast_real.yaml

# Ultra-fast testing with small sample (30 seconds)
python run_feature_discovery.py --config config/mcts_config_s5e6_fast_test.yaml

# Production feature discovery (hours)
python run_feature_discovery.py --config config/mcts_config_s5e6_production.yaml

# Titanic domain testing (30-60 seconds)
python run_feature_discovery.py --config config/mcts_config_titanic_test.yaml
```

### Session Management
```bash
# List recent sessions
python run_feature_discovery.py --list-sessions

# Resume last session
python run_feature_discovery.py --resume

# Resume specific session
python run_feature_discovery.py --resume SESSION_ID
```

## 🏗️ Architecture

### Modern Database System
- **Repository Pattern**: Clean separation between business logic and data access
- **DuckDB-Only**: High-performance analytical database (no SQLite fallbacks)
- **Connection Pooling**: Thread-safe connection management with automatic health checking
- **Migration System**: Version-controlled schema updates with rollback support
- **Type Safety**: Pydantic models for all database operations

### MCTS-Driven Feature Discovery
- **MCTSEngine**: Monte Carlo Tree Search with UCB1 selection algorithm
- **FeatureSpace**: Domain-agnostic feature operations framework
- **DomainModules**: Domain-specific feature operations (generic + fertilizer_s5e6 + titanic)
- **AutoGluonEvaluator**: Fast ML model evaluation using AutoGluon TabularPredictor
- **FeatureCacheManager**: MD5-based caching system for efficient feature storage

### Dataset Management System
- **Centralized Registration**: Register datasets once, use everywhere
- **Automatic Validation**: Hash-based integrity checking and metadata tracking
- **Secure Access**: Controlled dataset access patterns with security layer
- **Usage Tracking**: Monitor dataset utilization and performance
- **Legacy Support**: Backward compatibility with path-based configurations

## 📁 Directory Structure

```
minotaur/
├── 📁 config/           # Configuration files
│   ├── mcts_config.yaml                 # Base configuration (DO NOT MODIFY)
│   ├── mcts_config_s5e6_production.yaml # S5E6 production config
│   ├── mcts_config_s5e6_fast_real.yaml  # S5E6 fast real evaluation
│   ├── mcts_config_s5e6_fast_test.yaml  # S5E6 ultra-fast testing
│   └── mcts_config_titanic_test.yaml    # Titanic domain testing
│
├── 📄 run_feature_discovery.py         # **MAIN SCRIPT** - MCTS orchestration
│
├── 📁 src/              # Source code modules
│   ├── db/                               # Database abstraction layer
│   │   ├── core/                         # Connection management and base repositories
│   │   ├── models/                       # Pydantic data models
│   │   ├── repositories/                 # Data access layer
│   │   ├── migrations/                   # SQL migration files
│   │   └── config/                       # Database configuration
│   ├── domains/                          # Domain-specific feature operations
│   │   ├── generic.py                    # Universal statistical operations
│   │   ├── fertilizer_s5e6.py            # Agricultural NPK, stress indicators
│   │   └── titanic.py                    # Maritime features for testing
│   ├── discovery_db.py                   # Main database interface (refactored)
│   ├── db_service.py                     # High-level service orchestrating repositories
│   ├── dataset_manager.py                # Dataset lifecycle management
│   ├── session_output_manager.py         # Session-based file organization
│   ├── config_manager.py                 # Centralized configuration handling
│   ├── security.py                       # Dataset access security layer
│   ├── mcts_engine.py                    # Monte Carlo Tree Search implementation
│   ├── autogluon_evaluator.py            # ML model evaluation wrapper
│   ├── feature_space.py                  # Feature operation space management
│   └── analytics.py                      # Performance analytics and reporting
│
├── 📁 scripts/          # Database management utilities
│   ├── duckdb_manager.py                 # Modular DuckDB manager
│   ├── modules/                          # Management modules
│   │   ├── datasets.py                   # Dataset registration and management
│   │   ├── sessions.py                   # Session analysis and management
│   │   ├── features.py                   # Feature catalog analysis
│   │   ├── analytics.py                  # Analytics and reporting
│   │   ├── backup.py                     # Database backup system
│   │   ├── selfcheck.py                  # System validation
│   │   └── verification.py               # Data integrity verification
│   └── examples/                         # Traditional ML models (001-009)
│
├── 📁 tests/            # Comprehensive test suite
│   ├── unit/                             # Unit tests
│   ├── integration/                      # Integration tests
│   └── conftest.py                       # Pytest configuration
│
├── 📁 data/             # Database storage
│   ├── minotaur.duckdb                   # Main system database
│   ├── datasets/                         # Dataset-specific databases
│   └── backups/                          # Database backups
│
├── 📁 outputs/          # Generated outputs and reports
│   ├── sessions/                         # Session-based output directories
│   ├── reports/                          # Analytics reports
│   └── exports/                          # Exported features and models
│
├── 📁 .github/          # CI/CD configuration
│   └── workflows/                        # GitHub Actions
│
├── 📄 requirements.txt                   # Python dependencies
├── 📄 requirements-test.txt              # Testing dependencies
└── 📄 CLAUDE.md                         # Development guidelines
```

## 🔧 Configuration System

### Base + Override Architecture
The system uses a **base + override** configuration system:

- **Base**: `config/mcts_config.yaml` (DO NOT MODIFY)
- **Overrides**: Domain-specific configs that extend the base

### Available Configurations

| Profile | Config File | Purpose | Target Time |
|---------|-------------|---------|-------------|
| **Development** | `--test-mode` | Mock evaluator for debugging | 30 seconds |
| **Fast Test** | `mcts_config_s5e6_fast_test.yaml` | Real AutoGluon, 100 samples | 2-5 minutes |
| **Fast Real** | `mcts_config_s5e6_fast_real.yaml` | Real AutoGluon, 5% data | 5-10 minutes |
| **Production** | `mcts_config_s5e6_production.yaml` | Full S5E6 feature discovery | Hours |
| **Titanic Test** | `mcts_config_titanic_test.yaml` | Titanic domain validation | 30-60 seconds |

### Modern Dataset Configuration
```yaml
# New centralized system (preferred)
autogluon:
  dataset_name: 'playground-series-s5e6-2025'    # Registered dataset name
  target_metric: 'MAP@3'

# Legacy system (still supported for backward compatibility)
autogluon:
  train_path: "/mnt/ml/competitions/2025/playground-series-s5e6/train.csv"
  test_path: "/mnt/ml/competitions/2025/playground-series-s5e6/test.csv"
```

## 🗄️ Database Management

### DuckDB Manager - Modular System
```bash
# Show available modules and commands
python scripts/duckdb_manager.py

# Dataset management
python scripts/duckdb_manager.py datasets --list
python scripts/duckdb_manager.py datasets --register --help
python scripts/duckdb_manager.py datasets --show DATASET_NAME

# Session analysis
python scripts/duckdb_manager.py sessions --list
python scripts/duckdb_manager.py sessions --compare session1 session2

# Feature analysis
python scripts/duckdb_manager.py features --top 10
python scripts/duckdb_manager.py features --search npk

# System maintenance
python scripts/duckdb_manager.py backup --create
python scripts/duckdb_manager.py selfcheck --run
python scripts/duckdb_manager.py verification --verify-latest
```

### Database Architecture
```
data/
├── minotaur.duckdb                       # Main system database
├── datasets/
│   ├── playground-series-s5e6-2025/
│   │   └── dataset.duckdb                # S5E6 dataset cache
│   └── titanic/
│       └── dataset.duckdb                # Titanic dataset cache
└── backups/                              # Automated backups
```

## 📊 Performance Benchmarks

### MCTS Feature Discovery
- **Mock Mode**: 10 iterations in <30 seconds (development)
- **Small Dataset**: 3-5 iterations in 2-5 minutes (validation)
- **Production**: 100+ iterations over hours (full exploration)
- **Best Score Achieved**: 0.33453 MAP@3 (MCTS-discovered features)

### Data Loading Optimization
| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Data Loading | 0.25s (CSV) | 0.06s (Parquet) | **4.2x faster** |
| Memory Usage | 157MB | 0.2MB (optimized) | **785x reduction** |
| Cache Access | N/A | <0.001s | **Instant** |
| Full Pipeline | Hours | 2-5 minutes | **10-50x faster** |

### Traditional ML Models
| Model | MAP@3 Score | Description |
|-------|-------------|-------------|
| 001_baseline | 0.32065 | Baseline LightGBM |
| 003_xgboost | 0.33311 | XGBoost GPU |
| 005_optuna | 0.33453 | XGBoost + Optuna (best traditional) |
| MCTS_discovered | 0.33453+ | MCTS-automated features |

## 🧪 Testing Infrastructure

### Test Suite
```bash
# Run comprehensive test suite
pytest                                    # Full test suite with coverage
pytest tests/unit/                       # Unit tests only
pytest tests/integration/                # Integration tests only
pytest -m "not slow"                     # Skip slow tests
pytest -k "test_mcts"                    # Run specific test patterns

# Generate coverage report
pytest --cov=src --cov-report=html       # HTML coverage report
pytest --cov=src --cov-report=term       # Terminal coverage report
```

### GitHub Actions CI/CD
- Automated testing on push/PR
- Multi-environment testing
- Coverage reporting
- Security scanning

## 🎯 Development Workflow

### MCTS Feature Discovery Workflow
1. **Development**: Start with `--test-mode` for rapid iteration (30s)
2. **Validation**: Use fast configs for model validation (2-5min)
3. **Production**: Full exploration with comprehensive feature discovery (hours)
4. **Analysis**: Review analytics reports for insights and optimization
5. **Integration**: Export best features to existing model pipeline

### Database Development Workflow
1. **Schema Changes**: Create migration files in `src/db/migrations/`
2. **Model Updates**: Modify Pydantic models in `src/db/models/`
3. **Repository Changes**: Update repository methods in `src/db/repositories/`
4. **Service Integration**: Modify `src/db_service.py` for high-level operations
5. **Testing**: Run database tests with `pytest -m duckdb`

### Adding New Domain Features
1. **Create Domain Module**: Add new file in `src/domains/` (e.g., `new_domain.py`)
2. **Implement Operations**: Create feature operation classes with domain-specific methods
3. **Register Operations**: Add to `FeatureSpace` configuration
4. **Test Integration**: Create test configuration file in `config/`
5. **Validate Results**: Run with fast config first, then real evaluation

## 🔒 Security Features

- **Dataset Access Control**: Secure validation and access patterns
- **Path Sanitization**: Protection against directory traversal
- **Input Validation**: Comprehensive parameter validation
- **Error Handling**: Secure error messages without information leakage

## 📈 Generated Analytics

### Session Output Structure
```
outputs/sessions/session_YYYYMMDD_HHMMSS/
├── metadata/                             # Session configuration and metadata
├── logs/                                 # Session-specific logs
├── exports/                              # Generated code and data exports
├── reports/                              # Analytics reports and visualizations
└── summary.md                            # Session summary report
```

### Analytics Reports
- **Interactive Dashboard**: HTML reports with performance charts
- **Score Progression**: Real-time MCTS exploration progress
- **Operation Analysis**: Feature operation effectiveness ranking
- **Timing Analysis**: Performance profiling and bottleneck identification
- **Feature Impact**: Statistical significance of discovered features

## 🚨 Important Notes

### Configuration Management
- **Never modify** `config/mcts_config.yaml` - it's the base configuration
- **Create custom configs** that inherit from the base
- **Use dataset_name** instead of direct paths for modern dataset management
- **Validate configs** before running: `--validate-config`

### Dataset Requirements
- **Register datasets first** using `scripts/duckdb_manager.py datasets --register`
- **Verify registration** before running MCTS discovery
- **Use centralized dataset names** instead of hardcoded paths
- **Monitor dataset integrity** with verification modules

### Performance Optimization
- **Start with fast configs** for development and validation
- **Use connection pooling** - database connections are managed automatically
- **Monitor resource usage** during long-running sessions
- **Leverage session management** - resume interrupted sessions seamlessly

## 🤝 Contributing

1. **Test Changes**: Always run `pytest` before committing
2. **Follow Patterns**: Use repository pattern for database operations
3. **Document Features**: Update CLAUDE.md for new capabilities
4. **Security First**: Validate all inputs and secure data access
5. **Performance Aware**: Consider impact on MCTS exploration speed

The system is designed for competitive ML scenarios requiring rapid prototyping, automated feature discovery, and enterprise-grade data management with full auditability and reproducibility.