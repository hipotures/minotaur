# Minotaur - MCTS-Driven Feature Discovery System

Advanced automated feature engineering system using Monte Carlo Tree Search (MCTS) with enterprise-grade dataset management for agricultural fertilizer prediction.

> **Primary Focus**: Kaggle competition "Predicting Optimal Fertilizers" (Playground Series S5E6)  
> **Architecture**: Modern DuckDB-based system with repository pattern and connection pooling

## 📚 Documentation

### 🗂️ Complete System Documentation

This system has comprehensive documentation organized by component:

#### **🎯 MCTS System Documentation** (`docs/mcts/`)
- **[MCTS_OVERVIEW.md](docs/mcts/MCTS_OVERVIEW.md)** - Executive summary and quick start guide
- **[MCTS_IMPLEMENTATION.md](docs/mcts/MCTS_IMPLEMENTATION.md)** - Technical implementation details
- **[MCTS_DATA_FLOW.md](docs/mcts/MCTS_DATA_FLOW.md)** - Phase-by-phase data flow
- **[MCTS_VALIDATION.md](docs/mcts/MCTS_VALIDATION.md)** - Validation framework and testing
- **[MCTS_OPERATIONS.md](docs/mcts/MCTS_OPERATIONS.md)** - Configuration and troubleshooting

#### **⚙️ Feature Engineering Documentation** (`docs/features/`)
- **[FEATURES_OVERVIEW.md](docs/features/FEATURES_OVERVIEW.md)** - System architecture and quick start
- **[FEATURES_OPERATIONS.md](docs/features/FEATURES_OPERATIONS.md)** - Complete operations catalog
- **[FEATURES_INTEGRATION.md](docs/features/FEATURES_INTEGRATION.md)** - Pipeline and dataset integration
- **[FEATURES_DEVELOPMENT.md](docs/features/FEATURES_DEVELOPMENT.md)** - Custom development guide
- **[FEATURES_PERFORMANCE.md](docs/features/FEATURES_PERFORMANCE.md)** - Optimization and troubleshooting

#### **🛠️ System Configuration**
- **[AUTOGLUON_CONFIG_GUIDE.md](docs/AUTOGLUON_CONFIG_GUIDE.md)** - AutoGluon configuration reference
- **[SYSTEM_COMPLETION_SUMMARY.md](docs/SYSTEM_COMPLETION_SUMMARY.md)** - System implementation status
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines and project instructions

### 📖 Quick Navigation

**New Users**: Start with [MCTS_OVERVIEW.md](docs/mcts/MCTS_OVERVIEW.md) and [FEATURES_OVERVIEW.md](docs/features/FEATURES_OVERVIEW.md)

**Developers**: See implementation details in [MCTS_IMPLEMENTATION.md](docs/mcts/MCTS_IMPLEMENTATION.md) and [FEATURES_DEVELOPMENT.md](docs/features/FEATURES_DEVELOPMENT.md)

**Data Scientists**: Check operation catalogs in [FEATURES_OPERATIONS.md](docs/features/FEATURES_OPERATIONS.md)

**DevOps**: Review performance guides in [FEATURES_PERFORMANCE.md](docs/features/FEATURES_PERFORMANCE.md) and [MCTS_OPERATIONS.md](docs/mcts/MCTS_OPERATIONS.md)

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
python mcts.py --config config/mcts_config.yaml --new-session

# This will fail with dataset registration error, but creates the database
# Then list available datasets:
python manager.py datasets --list
```

### Dataset Management
```bash
# Register datasets (required for system operation)
python manager.py datasets --register \
  --dataset-name playground-series-s5e6-2025 \
  --auto \
  --dataset-path /path/to/your/dataset/

# Verify dataset registration
python manager.py datasets --show playground-series-s5e6-2025

# List all registered datasets
python manager.py datasets --list
```

### MCTS Feature Discovery
```bash
# Fast testing with S5E6 dataset (2-5 minutes)
python mcts.py --config config/mcts_config_s5e6_fast_real.yaml

# Ultra-fast testing with small sample (30 seconds)
python mcts.py --config config/mcts_config_s5e6_fast_test.yaml

# Production feature discovery (hours)
python mcts.py --config config/mcts_config_s5e6_production.yaml

# Titanic domain testing (30-60 seconds)
python mcts.py --config config/mcts_config_titanic_test.yaml
```

### Session Management
```bash
# List recent sessions
python mcts.py --list-sessions

# Resume last session
python mcts.py --resume

# Resume specific session
python mcts.py --resume SESSION_ID
```

## 🏗️ Architecture

### **🎯 MCTS-Driven Feature Discovery System**
- **Monte Carlo Tree Search**: UCB1 selection algorithm with configurable exploration
- **Automated Feature Engineering**: 100+ domain operations with signal detection
- **AutoGluon Integration**: Fast ML evaluation with MAP@3 optimization
- **Performance**: 50% speedup with new pipeline, 4.2x faster data loading

*📖 Detailed documentation: [MCTS_OVERVIEW.md](docs/mcts/MCTS_OVERVIEW.md)*

### **⚙️ Feature Engineering Framework**
- **Modular Architecture**: Generic + Custom domain operations
- **Signal Detection**: Automatic filtering of low-signal features (50% performance boost)
- **7 Generic Operations**: Statistical, polynomial, binning, ranking, temporal, text, categorical
- **2 Custom Domains**: Fertilizer S5E6 (agricultural), Titanic (maritime)
- **New Pipeline**: Lazy loading, memory optimization, parallel processing

*📖 Detailed documentation: [FEATURES_OVERVIEW.md](docs/features/FEATURES_OVERVIEW.md)*

### **🗄️ Enterprise Database System**
- **Repository Pattern**: Clean separation between business logic and data access
- **DuckDB-Only**: High-performance analytical database (no SQLite fallbacks)
- **Connection Pooling**: Thread-safe connection management with automatic health checking
- **Migration System**: Version-controlled schema updates with rollback support
- **Type Safety**: Pydantic models for all database operations

### **📊 Dataset Management System**
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
├── 📄 mcts.py                          # **MAIN SCRIPT** - MCTS orchestration
│
├── 📁 src/              # Source code modules
│   ├── db/                               # Database abstraction layer
│   │   ├── core/                         # Connection management and base repositories
│   │   ├── models/                       # Pydantic data models
│   │   ├── repositories/                 # Data access layer
│   │   ├── migrations/                   # SQL migration files
│   │   └── config/                       # Database configuration
│   ├── features/                         # Modern feature engineering system
│   │   ├── base.py                       # Abstract base classes and timing
│   │   ├── generic/                      # Domain-agnostic operations
│   │   │   ├── statistical.py            # Statistical aggregations
│   │   │   ├── polynomial.py             # Polynomial features
│   │   │   ├── binning.py                # Quantile binning
│   │   │   ├── ranking.py                # Rank transformations
│   │   │   ├── temporal.py               # Time-based features
│   │   │   ├── text.py                   # Text processing features
│   │   │   └── categorical.py            # Categorical encoding
│   │   └── custom/                       # Domain-specific operations
│   │       ├── kaggle_s5e6.py            # Fertilizer competition features
│   │       └── titanic.py                # Titanic dataset features
│   ├── domains/                          # Legacy domain system (backward compatibility)
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
├── 📁 docs/             # Comprehensive documentation
│   ├── mcts/                             # MCTS system documentation
│   │   ├── MCTS_OVERVIEW.md              # Executive summary and quick start
│   │   ├── MCTS_IMPLEMENTATION.md        # Technical implementation details
│   │   ├── MCTS_DATA_FLOW.md             # Phase-by-phase data flow
│   │   ├── MCTS_VALIDATION.md            # Validation framework and testing
│   │   └── MCTS_OPERATIONS.md            # Configuration and troubleshooting
│   ├── features/                         # Feature engineering documentation
│   │   ├── FEATURES_OVERVIEW.md          # System architecture and quick start
│   │   ├── FEATURES_OPERATIONS.md        # Complete operations catalog
│   │   ├── FEATURES_INTEGRATION.md       # Pipeline and dataset integration
│   │   ├── FEATURES_DEVELOPMENT.md       # Custom development guide
│   │   └── FEATURES_PERFORMANCE.md       # Optimization and troubleshooting
│   ├── AUTOGLUON_CONFIG_GUIDE.md         # AutoGluon configuration reference
│   └── SYSTEM_COMPLETION_SUMMARY.md      # System implementation status
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
  train_path: "/path/to/train.csv"
  test_path: "/path/to/test.csv"
```

## 🗄️ Database Management

### DuckDB Manager - Modular System
```bash
# Show available modules and commands
python manager.py

# Dataset management
python manager.py datasets --list
python manager.py datasets --register --help
python manager.py datasets --show DATASET_NAME

# Session analysis
python manager.py sessions --list
python manager.py sessions --compare session1 session2

# Feature analysis
python manager.py features --top 10
python manager.py features --search npk

# System maintenance
python manager.py backup --create
python manager.py selfcheck --run
python manager.py verification --verify-latest
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

### Adding New Feature Operations

#### **Generic Operations** (Domain-agnostic)
1. **Create Module**: Add new file in `src/features/generic/` (e.g., `graph_features.py`)
2. **Implement Class**: Extend `GenericFeatureOperation` with timing support
3. **Register Operation**: Add to `__init__.py` and configuration
4. **Test Integration**: Create unit tests and validate performance

*📖 Detailed guide: [FEATURES_DEVELOPMENT.md](docs/features/FEATURES_DEVELOPMENT.md)*

#### **Custom Domain Operations** (Problem-specific)
1. **Create Domain**: Add new file in `src/features/custom/` (e.g., `ecommerce.py`)
2. **Implement Operations**: Extend `BaseDomainFeatures` with domain knowledge
3. **Add Auto-Detection**: Configure domain detection rules
4. **Validate Results**: Test with representative datasets

*📖 Detailed guide: [FEATURES_DEVELOPMENT.md](docs/features/FEATURES_DEVELOPMENT.md)*

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
- **Register datasets first** using `manager.py datasets --register`
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

## 📋 System Status

### **Current Capabilities**
- ✅ **MCTS Feature Discovery**: Production-ready with 0.33453 MAP@3 score
- ✅ **Feature Engineering**: 100+ operations across 7 generic + 2 custom domains
- ✅ **Database System**: Enterprise-grade DuckDB with repository pattern
- ✅ **Performance Optimization**: 50% speedup with signal detection, 4.2x faster loading
- ✅ **Comprehensive Documentation**: 5-document suites for MCTS and Features
- ✅ **Testing Infrastructure**: Unit + integration tests with CI/CD

### **Recent Major Updates** (2025-06-30)
- 🆕 **New Feature Pipeline**: Modular `src/features/` system with lazy loading
- 🆕 **Signal Detection**: Automatic filtering of low-signal features
- 🆕 **Comprehensive Documentation**: Complete documentation suites for all systems
- 🆕 **Performance Monitoring**: Real-time timing and memory usage tracking
- 🆕 **Enhanced Testing**: Validation framework with timing and signal detection

*📖 Complete status: [SYSTEM_COMPLETION_SUMMARY.md](docs/SYSTEM_COMPLETION_SUMMARY.md)*

---

The system is designed for competitive ML scenarios requiring rapid prototyping, automated feature discovery, and enterprise-grade data management with full auditability and reproducibility.