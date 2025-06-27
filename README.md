# Minotaur - Fertilizer Prediction Models

Independent ML automation system for fertilizer recommendation with MCTS-driven feature discovery.

> **Data Source**: Read-only access to `/mnt/ml/competitions/2025/playground-series-s5e6/`  
> **Local Cache**: Automatic parquet conversion in `data/` for faster loading

## ⚠️ UWAGA: config/mcts_config.yaml
**NIE MODYFIKUJ** głównego pliku `config/mcts_config.yaml`!
Twórz własne konfiguracje dziedziczące z tego pliku.

## 📁 Directory Structure

```
minotaur/
├── 📁 config/           # Configuration files
│   ├── mcts_config.yaml                 # Main MCTS configuration (base - DO NOT MODIFY)
│   ├── mcts_config_s5e6_production.yaml # S5E6 production config
│   ├── mcts_config_s5e6_fast_real.yaml  # S5E6 fast real evaluation config
│   └── mcts_config_s5e6_fast_test.yaml  # S5E6 ultra-fast testing config
│
├── 📄 run_feature_discovery.py         # **MAIN SCRIPT** - MCTS feature discovery system
│
├── 📁 src/              # Source code modules
│   ├── __init__.py                     # Main imports
│   ├── domains/                        # Domain-specific features
│   │   ├── generic.py                  # Universal feature operations
│   │   └── fertilizer_s5e6.py          # S5E6 agricultural features
│   ├── feature_cache.py                # MD5-based feature caching
│   ├── data_utils.py                   # Data loading and sampling
│   ├── mcts_engine.py                  # MCTS algorithm
│   ├── autogluon_evaluator.py          # ML model evaluation
│   └── ...                             # Other core modules
│
├── 📁 examples/         # Example ML models (001-009)
│   ├── 001_fertilizer_prediction_gpu.py        # Baseline LightGBM (MAP@3: 0.32065)
│   ├── 002_fertilizer_prediction_optimized.py  # Over-engineered LightGBM (overfitting)
│   ├── 003_fertilizer_prediction_xgboost_gpu.py # XGBoost GPU (MAP@3: 0.33311)
│   ├── 004_fertilizer_prediction_catboost_gpu.py # CatBoost GPU (MAP@3: 0.32477)
│   ├── 005_fertilizer_prediction_xgboost_optuna.py # XGBoost + Optuna (MAP@3: 0.33453)
│   ├── 006_fertilizer_prediction_tabm.py        # TabM neural network
│   ├── 007_fertilizer_prediction_tabm_optimized.py
│   ├── 008_fertilizer_prediction_autogluon.py   # AutoGluon models
│   └── 009_fertilizer_prediction_autogluon_full_features.py
│
├── 📁 src/                     # Source code modules
│   ├── __init__.py
│   ├── mcts_engine.py           # Monte Carlo Tree Search engine
│   ├── autogluon_evaluator.py   # AutoGluon evaluation wrapper
│   ├── feature_space.py         # Feature operation space
│   ├── discovery_db.py          # SQLite database logging
│   ├── analytics.py             # Performance analytics
│   ├── timing.py                # Timing infrastructure
│   ├── data_utils.py            # Data loading and optimization
│   ├── mock_evaluator.py        # Mock evaluator for testing
│   └── synthetic_data.py        # Synthetic data generation
│
├── 📁 data/             # Data files and databases
│   ├── feature_discovery.db             # Main MCTS database
│   ├── feature_discovery_fast.db        # Fast evaluation database
│   ├── feature_discovery_test.db        # Testing database
│   ├── backups/                          # Database backups
│   └── *.csv                            # Generated datasets
│
├── 📁 outputs/          # Generated outputs and reports
│   ├── best_features_discovered.py      # Generated feature code
│   ├── best_features_fast.py           # Fast mode features
│   ├── discovery_report.html           # HTML reports
│   ├── reports/                         # Analytics reports
│   └── reports_fast/                    # Fast mode reports
│
├── 📁 logs/             # Log files and timing data
│   ├── mcts_discovery.log               # Main system logs
│   ├── mcts_discovery_fast.log          # Fast evaluation logs
│   ├── timing_data_*.json               # Timing statistics
│   └── archives/                        # Archived logs
│
├── 📁 docs/             # Documentation
│   ├── CONFIG_COMPARISON.md             # Configuration comparison guide
│   ├── MCTS_SYSTEM_GUIDE.md            # Comprehensive system guide
│   ├── AUTOGLUON_CONFIG_GUIDE.md       # AutoGluon configuration
│   └── SYSTEM_COMPLETION_SUMMARY.md    # Implementation summary
│
├── 📁 models/           # Trained model artifacts
│   ├── autogluon_models/               # AutoGluon saved models
│   └── autogluon_models_full/          # Full feature models
│
├── 📁 features/         # Generated feature cache
│   ├── train/                          # Training features
│   └── test/                           # Test features
│
├── feature_engineering.py              # Shared feature engineering module
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

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

### 1. MCTS Feature Discovery

```bash
# Navigate to the main directory
cd ~/DEV/minotaur/

# Ultra-fast testing (30 seconds)
python run_feature_discovery.py --config config/mcts_config_fast_test.yaml --test-mode

# Fast real evaluation (2-5 minutes)
python run_feature_discovery.py --config config/mcts_config_fast_real.yaml --real-autogluon

# Full production discovery (hours)
python run_feature_discovery.py --config config/mcts_config.yaml
```

### 2. Traditional Model Execution

```bash
# Run individual models
python examples/001_fertilizer_prediction_gpu.py      # Baseline
python examples/005_fertilizer_prediction_xgboost_optuna.py  # Best performing
```

### 3. Session Management

```bash
# List recent sessions
python run_feature_discovery.py --list-sessions

# Resume last session
python run_feature_discovery.py --resume

# Resume specific session
python run_feature_discovery.py --resume 67ae9bdd
```

## 🔗 Data Sources

### Read-Only Competition Data
- **Source**: `/mnt/ml/competitions/2025/playground-series-s5e6/`
- **Files**: `train.csv` (750K rows), `test.csv` (187.5K rows)
- **Access**: Read-only mount point

### Local Data Cache
- **Location**: `data/` directory
- **Format**: Parquet files for 4.2x faster loading
- **Auto-generation**: Converts CSV to parquet on first use
- **Cache files**: `train_data.parquet`, `test_data.parquet`, `*_features_*.parquet`

## 📊 Performance Benchmarks

| Model | MAP@3 Score | Description |
|-------|-------------|-------------|
| 001_baseline | 0.32065 | Baseline LightGBM |
| 003_xgboost | 0.33311 | XGBoost GPU |
| 005_optuna | 0.33453 | XGBoost + Optuna (best) |
| 004_catboost | 0.32477 | CatBoost GPU |

## 🔧 Configuration Modes

| Mode | Config File | Performance | Use Case |
|------|-------------|------------|----------|
| **Mock Testing** | `--test-mode` | 30 seconds | Development |
| **Fast Test** | `config/mcts_config_fast_test.yaml` | 30-60 seconds | Quick testing |
| **Fast Real** | `config/mcts_config_fast_real.yaml` | 2-5 minutes | Validation |
| **Production** | `config/mcts_config.yaml` | Hours | Competition |

## 📋 System Features

### MCTS Feature Discovery
- **Monte Carlo Tree Search** with UCB1 selection
- **AutoGluon Integration** for ML evaluation
- **SQLite Database** for complete exploration logging
- **Timing Infrastructure** for performance monitoring
- **Mock Evaluator** for fast development
- **Session Management** with resume capability

### Data Optimization
- **Parquet Support** (4.2x faster loading)
- **Memory Optimization** (11x memory reduction)
- **Smart Sampling** for fast testing
- **Feature Caching** with LRU eviction

### Analytics and Reporting
- **Interactive Dashboard** (HTML reports)
- **Performance Charts** (score progression, timing analysis)
- **Operation Analysis** (feature operation effectiveness)
- **Export Capabilities** (Python code, JSON, HTML)

## 🔍 Key Files

### Configuration Management
- **Base + Override System**: `config/mcts_config.yaml` + override configs
- **Deep Merge**: Automatic merging of configuration parameters
- **Validation**: Built-in configuration validation

### Core Components
- **`run_feature_discovery.py`**: Main MCTS orchestration script
- **`mcts_feature_discovery/`**: Complete MCTS system implementation
- **`scripts/feature_engineering.py`**: Shared feature engineering functions
- **`examples/`**: Traditional ML models (001-009) for comparison

### Generated Outputs
- **`outputs/best_features_discovered.py`**: Discovered feature code
- **`outputs/reports/mcts_analytics_report.html`**: Interactive dashboard
- **`logs/mcts_discovery.log`**: Complete system logs

## 🏗️ Development Workflow

1. **Development**: Start with `--test-mode` (30s iterations)
2. **Validation**: Use `config/mcts_config_fast_real.yaml` (2-5min)
3. **Production**: Full discovery with `config/mcts_config.yaml` (hours)
4. **Analysis**: Review analytics reports for insights
5. **Integration**: Export best features to existing models

## 📈 Evolution Timeline

The system evolved from basic models (15 features) to advanced MCTS-driven discovery (116+ features):

1. **Phase 1**: Basic LightGBM models (001-002)
2. **Phase 2**: Multi-algorithm comparison (003-005)
3. **Phase 3**: Neural networks and AutoGluon (006-009)
4. **Phase 4**: MCTS automation system
5. **Phase 5**: Complete analytics and optimization

## 🎯 Best Practices

- **Start Small**: Always begin with mock mode for development
- **Iterate Fast**: Use fast real mode for feature validation
- **Monitor Resources**: Watch memory and CPU usage during runs
- **Use Session Management**: Resume interrupted sessions seamlessly
- **Analyze Results**: Review analytics reports for optimization insights

The system is designed for competitive ML scenarios requiring rapid prototyping, automated feature discovery, and domain expertise integration.