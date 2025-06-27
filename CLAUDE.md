# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview - Minotaur

This is an **independent MCTS-driven feature discovery system** for agricultural fertilizer prediction:

**Primary Focus**: Automated feature engineering using Monte Carlo Tree Search (MCTS) with AutoGluon evaluation for Kaggle competition "Predicting Optimal Fertilizers" (Playground Series S5E6).

**Data Source**: Read-only access to `/mnt/ml/competitions/2025/playground-series-s5e6/` with automatic local parquet caching for 4.2x faster loading.

## Key Commands

### MCTS Feature Discovery Operations
```bash
# Ultra-fast testing with 100 samples (30 seconds)
python run_feature_discovery.py --config config/mcts_config_s5e6_fast_test.yaml --test-mode

# Fast real evaluation with 5% data (2-5 minutes)
python run_feature_discovery.py --config config/mcts_config_s5e6_fast_real.yaml --real-autogluon

# Production feature discovery with 80% data (hours)
python run_feature_discovery.py --config config/mcts_config_s5e6_production.yaml

# Session management
python run_feature_discovery.py --list-sessions
python run_feature_discovery.py --resume [SESSION_ID]
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
python examples/001_fertilizer_prediction_gpu.py      # Baseline LightGBM (MAP@3: 0.32065)
python examples/003_fertilizer_prediction_xgboost_gpu.py # XGBoost GPU (MAP@3: 0.33311)
python examples/005_fertilizer_prediction_xgboost_optuna.py # Best performing (MAP@3: 0.33453)
python examples/006_fertilizer_prediction_tabm.py # Neural network approach
```

### MCTS Feature Discovery - Configuration System
```bash
# Ultra-fast development with mock evaluator (30 seconds)
python run_feature_discovery.py --test-mode

# Fast real AutoGluon with S5E6 override config (2-5 minutes) 
# Uses mcts_config.yaml + mcts_config_s5e6_fast_real.yaml overrides
python run_feature_discovery.py --config config/mcts_config_s5e6_fast_real.yaml --real-autogluon

# Production feature discovery with S5E6 config (hours)
python run_feature_discovery.py --config config/mcts_config_s5e6_production.yaml

# Session management
python run_feature_discovery.py --list-sessions
python run_feature_discovery.py --resume [SESSION_ID]

# Configuration validation
python run_feature_discovery.py --config config/mcts_config_s5e6_fast_real.yaml --validate-config
```


## Architecture

### MCTS-Driven Feature Discovery System
- **MCTSEngine**: Monte Carlo Tree Search with UCB1 selection algorithm
- **FeatureSpace**: Domain-agnostic feature operations framework
- **DomainModules**: Domain-specific feature operations (generic + fertilizer_s5e6)
- **FeatureCacheManager**: MD5-based caching system for efficient feature storage
- **AutoGluonEvaluator**: Fast ML model evaluation using AutoGluon TabularPredictor
- **MockEvaluator**: Ultra-fast testing mode for development
- **Analytics**: Comprehensive performance monitoring and visualization
- **Database**: SQLite logging of complete exploration history

## MCTS Configuration System

The MCTS Feature Discovery uses a **base + override** configuration architecture:

### Configuration Files
- **mcts_config.yaml**: Base configuration with all default parameters (DO NOT MODIFY)
- **mcts_config_s5e6_production.yaml**: S5E6 production configuration
- **mcts_config_s5e6_fast_real.yaml**: S5E6 fast real-data evaluation
- **mcts_config_s5e6_fast_test.yaml**: S5E6 ultra-fast testing configuration
- **Custom configs**: Can create additional domain-specific override configs

### How Override System Works
```bash
# When you run:
python run_feature_discovery.py --config mcts_config_s5e6_fast_real.yaml

# System automatically:
# 1. Loads mcts_config.yaml (base configuration)
# 2. Loads mcts_config_s5e6_fast_real.yaml (overrides)
# 3. Deep merges overrides into base configuration
# 4. Uses final merged configuration with S5E6 paths and domain features
```

### Configuration Profiles
| Profile | Config File | Purpose | Target Time |
|---------|-------------|---------|-------------|
| **Development** | `--test-mode` | Mock evaluator for debugging | 30 seconds |
| **Fast Test** | `mcts_config_s5e6_fast_test.yaml` | Real AutoGluon, 100 samples | 2-5 minutes |
| **Fast Real** | `mcts_config_s5e6_fast_real.yaml` | Real AutoGluon, 5% data | 5-10 minutes |
| **Production** | `mcts_config_s5e6_production.yaml` | Full S5E6 feature discovery | Hours |

### Key Override Parameters
```yaml
# mcts_config_s5e6_fast_real.yaml key changes:
session:
  max_iterations: 5              # vs 20 in base
autogluon:
  train_path: "/mnt/ml/competitions/2025/playground-series-s5e6/train.csv"
  target_metric: 'MAP@3'
  included_model_types: ['XGB']  # vs ['XGB', 'GBM', 'CAT'] in base
  train_size: 0.05               # 5% of data vs null in base
feature_space:
  domain_module: 'domains.fertilizer_s5e6'
```

## Important Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-key"      # For OpenAI/OpenRouter APIs
export ANTHROPIC_API_KEY="your-key"   # For direct Anthropic API
```

### Key Configuration Files
- **config.yaml**: Main orchestrator settings (LLM, strategies, execution modes)
- **agents/*.yaml**: Individual agent prompts and constraints
- **fertilizer_models/requirements.txt**: Python ML dependencies

## Generated Code Structure

After running the orchestrator, generated code appears in:
```
sessions/[timestamp]_[strategy]/
├── prompt/         # Input prompts sent to each agent
├── response/       # Raw agent responses  
├── code/          # Extracted Python code
│   ├── requirements.txt
│   └── [timestamp].py
└── session_summary.md
```

## ML Model Evolution

The fertilizer_models directory contains 6 versions of the prediction model, showing evolution from basic (15 features) to advanced (116+ features):

1. **001**: Baseline LightGBM (MAP@3: 0.32065)
2. **002**: Over-engineered LightGBM (MAP@3: 0.32361, overfitting)  
3. **003**: XGBoost GPU (MAP@3: 0.33311)
4. **004**: CatBoost GPU (MAP@3: 0.32477)
5. **005**: XGBoost + Optuna optimization (MAP@3: 0.33453)
6. **006**: TabM neural network (in progress)

Each model includes comprehensive feature engineering with agricultural domain knowledge (NPK ratios, soil-crop interactions, environmental stress indicators).

## Key Constraints

### Agent Behavior
- **NO VISUALIZATION**: All agents explicitly avoid matplotlib/seaborn (adds unnecessary complexity)
- **GPU Acceleration**: Models prioritize GPU-enabled training (LightGBM, XGBoost, CatBoost)
- **Relative Paths**: Generated code uses paths relative to session directories
- **Competition Focus**: All models generate Kaggle-compatible submissions

### Feature Engineering
- **Agricultural Domain**: Heavy focus on NPK (Nitrogen, Phosphorous, Potassium) chemistry
- **Feature Caching**: Advanced models use caching for expensive feature computations
- **Memory Optimization**: Data type conversion (int64→int8, float64→float16)
- **Numerical Binning**: Converting numerical features to categorical for tree models

## MCTS-Driven Automated Feature Engineering System

**NEW**: Advanced MCTS-powered feature discovery system for automated feature engineering.

### MCTS System Components
- **MCTSEngine**: Monte Carlo Tree Search with UCB1 selection algorithm
- **FeatureSpace**: 6 categories of agricultural domain feature operations
- **AutoGluonEvaluator**: Fast ML model evaluation using AutoGluon TabularPredictor
- **MockEvaluator**: Ultra-fast testing mode (10 iterations in <30 seconds)
- **Analytics**: Comprehensive performance monitoring and visualization
- **Database**: SQLite logging of complete exploration history

### Quick Start - MCTS Feature Discovery
```bash
# 1. Ultra-fast testing with mock evaluator (30 seconds)
python run_feature_discovery.py --test-mode

# 2. Real AutoGluon validation with small dataset (2-5 minutes)
python run_feature_discovery.py --config config/mcts_config_fast_real.yaml --real-autogluon

# 3. Full production feature discovery (hours)
python run_feature_discovery.py --config config/mcts_config.yaml

# 4. Resume interrupted session
python run_feature_discovery.py --resume

# 5. Configuration validation
python run_feature_discovery.py --config config/mcts_config_fast_real.yaml --validate-config
```

### MCTS Configuration (mcts_config.yaml)
```yaml
# Session settings
session:
  mode: 'new'                    # 'new', 'continue', 'resume_best'
  max_iterations: 200            # Exploration limit
  max_runtime_hours: 12          # Safety timeout

# MCTS algorithm
mcts:
  exploration_weight: 1.4        # UCB1 exploration coefficient
  max_tree_depth: 8             # Feature operation depth
  selection_strategy: 'ucb1'     # Selection algorithm

# Testing modes
testing:
  use_mock_evaluator: false      # Enable for development
  use_small_dataset: true        # Small dataset for testing
  small_dataset_size: 5000       # Sample size

# AutoGluon evaluation
autogluon:
  target_metric: 'MAP@3'
  fast_eval:
    time_limit: 20              # Fast exploration phase
  thorough_eval:
    time_limit: 120             # Thorough exploitation phase
```

### Feature Operation Categories
1. **NPK Interactions**: Nutrient ratios, harmony, dominance patterns
2. **Environmental Stress**: Heat/drought stress, optimal conditions
3. **Agricultural Domain**: Crop-specific deficits, soil adjustments
4. **Statistical Aggregations**: Groupby statistics, rankings
5. **Feature Transformations**: Polynomial, log, binning
6. **Feature Selection**: Correlation filter, variance filter

### Performance Benchmarks
- **Mock Mode**: 10 iterations in <30 seconds (development)
- **Small Dataset**: 3-5 iterations in 2-5 minutes (validation)
- **Production**: 100+ iterations over hours (full exploration)
- **Best Score Achieved**: 0.33453 MAP@3 (MCTS-discovered features)

### Generated Analytics
```
reports/
├── mcts_analytics_report.html     # Interactive dashboard
├── exploration_history.csv        # Raw exploration data
├── score_progression.png          # Performance charts
├── operation_performance.png      # Feature operation analysis
├── timing_analysis.png           # Performance profiling
└── best_features_discovered.py   # Generated feature code
```

### MCTS Integration with Existing Models
The MCTS system integrates with existing fertilizer models through the shared `feature_engineering.py` module:
- Models 006-009 use `load_or_create_features()` for consistency
- MCTS discovers new feature combinations automatically
- Best discovered features can be exported as Python code
- Seamless integration with existing AutoGluon evaluation pipeline

### Fast Data Loading and Optimization
The system includes advanced data optimization for rapid iteration:
- **Parquet Support**: Automatic CSV → Parquet conversion (4.2x speedup)
- **Memory Optimization**: Intelligent dtype optimization (up to 11x memory reduction)
- **Smart Sampling**: Stratified sampling with configurable sizes (10K samples in <0.1s)
- **Intelligent Caching**: LRU cache with memory limits and automatic cleanup
- **GPU Acceleration**: XGBoost GPU support for fast evaluation

#### Performance Benchmarks
| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Data Loading | 0.25s (CSV) | 0.06s (Parquet) | **4.2x faster** |
| Memory Usage | 157MB | 0.2MB (optimized) | **785x reduction** |
| Cache Access | N/A | <0.001s | **Instant** |
| Full Pipeline | Hours | 2-5 minutes | **10-50x faster** |

### Monitoring and Debugging
```bash
# Real-time monitoring
tail -f mcts_discovery.log

# Check system resources
python -c "from src import performance_monitor; print('System OK')"

# Database inspection
sqlite3 feature_discovery.db "SELECT * FROM exploration_history ORDER BY evaluation_score DESC LIMIT 10;"
```

### Complete Documentation
- **MCTS_SYSTEM_GUIDE.md**: Comprehensive 50-page implementation guide
- **Covers**: Installation, configuration, usage, optimization, troubleshooting
- **API Reference**: Complete class and function documentation
- **Extension Guide**: How to add new feature operations and evaluators

## Development Workflow

### Traditional Orchestrator Workflow
1. **Configure LLM**: Set API keys and model preferences in config.yaml
2. **Run Orchestrator**: Choose strategy based on problem approach
3. **Review Generated Code**: Check sessions/[timestamp]_[strategy]/code/
4. **Execute ML Models**: Run generated Python code to train and submit
5. **Iterate**: Use different strategies or manual model improvements

### MCTS Feature Discovery Workflow
1. **Development**: Start with `--test-mode` for rapid iteration (30s)
2. **Validation**: Use `--real-autogluon` for model validation (2-5min)
3. **Production**: Full exploration with comprehensive feature discovery (hours)
4. **Analysis**: Review analytics reports for insights and optimization
5. **Integration**: Export best features to existing model pipeline

The system is designed for competitive ML scenarios where rapid prototyping, automated feature discovery, and domain expertise integration are crucial for performance.