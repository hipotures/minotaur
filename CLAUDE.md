# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview - Minotaur

This is an **independent MCTS-driven feature discovery system** for agricultural fertilizer prediction:

**Primary Focus**: Automated feature engineering using Monte Carlo Tree Search (MCTS) with AutoGluon evaluation for Kaggle competition "Predicting Optimal Fertilizers" (Playground Series S5E6).

**Data Source**: Read-only access to `/mnt/ml/competitions/2025/playground-series-s5e6/` with automatic local parquet caching for 4.2x faster loading.

**Database Architecture**: Advanced DuckDB-based system with repository pattern, connection pooling, and automatic migrations.

## Key Commands

### MCTS Feature Discovery Operations
```bash
# Ultra-fast testing with 100 samples (30 seconds)
python mcts.py --config config/mcts_config_s5e6_fast_test.yaml --test-mode

# Fast real evaluation with 5% data (2-5 minutes)
python mcts.py --config config/mcts_config_s5e6_fast_real.yaml --real-autogluon

# Production feature discovery with 80% data (hours)
python mcts.py --config config/mcts_config_s5e6_production.yaml

# Session management
python mcts.py --list-sessions
python mcts.py --resume [SESSION_ID]
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

# Generate coverage report
pytest --cov=src --cov-report=html       # HTML coverage report in htmlcov/
pytest --cov=src --cov-report=term       # Terminal coverage report

# Linting and code quality (if configured)
python -m py_compile src/discovery_db.py # Check syntax
python -m py_compile src/db_service.py   # Check refactored database layer
```

### MCTS Feature Discovery - Configuration System
```bash
# Ultra-fast development with mock evaluator (30 seconds)
python mcts.py --test-mode

# Fast real AutoGluon with S5E6 override config (2-5 minutes) 
# Uses mcts_config.yaml + mcts_config_s5e6_fast_real.yaml overrides
python mcts.py --config config/mcts_config_s5e6_fast_real.yaml --real-autogluon

# Production feature discovery with S5E6 config (hours)
python mcts.py --config config/mcts_config_s5e6_production.yaml

# Titanic domain testing and validation (30-60 seconds)
python mcts.py --config config/mcts_config_titanic_test.yaml

# Session management
python mcts.py --list-sessions
python mcts.py --resume [SESSION_ID]

# Configuration validation
python mcts.py --config config/mcts_config_s5e6_fast_real.yaml --validate-config
```

## Memories

### DuckDB Interaction
- Plik klient duckdb możesz tak wywołać bin/duckdb ma prawa wykonywania

[Rest of the file remains unchanged]