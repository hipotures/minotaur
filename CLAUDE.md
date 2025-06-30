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

## Documentation Structure

Comprehensive MCTS documentation is organized in `docs/mcts/`:

- **📋 [MCTS_OVERVIEW.md](docs/mcts/MCTS_OVERVIEW.md)** - Executive summary, quick start, and key concepts
- **🔧 [MCTS_IMPLEMENTATION.md](docs/mcts/MCTS_IMPLEMENTATION.md)** - Technical details, components, and APIs
- **📊 [MCTS_DATA_FLOW.md](docs/mcts/MCTS_DATA_FLOW.md)** - Phase-by-phase data flow with examples
- **✅ [MCTS_VALIDATION.md](docs/mcts/MCTS_VALIDATION.md)** - Validation framework and testing tools
- **⚙️ [MCTS_OPERATIONS.md](docs/mcts/MCTS_OPERATIONS.md)** - Configuration, performance, and troubleshooting

**Quick Navigation**:
- **New Users**: Start with [MCTS_OVERVIEW.md](docs/mcts/MCTS_OVERVIEW.md)
- **Developers**: See [MCTS_IMPLEMENTATION.md](docs/mcts/MCTS_IMPLEMENTATION.md) for technical details
- **Operations**: Check [MCTS_OPERATIONS.md](docs/mcts/MCTS_OPERATIONS.md) for configuration
- **QA/Testing**: Review [MCTS_VALIDATION.md](docs/mcts/MCTS_VALIDATION.md) for validation tools

Comprehensive Feature Engineering documentation is organized in `docs/features/`:

- **📋 [FEATURES_OVERVIEW.md](docs/features/FEATURES_OVERVIEW.md)** - Executive summary, architecture, and quick start
- **🔧 [FEATURES_OPERATIONS.md](docs/features/FEATURES_OPERATIONS.md)** - Complete catalog of all feature operations
- **🔗 [FEATURES_INTEGRATION.md](docs/features/FEATURES_INTEGRATION.md)** - Pipeline integration and dataset management
- **🛠️ [FEATURES_DEVELOPMENT.md](docs/features/FEATURES_DEVELOPMENT.md)** - Custom domain development guide
- **⚡ [FEATURES_PERFORMANCE.md](docs/features/FEATURES_PERFORMANCE.md)** - Performance optimization and troubleshooting

**Quick Navigation**:
- **Data Scientists**: Start with [FEATURES_OVERVIEW.md](docs/features/FEATURES_OVERVIEW.md) for system overview
- **ML Engineers**: Check [FEATURES_OPERATIONS.md](docs/features/FEATURES_OPERATIONS.md) for operation details
- **Developers**: See [FEATURES_DEVELOPMENT.md](docs/features/FEATURES_DEVELOPMENT.md) for creating custom features
- **DevOps**: Review [FEATURES_PERFORMANCE.md](docs/features/FEATURES_PERFORMANCE.md) for optimization

## Memories

### DuckDB Interaction
- Plik klient duckdb możesz tak wywołać bin/duckdb ma prawa wykonywania

### Feature Engineering Refactoring (2025-06-29)
- Migrated from `src/domains` to `src/features` with modular structure
- Added timing capabilities with DEBUG/INFO level logging
- Implemented no-signal feature detection (filters features with nunique() <= 1)
- Created backward compatibility wrappers for smooth migration

### Documentation Guidelines
- Jeśli problem lub zagadnienie odnoszą się do tematu features (ficzerów) musisz zapoznać się z docs/features/FEATURES_OVERVIEW.md

### MCTS Documentation Guidelines
- Jeśli problem lub zagadnienie odnoszą się do tematów związnych z działaniem MCTS musisz zapoznać się zdocs/mcts/MCTS_OVERVIEW.md

### AutoGluon Configuration Guidelines
- Jeśli problem lub zagadnienie odnoszą się do tematów związnych z działaniem autogluon musisz zapoznać się z docs/AUTOGLUON_CONFIG_GUIDE.md