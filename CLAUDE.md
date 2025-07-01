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

## Common Commands

### Core System Operations
```bash
# Main MCTS feature discovery
./mcts.py --config config/mcts_config_s5e6_fast_test.yaml    # Fast testing (2-5 min)
./mcts.py --config config/mcts_config_titanic_test.yaml      # Titanic validation (30-60s)
./mcts.py --test-mode                                        # Mock mode for debugging (30s)

# Database management
./manager.py datasets --list                                 # List registered datasets
./manager.py datasets --register --dataset-name NAME --auto --dataset-path PATH
./manager.py sessions --list                                 # Show recent sessions
./manager.py features --top 10                              # Top performing features

# Session management
./mcts.py --list-sessions                                    # List all sessions
./mcts.py --resume                                           # Resume last session
./mcts.py --resume SESSION_ID                               # Resume specific session
```

### Testing and Development
```bash
# Run tests
pytest                                                       # Full test suite
pytest tests/unit/                                          # Unit tests only
pytest tests/integration/                                   # Integration tests only
pytest --cov=src --cov-report=html                         # Coverage report

# Development workflow
./mcts.py --test-mode --validate-config                     # Validate config without running
python -m src.features.generic.statistical                  # Test feature operations
bin/duckdb data/minotaur.duckdb                            # Direct DuckDB access
```

### Performance Analysis
```bash
# System validation
./manager.py selfcheck --run                                # System health check
./manager.py verification --verify-latest                   # Data integrity check
./manager.py analytics --summary                            # Performance summary
./manager.py backup --create                                # Create database backup
```

## Custom Claude Code Commands

### Documentation Update Command
Update documentation based on Git history analysis:
```
/project:update-docs-from-git.md mcts      # Update MCTS documentation
/project:update-docs-from-git.md features  # Update features documentation
/project:update-docs-from-git.md db        # Update database documentation
```

This command analyzes recent commits and guides through updating the specified documentation.

## System Architecture Essentials

### MCTS Iteration Pattern
The system follows a 4-phase cycle per iteration:
1. **Selection**: UCB1 algorithm navigates tree to select promising nodes
2. **Expansion**: Create child nodes with new feature operations
3. **Evaluation**: AutoGluon evaluates feature combinations (MAP@3 metric)
4. **Backpropagation**: Update node statistics up to root

**Key Insight**: After ~13-15 iterations, MCTS shifts from exploration to exploitation - fewer new evaluations is normal behavior as it focuses on promising branches.

### Database Repository Pattern
All data access goes through repository layer:
- `SessionRepository`: MCTS session management
- `FeatureRepository`: Feature catalog and operations
- `ExplorationRepository`: Node exploration tracking
- `DatasetRepository`: Dataset registration and metadata

Connection pooling is automatic - never create direct DuckDB connections.

### Feature Operation System
Two-tier architecture:
- **Generic Operations** (`src/features/generic/`): Domain-agnostic (statistical, polynomial, binning, ranking, temporal, text, categorical)
- **Custom Operations** (`src/features/custom/`): Domain-specific (kaggle_s5e6, titanic)

All operations extend base classes with timing support and signal detection.

### Configuration System
Base + override pattern:
- **Base**: `config/mcts_config.yaml` (never modify)
- **Overrides**: Domain-specific configs that inherit from base
- **Dataset Registration**: Use `dataset_name` instead of direct paths

## Critical Performance Notes

### MCTS Behavior Understanding
- **100 iterations ≠ 100 evaluations**: Each iteration can evaluate multiple nodes
- **Early iterations**: High expansion rate, many evaluations per iteration
- **Later iterations**: Exploitation focus, fewer new evaluations (normal)
- **Tree depth limits**: `max_tree_depth` affects exploration scope

### Memory and Resource Management
- **Feature catalog caching**: Thread-safe lazy loading for 50-100x speedup
- **Connection pooling**: Automatic DuckDB connection management
- **Memory monitoring**: Built-in resource tracking and garbage collection
- **Signal detection**: Automatic filtering of low-signal features (nunique() <= 1)

### Data Loading Optimization
- **Parquet caching**: 4.2x faster than CSV loading
- **DuckDB sampling**: Efficient random sampling without full data loading
- **Column masking**: Use DuckDB queries instead of DataFrame copies in memory

## Memories

### DuckDB Interaction
- Direct DuckDB client: `bin/duckdb data/minotaur.duckdb` (has execute permissions)

### Feature Engineering Refactoring (2025-06-29)
- Migrated from `src/domains` to `src/features` with modular structure
- Added timing capabilities with DEBUG/INFO level logging
- Implemented no-signal feature detection (filters features with nunique() <= 1)
- Created backward compatibility wrappers for smooth migration

### MCTS Evaluation Issue Resolution (2025-07-01)
- Identified that "13 evaluations per 100 iterations" is normal MCTS behavior
- System actually performs 200+ total evaluations across all iterations
- Early iterations do more expansion, later focus on exploitation
- Configuration optimization: `expansion_budget` should be ≤ `max_children_per_node`

### Documentation Guidelines
- Features problems: Must read `docs/features/FEATURES_OVERVIEW.md`
- MCTS problems: Must read `docs/mcts/MCTS_OVERVIEW.md`
- AutoGluon problems: Must read `docs/AUTOGLUON_CONFIG_GUIDE.md`