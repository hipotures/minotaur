# MCTS-Driven Automated Feature Engineering System
## Complete Implementation Guide

### Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Performance Optimization](#performance-optimization)
7. [Analytics & Monitoring](#analytics--monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Extension Guide](#extension-guide)
10. [API Reference](#api-reference)

---

## System Overview

The MCTS-Driven Automated Feature Engineering System uses Monte Carlo Tree Search to automatically discover and evaluate feature combinations for machine learning tasks. The system is specifically optimized for the fertilizer prediction domain but can be adapted for other tabular ML problems.

### Key Features
- **MCTS-driven exploration**: Intelligent feature space exploration using UCB1 selection
- **AutoGluon integration**: Fast ML model evaluation and scoring
- **Comprehensive logging**: SQLite database tracking all exploration history
- **Mock evaluator**: Ultra-fast testing mode for development
- **Advanced analytics**: Real-time performance monitoring and visualization
- **Session management**: Resume/continue/restart capabilities
- **Timing infrastructure**: Detailed performance profiling
- **Feature caching**: Memory-optimized feature generation

### Performance Metrics
- **Mock Mode**: 10 iterations in <30 seconds (development/testing)
- **Real AutoGluon**: 3-5 iterations in 2-5 minutes (small dataset)
- **Production**: 100+ iterations over hours with comprehensive features
- **Best achieved score**: 0.33453 MAP@3 (XGBoost + Optuna optimization)

---

## Architecture

### Core Components

```
mcts_feature_discovery/
├── discovery_db.py       # SQLite database interface
├── mcts_engine.py        # Core MCTS algorithm
├── feature_space.py      # Feature operations manager
├── autogluon_evaluator.py # Real AutoGluon wrapper
├── mock_evaluator.py     # Fast mock evaluator
├── timing.py             # Performance monitoring
├── analytics.py          # Reporting and visualization
├── data_utils.py         # Data loading optimization
└── synthetic_data.py     # Testing data generation
```

### Integration Points
- **feature_engineering.py**: Shared feature generation module
- **CLAUDE.md**: System documentation and commands
- **mcts_config.yaml**: Centralized configuration
- **mcts.py**: Main orchestrator script

### Data Flow
```
[Raw Data] → [Feature Space] → [MCTS Engine] → [Evaluator] → [Database] → [Analytics]
    ↓              ↓               ↓              ↓            ↓            ↓
[CSV/Parquet] [Operations] [UCB1 Selection] [AutoGluon] [SQLite Log] [HTML Reports]
```

---

## Installation & Setup

### Prerequisites
```bash
# Core dependencies
pip install pandas numpy scikit-learn pyyaml psutil

# AutoGluon (for real evaluation)
pip install autogluon.tabular

# Visualization (optional)
pip install matplotlib seaborn

# Development tools
pip install pytest jupyter
```

### Quick Start
```bash
# 1. Navigate to fertilizer models directory
cd fertilizer_models/

# 2. Test with mock evaluator (30 seconds)
python mcts.py --test-mode

# 3. Test with real AutoGluon (2-5 minutes)
python mcts.py --real-autogluon

# 4. Full production run (hours)
python mcts.py
```

### Verify Installation
```bash
# Check configuration
python mcts.py --validate-config

# Test components individually
python -c "from mcts_feature_discovery import MCTSEngine, AUTOGLUON_AVAILABLE; print(f'AutoGluon: {AUTOGLUON_AVAILABLE}')"
```

---

## Configuration

### Main Configuration File: `mcts_config.yaml`

#### Session Management
```yaml
session:
  mode: 'new'                    # 'new', 'continue', 'resume_best'
  max_iterations: 200            # Total exploration limit
  max_runtime_hours: 12          # Safety timeout
  checkpoint_interval: 25        # Auto-save frequency
```

#### MCTS Algorithm Parameters
```yaml
mcts:
  exploration_weight: 1.4        # UCB1 exploration coefficient
  max_tree_depth: 8             # Maximum feature operation depth
  expansion_threshold: 3         # Visits before node expansion
  selection_strategy: 'ucb1'     # Selection algorithm
```

#### Testing Configuration
```yaml
testing:
  use_mock_evaluator: false      # Enable for fast testing
  mock_base_score: 0.30         # Mock evaluator baseline
  fast_test_mode: true          # Reduced iterations
  use_small_dataset: true       # Sample data for speed
  small_dataset_size: 5000      # Sample size
```

#### AutoGluon Evaluation
```yaml
autogluon:
  train_path: '../competitions/playground-series-s5e6/train.csv'
  test_path: '../competitions/playground-series-s5e6/test.csv'
  target_metric: 'MAP@3'
  
  # Fast evaluation (exploration phase)
  fast_eval:
    time_limit: 20              # Seconds per evaluation
    presets: 'medium_quality_faster_train'
    
  # Thorough evaluation (exploitation phase)  
  thorough_eval:
    time_limit: 120             # More thorough evaluation
    presets: 'good_quality_faster_inference'
```

### Command Line Overrides
```bash
# Override session mode
python mcts.py --new-session
python mcts.py --resume

# Testing modes
python mcts.py --test-mode      # Mock evaluator
python mcts.py --real-autogluon # Small dataset
```

---

## Usage Examples

### Example 1: Quick Development Testing
```bash
# Fast mock testing (30 seconds)
python mcts.py --test-mode

# Expected output:
# INFO - MCTS-Driven Automated Feature Engineering System
# INFO - Using mock evaluator for testing mode
# INFO - Generated comprehensive report with 7 files
# INFO - Best score achieved: 0.32145
```

### Example 2: Real AutoGluon Validation
```bash
# Small dataset real evaluation (2-5 minutes)
python mcts.py --real-autogluon

# Monitor progress:
tail -f mcts_discovery.log
```

### Example 3: Production Run
```bash
# Full feature discovery (hours)
python mcts.py --config production_config.yaml

# Session management:
python mcts.py --resume        # Continue last session
python mcts.py --new-session   # Start fresh
```

### Example 4: Custom Configuration
```python
# Custom config programmatically
from run_feature_discovery import FeatureDiscoveryRunner

config_overrides = {
    'session': {'max_iterations': 50},
    'testing': {'use_mock_evaluator': True},
    'mcts': {'exploration_weight': 2.0}
}

runner = FeatureDiscoveryRunner('mcts_config.yaml', config_overrides)
results = runner.run_discovery()

print(f"Best score: {results['search_results']['best_score']:.4f}")
print(f"Runtime: {results['total_runtime']:.2f}s")
```

---

## Performance Optimization

### Speed Optimization Strategies

#### 1. Mock Evaluator Development
```yaml
# In mcts_config.yaml for development
testing:
  use_mock_evaluator: true
  fast_test_mode: true
  mock_base_score: 0.30
  mock_score_variance: 0.05
```
- **Use case**: Algorithm development, testing, debugging
- **Speed**: 10 iterations in <30 seconds
- **Accuracy**: Realistic score simulation for development

#### 2. Small Dataset Mode
```yaml
# Real AutoGluon with reduced data
testing:
  use_mock_evaluator: false
  use_small_dataset: true
  small_dataset_size: 5000
```
- **Use case**: Model validation, hyperparameter testing
- **Speed**: 3-5 iterations in 2-5 minutes
- **Accuracy**: Real AutoGluon performance on subset

#### 3. AutoGluon Optimization
```yaml
autogluon:
  fast_eval:
    time_limit: 20              # Reduce for faster exploration
    excluded_model_types: ['KNN', 'NN_TORCH', 'FASTAI', 'XT']
  
  thorough_eval:
    time_limit: 60              # Balance speed vs accuracy
    num_bag_folds: 2            # Reduce cross-validation
```

#### 4. Feature Caching
```yaml
feature_space:
  cache_features: true
  max_cache_size_mb: 2048      # Adjust based on available memory
  lazy_loading: true           # Generate features on-demand
```

#### 5. Memory Management
```yaml
resources:
  max_memory_gb: 16           # Set based on system capacity
  force_gc_interval: 50       # Garbage collection frequency
  cleanup_temp_on_exit: true  # Automatic cleanup
```

### Performance Monitoring
```python
# Check timing statistics
from mcts_feature_discovery import get_timing_collector

timing = get_timing_collector()
stats = timing.get_stats()

print(f"Operations per minute: {stats['session']['operations_per_minute']:.1f}")
print(f"Average evaluation time: {stats['operations']['autogluon.evaluate_features']['avg_time']:.2f}s")
```

---

## Analytics & Monitoring

### Real-time Monitoring
```bash
# Monitor progress during execution
tail -f mcts_discovery.log

# Key metrics to watch:
# - Iteration progress
# - Best score improvements  
# - Evaluation times
# - Memory usage
```

### Generated Analytics Files

#### 1. HTML Report (`reports/mcts_analytics_report.html`)
- **Executive summary** with key metrics
- **Session performance** table  
- **Operation analysis** with success rates
- **Timing performance** breakdown
- **Recommendations** for optimization

#### 2. Data Exports
```
reports/
├── mcts_analytics_report.html     # Main dashboard
├── exploration_history.csv        # Raw exploration data
├── summary_statistics.json        # Key metrics
├── timing_analysis.json          # Performance data
├── score_progression.png          # Score over time chart
├── operation_performance.png      # Operation comparison
└── timing_analysis.png           # Timing breakdown
```

#### 3. Session Data
```
# Session results JSON
discovery_session_[ID].json       # Complete session data
timing_data_[ID].json             # Detailed timing logs
best_features_discovered.py       # Generated feature code
```

### Analytics API
```python
from mcts_feature_discovery import AnalyticsGenerator

# Generate custom reports
analytics = AnalyticsGenerator(config)
report_files = analytics.generate_comprehensive_report(
    db_path='feature_discovery.db',
    timing_data='timing_data_abc123.json'
)

# Access specific metrics
summary = analytics._calculate_summary_stats(session_data)
print(f"Best score: {summary['best_score']:.4f}")
print(f"Total runtime: {summary['total_runtime_minutes']:.1f} minutes")
```

---

## Troubleshooting

### Common Issues & Solutions

#### Issue 1: Mock Evaluator Not Working
```
Error: Using real AutoGluon despite --test-mode flag
```
**Solution**: Check config override application:
```python
# Verify override in mcts.py:
config_overrides = {
    'testing': {
        'use_mock_evaluator': True,
        'fast_test_mode': True
    }
}
```

#### Issue 2: AutoGluon 0.0 Scores
```
Warning: AutoGluon returning 0.0 scores consistently
```
**Solution**: Check data format and target column:
```python
# Verify target column exists and format
print(f"Target column: {evaluator.target_column}")
print(f"Data shape: {train_data.shape}")
print(f"Unique targets: {train_data[target_column].unique()}")
```

#### Issue 3: Memory Issues
```
Error: MemoryError during feature generation
```
**Solution**: Reduce memory usage:
```yaml
resources:
  max_memory_gb: 8              # Reduce limit
feature_space:
  max_features_per_node: 200    # Fewer features
  max_cache_size_mb: 1024       # Smaller cache
```

#### Issue 4: Database Cursor Errors
```
AttributeError: 'FeatureDiscoveryDB' object has no attribute 'cursor'
```
**Solution**: Ensure proper database connection management:
```python
# In discovery_db.py, avoid direct cursor access
# Use connection context managers properly
```

#### Issue 5: Configuration Validation Errors
```
Error: Missing required config section: 'mcts'
```
**Solution**: Validate config file:
```bash
python mcts.py --validate-config
```

### Debug Mode
```yaml
# Enable detailed debugging
advanced:
  debug_mode: true
  debug_save_all_features: true
  debug_detailed_timing: true
  
logging:
  level: 'DEBUG'
  log_autogluon_details: true
```

### Performance Diagnostics
```python
# Check system resources
from mcts_feature_discovery import performance_monitor

with performance_monitor("diagnosis"):
    # Run problematic code
    pass

# Check memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
```

---

## Extension Guide

### Adding New Feature Operations

#### 1. Define Operation
```python
# In feature_space.py
new_operation = FeatureOperation(
    name='custom_operation',
    category='custom_category',
    description='Custom feature operation',
    dependencies=['required_feature_1', 'required_feature_2'],
    computational_cost=0.5,
    output_features=['new_feature_1', 'new_feature_2']
)
```

#### 2. Implement Generation Logic
```python
def _apply_custom_operation(self, df: pd.DataFrame) -> pd.DataFrame:
    """Apply custom feature operation."""
    if all(col in df.columns for col in ['required_feature_1', 'required_feature_2']):
        df['new_feature_1'] = df['required_feature_1'] * 2
        df['new_feature_2'] = df['required_feature_2'] ** 0.5
    return df
```

#### 3. Register Operation
```python
# In _initialize_operations method
if 'custom_category' in self.enabled_categories:
    self.operations['custom_operation'] = new_operation
```

### Adding New Evaluators

#### 1. Implement Evaluator Interface
```python
class CustomEvaluator:
    def __init__(self, config):
        self.config = config
    
    def evaluate_features(self, features_df, node_depth=0, iteration=0):
        """Evaluate features and return score."""
        # Implement your evaluation logic
        return score
    
    def get_evaluation_statistics(self):
        """Return evaluation statistics."""
        return {"total_evaluations": self.eval_count}
    
    def cleanup(self):
        """Cleanup resources."""
        pass
```

#### 2. Register in Configuration
```yaml
# Custom evaluator selection
evaluation:
  evaluator_type: 'custom'
  custom_evaluator_class: 'path.to.CustomEvaluator'
```

### Adding New Metrics
```python
# In autogluon_evaluator.py
def _calculate_custom_metric(self, y_true, y_pred):
    """Calculate custom evaluation metric."""
    # Implement metric calculation
    return custom_score

# Register metric
if self.target_metric == 'CUSTOM':
    score = self._calculate_custom_metric(y_true, y_pred)
```

---

## API Reference

### Core Classes

#### MCTSEngine
```python
class MCTSEngine:
    def __init__(self, config: Dict[str, Any])
    def run_search(self, evaluator, feature_space, db, initial_features: Set[str]) -> Dict[str, Any]
    def get_best_path(self) -> List[FeatureNode]
    def get_tree_statistics(self) -> Dict[str, Any]
```

#### FeatureSpace
```python
class FeatureSpace:
    def __init__(self, config: Dict[str, Any])
    def get_available_operations(self, node) -> List[str]
    def generate_features_for_node(self, node) -> pd.DataFrame
    def update_operation_performance(self, operation_name: str, improvement: float, success: bool)
```

#### AutoGluonEvaluator
```python
class AutoGluonEvaluator:
    def __init__(self, config: Dict[str, Any])
    def evaluate_features(self, features_df: pd.DataFrame, node_depth: int = 0, iteration: int = 0) -> float
    def evaluate_final_features(self, features_df: pd.DataFrame) -> Dict[str, Any]
    def get_evaluation_statistics(self) -> Dict[str, Any]
```

#### FeatureDiscoveryDB
```python
class FeatureDiscoveryDB:
    def __init__(self, config: Dict[str, Any])
    def log_exploration(self, iteration: int, operation: str, score: float, **kwargs)
    def get_best_features(self, limit: int = 10) -> List[Dict[str, Any]]
    def export_best_features_code(self, output_file: str, limit: int = 10)
```

### Utility Functions

#### Timing
```python
@timed("operation_name", include_memory=True)
def my_function():
    pass

with timing_context("operation_name"):
    # Code to time
    pass
```

#### Analytics
```python
# Generate quick report
from mcts_feature_discovery import generate_quick_report
report_path = generate_quick_report('feature_discovery.db')

# Custom analytics
analytics = AnalyticsGenerator(config)
reports = analytics.generate_comprehensive_report(db_path, timing_data, session_id)
```

### Configuration Schema

#### Required Sections
- `session`: Session management parameters
- `mcts`: MCTS algorithm configuration  
- `autogluon`: AutoGluon evaluation settings
- `feature_space`: Feature operation parameters
- `database`: SQLite database configuration

#### Optional Sections
- `testing`: Testing and development settings
- `logging`: Logging configuration
- `resources`: Resource management
- `export`: Output and reporting settings
- `analytics`: Visualization parameters
- `advanced`: Experimental features

---

## Best Practices

### Development Workflow
1. **Start with mock testing** (`--test-mode`) for rapid iteration
2. **Validate with small dataset** (`--real-autogluon`) before production
3. **Monitor resource usage** during long runs
4. **Use session management** for checkpointing
5. **Review analytics reports** for optimization opportunities

### Production Deployment
1. **Configure appropriate timeouts** based on available compute
2. **Set memory limits** based on system capacity  
3. **Enable comprehensive logging** for debugging
4. **Use backup and recovery** features
5. **Monitor performance metrics** continuously

### Optimization Strategy
1. **Profile with timing infrastructure** to identify bottlenecks
2. **Tune MCTS parameters** (exploration weight, tree depth)
3. **Optimize AutoGluon settings** for speed/accuracy tradeoff
4. **Use feature caching** effectively
5. **Balance exploration vs exploitation** phases

---

This comprehensive guide covers all aspects of the MCTS Feature Discovery system. For additional support, refer to the generated analytics reports and timing logs for performance insights.