<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30 23:57
Compatible with commit: 601a407
Changes: Updated to reflect new feature engineering system with origin classification and auto-registration
-->

# MCTS Operations Guide

## ⚙️ Configuration Management

### Configuration Hierarchy

**Base Configuration**: `config/mcts_config.yaml`
- Contains all default settings
- Never modify this file directly
- Used as template for override configs

**Override Configurations**: Create specific configs that inherit from base
```yaml
# Example: config/mcts_config_my_experiment.yaml
# Only specify values that differ from base config

# Core dataset configuration
autogluon:
  dataset_name: 'my-registered-dataset'
  target_metric: 'accuracy'

# Experiment-specific settings
session:
  max_iterations: 100
  max_runtime_hours: 4.0

mcts:
  exploration_weight: 2.0  # Higher exploration
```

### Essential Configuration Sections

#### Dataset Configuration
```yaml
autogluon:
  # REQUIRED: Must match registered dataset name
  dataset_name: 'playground-series-s5e6-2025'
  
  # Target metric for optimization
  target_metric: 'MAP@3'  # Options: MAP@3, accuracy, f1, auc, rmse, mae
  
  # Performance tuning
  train_size: 10000      # Samples for training (null = all data)
  time_limit: 120        # AutoGluon training time limit (seconds)
  enable_gpu: true       # Enable GPU acceleration
  
  # Model selection for speed
  included_model_types: ['XGB', 'LGB']  # Limit to fast models
```

#### MCTS Parameters
```yaml
mcts:
  # Core MCTS algorithm settings
  exploration_weight: 1.4        # UCB1 exploration parameter (C)
  max_tree_depth: 8             # Maximum operation chaining
  expansion_threshold: 1        # Visits before node expansion
  max_children_per_node: 5      # Branching factor limit
  
  # Search termination
  max_nodes_in_memory: 10000    # Memory limit for tree
  min_visits_for_best: 3        # Minimum visits for best path
```

#### Session Management
```yaml
session:
  max_iterations: 50            # MCTS iterations
  max_runtime_hours: 2.0        # Wall clock time limit
  checkpoint_interval: 10       # Save progress every N iterations
```

#### Feature Space Control
```yaml
feature_space:
  max_features_per_node: 300    # Maximum columns per evaluation
  
  # Enable/disable feature categories
  enabled_categories:
    - 'statistical_aggregations'
    - 'polynomial_features'
    - 'binning_features'
    - 'agricultural_domain'      # Domain-specific features
  
  # Generic operation parameters
  generic_params:
    binning_bins: 5             # Number of quantile bins
    groupby_columns: ['Soil_Type', 'Crop_Type']
    aggregate_columns: ['Nitrogen', 'Phosphorous', 'Potassium']
```

#### Logging Configuration
```yaml
logging:
  level: 'DEBUG'                # DEBUG, INFO, WARNING, ERROR
  log_file: 'logs/minotaur.log'
  max_log_size_mb: 100
  backup_count: 5
  
  # Feature-specific logging
  log_mcts_details: true        # Enable detailed MCTS logging
  log_feature_generation: true  # Log feature operations
  log_database_operations: true # Log database queries
```

### Configuration Templates

#### Ultra-Fast Testing (30 seconds)
```yaml
# config/mcts_config_ultra_fast.yaml
session:
  max_iterations: 3
  max_runtime_hours: 0.05

autogluon:
  train_size: 100
  time_limit: 15
  included_model_types: ['XGB']

mcts:
  max_tree_depth: 2
  expansion_budget: 2
```

#### Production Discovery (2-4 hours)
```yaml
# config/mcts_config_production.yaml
session:
  max_iterations: 100
  max_runtime_hours: 4.0

autogluon:
  train_size: null  # Use all data
  time_limit: 300
  enable_gpu: true

mcts:
  max_tree_depth: 8
  exploration_weight: 1.4
```

#### Memory-Constrained Environment
```yaml
# config/mcts_config_low_memory.yaml
autogluon:
  train_size: 5000
  time_limit: 60

feature_space:
  max_features_per_node: 100
  
resources:
  max_memory_gb: 4
  force_gc_interval: 1
```

## 📊 Performance Characteristics

### Timing Benchmarks

#### Dataset Registration (One-time Setup)
```
Fertilizer Competition (S5E6):
├── CSV Loading: 30-60 seconds
├── Feature Generation: 3-8 minutes
├── DuckDB Storage: 1-2 minutes
├── No-signal Filtering: 30 seconds
└── Total: 5-12 minutes

Memory Usage:
├── Peak: 8-12GB (during feature generation)
├── Steady State: 500MB-1GB (DuckDB file)
└── Concurrent Sessions: +2GB per session
```

#### MCTS Search Performance
```
Per-Iteration Timing:
├── Node Selection (UCB1): <1ms
├── Feature Column Mapping: 1-5ms
├── DuckDB Query: 100-500ms
├── AutoGluon Training: 30-300 seconds
├── Database Logging: 10-50ms
└── Total: 30-300 seconds per iteration

Throughput:
├── Fast Config: 10-50 iterations/hour
├── Balanced Config: 5-20 iterations/hour
├── Production Config: 2-10 iterations/hour
```

### Performance Optimization Guidelines

#### Speed Optimization
```yaml
# Optimize for maximum speed
autogluon:
  train_size: 1000              # Small sample
  time_limit: 30                # Very short training
  included_model_types: ['XGB'] # Single fast model
  skip_final_evaluation: true   # Skip expensive evaluation
  
session:
  max_iterations: 20            # Fewer iterations
  
feature_space:
  max_features_per_node: 50     # Fewer features
```

#### Quality Optimization
```yaml
# Optimize for best results
autogluon:
  train_size: null              # All data
  time_limit: 600               # Longer training
  presets: 'high_quality'       # Best preset
  enable_gpu: true              # GPU acceleration
  
session:
  max_iterations: 200           # More exploration
  
mcts:
  exploration_weight: 1.0       # More exploitation
```

#### Memory Optimization
```yaml
# Optimize for low memory usage
resources:
  max_memory_gb: 4
  memory_check_interval: 1
  force_gc_interval: 1

autogluon:
  train_size: 5000
  ag_args_fit:
    num_cpus: 1                 # Limit CPU cores
    num_gpus: 0                 # Disable GPU

feature_space:
  max_features_per_node: 100
  max_cache_size_mb: 128
```

## 🚀 Operational Workflows

### Standard Production Workflow
```bash
# 1. Dataset Registration (one-time)
python manager.py datasets --register \
  --dataset-name production-dataset \
  --dataset-path /data/competition/ \
  --auto

# 2. Validate registration
python manager.py datasets --show production-dataset

# 3. Run MCTS discovery
python mcts.py --config config/mcts_config_production.yaml

# 4. Monitor progress
python manager.py sessions --list
python manager.py sessions --details LATEST_SESSION_ID

# 5. Validate results
python scripts/mcts/validate_mcts_correctness.py --latest

# 6. Analyze best features
python manager.py features --top 20
python manager.py analytics --best-features
```

### Development/Testing Workflow
```bash
# Quick development iteration
python mcts.py --config config/mcts_config_fast_test.yaml

# Validate changes
python scripts/mcts/validate_mcts_correctness.py --latest

# Check feature performance
python manager.py features --performance
```

### Session Management
```bash
# List all sessions
python manager.py sessions --list

# Resume interrupted session
python mcts.py --resume SESSION_ID

# Compare multiple sessions
python manager.py sessions --compare session1 session2

# Export session results
python manager.py sessions --export SESSION_ID --format json
```

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. Dataset Registration Failures

**Error**: `Target column 'target' not found in dataset`
```bash
# Solution: Check actual column names
python -c "import pandas as pd; print(pd.read_csv('train.csv').columns.tolist())"

# Update registration with correct target
python manager.py datasets --register \
  --dataset-name my-dataset \
  --dataset-path /data/ \
  --target-column actual_target_name
```

**Error**: `Feature generation failed: Memory error`
```bash
# Solution: Use smaller batch sizes or increase memory
export DUCKDB_MEMORY_LIMIT=8GB
python manager.py datasets --register ... --force-update
```

#### 2. MCTS Runtime Issues

**Error**: `AutoGluon evaluation timeout`
```yaml
# Solution: Increase time limits in config
autogluon:
  time_limit: 300  # Increase from default
  presets: 'medium_quality'  # Use faster preset
```

**Error**: `CUDA out of memory`
```yaml
# Solution: Disable GPU or reduce batch size
autogluon:
  enable_gpu: false
  train_size: 1000  # Smaller sample
```

**Error**: `Session interrupted, cannot resume`
```bash
# Solution: Check database integrity
python manager.py selfcheck --run
python manager.py selfcheck --fix

# Restart with new session
python mcts.py --config config/mcts_config.yaml
```

#### 3. Performance Issues

**Slow iterations (>10 minutes each)**:
```yaml
# Optimize for speed
autogluon:
  train_size: 5000
  time_limit: 60
  included_model_types: ['XGB']

feature_space:
  max_features_per_node: 100
```

**High memory usage**:
```yaml
# Memory optimization
resources:
  max_memory_gb: 4
  force_gc_interval: 1

autogluon:
  train_size: 1000
  ag_args_fit:
    num_cpus: 1
```

**Database size growth**:
```bash
# Clean old sessions
python manager.py sessions --cleanup --older-than 30d

# Optimize database
python manager.py db --vacuum
```

#### 4. Configuration Issues

**Error**: `Configuration validation failed`
```bash
# Validate config before running
python mcts.py --config my_config.yaml --validate-config

# Check for syntax errors
python -c "import yaml; yaml.safe_load(open('my_config.yaml'))"
```

**Error**: `Dataset not found: 'my-dataset'`
```bash
# Check registered datasets
python manager.py datasets --list

# Re-register if needed
python manager.py datasets --register ...
```

### Debugging Tools

#### Enable Verbose Logging
```yaml
logging:
  level: 'DEBUG'
  log_mcts_details: true
  log_feature_generation: true
  log_database_operations: true
```

#### Database Inspection
```bash
# Access database directly
bin/duckdb data/minotaur.duckdb

# Common debug queries
.tables
SELECT COUNT(*) FROM sessions;
SELECT * FROM exploration_history ORDER BY timestamp DESC LIMIT 10;
```

#### Memory Monitoring
```bash
# Monitor memory usage during MCTS
watch -n 5 'ps aux | grep python | grep mcts'

# System resource monitoring
htop  # or top on macOS
```

#### Log Analysis
```bash
# Filter MCTS logs by session
grep "session_20250630_132513" logs/mcts/session_20250630_132513.log

# Check for errors
grep -i error logs/minotaur.log | tail -20

# Monitor live logs
tail -f logs/minotaur.log
```

## 📈 Monitoring and Maintenance

### Health Checks
```bash
# Run system self-check
python manager.py selfcheck --run

# Verify database integrity
python manager.py verification --check-integrity

# Validate latest session
python scripts/mcts/validate_mcts_correctness.py --latest
```

### Regular Maintenance
```bash
# Create database backup
python manager.py backup --create

# Clean old sessions (keep last 30 days)
python manager.py sessions --cleanup --older-than 30d

# Optimize database performance
python manager.py db --vacuum
python manager.py db --analyze
```

### Performance Monitoring
```bash
# Session performance summary
python manager.py analytics --summary

# Feature operation statistics
python manager.py analytics --operation-stats

# Resource usage analysis
python manager.py analytics --resource-usage
```

## 🔐 Security and Best Practices

### Configuration Security
- Store sensitive configs outside repository
- Use environment variables for credentials
- Validate all user inputs
- Limit resource usage in production

### Data Security
- Encrypt backups for sensitive datasets
- Use secure paths for dataset registration
- Implement access controls for database
- Monitor for data leakage in logs

### Operational Security
- Regular security updates for dependencies
- Monitor system resource usage
- Implement proper logging without sensitive data
- Use read-only access where possible

### Example Production Setup
```yaml
# Production configuration template
database:
  path: '/secure/data/minotaur.duckdb'
  backup_location: '/secure/backups/'

logging:
  level: 'INFO'  # Avoid DEBUG in production
  log_file: '/var/log/minotaur/minotaur.log'

resources:
  max_memory_gb: 16
  max_cpu_cores: 8
  max_disk_usage_gb: 100

security:
  enable_audit_logging: true
  mask_sensitive_data: true
```

---

*For validation and testing, see [MCTS_VALIDATION.md](MCTS_VALIDATION.md)*  
*For technical details, see [MCTS_IMPLEMENTATION.md](MCTS_IMPLEMENTATION.md)*  
*For data flow analysis, see [MCTS_DATA_FLOW.md](MCTS_DATA_FLOW.md)*