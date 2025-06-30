<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30 15:30
Compatible with commit: TBD
Changes: Updated with current system configuration and dataset integration
-->

# AutoGluon Configuration Guide

## 📋 Overview

This guide provides comprehensive AutoGluon configuration for the Minotaur MCTS Feature Discovery System. It covers modern dataset integration, performance optimization, and phase-based evaluation strategies.

### 🎯 Quick Navigation
- **New Users**: Start with [Configuration Examples](#configuration-examples)
- **Performance**: See [Performance Optimization](#performance-optimization-tips)
- **Integration**: Check [Dataset Integration](#dataset-integration)
- **Troubleshooting**: Review [Common Issues](#troubleshooting)

## Configuration Structure

```yaml
autogluon:
  # Dataset Integration (Modern Approach)
  dataset_name: 'playground-series-s5e6-2025'  # Registered dataset
  target_metric: 'MAP@3'                        # Competition metric
  
  # Core AutoGluon configuration
  included_model_types: ['XGB']                 # Focus on specific algorithms
  presets: 'medium_quality_faster_train'       # Training speed preset
  enable_gpu: true                              # GPU acceleration
  train_size: 0.1                               # Fraction of training data (0.0-1.0)
  holdout_frac: 0.2                             # Validation split
  time_limit: 30                                # Base time limit in seconds
  
  # Phase-specific configurations
  fast_eval: { ... }                            # Exploration phase settings
  thorough_eval: { ... }                        # Exploitation phase settings
  final_eval: { ... }                           # Final evaluation settings
  
  # Advanced settings
  adaptive_time_limit: true                     # Dynamic time adjustment
  timeout_multiplier: 1.5                      # Increase when improving
  thorough_eval_threshold: 0.6                 # Switch phase at 60%
```

## 🚀 Dataset Integration

### Modern Dataset Configuration (Recommended)

```yaml
# Use registered dataset names
autogluon:
  dataset_name: 'playground-series-s5e6-2025'  # Centralized dataset management
  target_metric: 'MAP@3'                        # Competition-specific metric
  target_column: 'Fertilizer Type'              # Target prediction column
  
# Benefits:
# - Automatic data loading and caching
# - Consistent dataset versions across runs
# - Built-in validation and integrity checking
# - Performance optimizations (parquet caching)
```

### Legacy Path Configuration (Still Supported)

```yaml
# Direct file paths (backward compatibility)
autogluon:
  train_path: "/mnt/ml/competitions/2025/playground-series-s5e6/train.csv"
  test_path: "/mnt/ml/competitions/2025/playground-series-s5e6/test.csv"
  target_column: 'Fertilizer Type'
```

### Available Registered Datasets

```bash
# List all registered datasets
python manager.py datasets --list

# Common datasets:
# - 'playground-series-s5e6-2025'  # Fertilizer prediction
# - 'titanic'                      # Titanic survival (testing)
# - 'house-prices'                 # House price regression
```

## 🎯 Available Model Types

```yaml
# All available AutoGluon model types:
# included_model_types: ['GBM', 'CAT', 'XGB', 'FASTAI', 'RF', 'XT', 'NN_TORCH', 'KNN']

# Recommended configurations by speed:
# Ultra-fast: ['XGB']
# Fast: ['XGB', 'GBM'] 
# Balanced: ['XGB', 'GBM', 'CAT']
# Comprehensive: ['XGB', 'GBM', 'CAT', 'RF']
```

### Model Performance Characteristics

| Model | Speed | Accuracy | GPU Support | Memory Usage |
|-------|-------|----------|-------------|--------------|
| **XGB** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Low |
| **GBM** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Low |
| **CAT** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Medium |
| **RF** | ⭐⭐ | ⭐⭐⭐ | ❌ | Medium |
| **NN_TORCH** | ⭐ | ⭐⭐⭐⭐ | ✅ | High |
| **FASTAI** | ⭐ | ⭐⭐⭐⭐⭐ | ✅ | High |

## Available Presets

```yaml
# Available AutoGluon presets (ordered by training speed):
# - 'medium_quality_faster_train'           # Fastest training
# - 'medium_quality_faster_inference'       # Fast inference
# - 'good_quality_faster_inference'         # Balanced speed/quality
# - 'medium_quality'                        # Standard quality
# - 'good_quality'                          # Higher quality
# - 'high_quality'                          # Slow but accurate
# - 'best_quality'                          # Slowest, best accuracy

# Recommended presets by use case:
# Development/Testing: 'medium_quality_faster_train'
# Production Fast: 'good_quality_faster_inference'  
# Production Quality: 'good_quality'
# Competition: 'high_quality' or 'best_quality'
```

## Configuration Examples

### 1. Ultra-Fast Development Configuration

```yaml
# For rapid prototyping and testing
autogluon:
  dataset_name: 'playground-series-s5e6-2025'
  target_metric: 'MAP@3'
  
  included_model_types: ['XGB']
  presets: 'medium_quality_faster_train'
  enable_gpu: true
  train_size: 0.05                        # Only 5% of data
  time_limit: 15
  holdout_frac: 0.3
  
  fast_eval:
    time_limit: 10
    train_size: 0.02                      # Tiny dataset
    included_model_types: ['XGB']
    
# Expected performance: 30-60 seconds per evaluation
# Use case: Feature operation testing, rapid iteration
```

### 2. Balanced Speed/Quality Configuration

```yaml
# Production-ready with good performance balance
autogluon:
  dataset_name: 'playground-series-s5e6-2025'
  target_metric: 'MAP@3'
  
  included_model_types: ['XGB', 'GBM']
  presets: 'good_quality_faster_inference'
  enable_gpu: true
  train_size: 0.2                         # 20% of data
  time_limit: 60
  
  # Phase-based evaluation
  thorough_eval_threshold: 0.6            # Switch at 60% iterations
  adaptive_time_limit: true
  
  fast_eval:
    time_limit: 30
    train_size: 0.1
    included_model_types: ['XGB']
    
  thorough_eval:
    time_limit: 120
    train_size: 0.2
    included_model_types: ['XGB', 'GBM']
    
# Expected performance: 2-5 minutes per evaluation
# Use case: Standard MCTS feature discovery
```

### 3. Production Quality Configuration

```yaml
# High-quality feature discovery for competition submission
autogluon:
  dataset_name: 'playground-series-s5e6-2025'
  target_metric: 'MAP@3'
  
  included_model_types: ['XGB', 'GBM', 'CAT']
  presets: 'good_quality'
  enable_gpu: true
  train_size: 0.5                         # Half the data
  time_limit: 300                         # 5 minutes
  
  # Advanced phase management
  thorough_eval_threshold: 0.4            # Earlier switch to quality
  adaptive_time_limit: true
  timeout_multiplier: 2.0                 # Longer time for good features
  
  fast_eval:
    time_limit: 60
    train_size: 0.1
    included_model_types: ['XGB']
    
  thorough_eval:
    time_limit: 300
    train_size: 0.3
    included_model_types: ['XGB', 'GBM']
  
  final_eval:
    time_limit: 600                       # 10 minutes
    train_size: 1.0                       # Full dataset
    included_model_types: ['XGB', 'GBM', 'CAT', 'RF']
    presets: 'high_quality'
    
# Expected performance: 5-15 minutes per evaluation
# Use case: Final competition runs, best feature validation
```

## GPU Configuration

### Enabling GPU Acceleration

```yaml
autogluon:
  enable_gpu: true                        # Enable GPU for supported models
  
  # Models with GPU support: XGB, CAT, NN_TORCH, FASTAI
  included_model_types: ['XGB', 'CAT']    # GPU-enabled models only
```

### Performance Improvements with GPU

| Dataset Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 10K rows | 30s | 8s | **3.7x** |
| 50K rows | 120s | 25s | **4.8x** |
| 100K rows | 300s | 45s | **6.7x** |

## Data Sampling Configuration

### Train Size Parameter

```yaml
autogluon:
  train_size: 0.1                         # Use 10% of training data
  
  # Use cases:
  # 0.01-0.05: Ultra-fast development/testing
  # 0.05-0.1:  Fast iteration and debugging  
  # 0.1-0.3:   Balanced development
  # 0.3-0.7:   Production validation
  # 0.7-1.0:   Final evaluation
```

### Stratified Sampling

The system automatically maintains class distribution when sampling:

```python
# Automatic stratified sampling by target column
sampled_data = original_data.groupby('Fertilizer Type').apply(
    lambda x: x.sample(proportional_size)
)
```

## Time Limit Strategies

### Static Time Limits

```yaml
# Fixed time per evaluation
fast_eval:
  time_limit: 15                          # 15 seconds
thorough_eval:  
  time_limit: 60                          # 1 minute
final_eval:
  time_limit: 300                         # 5 minutes
```

### Adaptive Time Limits

```yaml
# Dynamic time adjustment based on progress
autogluon:
  adaptive_time_limit: true               # Enable adaptation
  timeout_multiplier: 1.5                 # Increase by 50% when improving
  
  # Logic: if recent_improvement detected:
  #   new_time_limit = base_time_limit * timeout_multiplier
```

## Performance Optimization Tips

### 1. Model Selection Strategy

```yaml
# Phase 1: Fast exploration (first 60% of iterations)
fast_eval:
  included_model_types: ['XGB']           # Single fastest model
  
# Phase 2: Quality validation (last 40% of iterations)  
thorough_eval:
  included_model_types: ['XGB', 'GBM']    # Add comparison model
  
# Phase 3: Final evaluation (best candidates only)
final_eval:
  included_model_types: ['XGB', 'GBM', 'CAT']  # Full model suite
```

### 2. Data Size Progression

```yaml
# Gradual increase in data size
fast_eval:
  train_size: 0.05                        # 5% for quick tests
thorough_eval:
  train_size: 0.15                        # 15% for validation  
final_eval:
  train_size: 0.3                         # 30% for final assessment
```

### 3. Bagging Configuration

```yaml
# Minimal bagging for speed
fast_eval:
  num_bag_folds: 1                        # No bagging
  num_bag_sets: 1
  
# Standard bagging for quality
thorough_eval:
  num_bag_folds: 2                        # Light bagging
  num_bag_sets: 1
  
# Full bagging for final evaluation
final_eval:
  num_bag_folds: 5                        # Full bagging
  num_bag_sets: 2
```

## Memory Management

### Memory-Efficient Settings

```yaml
autogluon:
  train_size: 0.1                         # Limit data size
  holdout_frac: 0.3                       # Larger holdout = less training data
  
data:
  memory_limit_mb: 200                    # Auto-sample if exceeded
  dtype_optimization: true                # Optimize data types
  use_small_dataset: true                 # Enable sampling
```

### Memory Usage by Configuration

| Configuration | Memory Usage | Speed | Quality |
|--------------|-------------|-------|---------|
| Ultra-Fast | 50MB | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Fast | 200MB | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Balanced | 500MB | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Quality | 1GB | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🔗 Integration with MCTS

### Phase-Based Evaluation Strategy

```yaml
# Automatic phase switching based on iteration progress
autogluon:
  thorough_eval_threshold: 0.6            # Switch at 60% completion
  adaptive_time_limit: true               # Dynamic time allocation
  timeout_multiplier: 1.5                # Increase time for improvements
  
# Example: 10 total iterations
# Iterations 1-6: Use fast_eval settings (exploration)
# Iterations 7-10: Use thorough_eval settings (exploitation)
```

### Feature Space Integration

```yaml
# Optimize for MCTS feature discovery
feature_space:
  max_features_per_node: 150              # Limit feature count
  use_new_pipeline: true                  # Modern feature system
  check_signal: true                      # Filter low-signal features
  
autogluon:
  dataset_name: 'playground-series-s5e6-2025'  # Centralized dataset
  included_model_types: ['XGB']               # Single model for consistency
  train_size: 0.1                             # Fast evaluation
  adaptive_time_limit: true                   # More time for promising features
```

### Signal Detection Integration

```yaml
# Enhanced feature validation
feature_space:
  check_signal: true                      # Enable signal detection
  min_signal_ratio: 0.01                  # 1% minimum unique values
  signal_sample_size: 1000                # Sample for large datasets
  
# Benefits:
# - 50% faster feature generation
# - Automatic filtering of constant features
# - Better AutoGluon evaluation quality
```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Errors**
   ```yaml
   # Solution: Reduce data size
   autogluon:
     train_size: 0.05                     # Smaller dataset
   data:
     memory_limit_mb: 100                 # Force smaller limit
   ```

2. **Slow Evaluation**
   ```yaml
   # Solution: Use faster configuration
   autogluon:
     included_model_types: ['XGB']        # Single model
     time_limit: 15                       # Short time limit
     presets: 'medium_quality_faster_train'
   ```

3. **Poor Model Quality**
   ```yaml
   # Solution: Increase resources gradually
   autogluon:
     train_size: 0.2                      # More data
     time_limit: 60                       # More time
     included_model_types: ['XGB', 'GBM'] # Multiple models
   ```

4. **Dataset Registration Errors**
   ```bash
   # Solution: Check dataset status
   python manager.py datasets --list
   python manager.py datasets --show DATASET_NAME
   
   # Re-register if needed
   python manager.py datasets --register --dataset-name NAME --auto
   ```

5. **Signal Detection Issues**
   ```yaml
   # Solution: Adjust signal detection settings
   feature_space:
     check_signal: true
     min_signal_ratio: 0.005             # Lower threshold
     signal_sample_size: 2000            # Larger sample
   ```

## 📈 Performance Metrics

### AutoGluon Evaluation Performance

| Configuration | Evaluation Time | Memory Usage | MAP@3 Score | Use Case |
|--------------|----------------|--------------|-------------|----------|
| **Ultra-Fast** | 30-60s | 50-100MB | 0.31-0.32 | Development |
| **Fast** | 2-5min | 200-400MB | 0.32-0.33 | Validation |
| **Balanced** | 5-10min | 500MB-1GB | 0.33-0.335 | Production |
| **Quality** | 10-30min | 1-2GB | 0.335+ | Competition |

### Dataset Loading Performance

| Method | Load Time | Memory | Cache | Improvement |
|--------|-----------|--------|-------|-------------|
| **CSV Direct** | 0.25s | 157MB | No | Baseline |
| **Parquet Cache** | 0.06s | 157MB | Yes | **4.2x faster** |
| **Optimized Sample** | 0.01s | 0.2MB | Yes | **25x faster** |

## 🎯 Best Practices

### Development Workflow

1. **Start with Dataset Registration**:
   ```bash
   python manager.py datasets --register --dataset-name my-dataset --auto
   ```

2. **Use Modern Configuration**:
   ```yaml
   autogluon:
     dataset_name: 'my-dataset'  # Not file paths
     target_metric: 'MAP@3'      # Competition metric
   ```

3. **Progressive Scaling**:
   - Development: Ultra-fast config (30s evaluations)
   - Validation: Fast config (2-5min evaluations)
   - Production: Balanced config (5-10min evaluations)
   - Competition: Quality config (10-30min evaluations)

### Performance Optimization

4. **Leverage GPU Acceleration**: Significant speedups for XGB and CAT models
5. **Use Signal Detection**: Enable `check_signal: true` for 50% speedup
6. **Monitor Memory Usage**: Use data limits and optimization to prevent OOM
7. **Phase-Based Strategy**: Different settings for exploration vs exploitation
8. **Enable Caching**: System automatically caches evaluations and datasets

### Integration Best Practices

9. **Feature Space Configuration**:
   ```yaml
   feature_space:
     use_new_pipeline: true    # Modern feature system
     check_signal: true        # Performance boost
     max_features_per_node: 150  # Prevent overfitting
   ```

10. **MCTS Integration**:
    ```yaml
    mcts:
      max_iterations: 10        # Start small
      exploration_weight: 1.4   # UCB1 parameter
    autogluon:
      adaptive_time_limit: true # More time for good features
    ```

## 📁 Example Complete Configurations

### Available Configuration Templates

```bash
# View available configurations
ls config/mcts_config*.yaml

# Key configurations:
# - mcts_config_s5e6_fast_test.yaml     # Ultra-fast testing (30s)
# - mcts_config_s5e6_fast_real.yaml     # Fast real evaluation (2-5min)
# - mcts_config_s5e6_production.yaml    # Production quality (hours)
# - mcts_config_titanic_test.yaml       # Titanic domain testing
```

### Complete Fast Real Configuration Example

```yaml
# config/mcts_config_s5e6_fast_real.yaml (excerpt)
autogluon:
  dataset_name: 'playground-series-s5e6-2025'
  target_metric: 'MAP@3'
  target_column: 'Fertilizer Type'
  
  included_model_types: ['XGB']
  presets: 'medium_quality_faster_train'
  enable_gpu: true
  train_size: 0.1
  time_limit: 60
  
  thorough_eval_threshold: 0.6
  adaptive_time_limit: true
  
  fast_eval:
    time_limit: 30
    train_size: 0.05
  
  thorough_eval:
    time_limit: 120
    train_size: 0.1

feature_space:
  use_new_pipeline: true
  check_signal: true
  enabled_categories:
    - 'statistical_aggregations'
    - 'polynomial_features'
    - 'kaggle_s5e6_domain'

mcts:
  max_iterations: 5
  exploration_weight: 1.4
```

---

*For MCTS configuration details, see [MCTS_OPERATIONS.md](mcts/MCTS_OPERATIONS.md)*  
*For feature configuration details, see [FEATURES_INTEGRATION.md](features/FEATURES_INTEGRATION.md)*