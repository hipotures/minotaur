# AutoGluon Configuration Guide for MCTS Feature Discovery

## Overview

This guide describes the enhanced AutoGluon configuration options for fast and efficient feature evaluation in the MCTS system.

## Configuration Structure

```yaml
autogluon:
  # Core AutoGluon configuration
  included_model_types: ['XGB']           # Focus on specific algorithms
  presets: 'medium_quality_faster_train'  # Training speed preset
  enable_gpu: true                        # GPU acceleration
  train_size: 0.1                         # Fraction of training data (0.0-1.0)
  holdout_frac: 0.2                       # Validation split
  time_limit: 30                          # Base time limit in seconds
  
  # Phase-specific configurations
  fast_eval: { ... }                      # Exploration phase settings
  thorough_eval: { ... }                  # Exploitation phase settings
  final_eval: { ... }                     # Final evaluation settings
```

## Available Model Types

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
autogluon:
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
```

### 2. Balanced Speed/Quality Configuration

```yaml
autogluon:
  included_model_types: ['XGB', 'GBM']
  presets: 'good_quality_faster_inference'
  enable_gpu: true
  train_size: 0.2                         # 20% of data
  time_limit: 60
  
  fast_eval:
    time_limit: 30
    train_size: 0.1
    included_model_types: ['XGB']
    
  thorough_eval:
    time_limit: 120
    train_size: 0.2
    included_model_types: ['XGB', 'GBM']
```

### 3. Production Quality Configuration

```yaml
autogluon:
  included_model_types: ['XGB', 'GBM', 'CAT']
  presets: 'good_quality'
  enable_gpu: true
  train_size: 0.5                         # Half the data
  time_limit: 300                         # 5 minutes
  
  final_eval:
    time_limit: 600                       # 10 minutes
    train_size: 1.0                       # Full dataset
    included_model_types: ['XGB', 'GBM', 'CAT', 'RF']
    presets: 'high_quality'
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

## Integration with MCTS

### Phase-Based Evaluation

```yaml
# Automatic phase switching based on iteration progress
autogluon:
  thorough_eval_threshold: 0.6            # Switch at 60% completion
  
# Example: 10 total iterations
# Iterations 1-6: Use fast_eval settings
# Iterations 7-10: Use thorough_eval settings
```

### Feature Discovery Optimization

```yaml
# Optimize for feature discovery speed
feature_space:
  max_features_per_node: 150              # Limit feature count
  
autogluon:
  included_model_types: ['XGB']           # Single model for consistency
  train_size: 0.1                         # Fast evaluation
  adaptive_time_limit: true               # More time for promising features
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

## Best Practices

1. **Start Small, Scale Up**: Begin with ultra-fast settings for development
2. **Use GPU When Available**: Significant speedups for XGB and CAT
3. **Progressive Data Sizing**: Increase train_size for promising features
4. **Monitor Memory Usage**: Use data limits to prevent OOM errors
5. **Phase-Based Strategy**: Different settings for exploration vs exploitation
6. **Cache Everything**: System automatically caches evaluations

## Example Complete Configuration

See `mcts_config_fast_real.yaml` for a complete working example optimized for speed and real AutoGluon evaluation.