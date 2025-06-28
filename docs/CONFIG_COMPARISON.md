# MCTS Configuration Comparison

## Quick Reference Table

| Parameter | Mock Mode | Fast Test | Fast Real | Production |
|-----------|-----------|-----------|-----------|------------|
| **Config File** | `--test-mode` | `mcts_config_fast_test.yaml` | `mcts_config_fast_real.yaml` | `mcts_config.yaml` |
| **Performance Target** | 30 seconds | 30-60 seconds | 2-5 minutes | Hours to days |
| **Iterations** | 10 | 3 | 5 | 20-200 |
| **Dataset Size** | Synthetic | 1K samples | 5K samples | Full dataset |
| **Memory Limit** | N/A | 50MB | 100MB | 2GB |
| **AutoGluon Models** | Mock | XGB only | XGB only | XGB, GBM, CAT |
| **Time Limits** | N/A | 8-15s | 15-60s | 20-600s |
| **GPU Usage** | No | Yes | Yes | Yes |
| **Use Case** | Development | Quick Testing | Feature Validation | Competition |

## Configuration System

The MCTS system uses a **base + override** configuration architecture:

1. **Base Configuration** (`mcts_config.yaml`): Contains all default parameters
2. **Override Configurations** (e.g., `mcts_config_fast_real.yaml`): Contains only parameters to override
3. **Automatic Merging**: System automatically loads base config and applies overrides

### How Override System Works

```bash
# This command:
python mcts.py --config mcts_config_fast_real.yaml

# Actually does:
# 1. Load mcts_config.yaml (base)
# 2. Load mcts_config_fast_real.yaml (overrides) 
# 3. Deep merge overrides into base
# 4. Use merged configuration
```

**Example Override Process:**
```yaml
# Base (mcts_config.yaml)
session:
  max_iterations: 20
  max_runtime_hours: 12
autogluon:
  time_limit: 120
  
# Override (mcts_config_fast_real.yaml)  
session:
  max_iterations: 5     # Override this
autogluon:
  time_limit: 30        # Override this

# Final merged config:
session:
  max_iterations: 5     # From override
  max_runtime_hours: 12 # From base (kept)
autogluon:
  time_limit: 30        # From override
```

## Configuration Details

### Mock Mode (Development)
```bash
python mcts.py --test-mode
```
- **Purpose**: Ultra-fast development and debugging
- **Evaluator**: Synthetic mock scores
- **Speed**: ~10 iterations in 30 seconds
- **Use Case**: Code development, testing, debugging

### Fast Test Mode (Quick Testing)
```bash
python mcts.py --config mcts_config_fast_test.yaml --real-autogluon
```
- **Purpose**: Ultra-fast testing with real AutoGluon
- **Configuration**: mcts_config.yaml + mcts_config_fast_test.yaml overrides
- **Evaluator**: Real XGBoost on tiny dataset (1K samples)
- **Speed**: ~3 iterations in 30-60 seconds
- **Use Case**: Quick testing, CI/CD, rapid iteration

### Fast Real Mode (Validation)
```bash
python mcts.py --config mcts_config_fast_real.yaml --real-autogluon
```
- **Purpose**: Quick validation with real AutoGluon
- **Configuration**: mcts_config.yaml + mcts_config_fast_real.yaml overrides
- **Evaluator**: Real XGBoost on small dataset (5K samples)
- **Speed**: ~5 iterations in 2-5 minutes
- **Use Case**: Feature validation, rapid prototyping

### Production Mode (Full Discovery)
```bash
python mcts.py --config mcts_config.yaml
```
- **Purpose**: Complete feature discovery
- **Evaluator**: Full AutoGluon suite on complete data
- **Speed**: Hours to days for comprehensive exploration
- **Use Case**: Competition submissions, final models

## Detailed Configuration Breakdown

### Session Settings
| Setting | Mock | Fast Test | Fast Real | Production |
|---------|------|-----------|-----------|------------|
| max_iterations | 10 | 3 | 5 | 20-200 |
| max_runtime_hours | N/A | 0.25 | 1 | 12-48 |
| checkpoint_interval | N/A | 1 | 2 | 10-25 |

### AutoGluon Settings
| Setting | Mock | Fast Test | Fast Real | Production |
|---------|------|-----------|-----------|------------|
| included_model_types | N/A | ['XGB'] | ['XGB'] | ['XGB', 'GBM', 'CAT'] |
| enable_gpu | N/A | true | true | true |
| train_size | N/A | 0.02 | 0.1 | 0.8 |
| time_limit | N/A | 8s | 30s | 120s |
| presets | N/A | medium_quality_faster_train | medium_quality_faster_train | good_quality_faster_inference |

### Data Settings
| Setting | Mock | Fast Test | Fast Real | Production |
|---------|------|-----------|-----------|------------|
| small_dataset_size | 1000 | 1000 | 5000 | N/A |
| memory_limit_mb | N/A | 50 | 100 | 2048 |
| dtype_optimization | N/A | true | true | true |
| prefer_parquet | N/A | true | true | true |

### Resource Settings
| Setting | Mock | Fast Test | Fast Real | Production |
|---------|------|-----------|-----------|------------|
| max_memory_gb | N/A | 1 | 2 | 16 |
| max_cpu_cores | N/A | 2 | 4 | -1 (all) |
| max_disk_usage_gb | N/A | 5 | 10 | 50 |

## Performance Expectations

### Mock Mode
- **Total Time**: ~30 seconds
- **Iterations/min**: ~20
- **Memory Usage**: <50MB
- **CPU Usage**: Minimal
- **Output**: Synthetic scores for testing

### Fast Real Mode
- **Total Time**: 2-5 minutes
- **Iterations/min**: ~1-2
- **Memory Usage**: 100-200MB
- **CPU Usage**: 4 cores + GPU
- **Output**: Real MAP@3 scores on small data

### Production Mode
- **Total Time**: Hours to days
- **Iterations/hour**: 1-10 (depends on data size)
- **Memory Usage**: 2-16GB
- **CPU Usage**: All available cores + GPU
- **Output**: Competition-ready feature discoveries

## When to Use Each Mode

### Use Mock Mode When:
- ✅ Developing new features
- ✅ Testing configuration changes
- ✅ Debugging code issues
- ✅ Learning the system
- ✅ CI/CD testing

### Use Fast Real Mode When:
- ✅ Validating feature engineering ideas
- ✅ Quick competitive analysis
- ✅ Prototyping new approaches
- ✅ Testing on new datasets
- ✅ Daily development workflow

### Use Production Mode When:
- ✅ Final competition submissions
- ✅ Comprehensive feature discovery
- ✅ Research publications
- ✅ Production model deployment
- ✅ Maximum accuracy required

## Configuration Migration Path

### Development → Fast Real
```bash
# From:
python mcts.py --test-mode

# To:
python mcts.py --config mcts_config_fast_real.yaml --real-autogluon
```

### Fast Real → Production
```bash
# From:
python mcts.py --config mcts_config_fast_real.yaml --real-autogluon

# To:
python mcts.py --config mcts_config.yaml
```

## Common Configuration Patterns

### Memory-Constrained Environment
```yaml
resources:
  max_memory_gb: 1
data:
  small_dataset_size: 2000
  memory_limit_mb: 50
autogluon:
  train_size: 0.02
```

### GPU-Accelerated Fast Mode
```yaml
autogluon:
  included_model_types: ['XGB']
  enable_gpu: true
  train_size: 0.1
  time_limit: 30
```

### CPU-Only Mode
```yaml
autogluon:
  included_model_types: ['GBM']  # No GPU dependency
  enable_gpu: false
  train_size: 0.05
```

## Troubleshooting Configuration Issues

### "Out of Memory" Errors
1. Reduce `data.small_dataset_size`
2. Lower `data.memory_limit_mb`
3. Decrease `autogluon.train_size`
4. Reduce `resources.max_memory_gb`

### "Too Slow" Performance
1. Use `--test-mode` for development
2. Reduce `session.max_iterations`
3. Lower `autogluon.time_limit`
4. Use only `['XGB']` models

### "Poor Quality" Results
1. Increase `autogluon.train_size`
2. Add more model types
3. Increase `autogluon.time_limit`
4. Use production configuration

## Best Practices

1. **Start Small**: Always begin with mock mode
2. **Iterate Fast**: Use fast real mode for feature development
3. **Scale Gradually**: Move to production only when needed
4. **Monitor Resources**: Watch memory and CPU usage
5. **Use Session Management**: Resume interrupted sessions
6. **Cache Everything**: System automatically optimizes data loading

## Configuration Validation

```bash
# Validate configuration syntax
python mcts.py --config mcts_config_fast_real.yaml --validate-config

# Test configuration with minimal run
python mcts.py --config mcts_config_fast_real.yaml --real-autogluon
```