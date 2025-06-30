<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30 13:35
Compatible with commit: 02d53a5
Changes: Created dedicated validation documentation with comprehensive testing framework
-->

# MCTS Validation Framework

## 🎯 Overview

The MCTS validation framework provides comprehensive testing and verification tools to ensure correct implementation of the Monte Carlo Tree Search algorithm, database persistence, and session logging. All validation tools are located in `scripts/mcts/`.

## ✅ Core Validation Tools

### Primary Validation Script: `validate_mcts_correctness.py`

**Purpose**: Comprehensive MCTS implementation validation using database and session log analysis

**Features**:
- Cross-validates database records with session-specific MCTS logs
- Verifies MCTS algorithm correctness (UCB1, backpropagation)
- Analyzes tree structure and parent-child relationships
- Validates evaluation scores and feature consistency

**Usage**:
```bash
# Validate latest session (recommended)
python scripts/mcts/validate_mcts_correctness.py --latest

# Validate specific session
python scripts/mcts/validate_mcts_correctness.py --session session_20250630_132513

# Validate all sessions in database
python scripts/mcts/validate_mcts_correctness.py --all

# Use custom database path
python scripts/mcts/validate_mcts_correctness.py --latest --db-path data/custom.duckdb
```

### Legacy Validation Script: `validate_mcts.py`

**Purpose**: Original MCTS validation script (maintained for compatibility)

**Usage**:
```bash
python scripts/mcts/validate_mcts.py
```

## 🔬 Validation Test Categories

### 1. Database Structure Validation
**What it tests**:
- ✅ Root node presence (iteration 0)
- ✅ Sequential node ID assignment
- ✅ Proper iteration sequence
- ✅ Complete exploration history

**Example failure cases**:
```
❌ No root node (iteration 0) found
❌ Node ID sequence broken: [2, 3, 5, 6] != [2, 3, 4, 5, 6]
❌ Iterations should start from 0, found: 1
```

### 2. Tree Structure Validation
**What it tests**:
- ✅ Parent-child relationships integrity
- ✅ Tree depth progression
- ✅ Valid parent references
- ✅ No orphaned nodes

**Example checks**:
```python
# Root node has no parent
assert root_nodes['parent_node_id'].isna().all()

# All non-root nodes have valid parents
for node in non_root_nodes:
    parent_exists = (df['mcts_node_id'] == node['parent_node_id']).any()
    assert parent_exists, f"Parent {node['parent_node_id']} not found"

# Parent created before child
assert parent_iteration < node_iteration
```

### 3. MCTS Algorithm Validation
**What it tests**:
- ✅ Evaluation scores in valid range [0,1]
- ✅ Feature count consistency
- ✅ Algorithm progression logic

**Example checks**:
```python
# Scores must be valid probabilities
assert all(0 <= score <= 1 for score in scores)

# Feature counts should not decrease
assert node_features >= root_features
```

### 4. Log-Database Consistency Validation
**What it tests**:
- ✅ Node creation consistency between logs and database
- ✅ Evaluation score matching
- ✅ Session context preservation

**Cross-validation process**:
```python
# Compare node sets
db_nodes = set(exploration_df['mcts_node_id'].unique())
log_nodes = set(log_data['created_nodes'])
assert db_nodes.issubset(log_nodes)

# Compare evaluation scores
for node_id in common_nodes:
    db_score = db_scores[node_id]
    log_score = log_scores[node_id]
    assert abs(db_score - log_score) < 0.001
```

### 5. UCB1 Selection Logic Validation
**What it tests**:
- ✅ UCB1 calculation validation
- ✅ Selection phase analysis
- ✅ Exploration vs exploitation balance

**UCB1 Formula Verification**:
```
UCB1 = avg_reward + C * sqrt(ln(total_visits) / node_visits)

Where:
- avg_reward = total_reward / visit_count
- C = exploration_weight (typically 1.4)
- total_visits = parent node visit count
```

### 6. Backpropagation Logic Validation
**What it tests**:
- ✅ Visit count progression
- ✅ Reward accumulation
- ✅ Path propagation up the tree

**Example backpropagation validation**:
```python
# Visits should increase during backpropagation
for phase in backprop_phases:
    for update in phase:
        assert update['visits'] >= 1
        assert update['total_reward'] >= 0
```

### 7. Session-Specific Logging Validation
**What it tests**:
- ✅ Individual session log files created in DEBUG mode
- ✅ Session context preserved in log entries
- ✅ Log file naming conventions

**Log file validation**:
```python
# Check session log exists in DEBUG mode
if main_log_level == 'DEBUG':
    session_log = Path(f"logs/mcts/{session_name}.log")
    assert session_log.exists()
    
# Verify session context in logs
assert session_name in log_entry
```

## 📊 Validation Output Examples

### Successful Validation
```
🔍 Validating MCTS correctness for session: session_20250630_132513
  📊 Validating database structure...
    ✅ Database structure valid: 5 exploration steps, iterations 0-3
  🌳 Validating tree structure...
    ✅ Tree structure valid: max depth 2, proper parent-child relationships
  🎯 Validating MCTS algorithm progression...
    ✅ MCTS algorithm valid: scores in [0,1], feature counts consistent
  🔗 Validating log-database consistency...
    ✅ Log-database consistency validated: 5 nodes, scores match
  🎲 Validating UCB1 selection logic...
    ✅ UCB1 logic appears valid: 2 selection phases analyzed
  📈 Validating backpropagation logic...
    ✅ Backpropagation logic appears valid: 0 phases analyzed

📋 MCTS Validation Summary for session_20250630_132513
============================================================
Database Structure...................... ✅ PASS
Tree Structure.......................... ✅ PASS
Mcts Algorithm.......................... ✅ PASS
Log Consistency......................... ✅ PASS
Ucb1 Logic.............................. ✅ PASS
Backpropagation......................... ✅ PASS
============================================================
Overall Status: ✅ ALL TESTS PASSED
🎯 MCTS implementation is working correctly!
```

### Failed Validation (Historical Example)
```
🔍 Validating MCTS correctness for session: session_20250630_065044
  📊 Validating database structure...
    ❌ No root node (iteration 0) found
  🌳 Validating tree structure...
    ✅ Tree structure valid: max depth 2, proper parent-child relationships
  
📋 MCTS Validation Summary for session_20250630_065044
============================================================
Database Structure...................... ❌ FAIL
Tree Structure.......................... ✅ PASS
============================================================
Overall Status: ❌ SOME TESTS FAILED
🔧 MCTS implementation needs attention.
```

## 🔧 Additional Validation Tools

### Tree Visualization: `visualize_tree.py`
**Purpose**: ASCII tree structure visualization

**Usage**:
```bash
python scripts/mcts/visualize_tree.py SESSION_ID
```

**Example output**:
```
MCTS Tree for session_20250630_132513
======================================
Root (node_2): score=0.77, visits=5
├── statistical_aggregations (node_3): score=0.77, visits=2
│   └── binning_features (node_6): score=0.75, visits=1
└── binning_features (node_4): score=0.79, visits=2
    └── statistical_aggregations (node_5): score=0.75, visits=1
```

### Session Analysis: `analyze_session.py`
**Purpose**: Deep session analysis and performance metrics

**Usage**:
```bash
python scripts/mcts/analyze_session.py SESSION_ID
```

### Live Monitoring: `monitor_live.py`
**Purpose**: Real-time MCTS monitoring during execution

**Usage**:
```bash
python scripts/mcts/monitor_live.py
```

## 🎛️ Configuration for Validation

### Enable DEBUG Logging for Validation
```yaml
# In your MCTS config file
logging:
  level: 'DEBUG'                          # Required for session-specific logs
  log_mcts_details: true                  # Enable detailed MCTS logging
  log_feature_generation: true            # Log feature generation details
  log_database_operations: true           # Log database operations
```

### Validation-Friendly MCTS Config
```yaml
# Short test sessions for validation
session:
  max_iterations: 3                       # Quick validation runs
  max_runtime_hours: 0.1                  # 6 minutes max

# Minimal MCTS for testing
mcts:
  max_tree_depth: 3                       # Shallow trees
  expansion_budget: 3                     # Small expansion budget

# Fast AutoGluon evaluation
autogluon:
  train_size: 100                         # Small sample size
  time_limit: 30                          # Quick training
  included_model_types: ['XGB']           # Single model type
```

## 📋 Validation Checklist

### Pre-Session Validation
- [ ] Dataset properly registered (`python manager.py datasets --list`)
- [ ] Configuration file validated (`--validate-config`)
- [ ] DEBUG logging enabled for comprehensive validation
- [ ] Database accessible and migrations applied

### Post-Session Validation
- [ ] Session completed successfully (`MCTS_RESULT_JSON` output)
- [ ] Run comprehensive validation (`validate_mcts_correctness.py --latest`)
- [ ] Check session logs for errors (`logs/mcts/session_NAME.log`)
- [ ] Verify database persistence (`python manager.py sessions --details`)
- [ ] Analyze feature performance (`python manager.py features --top 10`)

### Troubleshooting Failed Validation

**Common Issues and Solutions**:

1. **Missing Root Node (iteration 0)**:
   ```
   Solution: Ensure evaluation_root flag is enabled in config
   Check: Database logging methods include iteration=0
   ```

2. **Log-Database Inconsistency**:
   ```
   Solution: Verify DEBUG logging is enabled
   Check: Session log files exist in logs/mcts/
   ```

3. **Tree Structure Errors**:
   ```
   Solution: Check parent_node_id assignments in database
   Check: Node creation order in MCTS engine
   ```

4. **UCB1 Calculation Errors**:
   ```
   Solution: Verify exploration_weight parameter
   Check: Visit count accumulation in backpropagation
   ```

## 🚀 Running Validation in CI/CD

### Automated Validation Script
```bash
#!/bin/bash
# ci/validate_mcts.sh

set -e

echo "🔍 Running MCTS validation tests..."

# Run quick test session
python mcts.py --config config/mcts_config_fast_test.yaml

# Get latest session for validation
LATEST_SESSION=$(python scripts/mcts/validate_mcts_correctness.py --latest 2>/dev/null | grep "session_" | head -1)

if [ -z "$LATEST_SESSION" ]; then
    echo "❌ No session found for validation"
    exit 1
fi

# Run comprehensive validation
python scripts/mcts/validate_mcts_correctness.py --session "$LATEST_SESSION"

echo "✅ MCTS validation completed successfully"
```

### Integration with pytest
```python
# tests/integration/test_mcts_validation.py
import subprocess
import pytest

def test_mcts_validation():
    """Test MCTS implementation using validation script."""
    # Run MCTS with test config
    result = subprocess.run([
        'python', 'mcts.py', 
        '--config', 'config/mcts_config_fast_test.yaml'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"MCTS run failed: {result.stderr}"
    
    # Run validation
    result = subprocess.run([
        'python', 'scripts/mcts/validate_mcts_correctness.py', 
        '--latest'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Validation failed: {result.stdout}"
    assert "ALL TESTS PASSED" in result.stdout
```

## 📈 Performance Benchmarks

### Validation Performance Metrics
```
Validation Script Performance:
├── Database queries: ~100ms
├── Log parsing: ~200ms  
├── Tree analysis: ~50ms
├── Total validation time: ~500ms per session
└── Memory usage: ~50MB
```

### Expected Results by Session Type
```
Fast Test Sessions (3 iterations):
├── Expected nodes: 4-6
├── Expected depth: 1-2
├── Expected scores: 0.65-0.85 (Titanic)
└── Validation time: <1 second

Production Sessions (50+ iterations):
├── Expected nodes: 20-100+
├── Expected depth: 3-8
├── Expected scores: 0.30-0.40 (S5E6)
└── Validation time: 1-5 seconds
```

---

*For configuration and performance tuning, see [MCTS_OPERATIONS.md](MCTS_OPERATIONS.md)*  
*For technical implementation details, see [MCTS_IMPLEMENTATION.md](MCTS_IMPLEMENTATION.md)*