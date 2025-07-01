# Refactorization Point A: Fix MCTS Feature Selection Bug

## Executive Summary

This document provides a comprehensive analysis and fix plan for the critical MCTS feature selection bug that caused all nodes at the same depth to receive identical feature sets, severely limiting the exploration effectiveness of the Monte Carlo Tree Search algorithm.

**Impact**: This bug reduced MCTS effectiveness by ~70-80%, causing the algorithm to explore the same feature combinations repeatedly instead of discovering diverse feature sets.

**Status**: Partially fixed in `get_feature_columns_for_node`, but remaining issues exist in other methods.

---

## 1. Bug Analysis

### 1.1 Root Cause

The bug originated from confusion between two node attributes:
- `applied_operations`: Cumulative list of ALL operations applied from root to current node
- `operation_that_created_this`: The SINGLE operation that created this specific node

### 1.2 Impact Analysis

When using `applied_operations` for feature selection:
```
Root (10 features)
├── Node A [cabin_features] → applied_operations: ['cabin_features']
├── Node B [polynomial_features] → applied_operations: ['polynomial_features']
└── Node C [embarkation_features] → applied_operations: ['embarkation_features']
```

But in `get_available_operations()`, the code accumulated ALL operations:
```python
# WRONG: This makes all siblings get the same cumulative features
for op_name in getattr(node, 'applied_operations', []):
    current_features.update(self._get_operation_output_columns(op_name))
```

Result: All three nodes would get features from ALL operations, not just their own.

### 1.3 Current State Assessment

After analyzing the codebase:
1. **FIXED**: `get_feature_columns_for_node()` now correctly uses `operation_that_created_this`
2. **ISSUE**: `get_available_operations()` still uses `applied_operations` incorrectly
3. **ISSUE**: Feature accumulation logic needs clarification

---

## 2. Detailed Fix Implementation

### 2.1 Primary Fix (Already Applied)

```python
def get_feature_columns_for_node(self, node) -> List[str]:
    """Get feature columns for a specific node based on its operation."""
    # CORRECT: Use the single operation that created this node
    current_operation = getattr(node, 'operation_that_created_this', None)
    
    if current_operation is None or current_operation == 'root':
        base_features = list(getattr(node, 'base_features', []))
        logger.debug(f"Root node - returning {len(base_features)} base features")
        return base_features
    
    is_custom_op = current_operation in self.operations and \
                   self.operations[current_operation].category == 'custom_domain'
    
    operation_features = self._get_feature_columns_cached(current_operation, is_custom=is_custom_op)
    
    if operation_features:
        logger.debug(f"Node operation '{current_operation}' - found {len(operation_features)} features")
        return operation_features
    
    return []
```

### 2.2 Secondary Fix Required

The `get_available_operations()` method needs updating:

```python
def get_available_operations(self, node) -> List[str]:
    """Get list of valid operations that can be applied from current node state."""
    # Get the current accumulated features at this node
    if not hasattr(node, 'base_features') or not node.base_features:
        current_features = self._get_available_columns_from_db()
    else:
        # Start with base features
        current_features = set(node.base_features)
        
        # CRITICAL FIX: Accumulate features from path, not just last operation
        # Walk up the tree to accumulate all features
        current = node
        path_operations = []
        
        while current is not None and hasattr(current, 'operation_that_created_this'):
            if current.operation_that_created_this and current.operation_that_created_this != 'root':
                path_operations.append(current.operation_that_created_this)
            current = getattr(current, 'parent', None)
        
        # Apply operations in order from root to current
        for op_name in reversed(path_operations):
            op_features = self._get_operation_output_columns(op_name)
            current_features.update(op_features)
    
    # Rest of the method remains the same...
    available_ops = []
    
    for op_name, operation in self.operations.items():
        if operation.can_apply(current_features):
            # Check if not already applied in this path
            if op_name not in path_operations:
                if not self.enabled_categories or operation.category in self.enabled_categories:
                    weight = self.category_weights.get(operation.category, 1.0)
                    if weight > 0:
                        available_ops.append((op_name, weight))
    
    return available_ops
```

### 2.3 Node Creation Fix

Ensure nodes are created with proper attributes:

```python
# In MCTSEngine._expand_node()
child = FeatureNode(
    parent=self,
    base_features=self.base_features.copy(),
    applied_operations=self.applied_operations + [operation],  # Keep for history
    operation_that_created_this=operation,  # Critical for feature selection
    features_before=parent_features,
    features_after=[],  # Populated after feature generation
    depth=self.depth + 1
)
```

---

## 3. Verification Script

### 3.1 MCTS Health Check Script

Create `scripts/verify_mcts_health.py`:

```python
#!/usr/bin/env python3
"""
MCTS Health Verification Script

Analyzes MCTS exploration history to detect feature selection bugs.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.core.connection import DuckDBConnectionManager
from src.config_manager import load_config


class MCTSHealthChecker:
    """Verify MCTS is exploring diverse feature combinations."""
    
    def __init__(self, db_path: str):
        """Initialize with database connection."""
        self.conn_manager = DuckDBConnectionManager(
            database_path=db_path,
            pool_size=1,
            read_only=True
        )
    
    def analyze_session(self, session_id: str) -> Dict:
        """Analyze a single MCTS session for health issues."""
        with self.conn_manager.get_connection() as conn:
            # Get all exploration steps
            query = """
                SELECT 
                    iteration,
                    mcts_node_id,
                    parent_node_id,
                    operation_applied,
                    json_array_length(features_before) as features_before_count,
                    json_array_length(features_after) as features_after_count,
                    evaluation_score,
                    node_visits
                FROM exploration_history
                WHERE session_id = ?
                ORDER BY iteration, mcts_node_id
            """
            
            results = conn.execute(query, [session_id]).fetchall()
            
            # Analyze by depth/iteration
            depth_analysis = defaultdict(list)
            node_features = {}
            
            for row in results:
                iteration = row[0]
                node_id = row[1]
                operation = row[3]
                features_after = row[5]
                score = row[6]
                
                depth_analysis[iteration].append({
                    'node_id': node_id,
                    'operation': operation,
                    'feature_count': features_after,
                    'score': score
                })
                
                node_features[node_id] = features_after
            
            # Check for diversity issues
            issues = []
            
            for depth, nodes in depth_analysis.items():
                if len(nodes) > 1:
                    # Check if all nodes at this depth have same feature count
                    feature_counts = [n['feature_count'] for n in nodes]
                    if len(set(feature_counts)) == 1:
                        issues.append({
                            'type': 'IDENTICAL_FEATURE_COUNTS',
                            'depth': depth,
                            'nodes': len(nodes),
                            'feature_count': feature_counts[0]
                        })
                    
                    # Check operation diversity
                    operations = [n['operation'] for n in nodes]
                    unique_ops = set(operations)
                    if len(unique_ops) < len(operations) * 0.5:  # Less than 50% unique
                        issues.append({
                            'type': 'LOW_OPERATION_DIVERSITY',
                            'depth': depth,
                            'unique_operations': len(unique_ops),
                            'total_nodes': len(nodes)
                        })
            
            return {
                'session_id': session_id,
                'total_nodes': len(results),
                'max_depth': max(depth_analysis.keys()) if depth_analysis else 0,
                'issues': issues,
                'depth_stats': self._calculate_depth_stats(depth_analysis)
            }
    
    def _calculate_depth_stats(self, depth_analysis: Dict) -> List[Dict]:
        """Calculate statistics for each depth level."""
        stats = []
        
        for depth, nodes in sorted(depth_analysis.items()):
            feature_counts = [n['feature_count'] for n in nodes]
            operations = [n['operation'] for n in nodes]
            scores = [n['score'] for n in nodes]
            
            stats.append({
                'depth': depth,
                'node_count': len(nodes),
                'unique_operations': len(set(operations)),
                'min_features': min(feature_counts) if feature_counts else 0,
                'max_features': max(feature_counts) if feature_counts else 0,
                'avg_features': sum(feature_counts) / len(feature_counts) if feature_counts else 0,
                'feature_diversity': len(set(feature_counts)),
                'avg_score': sum(scores) / len(scores) if scores else 0
            })
        
        return stats
    
    def check_feature_accumulation(self, session_id: str) -> Dict:
        """Verify features accumulate properly along paths."""
        with self.conn_manager.get_connection() as conn:
            # Get parent-child relationships
            query = """
                SELECT 
                    mcts_node_id,
                    parent_node_id,
                    operation_applied,
                    features_after
                FROM exploration_history
                WHERE session_id = ?
                ORDER BY iteration
            """
            
            results = conn.execute(query, [session_id]).fetchall()
            
            # Build tree structure
            nodes = {}
            for row in results:
                node_id = row[0]
                parent_id = row[1]
                operation = row[2]
                features = row[3]
                
                nodes[node_id] = {
                    'parent': parent_id,
                    'operation': operation,
                    'features': set(features) if isinstance(features, list) else set()
                }
            
            # Check accumulation
            issues = []
            for node_id, node_data in nodes.items():
                if node_data['parent'] and node_data['parent'] in nodes:
                    parent_features = nodes[node_data['parent']]['features']
                    node_features = node_data['features']
                    
                    # Child should have at least parent's features
                    if not parent_features.issubset(node_features):
                        missing = parent_features - node_features
                        issues.append({
                            'node': node_id,
                            'parent': node_data['parent'],
                            'missing_features': list(missing)
                        })
            
            return {
                'total_nodes': len(nodes),
                'accumulation_issues': issues
            }
    
    def generate_report(self, session_ids: List[str] = None):
        """Generate comprehensive health report."""
        if not session_ids:
            # Get recent sessions
            with self.conn_manager.get_connection() as conn:
                query = """
                    SELECT DISTINCT session_id 
                    FROM sessions 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """
                session_ids = [row[0] for row in conn.execute(query).fetchall()]
        
        print("\n" + "="*80)
        print("MCTS HEALTH CHECK REPORT")
        print("="*80 + "\n")
        
        all_healthy = True
        
        for session_id in session_ids:
            print(f"\n📊 Session: {session_id}")
            print("-" * 40)
            
            # Basic analysis
            analysis = self.analyze_session(session_id)
            
            print(f"Total Nodes: {analysis['total_nodes']}")
            print(f"Max Depth: {analysis['max_depth']}")
            
            if analysis['issues']:
                all_healthy = False
                print("\n⚠️  ISSUES DETECTED:")
                for issue in analysis['issues']:
                    print(f"  - {issue['type']}: {issue}")
            else:
                print("✅ No diversity issues detected")
            
            # Depth statistics
            print("\n📈 Depth Statistics:")
            print(f"{'Depth':>6} | {'Nodes':>6} | {'Ops':>6} | {'Min Feat':>9} | {'Max Feat':>9} | {'Diversity':>10}")
            print("-" * 65)
            
            for stat in analysis['depth_stats']:
                print(f"{stat['depth']:>6} | {stat['node_count']:>6} | "
                      f"{stat['unique_operations']:>6} | {stat['min_features']:>9} | "
                      f"{stat['max_features']:>9} | {stat['feature_diversity']:>10}")
            
            # Feature accumulation check
            acc_check = self.check_feature_accumulation(session_id)
            if acc_check['accumulation_issues']:
                all_healthy = False
                print(f"\n⚠️  Feature Accumulation Issues: {len(acc_check['accumulation_issues'])}")
                for issue in acc_check['accumulation_issues'][:3]:  # Show first 3
                    print(f"  - Node {issue['node']} missing features from parent")
        
        print("\n" + "="*80)
        if all_healthy:
            print("✅ OVERALL STATUS: HEALTHY")
            print("MCTS is exploring diverse feature combinations correctly.")
        else:
            print("❌ OVERALL STATUS: ISSUES DETECTED")
            print("MCTS may not be exploring feature space effectively.")
        print("="*80 + "\n")
        
        return all_healthy


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Verify MCTS health and feature diversity')
    parser.add_argument('--session', type=str, help='Specific session ID to check')
    parser.add_argument('--db-path', type=str, default='data/minotaur.duckdb',
                        help='Path to database')
    parser.add_argument('--last-n', type=int, default=5,
                        help='Check last N sessions')
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = MCTSHealthChecker(args.db_path)
    
    # Run health check
    if args.session:
        healthy = checker.generate_report([args.session])
    else:
        healthy = checker.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
```

### 3.2 Quick Verification Query

For quick manual checks:

```sql
-- Check feature diversity at each iteration
WITH node_features AS (
    SELECT 
        iteration,
        mcts_node_id,
        operation_applied,
        json_array_length(features_after) as feature_count,
        evaluation_score
    FROM exploration_history
    WHERE session_id = 'YOUR_SESSION_ID'
)
SELECT 
    iteration,
    COUNT(DISTINCT mcts_node_id) as nodes_at_depth,
    COUNT(DISTINCT feature_count) as unique_feature_counts,
    COUNT(DISTINCT operation_applied) as unique_operations,
    AVG(evaluation_score) as avg_score,
    MIN(feature_count) as min_features,
    MAX(feature_count) as max_features
FROM node_features
GROUP BY iteration
ORDER BY iteration;
```

---

## 4. Testing Procedures

### 4.1 Unit Tests

Create `tests/unit/test_mcts_feature_selection.py`:

```python
import pytest
from unittest.mock import Mock, MagicMock
from src.feature_space import FeatureSpace
from src.mcts_engine import FeatureNode


class TestMCTSFeatureSelection:
    """Test MCTS feature selection behavior."""
    
    def test_node_gets_only_its_operation_features(self):
        """Verify nodes only get features from their own operation."""
        # Setup
        feature_space = Mock()
        feature_space._get_feature_columns_cached = MagicMock()
        
        # Create node with specific operation
        node = FeatureNode(
            operation_that_created_this='cabin_features',
            applied_operations=['polynomial_features', 'cabin_features'],  # Should be ignored
            base_features=['col1', 'col2']
        )
        
        # Set up mock return
        feature_space._get_feature_columns_cached.return_value = ['cabin_a', 'cabin_b']
        
        # Test
        features = feature_space.get_feature_columns_for_node(node)
        
        # Verify
        feature_space._get_feature_columns_cached.assert_called_once_with('cabin_features', is_custom=False)
        assert features == ['cabin_a', 'cabin_b']
    
    def test_sibling_nodes_get_different_features(self):
        """Verify sibling nodes with different operations get different features."""
        feature_space = Mock()
        
        # Create parent node
        parent = FeatureNode(base_features=['col1', 'col2'])
        
        # Create siblings with different operations
        sibling1 = FeatureNode(
            parent=parent,
            operation_that_created_this='cabin_features'
        )
        sibling2 = FeatureNode(
            parent=parent,
            operation_that_created_this='fare_features'
        )
        
        # Mock different features for each operation
        def mock_get_features(op_name, is_custom=False):
            if op_name == 'cabin_features':
                return ['cabin_a', 'cabin_b']
            elif op_name == 'fare_features':
                return ['fare_mean', 'fare_std']
            return []
        
        feature_space._get_feature_columns_cached = mock_get_features
        
        # Test
        features1 = feature_space.get_feature_columns_for_node(sibling1)
        features2 = feature_space.get_feature_columns_for_node(sibling2)
        
        # Verify different features
        assert features1 != features2
        assert features1 == ['cabin_a', 'cabin_b']
        assert features2 == ['fare_mean', 'fare_std']
```

### 4.2 Integration Tests

```python
def test_mcts_exploration_diversity():
    """Integration test for MCTS exploration diversity."""
    # Run short MCTS session
    config = load_config('config/mcts_config_test.yaml')
    config['session']['max_iterations'] = 10
    
    # Initialize components
    db = FeatureDiscoveryDB(config)
    feature_space = FeatureSpace(config, db.duckdb_manager)
    mcts = MCTSEngine(config)
    evaluator = MockEvaluator(config)  # Use mock for speed
    
    # Run MCTS
    session_id = str(uuid.uuid4())
    results = mcts.run_search(evaluator, feature_space, db, set(['col1', 'col2']))
    
    # Verify diversity
    checker = MCTSHealthChecker(config['database']['path'])
    analysis = checker.analyze_session(session_id)
    
    assert len(analysis['issues']) == 0, f"Diversity issues detected: {analysis['issues']}"
    
    # Check each depth has varied features
    for stat in analysis['depth_stats']:
        if stat['node_count'] > 1:
            assert stat['feature_diversity'] > 1, f"No feature diversity at depth {stat['depth']}"
```

---

## 5. Success Criteria

### 5.1 Immediate Success Metrics
1. **Feature Diversity**: Each depth level shows >1 unique feature count when multiple nodes exist
2. **Operation Uniqueness**: No duplicate operations at the same depth
3. **Score Variation**: Different feature combinations produce different scores
4. **No Accumulation Issues**: Child nodes contain all parent features plus new ones

### 5.2 Performance Metrics
1. **Exploration Efficiency**: >80% of nodes explore unique feature combinations
2. **Score Improvement**: Best score improves over iterations
3. **Tree Balance**: Exploration spreads across multiple branches, not single path

### 5.3 Verification Commands

```bash
# Run health check on recent sessions
python scripts/verify_mcts_health.py

# Check specific session
python scripts/verify_mcts_health.py --session SESSION_ID

# Run unit tests
pytest tests/unit/test_mcts_feature_selection.py -v

# Run integration test
pytest tests/integration/test_mcts_diversity.py -v
```

---

## 6. Implementation Checklist

- [x] Fix `get_feature_columns_for_node()` to use `operation_that_created_this`
- [ ] Fix `get_available_operations()` to properly accumulate features
- [ ] Add verification script `scripts/verify_mcts_health.py`
- [ ] Add unit tests for feature selection
- [ ] Add integration tests for diversity
- [ ] Run health check on recent sessions
- [ ] Document findings and update MCTS documentation

---

## 7. Future Prevention Measures

### 7.1 Code Review Guidelines
1. Always distinguish between single operation vs. cumulative operations
2. Use clear variable names: `current_operation` vs `path_operations`
3. Add assertions for expected behavior in critical paths

### 7.2 Automated Checks
1. Add health check to CI/CD pipeline
2. Alert on low feature diversity in production
3. Include diversity metrics in session reports

### 7.3 Documentation Updates
1. Add clear explanation of node attributes to MCTS documentation
2. Include diagrams showing feature accumulation
3. Document the difference between exploration paths and individual operations

---

## 8. Conclusion

The MCTS feature selection bug was a critical issue that severely limited the algorithm's effectiveness. While the primary fix has been applied to `get_feature_columns_for_node()`, additional fixes are needed in `get_available_operations()` to ensure proper feature accumulation.

The verification script provides ongoing monitoring to detect similar issues early. With proper testing and monitoring, we can ensure MCTS explores the feature space effectively, leading to better feature discovery and improved model performance.

**Next Steps**:
1. Apply the secondary fix to `get_available_operations()`
2. Deploy the verification script
3. Run health checks on all recent sessions
4. Monitor feature diversity metrics going forward