# MCTS Implementation Analysis Report

**Date**: 2025-06-30  
**Analyst**: Claude Code  
**System**: Minotaur - MCTS-driven Feature Discovery

## Executive Summary

After comprehensive analysis of the MCTS (Monte Carlo Tree Search) implementation in the Minotaur system, I have identified several critical issues that prevent the algorithm from functioning correctly. While the core MCTS phases are implemented, there are fundamental problems with tree structure tracking, node relationships, and the exploration-exploitation balance.

### Key Findings:
1. **Tree Structure Not Maintained**: Parent-child relationships are not tracked in the database (`parent_node_id` is always NULL)
2. **Node IDs Not Assigned**: The `node_id` field in `FeatureNode` is never populated
3. **Features Not Updated**: `features_before` and `features_after` show identical values
4. **Limited Tree Growth**: The tree doesn't expand properly beyond the first level
5. **UCB1 Implementation**: While mathematically correct, it's not being used effectively due to tree structure issues

## 1. Algorithm Implementation Analysis

### 1.1 MCTS Phases Verification

The implementation in `src/mcts_engine.py` contains all four required MCTS phases:

#### Selection (Lines 339-358)
```python
def selection(self) -> FeatureNode:
    """Selection phase: Navigate from root to a leaf using UCB1."""
    current = self.root
    path = [current]
    
    while not current.is_leaf() and current.visit_count >= self.expansion_threshold:
        current = current.select_best_child(self.exploration_weight)
        if current is None:
            break
        path.append(current)
    
    logger.debug(f"Selected node at depth {current.depth} with {current.visit_count} visits")
    return current
```

#### Expansion (Lines 360-411)
```python
def expansion_original(self, node: FeatureNode, available_operations: List[str]) -> List[FeatureNode]:
    """Expansion phase: Add new child nodes for unexplored operations."""
    # ... implementation details ...
```

#### Simulation (Lines 413-466)
```python
def simulation_original(self, node: FeatureNode, evaluator, feature_space) -> Tuple[float, float]:
    """Simulation phase: Evaluate the node using AutoGluon."""
    # Uses AutoGluon evaluation instead of random rollouts
```

#### Backpropagation (Lines 468-485)
```python
def backpropagation_original(self, node: FeatureNode, reward: float, evaluation_time: float) -> None:
    """Backpropagation phase: Update all ancestors with the reward."""
    current = node
    nodes_updated = 0
    
    while current is not None:
        current.update_reward(reward, evaluation_time)
        nodes_updated += 1
        current = current.parent
```

### 1.2 Issues with Implementation

Despite having all phases, the implementation has critical flaws:

1. **Backward Compatibility Wrappers** (Lines 261-303): The code has test mode wrappers that interfere with production usage
2. **Node ID Never Set**: The `node_id` field is defined but never assigned a value
3. **Tree Structure Not Persisted**: While parent-child relationships exist in memory, they're not properly saved to the database

## 2. UCB1 Implementation Analysis

### 2.1 UCB1 Formula (Lines 87-115)

The UCB1 implementation is mathematically correct:

```python
def ucb1_score(self, exploration_weight: float = 1.4, parent_visits: int = None) -> float:
    """Calculate UCB1 score for node selection with caching."""
    if self.visit_count == 0:
        return float('inf')  # Unvisited nodes have highest priority
    
    # ... parameter handling ...
    
    # Calculate exploration term with cached logarithm
    log_parent = math.log(parent_visits) if parent_visits > 0 else 0
    exploration_term = exploration_weight * exploration_term = exploration_weight * math.sqrt(log_parent / self.visit_count)
    
    score = self.average_reward + exploration_term
```

### 2.2 Exploration-Exploitation Balance Issues

From database analysis, all nodes show the same UCB1 scores:
- Score 1: `1.1655764556207766`
- Score 2: `1.467405903555487`

This indicates that:
1. Nodes are not being revisited (all have `node_visits = 1`)
2. Parent visit counts are not being tracked properly
3. The tree is essentially doing breadth-first search without proper MCTS behavior

## 3. Database Structure Analysis

### 3.1 Schema Review

The `exploration_history` table has appropriate columns:
```sql
- id (BIGINT) - Primary key
- session_id (VARCHAR) - Session tracking
- iteration (INTEGER) - MCTS iteration number
- parent_node_id (BIGINT) - Parent node reference (ALWAYS NULL - BUG)
- operation_applied (VARCHAR) - Feature operation name
- features_before/after (JSON) - Feature lists
- evaluation_score (DOUBLE) - AutoGluon score
- mcts_ucb1_score (DOUBLE) - UCB1 value
- node_visits (INTEGER) - Visit count (ALWAYS 1 - BUG)
```

### 3.2 Data Persistence Issues

Query results show:
```sql
SELECT iteration, operation_applied, evaluation_score, parent_node_id 
FROM exploration_history 
WHERE session_id = '720d5947-85c0-4746-b9ac-1604bd5f7cf3';

-- Results show parent_node_id is always NULL
-- node_visits is always 1
-- features_before equals features_after
```

## 4. Logging System Analysis

### 4.1 Current Logging

The logging captures basic information:
```
2025-06-30 11:15:00,608 - src.mcts_engine - INFO - Initialized MCTS tree with 17 base features
2025-06-30 11:15:01,651 - src.mcts_engine - DEBUG - Selected node at depth 0 with 1 visits
2025-06-30 11:15:01,651 - src.feature_space - DEBUG - Found 2 available operations for node at depth 0
```

### 4.2 Missing Critical Information

The logs lack:
1. Node IDs and parent-child relationships
2. UCB1 score calculations and selection reasoning
3. Tree structure visualization
4. Backpropagation paths
5. Feature changes between nodes

## 5. Tree Growth and Branch Selection

### 5.1 Observed Patterns

From session analysis:
- Iteration 1: 2 nodes explored (root expansion)
- Iteration 2: 1 node explored
- Iteration 3: 1 node explored

This shows the tree is not growing properly. Expected behavior:
- Nodes should be revisited based on UCB1 scores
- High-performing branches should receive more visits
- Tree should grow deeper over iterations

### 5.2 Current Behavior

The system appears to be:
1. Expanding root node with all available operations
2. Evaluating each child once
3. Not revisiting or deepening promising branches
4. Not maintaining proper visit counts

## 6. Specific Issues Identified

### Critical Bugs:

1. **Node ID Assignment** (src/mcts_engine.py:57)
   ```python
   node_id: Optional[int] = None  # Never assigned
   ```

2. **Parent Node ID Tracking** (src/mcts_engine.py:536)
   ```python
   parent_node_id=node.parent.node_id if node.parent else None
   # Always None because node_id is never set
   ```

3. **Feature Lists Not Updated** (src/feature_space.py)
   - `features_before` and `features_after` are identical in database
   - Feature generation results not properly tracked

4. **Visit Count Not Accumulated**
   - Database shows all nodes with `node_visits = 1`
   - Suggests backpropagation not updating database records

5. **Tree Structure Lost**
   - In-memory tree exists but not persisted
   - Cannot resume sessions with proper tree state

## 7. Recommendations

### 7.1 Immediate Fixes Required

1. **Implement Node ID Assignment**
   ```python
   # In FeatureNode.__post_init__
   self.node_id = self._generate_node_id()
   
   # Add counter or use database sequence
   _node_counter = 0
   def _generate_node_id(self):
       FeatureNode._node_counter += 1
       return FeatureNode._node_counter
   ```

2. **Fix Database Logging**
   ```python
   # Update node visit counts in database
   def update_node_in_database(self, node, db):
       db.execute("""
           UPDATE exploration_history 
           SET node_visits = node_visits + 1,
               total_reward = ?
           WHERE node_id = ?
       """, (node.total_reward, node.node_id))
   ```

3. **Track Feature Changes**
   ```python
   # Properly capture features after operation
   features_after = feature_space.get_features_after_operation(
       node.parent.current_features, 
       operation
   )
   ```

4. **Implement Tree Persistence**
   - Add `mcts_tree_nodes` table to store tree structure
   - Save parent-child relationships
   - Enable session resumption with full tree state

### 7.2 Algorithmic Improvements

1. **Fix Selection Logic**
   - Ensure nodes are properly selected based on UCB1
   - Track and update visit counts correctly
   - Implement proper tree traversal

2. **Improve Logging**
   ```python
   logger.debug(f"UCB1 Selection: Node {node.node_id} "
               f"(score={ucb1_score:.4f}, "
               f"visits={node.visit_count}, "
               f"avg_reward={node.average_reward:.4f})")
   ```

3. **Add Tree Visualization**
   ```python
   def log_tree_structure(self):
       """Log current tree structure for debugging"""
       for node in self.traverse_tree():
           indent = "  " * node.depth
           logger.debug(f"{indent}Node {node.node_id}: "
                       f"op={node.operation_that_created_this}, "
                       f"visits={node.visit_count}, "
                       f"score={node.average_reward:.4f}")
   ```

### 7.3 Testing Recommendations

1. Create unit tests for:
   - Node ID assignment
   - Parent-child tracking
   - UCB1 score calculation with various visit counts
   - Tree growth over multiple iterations

2. Integration tests for:
   - Database persistence of tree structure
   - Session resumption with tree state
   - Feature list updates through operations

## 8. Conclusion

The MCTS implementation contains the correct algorithmic structure but fails in execution due to missing node tracking, improper database updates, and lack of tree persistence. The system currently performs a shallow exploration without the key benefits of MCTS: focused exploration of promising branches and proper balance between exploration and exploitation.

Fixing these issues will require:
1. Implementing proper node identification
2. Fixing database update logic
3. Ensuring feature changes are tracked
4. Adding tree structure persistence
5. Improving logging for debugging

With these fixes, the system should properly implement MCTS for effective feature discovery.

## 9. Debugging and Testing Guide

### 9.1 Session Identification

Sessions in Minotaur use two identifiers:

1. **Human-readable name**: `session_YYYYMMDD_HHMMSS` (e.g., `session_20250630_100151`)
2. **UUID**: Full UUID stored in database (e.g., `720d5947-85c0-4746-b9ac-1604bd5f7cf3`)

To find sessions:
```bash
# List all sessions with details
./manager.py sessions --list

# Show specific session details
./manager.py sessions --show session_20250630_100151

# Get UUID for a session from database
bin/duckdb data/minotaur.duckdb -c "SELECT id, name FROM sessions WHERE name LIKE '%20250630%';"
```

### 9.2 Log File Analysis

#### Log Locations
- **Main log**: `logs/minotaur.log` - All application logs
- **Database log**: `logs/db.log` - Database operations
- **Debug database log**: `logs/db_debug.log` - Detailed DB operations

#### Key Log Patterns to Search

```bash
# MCTS phase transitions
grep -E "Selection|Expansion|Simulation|Backpropagation" logs/minotaur.log

# Node selection details
grep -E "Selected node|UCB1|exploration" logs/minotaur.log

# Tree growth
grep -E "depth|children|expanded" logs/minotaur.log

# Best score updates
grep -E "New best score|improvement" logs/minotaur.log

# Session start/end
grep -E "session_[0-9]{8}_[0-9]{6}" logs/minotaur.log
```

#### Enable Debug Logging
```yaml
# In config file
logging:
  level: 'DEBUG'  # Shows detailed MCTS operations
```

### 9.3 Database Inspection Methods

#### Connect to Database
```bash
# Interactive DuckDB shell
bin/duckdb data/minotaur.duckdb

# Single query
bin/duckdb data/minotaur.duckdb -c "SELECT * FROM sessions ORDER BY created_at DESC LIMIT 5;"
```

#### Essential Debugging Queries

```sql
-- 1. Find session UUID by name
SELECT id, name, created_at, status 
FROM sessions 
WHERE name = 'session_20250630_100151';

-- 2. Check MCTS tree structure for a session
SELECT 
    id,
    iteration,
    operation_applied,
    parent_node_id,
    node_visits,
    evaluation_score,
    mcts_ucb1_score,
    LENGTH(features_after::varchar) - LENGTH(features_before::varchar) as feature_diff
FROM exploration_history
WHERE session_id = 'YOUR_SESSION_UUID'
ORDER BY iteration, id;

-- 3. Verify node visit accumulation
SELECT 
    operation_applied,
    COUNT(*) as times_applied,
    SUM(node_visits) as total_visits,
    AVG(evaluation_score) as avg_score,
    MAX(evaluation_score) as best_score
FROM exploration_history
WHERE session_id = 'YOUR_SESSION_UUID'
GROUP BY operation_applied
ORDER BY total_visits DESC;

-- 4. Check parent-child relationships
WITH tree AS (
    SELECT 
        id,
        parent_node_id,
        operation_applied,
        iteration,
        evaluation_score
    FROM exploration_history
    WHERE session_id = 'YOUR_SESSION_UUID'
)
SELECT 
    t1.id as child_id,
    t1.operation_applied as child_op,
    t2.id as parent_id,
    t2.operation_applied as parent_op
FROM tree t1
LEFT JOIN tree t2 ON t1.parent_node_id = t2.id
ORDER BY t1.iteration, t1.id;

-- 5. Analyze UCB1 score progression
SELECT 
    iteration,
    operation_applied,
    mcts_ucb1_score,
    node_visits,
    evaluation_score,
    ROW_NUMBER() OVER (PARTITION BY iteration ORDER BY mcts_ucb1_score DESC) as ucb_rank
FROM exploration_history
WHERE session_id = 'YOUR_SESSION_UUID'
ORDER BY iteration, ucb_rank;

-- 6. Feature generation verification
SELECT 
    operation_applied,
    JSON_ARRAY_LENGTH(features_before) as features_before_count,
    JSON_ARRAY_LENGTH(features_after) as features_after_count,
    JSON_ARRAY_LENGTH(features_after) - JSON_ARRAY_LENGTH(features_before) as new_features
FROM exploration_history
WHERE session_id = 'YOUR_SESSION_UUID';
```

### 9.4 Testing Correct MCTS Behavior

#### Test 1: Node Revisitation
After fixes, nodes should be revisited based on UCB1 scores:

```sql
-- Should show some operations with node_visits > 1
SELECT operation_applied, node_visits 
FROM exploration_history 
WHERE session_id = ? AND node_visits > 1;
```

#### Test 2: Tree Depth Growth
Tree should grow deeper over iterations:

```sql
-- Should show increasing max depth
SELECT iteration, MAX(LENGTH(operation_applied) - LENGTH(REPLACE(operation_applied, '->', ''))) as depth
FROM exploration_history
WHERE session_id = ?
GROUP BY iteration;
```

#### Test 3: Parent-Child Tracking
All non-root nodes should have parents:

```sql
-- Should return 0 (no orphans except root)
SELECT COUNT(*) as orphan_count
FROM exploration_history
WHERE session_id = ? 
  AND parent_node_id IS NULL 
  AND operation_applied != 'root';
```

#### Test 4: Feature List Changes
Features should change after operations:

```sql
-- Should show different counts
SELECT 
    operation_applied,
    features_before::varchar != features_after::varchar as features_changed
FROM exploration_history
WHERE session_id = ?;
```

### 9.5 Real-time Monitoring During Execution

#### Monitor MCTS Progress
```bash
# Watch log file in real-time
tail -f logs/minotaur.log | grep -E "iteration|score|depth|UCB"

# Monitor database updates
watch -n 2 "bin/duckdb data/minotaur.duckdb -c \"SELECT iteration, COUNT(*) as nodes, AVG(evaluation_score) as avg_score FROM exploration_history WHERE session_id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1) GROUP BY iteration;\""
```

#### Create Debug View
```sql
-- Create a view for easier debugging
CREATE OR REPLACE VIEW mcts_debug AS
SELECT 
    eh.iteration,
    eh.operation_applied,
    eh.parent_node_id,
    eh.node_visits,
    eh.evaluation_score,
    eh.mcts_ucb1_score,
    s.name as session_name,
    JSON_ARRAY_LENGTH(eh.features_after) - JSON_ARRAY_LENGTH(eh.features_before) as features_added
FROM exploration_history eh
JOIN sessions s ON eh.session_id = s.id;

-- Use the view
SELECT * FROM mcts_debug WHERE session_name = 'session_20250630_100151';
```

### 9.6 Validation After Fixes

Run this validation script after implementing fixes:

```python
# validation_test.py
import duckdb
import json

def validate_mcts_fixes(session_id):
    conn = duckdb.connect('data/minotaur.duckdb')
    
    # Test 1: Node IDs assigned
    result = conn.execute("""
        SELECT COUNT(*) as null_nodes 
        FROM exploration_history 
        WHERE session_id = ? AND id IS NULL
    """, [session_id]).fetchone()
    print(f"✓ Node IDs: {result[0] == 0}")
    
    # Test 2: Parent relationships
    result = conn.execute("""
        SELECT COUNT(*) as orphans 
        FROM exploration_history 
        WHERE session_id = ? 
        AND parent_node_id IS NULL 
        AND operation_applied != 'root'
    """, [session_id]).fetchone()
    print(f"✓ Parent tracking: {result[0] == 0}")
    
    # Test 3: Visit accumulation
    result = conn.execute("""
        SELECT MAX(node_visits) as max_visits 
        FROM exploration_history 
        WHERE session_id = ?
    """, [session_id]).fetchone()
    print(f"✓ Visit accumulation: {result[0] > 1}")
    
    # Test 4: Feature changes
    result = conn.execute("""
        SELECT COUNT(*) as unchanged 
        FROM exploration_history 
        WHERE session_id = ? 
        AND features_before::varchar = features_after::varchar
        AND operation_applied != 'root'
    """, [session_id]).fetchone()
    print(f"✓ Feature generation: {result[0] == 0}")
    
    conn.close()

# Run validation
validate_mcts_fixes('YOUR_SESSION_UUID')
```

### 9.7 Performance Metrics to Track

Monitor these metrics to ensure MCTS is working efficiently:

```sql
-- MCTS efficiency metrics
WITH session_stats AS (
    SELECT 
        session_id,
        MAX(iteration) as total_iterations,
        COUNT(DISTINCT operation_applied) as unique_operations,
        MAX(evaluation_score) as best_score,
        AVG(evaluation_score) as avg_score,
        SUM(node_visits) as total_visits,
        MAX(node_visits) as max_visits_single_node
    FROM exploration_history
    WHERE session_id = ?
    GROUP BY session_id
)
SELECT 
    total_iterations,
    unique_operations,
    best_score,
    avg_score,
    total_visits / total_iterations as avg_nodes_per_iteration,
    max_visits_single_node,
    CASE 
        WHEN max_visits_single_node > 1 THEN 'MCTS Working'
        ELSE 'MCTS Not Working - No Revisitation'
    END as status
FROM session_stats;
```

### 9.8 Common Issues and Solutions

| Symptom | Check | Solution |
|---------|-------|----------|
| All parent_node_id NULL | `SELECT COUNT(*) FROM exploration_history WHERE parent_node_id IS NOT NULL` | Implement node ID assignment |
| node_visits always 1 | `SELECT MAX(node_visits) FROM exploration_history WHERE session_id = ?` | Fix backpropagation to update existing records |
| Same UCB1 scores | `SELECT DISTINCT mcts_ucb1_score FROM exploration_history` | Ensure visit counts are tracked |
| No tree depth growth | Check iteration vs depth correlation | Fix selection logic to traverse tree |
| Features unchanged | Compare features_before/after | Fix feature generation tracking |

### 9.9 Interpreting MCTS Results

#### Understanding UCB1 Scores
```sql
-- Analyze UCB1 components
SELECT 
    operation_applied,
    node_visits,
    evaluation_score as exploitation_term,
    mcts_ucb1_score,
    mcts_ucb1_score - evaluation_score as exploration_term
FROM exploration_history
WHERE session_id = ?
ORDER BY mcts_ucb1_score DESC;
```

High exploration term = node is underexplored
High exploitation term = node has good performance

#### Tree Shape Analysis
```sql
-- Visualize tree shape
WITH RECURSIVE tree_structure AS (
    -- Base case: root nodes
    SELECT 
        id,
        operation_applied,
        parent_node_id,
        0 as depth,
        CAST(operation_applied AS VARCHAR) as path
    FROM exploration_history
    WHERE session_id = ? AND parent_node_id IS NULL
    
    UNION ALL
    
    -- Recursive case
    SELECT 
        e.id,
        e.operation_applied,
        e.parent_node_id,
        t.depth + 1,
        t.path || ' -> ' || e.operation_applied
    FROM exploration_history e
    JOIN tree_structure t ON e.parent_node_id = t.id
    WHERE e.session_id = ?
)
SELECT 
    depth,
    COUNT(*) as nodes_at_depth,
    STRING_AGG(operation_applied, ', ') as operations
FROM tree_structure
GROUP BY depth
ORDER BY depth;
```

### 9.10 Step-by-Step Debugging Process

1. **Start New Test Session**
```bash
# Run with debug config
python mcts.py --config config/mcts_debug_test.yaml

# Note the session name from output
# Example: session_20250630_111500
```

2. **Get Session UUID**
```bash
bin/duckdb data/minotaur.duckdb -c "SELECT id FROM sessions WHERE name = 'session_20250630_111500';"
# Copy the UUID for next steps
```

3. **Monitor in Real-Time**
```bash
# Terminal 1: Watch logs
tail -f logs/minotaur.log | grep -E "session_20250630_111500.*UCB|score|depth"

# Terminal 2: Watch database
watch -n 1 "bin/duckdb data/minotaur.duckdb -c \"SELECT iteration, COUNT(*) as nodes, MAX(node_visits) as max_visits FROM exploration_history WHERE session_id = 'UUID' GROUP BY iteration;\""
```

4. **Post-Run Analysis**
```bash
# Export session data
./manager.py sessions --export json --output-file session_analysis.json

# Run validation script
python validation_test.py UUID

# Generate tree visualization
bin/duckdb data/minotaur.duckdb -c "WITH tree AS (...) SELECT * FROM tree;" > tree_structure.txt
```

### 9.11 Expected Behavior After Fixes

#### Correct MCTS Pattern
```
Iteration 1: Explore 2-3 children of root
Iteration 2: Revisit best node from iteration 1, explore its children
Iteration 3: Mix of revisiting promising nodes and exploring new ones
...
Later iterations: Focus on best branches with occasional exploration
```

#### Database Should Show:
1. **Increasing node_visits** for promising branches
2. **Non-null parent_node_id** for all non-root nodes
3. **Different features_before/after** for each operation
4. **Varying UCB1 scores** reflecting visit counts
5. **Tree depth growth** in later iterations

#### Logs Should Show:
```
[DEBUG] Selected node at depth 2 with 3 visits
[DEBUG] UCB1 scores: node_5 (1.845), node_3 (1.623), node_7 (inf)
[DEBUG] Expanding node_7 with operation 'polynomial_features'
[DEBUG] Backpropagated reward 0.856 through 3 nodes
[INFO] New best score: 0.856 at iteration 5
```

## Appendix: Example Database Queries for Validation

```sql
-- Check tree structure
SELECT 
    iteration,
    operation_applied,
    parent_node_id,
    node_visits,
    evaluation_score,
    mcts_ucb1_score
FROM exploration_history
WHERE session_id = ?
ORDER BY iteration, id;

-- Verify node relationships
SELECT 
    COUNT(*) as orphan_nodes
FROM exploration_history
WHERE parent_node_id IS NULL 
    AND operation_applied != 'root';

-- Check visit accumulation
SELECT 
    operation_applied,
    SUM(node_visits) as total_visits,
    AVG(evaluation_score) as avg_score
FROM exploration_history
WHERE session_id = ?
GROUP BY operation_applied;
```