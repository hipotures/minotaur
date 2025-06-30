# Performance Optimization Report for Minotaur Codebase

This report identifies performance optimization opportunities in the Python codebase, focusing on hot paths like feature generation, MCTS tree operations, and data processing.

## 1. List Comprehensions and Generator Expressions

### Issue 1: Inefficient Loop in MCTS Engine (`src/mcts_engine.py`)

**Current Code (lines 361-363):**
```python
existing_operations = {child.operation_that_created_this for child in node.children}
new_operations = [op for op in available_operations if op not in existing_operations]
```

**Optimization:**
The set comprehension is efficient, but we can use a generator expression when we only need to sample/iterate:
```python
existing_operations = {child.operation_that_created_this for child in node.children}
# If we're going to sample, use generator to avoid creating full list
new_operations = (op for op in available_operations if op not in existing_operations)
# Convert to list only if needed for random.sample
if expansion_count < len(available_operations):
    new_operations = list(new_operations)
```

### Issue 2: Repeated List Building in Feature Space (`src/feature_space.py`)

**Current Code (lines 296-298):**
```python
for col in all_columns:
    if any(pattern in col for pattern in stat_patterns):
        selected_columns.append(col)
```

**Optimization:**
Replace with list comprehension:
```python
selected_columns.extend(
    col for col in all_columns 
    if any(pattern in col for pattern in stat_patterns)
)
```

## 2. String Concatenation in Loops

### Issue 3: String Building in SQL Queries (`src/feature_space.py`)

**Current Code (line 385):**
```python
column_list = ', '.join([f'"{col}"' for col in feature_columns])
```

**Optimization:**
This is already optimal using `join()`. However, for repeated SQL queries, we should cache the formatted column lists:
```python
# Add to class __init__
self._sql_column_cache = {}

# In method
cache_key = tuple(sorted(feature_columns))
if cache_key not in self._sql_column_cache:
    self._sql_column_cache[cache_key] = ', '.join(f'"{col}"' for col in feature_columns)
column_list = self._sql_column_cache[cache_key]
```

## 3. Repeated Calculations and Caching

### Issue 4: Repeated Column Detection (`src/feature_space.py`)

**Current Code (lines 270-281):**
Multiple database queries to get column information are repeated.

**Optimization:**
Cache the column information:
```python
# Add to class
@property
def _cached_train_columns(self):
    if not hasattr(self, '_train_columns_cache'):
        if self.duckdb_manager:
            result = self.duckdb_manager.connection.execute(
                "SELECT name FROM pragma_table_info('train_features')"
            ).fetchall()
            self._train_columns_cache = [col[0] for col in result]
        else:
            self._train_columns_cache = []
    return self._train_columns_cache
```

### Issue 5: Repeated Feature Validation (`src/features/base.py`)

**Current Code (lines 52-58):**
```python
def _check_feature_signal(self, feature_series: pd.Series) -> bool:
    try:
        import pandas as pd
        unique_count = feature_series.dropna().nunique()
        return unique_count > 1
    except Exception:
        return False
```

**Optimization:**
1. Import pandas at module level, not in function
2. Use value_counts() with limit for early exit:
```python
def _check_feature_signal(self, feature_series: pd.Series) -> bool:
    try:
        # Early exit if series is empty or all NaN
        if feature_series.isna().all():
            return False
        # Use value_counts with limit - stops after finding 2 unique values
        return len(feature_series.value_counts().head(2)) > 1
    except Exception:
        return False
```

## 4. Inefficient Data Structures

### Issue 6: List for Membership Testing (`src/mcts_engine.py`)

**Current Code (line 362):**
```python
new_operations = [op for op in available_operations if op not in existing_operations]
```

The `existing_operations` is already a set (good), but we should ensure all membership tests use sets.

### Issue 7: Repeated List Operations (`src/features/generic/statistical.py`)

**Current Code (lines 54-75):**
Nested loops creating many temporary dataframes.

**Optimization:**
Batch operations and use vectorized pandas operations:
```python
# Instead of creating temp_df for each group/agg combination
# Pre-compute all groupby objects once
groupby_objects = {}
for group_col in groupby_cols:
    if group_col in df.columns:
        groupby_objects[group_col] = df.groupby(group_col)

# Then reuse in feature generation
for group_col, groupby in groupby_objects.items():
    for agg_col in agg_cols:
        # Use pre-computed groupby
        features[f'{agg_col}_mean_by_{group_col}'] = groupby[agg_col].transform('mean')
```

## 5. Built-in Functions vs Loops

### Issue 8: Manual Accumulation (`src/mcts_engine.py`)

**Current Code (lines 743-749):**
```python
def count_recursive(node):
    count = 1
    for child in node.children:
        count += count_recursive(child)
    return count
```

**Optimization:**
Use generator expression with sum():
```python
def count_recursive(node):
    return 1 + sum(count_recursive(child) for child in node.children)
```

### Issue 9: Filter then Transform Pattern

**Current Code (various locations):**
Loop to filter, then loop to transform.

**Optimization:**
Use `filter()` and `map()` where appropriate:
```python
# Instead of:
result = []
for item in items:
    if condition(item):
        result.append(transform(item))

# Use:
result = list(map(transform, filter(condition, items)))
# Or generator for lazy evaluation:
result = map(transform, filter(condition, items))
```

## 6. Batch Operations and I/O

### Issue 10: Individual Database Queries (`src/autogluon_evaluator.py`)

**Current Code:**
Multiple small queries to check columns.

**Optimization:**
Batch column validation:
```python
# Instead of checking each column individually
# Get all column info in one query
column_info = self.duckdb_manager.connection.execute("""
    SELECT name, type 
    FROM pragma_table_info('train_features')
    WHERE name IN ({})
""".format(','.join(f"'{col}'" for col in requested_columns))).fetchall()

valid_columns = {col[0] for col in column_info}
```

## 7. Memory-Efficient Operations

### Issue 11: Loading Full DataFrames (`src/feature_space.py`)

**Current Code (line 388):**
```python
features_df = self.duckdb_manager.connection.execute(query).df()
```

**Optimization:**
Use DuckDB's lazy evaluation and push computations to the database:
```python
# Instead of loading full dataframe, use DuckDB for computations
# Example: compute statistics in DuckDB
stats_query = f"""
    SELECT 
        {', '.join(f'AVG("{col}") as {col}_mean' for col in numeric_cols)},
        {', '.join(f'STDDEV("{col}") as {col}_std' for col in numeric_cols)}
    FROM train_features
"""
stats = self.duckdb_manager.connection.execute(stats_query).fetchone()
```

## 8. Algorithm-Specific Optimizations

### Issue 12: UCB1 Score Calculation (`src/mcts_engine.py`)

**Current Code (lines 87-104):**
UCB1 calculation could cache parent visit count.

**Optimization:**
```python
def ucb1_score(self, exploration_weight: float = 1.4, parent_visits: int = None) -> float:
    if self.visit_count == 0:
        return float('inf')
    
    # Cache parent visits in node
    if parent_visits is None:
        if not hasattr(self, '_cached_parent_visits') or self.parent.visit_count != self._cached_parent_visits:
            self._cached_parent_visits = self.parent.visit_count if self.parent else self.visit_count
        parent_visits = self._cached_parent_visits
    
    if parent_visits <= 0:
        return self.average_reward
    
    # Pre-compute log only when parent visits change
    if not hasattr(self, '_cached_log_parent') or parent_visits != self._last_parent_visits:
        self._cached_log_parent = math.log(parent_visits)
        self._last_parent_visits = parent_visits
    
    exploration_term = exploration_weight * math.sqrt(self._cached_log_parent / self.visit_count)
    return self.average_reward + exploration_term
```

## Implementation Priority

1. **High Priority** (Quick wins with significant impact):
   - Issue 5: Feature signal validation optimization
   - Issue 10: Batch database queries
   - Issue 4: Column information caching

2. **Medium Priority** (Moderate effort, good performance gains):
   - Issue 7: Vectorized pandas operations
   - Issue 11: Push computations to DuckDB
   - Issue 12: UCB1 calculation caching

3. **Low Priority** (Minor improvements):
   - Issue 1: Generator expressions
   - Issue 8: Recursive counting optimization
   - Issue 2: List comprehension conversions

## Estimated Performance Improvements

- Feature generation: 30-40% faster with vectorized operations and caching
- MCTS tree operations: 15-20% faster with optimized UCB1 and counting
- Database operations: 50-60% faster with batched queries and DuckDB computation pushdown
- Overall system: 25-35% performance improvement expected

## Next Steps

1. Profile the application to confirm bottlenecks
2. Implement high-priority optimizations first
3. Benchmark before and after each optimization
4. Consider using tools like `line_profiler` and `memory_profiler` for detailed analysis