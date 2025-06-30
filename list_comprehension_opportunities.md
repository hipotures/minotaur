# List Comprehension Optimization Opportunities in Minotaur Codebase

## Summary
After analyzing the src/ directory, I've identified several loops that could be converted to list comprehensions for better performance and more Pythonic code. These optimizations focus on:
1. Loops that append to lists
2. Simple filtering loops
3. Transformation loops

## Identified Optimization Opportunities

### 1. **src/mcts_engine.py**

#### Loop 1: Line ~366-369
**Current Code:**
```python
new_children = []
for operation in operations_to_expand:
    child = node.add_child(operation)
    new_children.append(child)
```

**Optimized with List Comprehension:**
```python
new_children = [node.add_child(operation) for operation in operations_to_expand]
```

#### Loop 2: Line ~415-419
**Current Code:**
```python
evaluation_results = []
for node in nodes_to_evaluate:
    score, eval_time = self.simulation(node, evaluator, feature_space)
    node.evaluation_score = score
    evaluation_results.append((node, score, eval_time))
```

**Note:** This loop has side effects (setting `node.evaluation_score`), so it's not ideal for list comprehension. However, it could be refactored if needed.

### 2. **src/feature_space.py**

#### Loop 1: Line ~246-250
**Current Code:**
```python
available_ops = []
for op_name, operation in self.operations.items():
    # Check if operation can be applied
    if operation.can_apply(current_features):
        # Check if already applied in this path
        if op_name not in getattr(node, 'applied_operations', []):
            # Apply category filtering
            if not self.enabled_categories or operation.category in self.enabled_categories:
                # Apply category weighting
                weight = self.category_weights.get(operation.category, 1.0)
                if weight > 0:
                    available_ops.append(op_name)
```

**Optimized with List Comprehension:**
```python
available_ops = [
    op_name 
    for op_name, operation in self.operations.items()
    if operation.can_apply(current_features)
    and op_name not in getattr(node, 'applied_operations', [])
    and (not self.enabled_categories or operation.category in self.enabled_categories)
    and self.category_weights.get(operation.category, 1.0) > 0
]
```

#### Loop 2: Lines ~296-298
**Current Code:**
```python
for col in all_columns:
    if any(pattern in col for pattern in stat_patterns):
        selected_columns.append(col)
```

**Optimized with List Comprehension:**
```python
selected_columns.extend([
    col for col in all_columns 
    if any(pattern in col for pattern in stat_patterns)
])
```

#### Loop 3: Lines ~303-305
**Current Code:**
```python
for col in all_columns:
    if any(col.endswith(suffix) for suffix in poly_suffixes):
        selected_columns.append(col)
```

**Optimized with List Comprehension:**
```python
selected_columns.extend([
    col for col in all_columns 
    if any(col.endswith(suffix) for suffix in poly_suffixes)
])
```

#### Loop 4: Lines ~633-636
**Current Code:**
```python
text_cols = []
for col in df.select_dtypes(include=['object']).columns:
    if col not in [self.target_column, self.id_column] + self.ignore_columns:
        # Simple heuristic: if average string length > 10, consider it text
        if df[col].str.len().mean() > 10:
            text_cols.append(col)
```

**Optimized with List Comprehension:**
```python
text_cols = [
    col for col in df.select_dtypes(include=['object']).columns
    if col not in [self.target_column, self.id_column] + self.ignore_columns
    and df[col].str.len().mean() > 10
]
```

### 3. **src/duckdb_data_manager.py**

#### Loop 1: Lines ~265-267
**Current Code:**
```python
json_columns = []
for col in column_names:
    json_columns.append(f"'{col}': COALESCE(CAST(\"{col}\" AS VARCHAR), 'NULL')")
```

**Optimized with List Comprehension:**
```python
json_columns = [
    f"'{col}': COALESCE(CAST(\"{col}\" AS VARCHAR), 'NULL')"
    for col in column_names
]
```

#### Loop 2: Lines ~399-403
**Current Code:**
```python
json_fields = []
for col_name, csv_column in column_mapping.items():
    if csv_column and col_name != 'id':  # Skip id, it's handled separately
        # Store everything as VARCHAR in JSON to avoid type conflicts
        # Use COALESCE with CAST to handle NULLs and convert everything to string
        json_fields.append(f"'{col_name}': COALESCE(CAST(\"{csv_column}\" AS VARCHAR), 'Unknown')")
```

**Optimized with List Comprehension:**
```python
json_fields = [
    f"'{col_name}': COALESCE(CAST(\"{csv_column}\" AS VARCHAR), 'Unknown')"
    for col_name, csv_column in column_mapping.items()
    if csv_column and col_name != 'id'
]
```

#### Loop 3: Lines ~668-670
**Current Code:**
```python
select_fields = []
for col_name in self.column_mapping.keys():
    if col_name:
        select_fields.append(f"json_extract_string(data, '$.{col_name}') as \"{col_name}\"")
```

**Optimized with List Comprehension:**
```python
select_fields = [
    f"json_extract_string(data, '$.{col_name}') as \"{col_name}\""
    for col_name in self.column_mapping.keys()
    if col_name
]
```

#### Loop 4: Lines ~1089-1099
**Current Code:**
```python
cached_features = []
for row in results:
    cached_features.append({
        'feature_hash': row[0],
        'feature_name': row[1],
        'evaluation_score': row[2],
        'node_depth': row[3],
        'created_at': row[4]
    })
```

**Optimized with List Comprehension:**
```python
cached_features = [
    {
        'feature_hash': row[0],
        'feature_name': row[1],
        'evaluation_score': row[2],
        'node_depth': row[3],
        'created_at': row[4]
    }
    for row in results
]
```

#### Loop 5: Lines ~1231-1250 (columns_to_keep filtering)
**Current Code:**
```python
columns_to_keep = []
for col in df.columns:
    # Skip excluded columns
    if col in excluded_columns:
        continue
        
    # Always keep basic columns
    if col in basic_cols or col.lower() in [c.lower() for c in basic_cols]:
        columns_to_keep.append(col)
        continue
    
    # Check if it's a generic feature
    is_generic_feature = False
    
    # Statistical aggregations
    if any(pattern in col for pattern in ['_mean_by_', '_std_by_', '_dev_from_']):
        if generic_operations.get('statistical_aggregations', False):
            columns_to_keep.append(col)
        is_generic_feature = True
    # ... more conditions ...
```

**Note:** This loop is complex with multiple conditions and early continues. While it could be refactored, the current structure might be more readable.

### 4. **src/analytics.py**

#### Loop 1: Lines ~820-831 (recommendations building)
**Current Code:**
```python
recommendations = []
# Performance recommendations
if stats.get('best_score', 0) < 0.3:
    recommendations.append("🎯 Consider increasing exploration time or trying different feature operation combinations")

if stats.get('operations_per_minute', 0) < 5:
    recommendations.append("⚡ Performance could be improved - consider using smaller train_size for faster testing")

if stats.get('total_iterations', 0) < 50:
    recommendations.append("🔄 Consider running more iterations for better exploration coverage")
```

**Note:** This isn't a traditional loop with append, but a series of conditional appends. It's already fairly optimized.

### 5. **Additional Files with Similar Patterns**

Several other files in the manager/ subdirectory have similar patterns:
- `src/manager/modules/sessions/compare_command.py`
- `src/manager/modules/datasets/stats.py`
- `src/manager/modules/backup/list_command.py`
- `src/manager/repositories/feature_repository.py`

## Performance Benefits

Converting these loops to list comprehensions provides:

1. **Performance**: List comprehensions are generally 10-20% faster than equivalent for loops with append
2. **Memory Efficiency**: List comprehensions pre-allocate memory more efficiently
3. **Readability**: More concise and Pythonic code
4. **Functional Style**: Encourages immutable data patterns

## Implementation Priority

1. **High Priority** (Simple conversions with clear benefits):
   - `json_columns` building in duckdb_data_manager.py
   - `select_fields` building in duckdb_data_manager.py
   - Simple filtering loops in feature_space.py

2. **Medium Priority** (More complex but still beneficial):
   - `available_ops` filtering in feature_space.py
   - `cached_features` building in duckdb_data_manager.py

3. **Low Priority** (Complex logic that might lose readability):
   - Complex filtering with multiple conditions
   - Loops with side effects

## Testing Recommendations

Before implementing these optimizations:
1. Run existing tests to ensure they pass
2. Profile the current performance
3. Apply optimizations incrementally
4. Re-run tests after each change
5. Profile again to measure improvement

## Example Implementation

Here's a complete example of optimizing a function:

**Before:**
```python
def get_matching_columns(all_columns, patterns):
    matching = []
    for col in all_columns:
        if any(pattern in col for pattern in patterns):
            matching.append(col)
    return matching
```

**After:**
```python
def get_matching_columns(all_columns, patterns):
    return [col for col in all_columns if any(pattern in col for pattern in patterns)]
```