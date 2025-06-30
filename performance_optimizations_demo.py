#!/usr/bin/env python3
"""
Performance Optimization Demonstrations

This script shows before/after examples of the performance optimizations
identified in the codebase analysis.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Set, Dict, Any
import functools


def timing_decorator(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}: {end - start:.4f} seconds")
        return result
    return wrapper


# 1. List Comprehensions vs Loops
print("1. LIST COMPREHENSIONS VS LOOPS")
print("-" * 50)

@timing_decorator
def filter_with_loop(items: List[str], patterns: List[str]) -> List[str]:
    """Original: Using loop with append"""
    result = []
    for item in items:
        if any(pattern in item for pattern in patterns):
            result.append(item)
    return result

@timing_decorator
def filter_with_comprehension(items: List[str], patterns: List[str]) -> List[str]:
    """Optimized: Using list comprehension"""
    return [item for item in items if any(pattern in item for pattern in patterns)]

@timing_decorator
def filter_with_generator(items: List[str], patterns: List[str]) -> List[str]:
    """Optimized: Using generator expression (for large datasets)"""
    return list(item for item in items if any(pattern in item for pattern in patterns))

# Test data
test_items = [f"column_{i}_data" for i in range(10000)]
test_patterns = ["_1", "_2", "_3", "_4", "_5"]

result1 = filter_with_loop(test_items, test_patterns)
result2 = filter_with_comprehension(test_items, test_patterns)
result3 = filter_with_generator(test_items, test_patterns)
print(f"Results match: {result1 == result2 == result3}")
print()


# 2. String Operations
print("2. STRING CONCATENATION")
print("-" * 50)

@timing_decorator
def build_sql_naive(columns: List[str]) -> str:
    """Original: String concatenation in loop"""
    sql = "SELECT "
    for i, col in enumerate(columns):
        if i > 0:
            sql += ", "
        sql += f'"{col}"'
    sql += " FROM table"
    return sql

@timing_decorator
def build_sql_join(columns: List[str]) -> str:
    """Optimized: Using join()"""
    column_list = ', '.join(f'"{col}"' for col in columns)
    return f"SELECT {column_list} FROM table"

@timing_decorator
def build_sql_cached(columns: List[str], cache: Dict[tuple, str]) -> str:
    """Optimized: With caching for repeated calls"""
    cache_key = tuple(sorted(columns))
    if cache_key not in cache:
        cache[cache_key] = ', '.join(f'"{col}"' for col in columns)
    return f"SELECT {cache[cache_key]} FROM table"

# Test data
test_columns = [f"col_{i}" for i in range(100)]
cache = {}

sql1 = build_sql_naive(test_columns)
sql2 = build_sql_join(test_columns)
sql3 = build_sql_cached(test_columns, cache)
# Second call to show cache benefit
sql4 = build_sql_cached(test_columns, cache)
print(f"Results match: {sql1 == sql2 == sql3}")
print()


# 3. Feature Signal Checking
print("3. FEATURE SIGNAL VALIDATION")
print("-" * 50)

@timing_decorator
def check_signal_original(series: pd.Series) -> bool:
    """Original: Using nunique() on full series"""
    try:
        unique_count = series.dropna().nunique()
        return unique_count > 1
    except Exception:
        return False

@timing_decorator
def check_signal_optimized(series: pd.Series) -> bool:
    """Optimized: Early exit with value_counts()"""
    try:
        if series.isna().all():
            return False
        # value_counts with limit stops after finding n unique values
        return len(series.value_counts().head(2)) > 1
    except Exception:
        return False

# Test data
test_series_varied = pd.Series(np.random.randint(0, 100, 100000))
test_series_constant = pd.Series([1] * 100000)
test_series_binary = pd.Series([0, 1] * 50000)

print("Varied data:")
signal1 = check_signal_original(test_series_varied)
signal2 = check_signal_optimized(test_series_varied)
print(f"Results match: {signal1 == signal2}")

print("\nConstant data:")
signal1 = check_signal_original(test_series_constant)
signal2 = check_signal_optimized(test_series_constant)
print(f"Results match: {signal1 == signal2}")
print()


# 4. Set vs List for Membership Testing
print("4. SET VS LIST FOR MEMBERSHIP TESTING")
print("-" * 50)

@timing_decorator
def filter_operations_list(available: List[str], existing: List[str]) -> List[str]:
    """Original: Using list for membership testing"""
    return [op for op in available if op not in existing]

@timing_decorator
def filter_operations_set(available: List[str], existing: Set[str]) -> List[str]:
    """Optimized: Using set for membership testing"""
    return [op for op in available if op not in existing]

# Test data
available_ops = [f"operation_{i}" for i in range(1000)]
existing_ops_list = [f"operation_{i}" for i in range(0, 1000, 2)]
existing_ops_set = set(existing_ops_list)

result1 = filter_operations_list(available_ops, existing_ops_list)
result2 = filter_operations_set(available_ops, existing_ops_set)
print(f"Results match: {result1 == result2}")
print()


# 5. Vectorized Operations
print("5. VECTORIZED PANDAS OPERATIONS")
print("-" * 50)

@timing_decorator
def compute_features_loop(df: pd.DataFrame, groupby_cols: List[str], agg_cols: List[str]) -> Dict[str, pd.Series]:
    """Original: Nested loops with repeated groupby"""
    features = {}
    for group_col in groupby_cols:
        for agg_col in agg_cols:
            temp_df = df[[group_col, agg_col]].copy()
            group_mean = temp_df.groupby(group_col)[agg_col].transform('mean')
            features[f'{agg_col}_mean_by_{group_col}'] = group_mean
    return features

@timing_decorator
def compute_features_vectorized(df: pd.DataFrame, groupby_cols: List[str], agg_cols: List[str]) -> Dict[str, pd.Series]:
    """Optimized: Pre-compute groupby objects"""
    features = {}
    # Pre-compute all groupby objects once
    groupby_objects = {col: df.groupby(col) for col in groupby_cols}
    
    for group_col, groupby in groupby_objects.items():
        for agg_col in agg_cols:
            features[f'{agg_col}_mean_by_{group_col}'] = groupby[agg_col].transform('mean')
    return features

# Test data
np.random.seed(42)
test_df = pd.DataFrame({
    'group1': np.random.choice(['A', 'B', 'C'], 10000),
    'group2': np.random.choice(['X', 'Y', 'Z'], 10000),
    'value1': np.random.randn(10000),
    'value2': np.random.randn(10000),
    'value3': np.random.randn(10000),
})

features1 = compute_features_loop(test_df, ['group1', 'group2'], ['value1', 'value2', 'value3'])
features2 = compute_features_vectorized(test_df, ['group1', 'group2'], ['value1', 'value2', 'value3'])
print(f"Results match: {all(features1[k].equals(features2[k]) for k in features1)}")
print()


# 6. Recursive Counting
print("6. RECURSIVE COUNTING OPTIMIZATION")
print("-" * 50)

class TreeNode:
    def __init__(self):
        self.children = []
    
    def add_child(self):
        child = TreeNode()
        self.children.append(child)
        return child

@timing_decorator
def count_nodes_loop(node: TreeNode) -> int:
    """Original: Manual accumulation"""
    count = 1
    for child in node.children:
        count += count_nodes_loop(child)
    return count

@timing_decorator
def count_nodes_sum(node: TreeNode) -> int:
    """Optimized: Using sum() with generator"""
    return 1 + sum(count_nodes_sum(child) for child in node.children)

# Build test tree
root = TreeNode()
for i in range(5):
    child = root.add_child()
    for j in range(5):
        subchild = child.add_child()
        for k in range(5):
            subchild.add_child()

count1 = count_nodes_loop(root)
count2 = count_nodes_sum(root)
print(f"Node counts match: {count1 == count2} (count: {count1})")
print()


# 7. Batch Database Operations (Simulated)
print("7. BATCH DATABASE OPERATIONS")
print("-" * 50)

@timing_decorator
def validate_columns_individual(columns: List[str], valid_columns: Set[str]) -> List[str]:
    """Original: Check each column individually"""
    result = []
    for col in columns:
        # Simulate database query
        time.sleep(0.001)  # 1ms per query
        if col in valid_columns:
            result.append(col)
    return result

@timing_decorator
def validate_columns_batch(columns: List[str], valid_columns: Set[str]) -> List[str]:
    """Optimized: Single batch query"""
    # Simulate single batch query
    time.sleep(0.005)  # 5ms for one query
    return [col for col in columns if col in valid_columns]

# Test data
requested_columns = [f"col_{i}" for i in range(50)]
valid_columns = {f"col_{i}" for i in range(0, 50, 2)}

result1 = validate_columns_individual(requested_columns, valid_columns)
result2 = validate_columns_batch(requested_columns, valid_columns)
print(f"Results match: {result1 == result2}")
print()


print("\nSUMMARY OF OPTIMIZATIONS:")
print("=" * 50)
print("1. Use list comprehensions and generator expressions instead of loops with append")
print("2. Use join() for string concatenation, with caching for repeated operations")
print("3. Implement early exit strategies for validation functions")
print("4. Always use sets for membership testing, not lists")
print("5. Pre-compute expensive operations (like groupby) and reuse them")
print("6. Use built-in functions like sum() with generators for cleaner, faster code")
print("7. Batch database/API operations instead of making individual calls")