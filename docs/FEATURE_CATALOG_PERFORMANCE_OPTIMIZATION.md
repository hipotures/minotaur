# Feature Catalog Query Performance Optimization

## Executive Summary

The current MCTS feature discovery system performs redundant database queries to the `feature_catalog` table for every node evaluation. A single SQL query executed once at initialization can provide all necessary data for the entire MCTS session, significantly improving performance and reducing database load.

## Problem Analysis

### Current Implementation

The system currently uses two separate methods in different classes to fetch feature names:

#### 1. `_get_initial_features()` - Class: `FeatureDiscoveryRunner` (mcts.py)
- **Purpose**: Initialize root node with base features
- **Frequency**: Called once at startup
- **Location**: `/home/xai/DEV/minotaur/mcts.py:300-356`
- **SQL Query**:
```sql
SELECT feature_name 
FROM feature_catalog 
WHERE origin = 'train' AND is_active = TRUE
  AND feature_name NOT IN ('survived', 'passengerid')
```

#### 2. `get_feature_columns_for_node()` - Class: `FeatureSpace` (feature_space.py)
- **Purpose**: Get feature list for each MCTS node during evaluation
- **Frequency**: Called for every node evaluation (N times per session)
- **Location**: `/home/xai/DEV/minotaur/src/feature_space.py:306-370`
- **SQL Queries**:

**For custom operations:**
```sql
SELECT DISTINCT feature_name 
FROM feature_catalog 
WHERE operation_name = 'titanic Custom Features'
```

**For generic operations:**
```sql
SELECT DISTINCT feature_name 
FROM feature_catalog 
WHERE LOWER(REPLACE(operation_name, ' ', '_')) = LOWER('statistical_aggregations')
```

### Performance Issues

#### 1. **Redundant Database Queries**
- Every MCTS node evaluation triggers a separate SQL query
- For a 5-iteration MCTS session: ~5-15 database roundtrips
- For a typical session: potentially 50+ redundant queries to same static data

#### 2. **Database Connection Overhead**
- Each query requires DuckDB connection setup/teardown
- Network latency for each individual query
- Unnecessary database lock contention

#### 3. **Identical Data Retrieval**
- `feature_catalog` data is static during MCTS session
- Same queries often repeated for similar node operations
- No caching mechanism between queries

### Evidence from Database

The complete `feature_catalog` contains all information needed for both methods:

```sql
-- Single query that provides ALL necessary data:
SELECT feature_name, origin, operation_name 
FROM feature_catalog 
WHERE is_active = TRUE;

-- Results show clear data partitioning:
┌──────────────────────────────┬─────────┬─────────────────────────┐
│         feature_name         │ origin  │     operation_name      │
├──────────────────────────────┼─────────┼─────────────────────────┤
│ passengerid                  │ train   │ train_features          │  ← Root node data
│ survived                     │ train   │ train_features          │
│ pclass                       │ train   │ train_features          │
│ name                         │ train   │ train_features          │
│ ...                          │ train   │ train_features          │
├──────────────────────────────┼─────────┼─────────────────────────┤
│ is_first_class               │ custom  │ titanic Custom Features │  ← Custom node data
│ family_size                  │ custom  │ titanic Custom Features │
│ ...                          │ custom  │ titanic Custom Features │
├──────────────────────────────┼─────────┼─────────────────────────┤
│ age_mean_by_sex              │ generic │ statistical_aggregations│  ← Generic node data
│ fare_std_by_pclass           │ generic │ statistical_aggregations│
│ ...                          │ generic │ text_features           │
└──────────────────────────────┴─────────┴─────────────────────────┘
```

## Root Cause Analysis

### Architectural Issue
The current design treats feature catalog queries as dynamic operations, but the data is fundamentally **static during MCTS execution**:

1. **Data Immutability**: `feature_catalog` doesn't change during MCTS session
2. **Predictable Access Patterns**: All possible operation_names are known at startup
3. **Small Data Size**: Entire catalog fits easily in memory (~1000 features max)

### Cross-Class Communication Gap
- `FeatureDiscoveryRunner` and `FeatureSpace` are separate classes
- No shared caching mechanism between them
- Each implements its own database access pattern

## Proposed Solution

### Phase 1: Unified Feature Catalog Loading

#### 1. **Single SQL Query at Initialization**

Move feature catalog loading to `FeatureSpace.__init__()`:

```python
class FeatureSpace:
    def __init__(self, config: Dict[str, Any], duckdb_manager=None):
        # ... existing initialization ...
        
        # Load entire feature catalog once
        self.feature_catalog_cache = self._load_feature_catalog()
        
    def _load_feature_catalog(self) -> Dict[str, List[str]]:
        """
        Load entire feature catalog once and organize by access patterns.
        
        Returns:
            Dict mapping access keys to feature lists:
            - 'train_features': Features for root node
            - 'statistical_aggregations': Features for statistical operations
            - 'text_features': Features for text operations
            - etc.
        """
        if not hasattr(self, 'duckdb_manager') or self.duckdb_manager is None:
            logger.warning("No DuckDB connection - feature catalog cache disabled")
            return {}
        
        try:
            # Single comprehensive query
            catalog_query = """
                SELECT feature_name, origin, operation_name 
                FROM feature_catalog 
                WHERE is_active = TRUE
            """
            results = self.duckdb_manager.connection.execute(catalog_query).fetchall()
            
            # Organize by access patterns
            catalog_cache = {}
            forbidden_columns = {self.id_column, self.target_column} | set(self.ignore_columns or [])
            
            for feature_name, origin, operation_name in results:
                # Skip forbidden columns for all caches
                if feature_name in forbidden_columns:
                    continue
                    
                # Create normalized operation key
                if origin == 'train':
                    cache_key = 'train_features'
                else:
                    # Normalize operation names (handle spaces/underscores)
                    cache_key = operation_name.lower().replace(' ', '_')
                
                if cache_key not in catalog_cache:
                    catalog_cache[cache_key] = []
                catalog_cache[cache_key].append(feature_name)
            
            logger.info(f"Loaded feature catalog: {len(results)} total features, "
                       f"{len(catalog_cache)} operation groups")
            for key, features in catalog_cache.items():
                logger.debug(f"  {key}: {len(features)} features")
                
            return catalog_cache
            
        except Exception as e:
            logger.error(f"Failed to load feature catalog: {e}")
            return {}
```

#### 2. **Update Feature Access Methods**

**Replace `get_feature_columns_for_node()` with memory lookup:**

```python
def get_feature_columns_for_node(self, node) -> List[str]:
    """Get feature columns using cached catalog data."""
    
    # Get the current operation for this node
    current_operation = getattr(node, 'operation_that_created_this', None)
    
    # For root node, return cached train features
    if current_operation is None or current_operation == 'root':
        return self.feature_catalog_cache.get('train_features', [])
    
    # For child nodes, lookup by operation name
    if not self.feature_catalog_cache:
        logger.warning("Feature catalog cache not available - falling back to database")
        return self._legacy_get_feature_columns_for_node(node)
    
    # Check if this is a custom domain operation
    is_custom_op = current_operation in self.operations and self.operations[current_operation].category == 'custom_domain'
    
    if is_custom_op and self.dataset_name:
        # For custom operations, use domain-specific key
        cache_key = f"{self.dataset_name.lower()}_custom_features"
        operation_features = self.feature_catalog_cache.get(cache_key, [])
    else:
        # For generic operations, normalize the name
        cache_key = current_operation.lower().replace(' ', '_')
        operation_features = self.feature_catalog_cache.get(cache_key, [])
    
    if operation_features:
        logger.debug(f"Node operation '{current_operation}' - found {len(operation_features)} features from cache")
        return operation_features
    else:
        logger.warning(f"No cached features found for operation '{current_operation}' - falling back to pattern matching")
        return self._get_features_by_pattern_fallback(current_operation, node)
```

#### 3. **Provide Root Features to MCTS**

**Update `FeatureDiscoveryRunner._get_initial_features()`:**

```python
def _get_initial_features(self) -> set:
    """Get initial feature set using FeatureSpace cache."""
    
    # Use FeatureSpace cache instead of direct database access
    if hasattr(self, 'feature_space') and self.feature_space:
        train_features = self.feature_space.feature_catalog_cache.get('train_features', [])
        if train_features:
            logger.info(f"Loaded {len(train_features)} initial features from cache: {sorted(train_features)}")
            return set(train_features)
    
    # Fallback to legacy method if cache not available
    logger.warning("Feature cache not available - using legacy database query")
    return self._legacy_get_initial_features()
```

### Phase 2: Performance Enhancements

#### 1. **Memory Optimization**

```python
def _optimize_catalog_cache(self, catalog_cache: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Optimize memory usage of catalog cache."""
    
    # Convert lists to tuples (immutable, less memory)
    optimized_cache = {
        key: tuple(sorted(features))  # Sort for consistent ordering
        for key, features in catalog_cache.items()
    }
    
    # Pre-compute commonly used sets
    optimized_cache['_all_features'] = tuple(
        feature for features in catalog_cache.values() for feature in features
    )
    
    return optimized_cache
```

#### 2. **Cache Validation**

```python
def _validate_catalog_cache(self) -> bool:
    """Validate that cache contains expected data."""
    
    if not self.feature_catalog_cache:
        return False
        
    # Check for required keys
    required_keys = ['train_features']
    missing_keys = [key for key in required_keys if key not in self.feature_catalog_cache]
    
    if missing_keys:
        logger.warning(f"Catalog cache missing required keys: {missing_keys}")
        return False
        
    # Check for reasonable data sizes
    train_features = self.feature_catalog_cache.get('train_features', [])
    if len(train_features) < 5:  # Sanity check
        logger.warning(f"Suspiciously few train features in cache: {len(train_features)}")
        return False
        
    return True
```

### Phase 3: Backward Compatibility

#### 1. **Legacy Method Preservation**

```python
def _legacy_get_feature_columns_for_node(self, node) -> List[str]:
    """Legacy database-based feature retrieval (fallback only)."""
    # Keep existing implementation as fallback
    # ... existing get_feature_columns_for_node logic ...
```

#### 2. **Graceful Degradation**

```python
def get_feature_columns_for_node(self, node) -> List[str]:
    """Get feature columns with automatic fallback."""
    
    # Try cache first
    if self._validate_catalog_cache():
        return self._get_feature_columns_from_cache(node)
    
    # Fall back to database
    logger.info("Using legacy database queries for feature retrieval")
    return self._legacy_get_feature_columns_for_node(node)
```

## Expected Performance Improvements

### 1. **Query Reduction**
- **Before**: 1 + N queries (where N = number of nodes evaluated)
- **After**: 1 query total per MCTS session
- **Improvement**: 80-95% reduction in database queries

### 2. **Response Time**
- **Before**: ~5-10ms per node evaluation (database overhead)
- **After**: ~0.1ms per node evaluation (memory lookup)
- **Improvement**: 50-100x faster feature retrieval

### 3. **Resource Usage**
- **Before**: Repeated DuckDB connection overhead
- **After**: Single connection, minimal memory usage (~1-10KB cache)
- **Improvement**: Reduced database lock contention, lower CPU usage

### 4. **Scalability**
- **Before**: Performance degrades with number of MCTS iterations
- **After**: Constant-time feature lookups regardless of iterations
- **Improvement**: Linear scaling with MCTS complexity

## Implementation Plan

### Phase 1: Core Implementation (High Priority)
- [ ] Implement `_load_feature_catalog()` in `FeatureSpace.__init__()`
- [ ] Update `get_feature_columns_for_node()` to use cache
- [ ] Modify `_get_initial_features()` to use FeatureSpace cache
- [ ] Add cache validation logic

### Phase 2: Optimization (Medium Priority)
- [ ] Memory optimization for catalog cache
- [ ] Performance metrics and monitoring
- [ ] Cache refresh mechanisms (if needed)

### Phase 3: Enhancement (Low Priority)
- [ ] Advanced caching strategies (LRU, etc.)
- [ ] Cache persistence across sessions
- [ ] Dynamic cache updates

## Success Metrics

### Performance Metrics
1. **Database Query Count**: Reduce from ~N to 1 per session
2. **Feature Retrieval Time**: Reduce from ~5-10ms to ~0.1ms per lookup
3. **Memory Usage**: Cache should use <10KB for typical datasets
4. **MCTS Session Time**: Overall reduction in session duration

### Functional Metrics
1. **Zero Regressions**: Identical feature sets returned by new vs old methods
2. **Backward Compatibility**: Graceful fallback for edge cases
3. **Error Handling**: Robust behavior when cache fails

## Risk Assessment

### Low Risk
- **Data Consistency**: Cache built from same database queries
- **Memory Usage**: Small cache size (~1000 features max)
- **Compatibility**: Fallback mechanisms preserve existing behavior

### Medium Risk
- **Cache Invalidation**: Need to handle dynamic catalog changes (rare)
- **Initialization Timing**: Ensure cache ready before first node evaluation

### Mitigation Strategies
- Comprehensive unit tests comparing cache vs database results
- Performance benchmarks on realistic datasets
- Gradual rollout with feature flags

## Conclusion

This optimization addresses a fundamental performance bottleneck in the MCTS feature discovery system. By leveraging the static nature of feature catalog data during MCTS execution, we can achieve significant performance improvements while maintaining identical functionality.

The proposed solution is low-risk, backward-compatible, and provides clear measurable benefits. Implementation should prioritize the core caching mechanism with gradual enhancement over time.

The optimization aligns with software engineering best practices:
- **DRY Principle**: Eliminate redundant database queries
- **Performance**: Optimize critical path operations
- **Maintainability**: Centralize feature catalog access logic
- **Scalability**: Constant-time lookups regardless of session complexity