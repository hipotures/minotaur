# Feature Operation Mapping Issue - Analysis and Solution

## Executive Summary

The MCTS feature discovery system has a fundamental mismatch between how custom features are registered in the `feature_catalog` table and how they are queried during node evaluation. This causes incorrect feature set assignment and suboptimal MCTS exploration.

## Problem Description

### Current Broken Flow

1. **Feature Registration** (during dataset import):
   ```
   Custom features from different operations:
   - family_size_features() → generates: family_size, is_alone, is_small_family
   - age_features() → generates: age_group, is_child, is_elderly  
   - fare_features() → generates: fare_log, fare_category, fare_vs_class_mean
   
   ALL registered as: operation_name = 'titanic Custom Features'
   ```

2. **MCTS Node Creation** (during tree expansion):
   ```
   FeatureSpace creates individual operations:
   - Node 1: operation='family_size_features'
   - Node 2: operation='age_features' 
   - Node 3: operation='fare_features'
   ```

3. **Feature Query** (during node evaluation):
   ```sql
   -- Node with operation='family_size_features' queries:
   SELECT feature_name FROM feature_catalog WHERE operation_name = 'family_size_features'
   
   -- Result: EMPTY! (because all custom features have operation_name = 'titanic Custom Features')
   ```

4. **Fallback Behavior**:
   - Falls back to pattern matching
   - All custom operations get the same feature set
   - MCTS cannot distinguish between different custom feature groups

### Evidence from Database

```sql
-- Current incorrect state:
SELECT feature_name, operation_name FROM feature_catalog WHERE origin='custom';

family_size          | titanic Custom Features  ❌ Should be: family_size_features  
is_alone            | titanic Custom Features  ❌ Should be: family_size_features
age_group           | titanic Custom Features  ❌ Should be: age_features
is_child            | titanic Custom Features  ❌ Should be: age_features
fare_log            | titanic Custom Features  ❌ Should be: fare_features
```

## Impact Analysis

### 1. **MCTS Exploration Degradation**
- Different custom operations receive identical feature sets
- UCB1 selection cannot differentiate between custom feature groups
- Reduces exploration diversity and effectiveness

### 2. **Feature Discovery Limitations**
- Cannot isolate performance of specific custom feature groups
- Mixed signal attribution across different domain operations
- Suboptimal feature combination discovery

### 3. **System Inconsistency**
- Generic features work correctly (proper operation_name mapping)
- Custom features are broken (all mapped to single operation_name)
- Inconsistent behavior between feature types

## Root Cause Analysis

### Primary Issue: `BaseDomainFeatures.get_operation_name()`

**File**: `src/features/custom/base.py:104-106`

```python
def get_operation_name(self) -> str:
    """Return the name of this feature operation."""
    return f"{self.domain_name} Custom Features"  # ❌ WRONG: All operations get same name
```

### Secondary Issue: Registration Architecture

The current custom feature registration uses a single `generate_all_features()` call that:
1. Executes all operation methods (`get_family_size_features`, `get_age_features`, etc.)
2. Combines all results into one feature dictionary
3. Registers everything under one `operation_name`

This architecture loses the mapping between individual operations and their features.

## Proposed Solution

### Phase 1: Fix Registration Architecture

**1. Update Custom Feature Registration Process**

Modify `BaseDomainFeatures.generate_all_features()` to register each operation separately:

```python
def generate_all_features(self, df: pd.DataFrame, auto_register: bool = True, **kwargs) -> Dict[str, pd.Series]:
    all_features = {}
    
    for operation_name, operation_func in self._operation_registry.items():
        try:
            # Generate features for this specific operation
            features = operation_func(df, **kwargs)
            all_features.update(features)
            
            # Register each operation separately if auto-registration enabled
            if self._auto_registration_enabled and features and auto_register:
                self._register_operation_features(operation_name, features, **kwargs)
                
        except Exception as e:
            logger.error(f"Error generating {operation_name} features: {e}")
    
    return all_features

def _register_operation_features(self, operation_name: str, features: Dict[str, pd.Series], **kwargs):
    """Register features for a specific operation with correct operation_name."""
    # Use the specific operation name, not generic domain name
    # This will map: 'family_size_features' → features from get_family_size_features()
    self._auto_register_custom_operation_metadata(
        features, 
        operation_name=operation_name,  # ✅ Use specific operation name
        **kwargs
    )
```

**2. Update Auto-Registration Method**

Modify `_auto_register_custom_operation_metadata()` to accept operation_name parameter:

```python
def _auto_register_custom_operation_metadata(self, features: Dict[str, pd.Series], 
                                           operation_name: str = None, 
                                           **kwargs):
    # Use provided operation_name instead of self.get_operation_name()
    operation_name_to_use = operation_name or self.get_operation_name()
    
    # Register in feature_catalog with specific operation_name
    conn.execute("""
        INSERT INTO feature_catalog (feature_name, operation_name, ...)
        VALUES (?, ?, ...)
    """, [feature_name, operation_name_to_use, ...])
```

### Phase 2: Migration Strategy

**1. Create Migration Script**

```python
#!/usr/bin/env python3
"""
Migrate existing custom features to use specific operation names.
"""

def migrate_custom_features(dataset_name: str):
    """Re-process custom features with correct operation names."""
    
    # 1. Delete existing custom features
    conn.execute("DELETE FROM feature_catalog WHERE origin = 'custom'")
    
    # 2. Re-generate with new registration logic
    custom_ops = get_custom_operations(dataset_name)
    for operation_name, operation_func in custom_ops._operation_registry.items():
        features = operation_func(df)
        register_features_with_operation_name(features, operation_name)
```

**2. Backward Compatibility**

Keep fallback logic in `get_feature_columns_for_node()` for datasets that haven't been migrated:

```python
# Try specific operation name first
result = query_by_operation_name(current_operation)

if not result and is_custom_operation:
    # Fallback to legacy naming for unmigrated datasets
    result = query_by_legacy_name(f"{dataset_name} Custom Features")
```

### Phase 3: Verification

**1. Database State Verification**

After migration, verify correct mapping:

```sql
-- Should show specific operation names:
SELECT feature_name, operation_name FROM feature_catalog WHERE origin='custom';

family_size          | family_size_features    ✅
is_alone            | family_size_features    ✅  
age_group           | age_features           ✅
is_child            | age_features           ✅
fare_log            | fare_features          ✅
```

**2. MCTS Behavior Verification**

- Different custom operations should receive different feature sets
- No more "No features found in database" warnings for custom operations
- Improved exploration diversity in MCTS tree

## Implementation Priority

### High Priority
- [ ] Fix `BaseDomainFeatures.generate_all_features()` registration logic
- [ ] Create migration script for existing datasets
- [ ] Update titanic dataset with correct feature mappings

### Medium Priority  
- [ ] Add verification tests for feature-operation mapping
- [ ] Update documentation for custom feature development
- [ ] Implement backward compatibility safeguards

### Low Priority
- [ ] Performance optimization for registration process
- [ ] Advanced migration tooling for bulk dataset updates

## Success Metrics

1. **Zero fallback queries** for custom operations (all features found in database)
2. **Distinct feature sets** for each custom operation type
3. **Improved MCTS exploration** metrics (feature diversity, convergence)
4. **Consistent behavior** between generic and custom feature systems

## Conclusion

This issue represents a fundamental architectural flaw in the custom feature registration system. The proposed solution maintains the existing API while fixing the underlying data consistency problem. The migration approach ensures smooth transition for existing datasets while establishing correct behavior for future development.

The fix will significantly improve MCTS exploration effectiveness and provide better feature discovery capabilities for domain-specific operations.