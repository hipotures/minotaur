# TODO - Minotaur Development Tasks

## 🚨 Critical Issues

### Feature Filtering Architecture - Dual System Problem

**Problem Description:**
Minotaur currently has two separate feature filtering systems that create maintenance overhead and potential inconsistencies:

**System 1 (Legacy - feature_space.py):**
- Uses static pattern matching for forbidden columns (`'target', 'label', 'price', 'id'`)
- Currently active in dataset registration process
- Does NOT know actual target_column/id_column values from configuration
- Leads to data leakage when target column doesn't match static patterns (e.g., 'survived')

**System 2 (New - generator.py):**
- Uses actual target_column/id_column values from configuration
- Properly prevents data leakage
- Currently unused (use_new_pipeline=False by default)
- Has correct forbidden column passing to operations

**Current Status:**
- **Quick Fix Applied**: System 1 modified to pass real forbidden columns to categorical operations
- **Architectural Issue Remains**: Still dual maintenance burden

**Impact:**
- Data leakage in ML pipelines when target column has non-standard names
- Feature count mismatches between train/test datasets
- Maintenance overhead requiring updates in two places
- Technical debt accumulation

**Long-term Solution Needed:**
1. **Unify to single filtering system** (prefer System 2 architecture)
2. **Remove auto-detection fallbacks** in feature operations when forbidden columns provided
3. **Migrate all operations** to use passed forbidden_columns parameter
4. **Remove static pattern matching** from base.py
5. **Enable new pipeline by default** after thorough testing

**Priority:** High (architectural debt affecting ML pipeline integrity)
**Estimated Effort:** 2-4 hours for complete unification
**Risk:** Medium (requires testing across all feature operations)

---

## 📋 Other Tasks

### Feature Generation Configuration Refactoring

**Problem Description:**
The current feature generation code has hardcoded parameters that should be moved to configuration:

```python
generic_features(test_df, check_signal=False, target_column=None, id_column=id_column, auto_register=False)
```

**Issues:**
- Parameters like `check_signal`, `target_column`, `id_column`, and `auto_register` are hardcoded
- Different behavior for train vs test data is scattered in code rather than configured
- Makes it difficult to adjust behavior without modifying source code

**Proposed Solution:**
1. Create a configuration section for feature generation parameters
2. Support different profiles for train/test data generation
3. Move all hardcoded parameters to YAML configuration
4. Example configuration structure:
   ```yaml
   feature_generation:
     train_profile:
       check_signal: true
       auto_register: true
       use_target_column: true
     test_profile:
       check_signal: false
       auto_register: false
       use_target_column: false
   ```

**Priority:** Medium
**Estimated Effort:** 1-2 hours
**Impact:** Improved maintainability and flexibility

### Feature Catalog Synchronization Check

**Problem Description:**
Need to implement a validation check to ensure feature_catalog table stays synchronized with actual feature columns in train_features/test_features tables.

**Background:**
Fixed critical bug where features with no signal were being registered in feature_catalog before signal validation, causing MCTS to attempt using non-existent features (resulting in score=0.0 nodes).

**Solution Implemented:**
- Modified `src/features/base.py` to filter features by signal BEFORE catalog registration
- Both GenericFeatureOperation and CustomFeatureOperation now only register features that pass signal validation
- Improved logging to show "valid features (out of X generated)" counts
- Removed redundant catalog cleanup from `src/dataset_importer.py` to resolve contradiction between signal filtering mechanisms

**Required Follow-up Task:**
Create validation script that compares:
1. Operations in `feature_catalog` table vs actual columns in `train_features`
2. Identifies orphaned catalog entries (operations without corresponding columns)  
3. Identifies missing catalog entries (columns without catalog registration)
4. Provides cleanup recommendations

**Implementation Details:**
- Script should scan all dataset.duckdb files in cache directories
- Report discrepancies per dataset with specific operation names
- Suggest commands to fix synchronization issues
- Run as part of health check system

**Fixed in commits:** 41a0323 (initial fix), f4028b9 (contradiction resolution)

**Priority:** Medium  
**Estimated Effort:** 1-2 hours
**Impact:** Prevents MCTS evaluation failures due to catalog inconsistencies

---

(Future TODO items can be added here)