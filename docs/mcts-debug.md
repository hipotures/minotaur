# MCTS Feature Selection and Debug System Documentation

## Overview

This document describes how the Monte Carlo Tree Search (MCTS) system works in Minotaur for feature selection, how it integrates with DuckDB for memory-efficient operations, and the debug output system.

## Architecture

### 1. Dataset Registration Phase

During dataset registration, the system creates feature tables using SQL:

```sql
-- Create comprehensive feature table
CREATE TABLE train_features AS 
SELECT t.*, g.*, c.*
FROM train t, train_generic g, train_custom c 
WHERE t.rowid = g.rowid AND g.rowid = c.rowid

-- Similar for test data
CREATE TABLE test_features AS 
SELECT t.*, g.*, c.*
FROM test t, test_generic g, test_custom c 
WHERE t.rowid = g.rowid AND g.rowid = c.rowid
```

This creates complete feature tables with:
- Original columns from train/test
- Generic features (statistical, polynomial, binning, etc.)
- Custom domain features (e.g., Titanic-specific, fertilizer-specific)

### 2. MCTS Feature Selection Process

#### How MCTS Selects Features

1. **Initial State**: MCTS starts with base features (original dataset columns)

2. **Feature Operations**: MCTS explores different feature engineering operations:
   - Statistical aggregations
   - Polynomial transformations
   - Binning operations
   - Ranking features
   - Custom domain operations

3. **Node Expansion**: Each MCTS node represents a state with:
   - Applied operations history
   - Current feature set
   - Evaluation score

4. **Operation Selection**: Available operations are determined by:
   - Current features in the node
   - Operations not yet applied
   - Category weights and computational costs

#### Memory-Efficient Data Loading

Instead of loading entire datasets, each MCTS iteration:

1. **Selects specific columns** based on current node's features
2. **Executes SQL query** to fetch only needed columns:
   ```sql
   SELECT PassengerId, Survived, feature1, feature2, ... 
   FROM train_features
   WHERE ... (optional sampling)
   ```
3. **Processes data** through AutoGluon
4. **Releases memory** after evaluation

### 3. Data Flow

```
Dataset Registration:
CSV files → DuckDB tables → train/test + generic + custom → train_features/test_features

MCTS Iteration:
1. MCTS selects features for node
2. SQL SELECT specific columns from train_features
3. AutoGluon evaluation
4. Update MCTS tree with score
5. Release memory
6. Next iteration

MCTS Completion:
1. Identify best iteration and features
2. SQL SELECT best features from test_features
3. Save for predictions (if DEBUG enabled)
```

## Debug File System

### File Naming Convention

Debug files follow this pattern: `/tmp/{dataset_name}-{type}-{iteration:04d}.csv`

Where:
- `dataset_name`: Name of the dataset (e.g., "titanic", "playground-series-s5e6-2025")
- `type`: Either "train" or "test"
- `iteration`: 4-digit iteration number

### Debug File Types

1. **Registration Phase** (iteration 0000):
   - `/tmp/{dataset_name}-train-0000.csv`: Complete train_features table
   - `/tmp/{dataset_name}-test-0000.csv`: Complete test_features table
   - Generated only if DEBUG is enabled during registration

2. **MCTS Iterations** (iteration 0001+):
   - `/tmp/{dataset_name}-train-{iteration:04d}.csv`: Training data with selected features for that iteration
   - Only training files are saved during iterations

3. **MCTS Completion**:
   - `/tmp/{dataset_name}-test-{best_iteration:04d}.csv`: Test data with best features
   - The iteration number indicates which iteration achieved the best score
   - The presence of a test file indicates it contains the best feature set

### Example Debug Files

For Titanic dataset with best score at iteration 42:
```
/tmp/titanic-train-0000.csv    # All features (registration)
/tmp/titanic-test-0000.csv     # All features (registration)
/tmp/titanic-train-0001.csv    # Iteration 1 features
/tmp/titanic-train-0002.csv    # Iteration 2 features
...
/tmp/titanic-train-0042.csv    # Iteration 42 features
/tmp/titanic-test-0042.csv     # Best features for prediction
```

## Feature Selection Details

### Feature Categories

1. **Base Features**: Original dataset columns
2. **Statistical Features**: Aggregations by groups (mean, std, count)
3. **Polynomial Features**: Squared, log, sqrt transformations
4. **Binning Features**: Discretization into bins
5. **Ranking Features**: Percentiles, quartiles, dense rank
6. **Custom Features**: Domain-specific (e.g., Titanic titles, fertilizer stress indicators)

### Feature Filtering

Features are filtered at multiple levels:
1. **Generation time**: No-signal features (nunique <= 1) are removed
2. **Pre-AutoGluon**: Additional validation for constant/low-variance features
3. **Column limits**: Maximum features per node constraints

### Forbidden Columns

The following columns are never used as features:
- Target column (e.g., "Survived", "Target")
- ID column (e.g., "PassengerId", "Id")
- User-specified ignored columns

## Session Management

### New Sessions
- Start with empty MCTS tree
- Initialize best_score to None
- Generate features from scratch

### Resumed Sessions
- Load previous MCTS tree state
- Restore best_score and best_features
- Continue exploration from last checkpoint

### Best Score Tracking
- Maintained throughout session
- Updated when better score found
- Persisted in session checkpoints
- Used to determine which test file to generate

## SQL Query Examples

### Loading specific features for MCTS iteration:
```sql
-- Get selected columns from train_features
SELECT PassengerId, Survived, Age, Fare_mean_by_Pclass, Age_squared
FROM train_features
LIMIT 1000  -- Optional sampling
```

### Getting test data for best features:
```sql
-- Get same columns from test_features
SELECT PassengerId, Age, Fare_mean_by_Pclass, Age_squared
FROM test_features
```

### Checking available columns:
```sql
-- List all columns in train_features
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'train_features'
```

## Performance Considerations

1. **Memory Usage**: Only selected columns loaded per iteration
2. **DuckDB Efficiency**: Columnar storage optimized for analytical queries
3. **Feature Caching**: Generated features stored in database, not regenerated
4. **Parallel Processing**: Multiple MCTS iterations can run concurrently
5. **Cleanup**: Memory released after each iteration

## Debug Mode

Debug mode is controlled by logging level:
- `DEBUG` level: Saves all intermediate CSV files
- `INFO` level: No CSV files saved (production mode)

To enable debug output:
```yaml
logging:
  level: 'DEBUG'
```

## Common Issues and Solutions

### Issue: Out of Memory
**Solution**: Reduce features per node or use sampling

### Issue: Slow Feature Generation
**Solution**: Check if features are cached in DuckDB tables

### Issue: Missing Test File
**Solution**: Ensure MCTS completed successfully and best score was found

### Issue: Incorrect Column Selection
**Solution**: Verify feature names match between train_features and test_features