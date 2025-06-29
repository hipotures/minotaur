# Feature Engineering Documentation

## Overview

The feature engineering system provides a modular, extensible architecture for generating features with enhanced timing, logging, and automatic filtering of no-signal features.

## New Structure

```
src/features/
├── __init__.py              # Main package initialization
├── base.py                  # Abstract base classes and mixins
├── generic/                 # Universal feature operations
│   ├── __init__.py
│   ├── statistical.py       # Statistical aggregations
│   ├── polynomial.py        # Polynomial transformations
│   ├── binning.py          # Discretization features
│   ├── ranking.py          # Ranking and percentile features
│   ├── temporal.py         # Time-based features
│   ├── text.py             # NLP and text mining features
│   └── categorical.py      # Advanced categorical encoding
└── custom/                  # Domain-specific features
    ├── __init__.py
    ├── base.py             # Base class for custom features
    ├── titanic.py          # Titanic dataset features
    └── kaggle_s5e6.py      # Fertilizer prediction features
```

## Key Components

### 1. Base Classes (`src/features/base.py`)

#### FeatureTimingMixin
Provides timing capabilities for feature operations:
- `_time_feature(name)`: Context manager for timing individual features (DEBUG level)
- `log_timing_summary()`: Logs class-level timing summary (INFO level)

#### AbstractFeatureOperation
Abstract base class defining the interface for all feature operations:
- `generate_features()`: Main method to generate features
- `get_operation_name()`: Returns operation name
- `validate_input()`: Validates input dataframe

#### GenericFeatureOperation
Base class for generic features with timing and error handling.

#### CustomFeatureOperation
Base class for domain-specific features with timing support.

### 2. Generic Feature Modules

#### Statistical Features (`statistical.py`)
- Group-based aggregations (mean, std, count)
- Deviation from group statistics
- Frequency encoding

#### Polynomial Features (`polynomial.py`)
- Squared and cubic transformations
- Square root and logarithm features
- Multiplicative interactions

#### Binning Features (`binning.py`)
- Quantile-based binning
- Equal-width binning
- Custom bin edges

#### Ranking Features (`ranking.py`)
- Dense rank
- Percentile rank
- Quartile indicators

#### Temporal Features (`temporal.py`)
- DateTime component extraction
- Cyclical encoding for periodic features
- Lag features for time series
- Rolling window statistics

#### Text Features (`text.py`)
- Basic text statistics (length, word count)
- Character type analysis
- Pattern detection (emails, URLs, phones)
- Text complexity metrics

#### Categorical Features (`categorical.py`)
- Frequency and count encoding
- Target encoding with smoothing
- One-hot encoding for low cardinality
- Label encoding

### 3. Custom Feature Modules

Each custom module inherits from `BaseDomainFeatures` and implements:
- Domain-specific feature generation methods
- Operation registry for selective feature generation
- Timing integration for all operations

## Usage Examples

### Using Generic Features

```python
from src.features.generic.statistical import get_statistical_aggregations
from src.features.generic.temporal import get_temporal_features

# Statistical features
stat_features = get_statistical_aggregations(
    df, 
    groupby_cols=['category'], 
    agg_cols=['price', 'quantity']
)

# Temporal features
temporal_features = get_temporal_features(
    df,
    datetime_columns=['date'],
    numeric_columns=['sales'],
    lags=[1, 7, 30]
)
```

### Using Custom Features

```python
from src.features.custom.kaggle_s5e6 import CustomFeatureOperations

# Create instance
fertilizer_ops = CustomFeatureOperations()

# Generate all features
all_features = fertilizer_ops.generate_all_features(df)

# Or generate specific features
stress_features = fertilizer_ops.generate_specific_features(
    df, 
    'environmental_stress_features'
)
```

## Backward Compatibility

The old import paths are maintained with deprecation warnings:

```python
# Old way (still works with warning)
from src.domains.generic import GenericFeatureOperations

# New way (recommended)
from src.features.generic.statistical import StatisticalFeatures
```

## Timing and Logging

### Timing Levels

1. **DEBUG Level**: Individual feature generation times
   ```
   DEBUG: Generated feature 'temperature_squared' in 0.003s
   DEBUG: Generated feature 'humidity_log' in 0.002s
   ```

2. **INFO Level**: Class-level summaries
   ```
   INFO: Polynomial Features: Generated 15 features in 0.045s (avg: 0.003s/feature)
   ```

### Performance Monitoring

Each feature operation tracks:
- Total time for the operation
- Time per individual feature
- Number of features generated
- Success/failure status

## Migration Guide

### For Generic Features

Replace monolithic imports:
```python
# Old
from src.domains.generic import GenericFeatureOperations
features = GenericFeatureOperations.get_polynomial_features(df, cols)

# New
from src.features.generic.polynomial import get_polynomial_features
features = get_polynomial_features(df, cols)
```

### For Custom Features

The class name remains `CustomFeatureOperations` for compatibility:
```python
# Old
from src.domains.kaggle_s5e6 import CustomFeatureOperations

# New (but old still works)
from src.features.custom.kaggle_s5e6 import CustomFeatureOperations
```

## Benefits of Refactoring

1. **Modularity**: Each feature type in its own file
2. **Extensibility**: Easy to add new feature types
3. **Performance Monitoring**: Detailed timing for optimization
4. **Code Reuse**: Shared base classes and utilities
5. **Better Testing**: Can test each module independently
6. **Enhanced Features**: New temporal, text, and categorical features
7. **Maintainability**: Clear separation of concerns

## Adding New Features

### Adding New Generic Feature Class

To create a new type of generic features:

1. Create a new module in `src/features/generic/`
2. Inherit from `GenericFeatureOperation`
3. Implement required methods:
   - `get_operation_name()`: Return descriptive name
   - `_generate_features_impl()`: Main feature generation logic
4. Use `_time_feature()` context manager for timing
5. Add to `src/features/generic/__init__.py` exports

Example - Graph features:
```python
# src/features/generic/graph.py
from ..base import GenericFeatureOperation

class GraphFeatures(GenericFeatureOperation):
    def get_operation_name(self) -> str:
        return "Graph Features"
    
    def _generate_features_impl(self, df, **kwargs):
        features = {}
        
        # Time each feature generation
        with self._time_feature('node_degree'):
            features['node_degree'] = self.calculate_degree(df)
            
        with self._time_feature('clustering_coefficient'):
            features['clustering_coef'] = self.calculate_clustering(df)
            
        return features
    
    def calculate_degree(self, df):
        # Implementation here
        pass
```

### Adding Features to Existing Generic Class

To add features to an existing generic module:

1. Open the relevant module (e.g., `statistical.py`)
2. Add new feature generation in `_generate_features_impl()`
3. Wrap with `_time_feature()` for timing
4. Follow existing patterns for consistency

Example - Adding to statistical features:
```python
# In statistical.py
def _generate_features_impl(self, df, **kwargs):
    features = {}
    # ... existing code ...
    
    # Add new feature
    with self._time_feature('median_by_group'):
        for group_col in groupby_cols:
            for agg_col in agg_cols:
                if self._validate_columns(df, group_col, agg_col):
                    group_median = df.groupby(group_col)[agg_col].transform('median')
                    features[f'{agg_col}_median_by_{group_col}'] = group_median
    
    return features
```

### Adding New Custom Domain Features

To create features for a new dataset/domain:

1. Create a new module in `src/features/custom/`
2. Name it appropriately (e.g., `dataset_name.py`)
3. Create class named `CustomFeatureOperations` (for compatibility)
4. Inherit from `BaseDomainFeatures`
5. Implement feature methods and register them

Example - Stock market features:
```python
# src/features/custom/stock_market.py
from .base import BaseDomainFeatures

class CustomFeatureOperations(BaseDomainFeatures):
    """Custom features for stock market prediction."""
    
    def __init__(self):
        super().__init__('stock_market')
    
    def _register_operations(self):
        """Register all available operations."""
        self._operation_registry = {
            'technical_indicators': self.get_technical_indicators,
            'market_sentiment': self.get_market_sentiment,
            'volatility_features': self.get_volatility_features,
        }
    
    def get_technical_indicators(self, df, **kwargs):
        """Generate technical analysis indicators."""
        features = {}
        
        if 'close' in df.columns:
            # Moving averages
            with self._time_feature('sma_20'):
                features['sma_20'] = df['close'].rolling(20).mean()
                
            with self._time_feature('ema_12'):
                features['ema_12'] = df['close'].ewm(span=12).mean()
                
            # RSI
            with self._time_feature('rsi'):
                features['rsi'] = self.calculate_rsi(df['close'])
                
        return features
```

### Adding Features to Existing Custom Domain

To add features to existing custom domain:

1. Open the domain file (e.g., `kaggle_s5e6.py`)
2. Add new method for feature generation
3. Register it in `_register_operations()`
4. Use `_time_feature()` for timing

Example:
```python
def _register_operations(self):
    self._operation_registry = {
        # ... existing operations ...
        'weather_extremes': self.get_weather_extremes,  # New!
    }

def get_weather_extremes(self, df, **kwargs):
    """Detect extreme weather conditions."""
    features = {}
    
    with self._time_feature('extreme_heat_days'):
        features['extreme_heat_days'] = (df['temperature'] > 40).astype(int)
    
    return features
```

## Logging System

### What is Logged

1. **Feature Generation Timing** (DEBUG level):
   - Each individual feature generation time
   - Format: `Generated feature 'feature_name' in X.XXXs`

2. **Operation Summaries** (INFO level):
   - Total features generated per operation
   - Total time and average time per feature
   - Format: `Operation Name: Generated N features in X.XXXs (avg: Y.YYYs/feature)`

3. **No-Signal Feature Filtering** (INFO level):
   - Features with all identical values are filtered
   - Format: `Skipping no-signal feature 'feature_name' from operation_type - all values identical`
   - Summary: `Filtered out N no-signal features from operation_type, kept M`

4. **Errors and Warnings** (ERROR/WARNING level):
   - Feature generation failures
   - Invalid input data
   - Missing columns

### Logging Configuration

Configure logging level in your application:
```python
import logging

# Show all timing details
logging.basicConfig(level=logging.DEBUG)

# Show only summaries (recommended for production)
logging.basicConfig(level=logging.INFO)
```

## No-Signal Feature Detection

### How It Works

The system automatically filters out features that provide no discriminative value:

1. **Detection**: After generating features, the system checks each feature
2. **Criteria**: A feature is considered "no-signal" if:
   - All non-NaN values are identical
   - `nunique() <= 1` after dropping NaN values
3. **Action**: No-signal features are removed from the final output
4. **Logging**: Filtered features are logged at INFO level

### Implementation

The filtering happens in `FeatureSpace._filter_no_signal_features()`:
```python
def has_signal(self, feature_series: pd.Series) -> bool:
    """Check if feature has different values."""
    try:
        unique_count = feature_series.dropna().nunique()
        return unique_count > 1
    except Exception as e:
        logger.warning(f"Error checking feature signal: {e}")
        return False  # Conservative: treat as no signal if error
```

### Examples of No-Signal Features

- Constant values: All rows have value 1.0
- Single category: Categorical feature with only one unique value
- All NaN: Feature generation resulted in all missing values
- Computation artifacts: Division by zero resulting in all inf/nan

## Best Practices

### Feature Naming

- Use descriptive names: `{source}_{operation}_{target}`
- Examples:
  - `price_mean_by_category`
  - `temperature_rolling_mean_7`
  - `description_word_count`

### Error Handling

- Always validate input columns exist
- Handle NaN values appropriately
- Use try-except blocks for complex calculations
- Log errors with context

### Performance

- Use vectorized pandas operations
- Avoid loops over rows when possible
- Cache expensive computations
- Use `_time_feature()` to identify bottlenecks

### Testing

- Test with various data types
- Include edge cases (empty df, all NaN, single row)
- Verify no-signal detection works correctly
- Check timing logs are generated