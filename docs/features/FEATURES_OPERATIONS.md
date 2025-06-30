<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30 14:25
Compatible with commit: TBD
Changes: Created comprehensive feature operations catalog with examples and usage
-->

# Feature Operations - Comprehensive Catalog

## 🔧 Generic Feature Operations

Generic operations are domain-agnostic and work across different datasets and problem types. They are located in `src/features/generic/` and inherit from `GenericFeatureOperation`.

### Statistical Aggregations (`statistical.py`)

**Purpose**: Group-based statistical features for categorical-numeric combinations

**Class**: `StatisticalFeatures`  
**Operation Name**: "Statistical Aggregations"

#### Available Features

**Group Aggregations**:
```python
# Mean aggregations
Nitrogen_mean_by_Soil_Type
Temperature_mean_by_Crop_Type

# Standard deviation aggregations  
Phosphorous_std_by_Soil_Type
Humidity_std_by_Crop_Type

# Count aggregations
count_by_Soil_Type
count_by_Crop_Type

# Min/Max aggregations
Potassium_min_by_Soil_Type
Temperature_max_by_Crop_Type
```

**Deviation Features**:
```python
# Deviation from group mean
Nitrogen_dev_from_Soil_Type_mean
Temperature_dev_from_Crop_Type_mean

# Normalized deviation (z-score)
Phosphorous_zscore_by_Soil_Type
Humidity_zscore_by_Crop_Type
```

**Frequency Encoding**:
```python
# Category frequency
Soil_Type_frequency
Crop_Type_frequency

# Category count
Soil_Type_count
Crop_Type_count
```

#### Usage Example
```python
from src.features.generic.statistical import StatisticalFeatures

# Initialize with timing
stat_features = StatisticalFeatures()

# Generate features
features = stat_features.generate_features(
    df,
    groupby_cols=['Soil_Type', 'Crop_Type'],
    agg_cols=['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature']
)

# Results: ~30-50 features depending on column combinations
```

#### Configuration Parameters
- `groupby_cols`: Categorical columns for grouping (auto-detected if None)
- `agg_cols`: Numeric columns to aggregate (auto-detected if None)
- `include_deviations`: Generate deviation features (default: True)
- `include_frequency`: Generate frequency features (default: True)

### Polynomial Features (`polynomial.py`)

**Purpose**: Mathematical transformations and interactions

**Class**: `PolynomialFeatures`  
**Operation Name**: "Polynomial Features"

#### Available Features

**Power Transformations**:
```python
# Squared features
Nitrogen_squared
Temperature_squared
Humidity_squared

# Cubic features (if degree > 2)
Nitrogen_cubed
Temperature_cubed

# Square root features
Nitrogen_sqrt
Temperature_sqrt

# Logarithmic features
Nitrogen_log
Temperature_log (with +1 to handle zeros)
```

**Multiplicative Interactions**:
```python
# Two-way interactions
Nitrogen_x_Phosphorous
Temperature_x_Humidity
Nitrogen_x_Temperature

# Three-way interactions (if enabled)
Nitrogen_x_Phosphorous_x_Potassium
```

#### Usage Example
```python
from src.features.generic.polynomial import PolynomialFeatures

poly_features = PolynomialFeatures()

features = poly_features.generate_features(
    df,
    numeric_cols=['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature'],
    degree=2,
    include_interactions=True
)

# Results: ~20-40 features depending on numeric columns
```

#### Configuration Parameters
- `numeric_cols`: Numeric columns for transformations (auto-detected if None)
- `degree`: Maximum polynomial degree (default: 2)
- `include_interactions`: Generate interaction features (default: True)
- `max_interaction_features`: Limit interactions to first N features (default: 3)

### Binning Features (`binning.py`)

**Purpose**: Discretization and categorical encoding

**Class**: `BinningFeatures`  
**Operation Name**: "Binning Features"

#### Available Features

**Quantile Binning**:
```python
# Quartiles (4 bins)
Nitrogen_quartile_1
Nitrogen_quartile_2
Nitrogen_quartile_3
Nitrogen_quartile_4

# Custom bins (5 bins default)
Temperature_bin_1
Temperature_bin_2
Temperature_bin_3
Temperature_bin_4
Temperature_bin_5
```

**Equal-Width Binning**:
```python
# Equal intervals
Humidity_equal_bin_1
Humidity_equal_bin_2
Humidity_equal_bin_3
```

**Extreme Value Indicators**:
```python
# High/low indicators
Nitrogen_is_high  # Top 10%
Temperature_is_low  # Bottom 10%
```

#### Usage Example
```python
from src.features.generic.binning import BinningFeatures

binning_features = BinningFeatures()

features = binning_features.generate_features(
    df,
    numeric_cols=['Nitrogen', 'Phosphorous', 'Temperature'],
    n_bins=5,
    strategy='quantile'  # or 'uniform'
)

# Results: ~15-30 binary features
```

#### Configuration Parameters
- `numeric_cols`: Columns to discretize (auto-detected if None)
- `n_bins`: Number of bins per column (default: 5)
- `strategy`: 'quantile' or 'uniform' (default: 'quantile')
- `include_extremes`: Generate extreme value indicators (default: True)

### Ranking Features (`ranking.py`)

**Purpose**: Rank and percentile-based features

**Class**: `RankingFeatures`  
**Operation Name**: "Ranking Features"

#### Available Features

**Dense Ranking**:
```python
# Overall ranking
Nitrogen_rank
Temperature_rank
Phosphorous_rank

# Group ranking
Nitrogen_rank_by_Soil_Type
Temperature_rank_by_Crop_Type
```

**Percentile Ranking**:
```python
# Percentiles (0-100)
Nitrogen_percentile
Temperature_percentile

# Group percentiles
Humidity_percentile_by_Soil_Type
```

**Quartile Indicators**:
```python
# Quartile membership
Nitrogen_is_Q1  # First quartile
Nitrogen_is_Q4  # Fourth quartile
```

#### Usage Example
```python
from src.features.generic.ranking import RankingFeatures

rank_features = RankingFeatures()

features = rank_features.generate_features(
    df,
    numeric_cols=['Nitrogen', 'Phosphorous', 'Temperature'],
    groupby_cols=['Soil_Type']
)

# Results: ~20-40 features
```

### Temporal Features (`temporal.py`)

**Purpose**: Date/time component extraction and time series features

**Class**: `TemporalFeatures`  
**Operation Name**: "Temporal Features"

#### Available Features

**DateTime Components**:
```python
# Date components
date_year
date_month
date_day
date_dayofweek
date_quarter

# Time components
time_hour
time_minute
time_second
```

**Cyclical Encoding**:
```python
# Sine/cosine encoding for periodic features
month_sin
month_cos
hour_sin
hour_cos
dayofweek_sin
dayofweek_cos
```

**Time Series Features**:
```python
# Lag features
temperature_lag_1
temperature_lag_7
temperature_lag_30

# Rolling statistics
temperature_rolling_mean_7
temperature_rolling_std_7
temperature_rolling_min_30
temperature_rolling_max_30
```

#### Usage Example
```python
from src.features.generic.temporal import TemporalFeatures

temporal_features = TemporalFeatures()

features = temporal_features.generate_features(
    df,
    datetime_columns=['date'],
    numeric_columns=['temperature', 'humidity'],
    lags=[1, 7, 30],
    rolling_windows=[7, 30]
)

# Results: ~30-60 features depending on datetime columns
```

### Text Features (`text.py`)

**Purpose**: Natural language processing and text analysis

**Class**: `TextFeatures`  
**Operation Name**: "Text Features"

#### Available Features

**Basic Text Statistics**:
```python
# Length features
description_length
description_word_count
description_char_count

# Character analysis
description_uppercase_count
description_digit_count
description_space_count
```

**Pattern Detection**:
```python
# Pattern indicators
description_has_email
description_has_url
description_has_phone
description_has_number
```

**Text Complexity**:
```python
# Readability metrics
description_avg_word_length
description_sentence_count
description_complexity_score
```

#### Usage Example
```python
from src.features.generic.text import TextFeatures

text_features = TextFeatures()

features = text_features.generate_features(
    df,
    text_columns=['description', 'title'],
    include_patterns=True,
    include_complexity=True
)

# Results: ~15-30 features per text column
```

### Categorical Features (`categorical.py`)

**Purpose**: Advanced categorical encoding techniques

**Class**: `CategoricalFeatures`  
**Operation Name**: "Categorical Features"

#### Available Features

**Frequency Encoding**:
```python
# Category frequency
category_frequency
category_count
category_probability
```

**Target Encoding** (when target provided):
```python
# Smoothed target encoding
category_target_mean_smoothed
category_target_std_smoothed
```

**One-Hot Encoding** (for low cardinality):
```python
# Binary indicators
category_value1
category_value2
category_value3
```

**Label Encoding**:
```python
# Ordinal encoding
category_label_encoded
category_ordinal
```

#### Usage Example
```python
from src.features.generic.categorical import CategoricalFeatures

cat_features = CategoricalFeatures()

features = cat_features.generate_features(
    df,
    categorical_columns=['Soil_Type', 'Crop_Type'],
    target_column='Fertilizer_Name',  # For target encoding
    max_cardinality_for_onehot=10
)

# Results: Variable, depending on cardinality
```

## 🎯 Custom Domain Operations

Custom operations are domain-specific and located in `src/features/custom/`. Each domain has its own `CustomFeatureOperations` class.

### Fertilizer S5E6 (`kaggle_s5e6.py`)

**Purpose**: Agricultural domain features for fertilizer prediction

**Class**: `CustomFeatureOperations`  
**Domain**: 'fertilizer_s5e6'

#### Operation Categories

**Environmental Stress Features** (`environmental_stress_features`):
```python
# Temperature stress indicators
temp_stress_extreme_cold  # < 15°C
temp_stress_cold         # 15-20°C
temp_stress_optimal      # 20-30°C
temp_stress_hot          # 30-35°C
temp_stress_extreme_hot  # > 35°C

# Moisture stress indicators
moisture_stress_drought   # < 30%
moisture_stress_low      # 30-50%
moisture_stress_optimal  # 50-70%
moisture_stress_high     # > 70%

# Humidity stress indicators
humidity_stress_low      # < 40%
humidity_stress_optimal  # 40-60%
humidity_stress_high     # > 60%
```

**Soil-Crop Interaction Features** (`soil_crop_interaction_features`):
```python
# Soil-crop compatibility
soil_crop_compatibility_score
sandy_crop_interaction
clay_crop_interaction
loamy_crop_interaction

# Crop-specific soil preferences
wheat_soil_preference
corn_soil_preference
rice_soil_preference
```

**Nutrient Deficiency Features** (`nutrient_deficiency_features`):
```python
# NPK ratios
NP_ratio                 # Nitrogen/Phosphorous
PK_ratio                 # Phosphorous/Potassium
NK_ratio                 # Nitrogen/Potassium

# Nutrient balance indicators
nutrient_balance         # (N+P+K)/3
nutrient_imbalance_score
npk_deficiency_indicator

# Individual deficiencies
nitrogen_deficiency      # Below optimal range
phosphorous_deficiency
potassium_deficiency
```

**Agricultural Recommendation Features** (`agricultural_recommendation_features`):
```python
# Environmental conditions
growing_season_indicator
optimal_growing_conditions
stress_condition_count

# Fertilizer recommendations
high_nitrogen_need
balanced_fertilizer_need
specific_nutrient_need

# Yield prediction features
expected_yield_category
growth_potential_score
```

#### Usage Example
```python
from src.features.custom.kaggle_s5e6 import CustomFeatureOperations

fertilizer_features = CustomFeatureOperations()

# Generate all features
all_features = fertilizer_features.generate_all_features(df)

# Generate specific operation
stress_features = fertilizer_features.generate_specific_features(
    df, 
    'environmental_stress_features'
)

# Results: ~50-80 domain-specific features
```

### Titanic (`titanic.py`)

**Purpose**: Social and demographic features for survival prediction

**Class**: `CustomFeatureOperations`  
**Domain**: 'titanic'

#### Available Features

**Social Class Features**:
```python
# Title extraction
title_Mr
title_Mrs
title_Miss
title_Master
title_Dr
title_Rev
title_rare

# Social indicators
is_married_woman
is_child
is_elderly
social_class_indicator
```

**Family Structure Features**:
```python
# Family size analysis
family_size              # SibSp + Parch + 1
is_alone                # Family size = 1
small_family            # 2-4 members
large_family            # 5+ members

# Family survival features
family_survival_rate
has_surviving_family
family_death_rate
```

**Survival Indicators**:
```python
# Age group indicators
child_under_12
teenager
young_adult
middle_aged
elderly

# Demographic combinations
woman_and_child
man_first_class
third_class_male
```

#### Usage Example
```python
from src.features.custom.titanic import CustomFeatureOperations

titanic_features = CustomFeatureOperations()

features = titanic_features.generate_all_features(df)

# Results: ~30-50 domain-specific features
```

## 📊 Feature Generation Pipeline

### Signal Detection Process

**During Generation**:
1. Feature is generated
2. Signal check performed (nunique() > 1)
3. If no signal: feature discarded, logged
4. If signal: feature kept, timing recorded

**Signal Check Implementation**:
```python
def has_signal(self, feature_series: pd.Series) -> bool:
    """Check if feature has discriminative value."""
    try:
        # Sample large datasets for performance
        if len(feature_series) > self._signal_sample_size:
            sample = feature_series.sample(self._signal_sample_size)
        else:
            sample = feature_series
            
        unique_count = sample.dropna().nunique()
        return unique_count > 1
    except Exception:
        return False  # Conservative: treat as no signal
```

### Timing and Performance Logging

**DEBUG Level** (Individual Features):
```
DEBUG: Generated feature 'Nitrogen_squared' in 0.003s
DEBUG: Generated feature 'NP_ratio' in 0.002s [no signal, discarded]
DEBUG: Generated feature 'temperature_stress_optimal' in 0.001s
```

**INFO Level** (Operation Summaries):
```
INFO: Statistical Aggregations: Generated 45 features in 0.234s (avg: 0.005s/feature)
INFO: Polynomial Features: Generated 23 features in 0.089s (avg: 0.004s/feature)
INFO: Filtered out 12 no-signal features from statistical_aggregations, kept 33
```

### Memory Management

**Efficient Column Detection**:
```python
# Automatic column type detection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
```

**Feature Limit Management**:
```python
# Prevent excessive feature generation
if len(features) > max_features_per_operation:
    logger.warning(f"Limiting features to {max_features_per_operation}")
    features = dict(list(features.items())[:max_features_per_operation])
```

## 🛠️ API Reference

### Generic Feature Base Class

```python
class GenericFeatureOperation(AbstractFeatureOperation, FeatureTimingMixin):
    """Base class for all generic feature operations."""
    
    def generate_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Main entry point for feature generation."""
        
    def get_operation_name(self) -> str:
        """Return descriptive name for the operation."""
        
    def _generate_features_impl(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Implementation-specific feature generation."""
```

### Custom Feature Base Class

```python
class BaseDomainFeatures(FeatureTimingMixin):
    """Base class for domain-specific feature operations."""
    
    def __init__(self, domain_name: str):
        """Initialize with domain name."""
        
    def generate_all_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate all available features for the domain."""
        
    def generate_specific_features(self, df: pd.DataFrame, operation: str, **kwargs) -> Dict[str, pd.Series]:
        """Generate features for specific operation."""
        
    def get_available_operations(self) -> List[str]:
        """Get list of available operations."""
```

### Function-Based API

```python
# Direct function access for generic operations
from src.features.generic.statistical import get_statistical_aggregations
from src.features.generic.polynomial import get_polynomial_features

# Usage
stat_features = get_statistical_aggregations(
    df, 
    groupby_cols=['category'], 
    agg_cols=['price', 'quantity']
)

poly_features = get_polynomial_features(
    df,
    numeric_cols=['price', 'quantity'],
    degree=2
)
```

## 📈 Performance Characteristics

### Generation Speed by Operation Type

| Operation Type | Small Dataset (1K) | Medium Dataset (100K) | Large Dataset (1M+) |
|----------------|--------------------|-----------------------|---------------------|
| Statistical | 0.1-0.5s | 2-8s | 20-60s |
| Polynomial | 0.05-0.2s | 1-4s | 10-30s |
| Binning | 0.05-0.3s | 1-6s | 15-45s |
| Ranking | 0.1-0.4s | 3-10s | 30-90s |
| Temporal | 0.2-1s | 5-15s | 50-150s |
| Text | 0.5-2s | 10-30s | 100-300s |
| Custom Domain | 0.1-0.8s | 3-12s | 30-120s |

### Signal Detection Statistics

| Dataset Type | Generated Features | Signal Rate | Filtered Out |
|--------------|-------------------|-------------|--------------|
| Tabular (numeric) | 200-400 | 70-85% | 15-30% |
| Mixed (categorical) | 150-300 | 60-75% | 25-40% |
| Text-heavy | 100-250 | 50-70% | 30-50% |
| Time series | 300-500 | 75-90% | 10-25% |

### Memory Usage

| Operation | Memory per 100K rows | Peak Memory | Notes |
|-----------|---------------------|-------------|-------|
| Statistical | 50-200MB | 300MB | Depends on group cardinality |
| Polynomial | 100-400MB | 500MB | Quadratic growth with features |
| Binning | 20-80MB | 150MB | Linear with bins |
| Text | 200-800MB | 1GB | Depends on text length |

---

*For integration and pipeline details, see [FEATURES_INTEGRATION.md](FEATURES_INTEGRATION.md)*  
*For development guides, see [FEATURES_DEVELOPMENT.md](FEATURES_DEVELOPMENT.md)*  
*For performance optimization, see [FEATURES_PERFORMANCE.md](FEATURES_PERFORMANCE.md)*