<!-- 
Documentation Status: CURRENT
Last Updated: 2025-07-02 13:00
Compatible with commit: 9baad51
Changes: Added origin field and auto-registration API guide for developers
-->

# Features Development - Custom Operations Guide

## 🛠️ Development Overview

This guide provides comprehensive instructions for developing new feature operations, both generic (domain-agnostic) and custom (domain-specific). The Minotaur feature system is designed for extensibility with clear patterns for adding new capabilities.

### Development Patterns

```
Generic Operations          Custom Domain Operations
      ↓                            ↓
Domain-agnostic            Competition-specific
Reusable across datasets   Optimized for problem domain
Statistical/Mathematical   Business logic/Domain knowledge
```

## 📋 Origin Field & Auto-Registration (Important for Developers)

### Understanding Feature Origins

All features in Minotaur are automatically classified by **origin** during generation:

- **`origin='train'`**: Original dataset columns (handled automatically during dataset import)
- **`origin='generic'`**: Your generic operations (domain-agnostic)
- **`origin='custom'`**: Your custom operations (domain-specific)

### Auto-Registration API

**All feature operations inherit auto-registration functionality**:

```python
def generate_features(self, df: pd.DataFrame, 
                     auto_register: bool = True, 
                     origin: str = 'generic',
                     **kwargs) -> Dict[str, pd.Series]:
    """
    Generate features with automatic catalog registration.
    
    Args:
        df: Input dataframe
        auto_register: Whether to register features in catalog
        origin: Feature origin classification ('generic', 'custom', 'train')
        **kwargs: Operation-specific parameters
    """
```

**Developer Guidelines**:
1. **Default Values**: Use `origin='generic'` for generic ops, `origin='custom'` for custom ops
2. **Auto-Registration**: Enable by default (`auto_register=True`) for production use
3. **Testing**: Disable auto-registration (`auto_register=False`) in unit tests
4. **Feature Names**: Ensure unique, descriptive names for catalog clarity

### Integration with Feature Catalog

Your operations automatically integrate with the dynamic feature catalog:

```python
# Your operation is automatically registered
operation_name = self.get_operation_name()  # e.g., "Graph Features"
category = self._get_category()             # e.g., "graph_analysis"

# Features are registered with metadata
for feature_name, feature_series in features.items():
    # Automatic registration includes:
    # - feature_name: Your feature name
    # - operation_name: Your operation name
    # - category: Dynamic category mapping
    # - origin: Specified origin type
    # - description: Auto-generated or custom
```

## 🔧 Generic Feature Development

Generic features are domain-agnostic operations that work across different datasets and problem types. They are located in `src/features/generic/`.

### Creating a New Generic Operation

#### Step 1: Create the Module File

Create a new file in `src/features/generic/`:

```python
# src/features/generic/graph_features.py
"""
Graph Feature Operations

Graph-based features for relational and network data including:
- Node degree and centrality measures
- Community detection features
- Path-based features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from ..base import GenericFeatureOperation

logger = logging.getLogger(__name__)


class GraphFeatures(GenericFeatureOperation):
    """Generate graph-based features from relational data."""
    
    def get_operation_name(self) -> str:
        return "Graph Features"
    
    def _generate_features_impl(self, df: pd.DataFrame,
                               node_col: str = None,
                               edge_cols: List[str] = None,
                               **kwargs) -> Dict[str, pd.Series]:
        """
        Generate graph features from relational data.
        
        Args:
            df: Input dataframe
            node_col: Column representing nodes (auto-detected if None)
            edge_cols: Columns representing edges/relationships
            
        Returns:
            Dictionary of feature name -> pandas Series
        """
        features = {}
        
        # Auto-detect node column if not provided
        if node_col is None:
            node_col = self._detect_node_column(df)
        
        if node_col is None:
            logger.warning("No suitable node column found")
            return features
        
        # Generate degree features
        with self._time_feature('node_degree', features):
            features['node_degree'] = self._calculate_node_degree(df, node_col)
        
        # Generate centrality features
        with self._time_feature('node_centrality', features):
            features['node_centrality'] = self._calculate_centrality(df, node_col)
        
        # Generate clustering features
        with self._time_feature('clustering_coefficient', features):
            features['clustering_coefficient'] = self._calculate_clustering(df, node_col)
        
        return features
    
    def _detect_node_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect the most likely node identifier column."""
        # Look for columns with 'id' in name
        for col in df.columns:
            if 'id' in col.lower() and df[col].nunique() > 1:
                return col
        
        # Fallback to first categorical column with high cardinality
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() > 10:
                return col
        
        return None
    
    def _calculate_node_degree(self, df: pd.DataFrame, node_col: str) -> pd.Series:
        """Calculate node degree (number of connections)."""
        # Count occurrences of each node
        degree_counts = df[node_col].value_counts()
        return df[node_col].map(degree_counts).fillna(0)
    
    def _calculate_centrality(self, df: pd.DataFrame, node_col: str) -> pd.Series:
        """Calculate simple centrality measure."""
        # Simplified centrality based on degree normalized by max possible
        degree = self._calculate_node_degree(df, node_col)
        max_degree = degree.max()
        return degree / max_degree if max_degree > 0 else degree
    
    def _calculate_clustering(self, df: pd.DataFrame, node_col: str) -> pd.Series:
        """Calculate clustering coefficient approximation."""
        # Simplified clustering based on co-occurrence patterns
        # This is a placeholder - real implementation would need edge data
        degree = self._calculate_node_degree(df, node_col)
        # Simple heuristic: higher degree nodes have lower clustering
        return 1.0 / (1.0 + degree * 0.1)


# Convenience function for direct access
def get_graph_features(df: pd.DataFrame, 
                      node_col: str = None,
                      edge_cols: List[str] = None,
                      **kwargs) -> Dict[str, pd.Series]:
    """
    Generate graph features for relational data.
    
    Args:
        df: Input dataframe
        node_col: Node identifier column
        edge_cols: Edge/relationship columns
        
    Returns:
        Dictionary of feature names to Series
    """
    graph_features = GraphFeatures()
    return graph_features.generate_features(
        df, 
        node_col=node_col,
        edge_cols=edge_cols,
        **kwargs
    )
```

#### Step 2: Add to Module Exports

Update `src/features/generic/__init__.py`:

```python
# src/features/generic/__init__.py
"""Generic feature operations package."""

from .statistical import StatisticalFeatures, get_statistical_aggregations
from .polynomial import PolynomialFeatures, get_polynomial_features
from .binning import BinningFeatures, get_binning_features
from .ranking import RankingFeatures, get_ranking_features
from .temporal import TemporalFeatures, get_temporal_features
from .text import TextFeatures, get_text_features
from .categorical import CategoricalFeatures, get_categorical_features
from .graph_features import GraphFeatures, get_graph_features  # New addition

__all__ = [
    # Classes
    'StatisticalFeatures', 'PolynomialFeatures', 'BinningFeatures',
    'RankingFeatures', 'TemporalFeatures', 'TextFeatures', 
    'CategoricalFeatures', 'GraphFeatures',
    
    # Functions
    'get_statistical_aggregations', 'get_polynomial_features', 
    'get_binning_features', 'get_ranking_features',
    'get_temporal_features', 'get_text_features',
    'get_categorical_features', 'get_graph_features'
]
```

#### Step 3: Add Configuration Support

Update feature space configuration to include the new operation:

```yaml
# In MCTS or dataset configuration
feature_space:
  enabled_categories:
    - 'statistical_aggregations'
    - 'polynomial_features'
    - 'graph_features'  # New category
    
  # Operation-specific parameters
  generic_params:
    graph_features:
      node_col: 'user_id'
      edge_cols: ['friend_id', 'follower_id']
      max_degree: 1000
```

#### Step 4: Test the New Operation

```python
# Test the new operation
from src.features.generic.graph_features import get_graph_features

# Sample data
df = pd.DataFrame({
    'user_id': [1, 2, 3, 1, 2, 3, 4, 5],
    'friend_id': [2, 3, 1, 4, 5, 4, 1, 2],
    'activity_score': [10, 20, 15, 25, 30, 18, 22, 28]
})

# Generate features
features = get_graph_features(df, node_col='user_id')
print(f"Generated {len(features)} graph features")

# Check timing information (DEBUG logging)
# DEBUG: Generated feature 'node_degree' in 0.003s
# DEBUG: Generated feature 'node_centrality' in 0.002s
# INFO: Graph Features: Generated 3 features in 0.008s (avg: 0.003s/feature)
```

### Generic Operation Best Practices

#### Error Handling and Validation

```python
def _generate_features_impl(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Generate features with proper error handling."""
    features = {}
    
    # Validate inputs
    if df.empty:
        logger.warning("Empty dataframe provided")
        return features
    
    # Validate required columns exist
    required_cols = kwargs.get('required_columns', [])
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return features
    
    try:
        # Feature generation with error handling
        with self._time_feature('feature_name', features):
            # Handle potential errors gracefully
            try:
                features['feature_name'] = self._safe_calculation(df)
            except Exception as e:
                logger.warning(f"Failed to generate feature 'feature_name': {e}")
                # Continue with other features
    
    except Exception as e:
        logger.error(f"Critical error in {self.get_operation_name()}: {e}")
        return {}
    
    return features

def _safe_calculation(self, df: pd.DataFrame) -> pd.Series:
    """Perform calculation with proper error handling."""
    try:
        # Actual calculation
        result = df['numeric_col'].apply(lambda x: x ** 2)
        
        # Handle edge cases
        result = result.replace([np.inf, -np.inf], np.nan)
        
        return result
    except Exception as e:
        logger.warning(f"Calculation failed: {e}")
        return pd.Series([np.nan] * len(df))
```

#### Performance Optimization

```python
def _generate_features_impl(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Optimized feature generation."""
    features = {}
    
    # Pre-compute common values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Vectorized operations
    with self._time_feature('vectorized_feature', features):
        # Use pandas vectorized operations
        features['vectorized_feature'] = df[numeric_cols].sum(axis=1)
    
    # Avoid loops when possible
    with self._time_feature('grouped_feature', features):
        # Use groupby instead of loops
        if 'category_col' in df.columns:
            grouped = df.groupby('category_col')['numeric_col'].transform('mean')
            features['grouped_feature'] = grouped
    
    # Memory-efficient operations
    with self._time_feature('memory_efficient', features):
        # Process in chunks for large datasets
        if len(df) > 100000:
            chunk_size = 10000
            results = []
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                chunk_result = self._process_chunk(chunk)
                results.append(chunk_result)
            features['memory_efficient'] = pd.concat(results)
        else:
            features['memory_efficient'] = self._process_chunk(df)
    
    return features
```

## 🎯 Custom Domain Development

Custom domain features are competition or problem-specific operations that encode domain knowledge. They are located in `src/features/custom/`.

### Creating a New Custom Domain

#### Step 1: Create Domain Module

```python
# src/features/custom/ecommerce.py
"""
E-commerce Domain Custom Feature Operations

Domain-specific features for e-commerce/retail datasets including:
- Customer behavior indicators
- Product recommendation features
- Sales pattern analysis
- Seasonal demand features
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base import BaseDomainFeatures

logger = logging.getLogger(__name__)


class CustomFeatureOperations(BaseDomainFeatures):
    """Custom feature operations for e-commerce datasets."""
    
    def __init__(self):
        """Initialize e-commerce feature operations."""
        super().__init__('ecommerce')
    
    def _register_operations(self):
        """Register all available operations for e-commerce domain."""
        self._operation_registry = {
            'customer_behavior_features': self.get_customer_behavior_features,
            'product_recommendation_features': self.get_product_recommendation_features,
            'sales_pattern_features': self.get_sales_pattern_features,
            'seasonal_demand_features': self.get_seasonal_demand_features,
        }
    
    def get_customer_behavior_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Customer behavior and loyalty indicators."""
        features = {}
        
        # Purchase frequency features
        if 'customer_id' in df.columns and 'order_date' in df.columns:
            with self._time_feature('purchase_frequency'):
                purchase_counts = df.groupby('customer_id')['order_date'].count()
                features['purchase_frequency'] = df['customer_id'].map(purchase_counts)
            
            with self._time_feature('days_since_last_purchase'):
                if pd.api.types.is_datetime64_any_dtype(df['order_date']):
                    latest_dates = df.groupby('customer_id')['order_date'].max()
                    days_since = (df['order_date'].max() - latest_dates).dt.days
                    features['days_since_last_purchase'] = df['customer_id'].map(days_since)
        
        # Spending behavior
        if 'customer_id' in df.columns and 'order_total' in df.columns:
            with self._time_feature('avg_order_value'):
                avg_spending = df.groupby('customer_id')['order_total'].mean()
                features['avg_order_value'] = df['customer_id'].map(avg_spending)
            
            with self._time_feature('total_customer_value'):
                total_spending = df.groupby('customer_id')['order_total'].sum()
                features['total_customer_value'] = df['customer_id'].map(total_spending)
            
            with self._time_feature('is_high_value_customer'):
                high_value_threshold = df['order_total'].quantile(0.8)
                avg_spending = df.groupby('customer_id')['order_total'].mean()
                features['is_high_value_customer'] = (
                    df['customer_id'].map(avg_spending) > high_value_threshold
                ).astype(int)
        
        # Customer segments
        if 'customer_age' in df.columns:
            with self._time_feature('customer_generation'):
                features['is_gen_z'] = (df['customer_age'] <= 25).astype(int)
                features['is_millennial'] = df['customer_age'].between(26, 41).astype(int)
                features['is_gen_x'] = df['customer_age'].between(42, 57).astype(int)
                features['is_boomer'] = (df['customer_age'] >= 58).astype(int)
        
        return features
    
    def get_product_recommendation_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Product recommendation and affinity features."""
        features = {}
        
        # Product popularity
        if 'product_id' in df.columns:
            with self._time_feature('product_popularity'):
                product_counts = df['product_id'].value_counts()
                features['product_popularity'] = df['product_id'].map(product_counts)
            
            with self._time_feature('product_popularity_rank'):
                product_ranks = df['product_id'].value_counts().rank(ascending=False)
                features['product_popularity_rank'] = df['product_id'].map(product_ranks)
        
        # Category affinity
        if 'customer_id' in df.columns and 'product_category' in df.columns:
            with self._time_feature('customer_category_affinity'):
                category_counts = df.groupby(['customer_id', 'product_category']).size()
                customer_total = df.groupby('customer_id').size()
                affinity = category_counts / customer_total
                features['customer_category_affinity'] = df.apply(
                    lambda row: affinity.get((row['customer_id'], row['product_category']), 0),
                    axis=1
                )
        
        # Cross-selling opportunities
        if 'customer_id' in df.columns and 'product_category' in df.columns:
            with self._time_feature('cross_sell_potential'):
                customer_categories = df.groupby('customer_id')['product_category'].nunique()
                features['cross_sell_potential'] = df['customer_id'].map(customer_categories)
        
        return features
    
    def get_sales_pattern_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Sales patterns and trends."""
        features = {}
        
        # Time-based patterns
        if 'order_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['order_date']):
            with self._time_feature('is_weekend_sale'):
                features['is_weekend_sale'] = (df['order_date'].dt.dayofweek >= 5).astype(int)
            
            with self._time_feature('is_month_end_sale'):
                features['is_month_end_sale'] = (df['order_date'].dt.day >= 25).astype(int)
            
            with self._time_feature('quarter'):
                features['quarter'] = df['order_date'].dt.quarter
        
        # Price-based patterns
        if 'product_price' in df.columns and 'discount_amount' in df.columns:
            with self._time_feature('discount_percentage'):
                features['discount_percentage'] = (
                    df['discount_amount'] / df['product_price'] * 100
                ).fillna(0)
            
            with self._time_feature('is_heavily_discounted'):
                features['is_heavily_discounted'] = (
                    features['discount_percentage'] > 20
                ).astype(int)
        
        return features
    
    def get_seasonal_demand_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Seasonal demand and holiday effects."""
        features = {}
        
        if 'order_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['order_date']):
            # Holiday proximity features
            with self._time_feature('days_to_christmas'):
                christmas = pd.to_datetime(f"{df['order_date'].dt.year}-12-25")
                features['days_to_christmas'] = (christmas - df['order_date']).dt.days
            
            with self._time_feature('days_to_black_friday'):
                # Black Friday is 4th Thursday of November
                year = df['order_date'].dt.year
                november_first = pd.to_datetime(f"{year}-11-01")
                # Find first Thursday, then add 3 weeks
                first_thursday = november_first + pd.Timedelta(days=(3 - november_first.dt.dayofweek) % 7)
                black_friday = first_thursday + pd.Timedelta(weeks=3)
                features['days_to_black_friday'] = (black_friday - df['order_date']).dt.days
            
            # Seasonal indicators
            with self._time_feature('is_holiday_season'):
                month = df['order_date'].dt.month
                features['is_holiday_season'] = (
                    (month == 11) | (month == 12) | (month == 1)
                ).astype(int)
            
            with self._time_feature('is_back_to_school_season'):
                month = df['order_date'].dt.month
                features['is_back_to_school_season'] = (
                    (month == 8) | (month == 9)
                ).astype(int)
        
        return features


# Test function for development
def test_ecommerce_features():
    """Test function for e-commerce features."""
    # Sample e-commerce data
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 1, 2, 3, 4, 5] * 10,
        'product_id': ['A', 'B', 'C', 'A', 'D', 'E', 'F', 'G'] * 10,
        'product_category': ['Electronics', 'Clothing', 'Books'] * 27 + ['Electronics'],
        'order_date': pd.date_range('2024-01-01', periods=80, freq='D'),
        'order_total': [50, 75, 100, 25, 150, 200, 80, 120] * 10,
        'customer_age': [25, 35, 45, 25, 55, 65, 30, 40] * 10,
        'product_price': [60, 80, 110, 30, 160, 220, 90, 130] * 10,
        'discount_amount': [10, 5, 10, 5, 10, 20, 10, 10] * 10
    })
    
    # Test feature generation
    ecommerce_ops = CustomFeatureOperations()
    
    # Test all operations
    all_features = ecommerce_ops.generate_all_features(df)
    print(f"Total features generated: {len(all_features)}")
    
    # Test specific operation
    behavior_features = ecommerce_ops.generate_specific_features(
        df, 'customer_behavior_features'
    )
    print(f"Customer behavior features: {len(behavior_features)}")
    
    return all_features


if __name__ == "__main__":
    test_ecommerce_features()
```

#### Step 2: Register Domain in Configuration

```yaml
# Update feature space configuration to include new domain
feature_space:
  enabled_categories:
    - 'statistical_aggregations'
    - 'polynomial_features'
    - 'ecommerce_domain'  # New custom domain
  
  # Domain detection rules
  domain_detection:
    ecommerce:
      required_columns: ['customer_id', 'product_id']
      optional_columns: ['order_date', 'order_total', 'product_category']
      indicators: ['purchase', 'order', 'customer', 'product']
```

#### Step 3: Add Auto-Detection Logic

```python
# In src/feature_space.py or dataset manager
def detect_custom_domains(df: pd.DataFrame) -> List[str]:
    """Auto-detect applicable custom domains based on dataset schema."""
    domains = []
    
    columns = [col.lower() for col in df.columns]
    
    # E-commerce detection
    ecommerce_indicators = ['customer', 'product', 'order', 'purchase', 'price']
    if any(indicator in ' '.join(columns) for indicator in ecommerce_indicators):
        domains.append('ecommerce_domain')
    
    # Agricultural detection
    agriculture_indicators = ['nitrogen', 'phosphorous', 'potassium', 'soil', 'crop']
    if any(indicator in ' '.join(columns) for indicator in agriculture_indicators):
        domains.append('agricultural_domain')
    
    # Financial detection
    finance_indicators = ['credit', 'loan', 'income', 'debt', 'balance']
    if any(indicator in ' '.join(columns) for indicator in finance_indicators):
        domains.append('financial_domain')
    
    return domains
```

### Custom Domain Best Practices

#### Domain Knowledge Integration

```python
def get_domain_specific_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Integrate deep domain knowledge into features."""
    features = {}
    
    # Use domain-specific thresholds
    HIGH_VALUE_THRESHOLD = 1000  # Domain knowledge: $1000+ is high value
    LOYALTY_THRESHOLD = 5        # Domain knowledge: 5+ purchases indicates loyalty
    
    # Business logic encoding
    with self._time_feature('customer_segment'):
        # Multi-dimensional customer segmentation
        conditions = [
            (df['total_spend'] >= HIGH_VALUE_THRESHOLD) & (df['purchase_count'] >= LOYALTY_THRESHOLD),
            (df['total_spend'] >= HIGH_VALUE_THRESHOLD) & (df['purchase_count'] < LOYALTY_THRESHOLD),
            (df['total_spend'] < HIGH_VALUE_THRESHOLD) & (df['purchase_count'] >= LOYALTY_THRESHOLD),
        ]
        choices = ['high_value_loyal', 'high_value_new', 'low_value_loyal']
        features['customer_segment'] = np.select(conditions, choices, default='low_value_new')
    
    # Industry-specific calculations
    with self._time_feature('customer_lifetime_value'):
        # CLV = Average Order Value × Purchase Frequency × Customer Lifespan
        avg_order = df.groupby('customer_id')['order_total'].mean()
        purchase_freq = df.groupby('customer_id')['order_date'].count()
        # Simplified CLV calculation
        features['customer_lifetime_value'] = (
            df['customer_id'].map(avg_order) * 
            df['customer_id'].map(purchase_freq) * 
            2.5  # Assumed 2.5 year average lifespan
        )
    
    return features
```

#### Configuration-Driven Features

```python
def get_configurable_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Generate features based on configuration parameters."""
    features = {}
    config = kwargs.get('domain_config', {})
    
    # Configurable thresholds
    high_value_threshold = config.get('high_value_threshold', 1000)
    loyalty_threshold = config.get('loyalty_threshold', 5)
    recency_days = config.get('recency_days', 30)
    
    # RFM analysis (Recency, Frequency, Monetary)
    if all(col in df.columns for col in ['customer_id', 'order_date', 'order_total']):
        with self._time_feature('rfm_score'):
            # Recency
            latest_order = df.groupby('customer_id')['order_date'].max()
            recency = (df['order_date'].max() - latest_order).dt.days
            
            # Frequency
            frequency = df.groupby('customer_id')['order_date'].count()
            
            # Monetary
            monetary = df.groupby('customer_id')['order_total'].sum()
            
            # Score each dimension (1-5 scale)
            recency_score = pd.cut(recency, bins=5, labels=[5,4,3,2,1])
            frequency_score = pd.cut(frequency, bins=5, labels=[1,2,3,4,5])
            monetary_score = pd.cut(monetary, bins=5, labels=[1,2,3,4,5])
            
            # Combined RFM score
            rfm_combined = (
                recency_score.astype(int) * 100 +
                frequency_score.astype(int) * 10 +
                monetary_score.astype(int)
            )
            
            features['rfm_score'] = df['customer_id'].map(rfm_combined)
    
    return features
```

## 🧪 Testing and Validation

### Unit Testing Framework

```python
# tests/test_features/test_generic/test_graph_features.py
import pytest
import pandas as pd
import numpy as np
from src.features.generic.graph_features import GraphFeatures, get_graph_features


class TestGraphFeatures:
    """Test cases for graph feature operations."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample dataset for testing."""
        return pd.DataFrame({
            'user_id': [1, 2, 3, 1, 2, 3, 4, 5],
            'friend_id': [2, 3, 1, 4, 5, 4, 1, 2],
            'activity_score': [10, 20, 15, 25, 30, 18, 22, 28],
            'timestamp': pd.date_range('2024-01-01', periods=8)
        })
    
    def test_feature_generation(self, sample_data):
        """Test basic feature generation."""
        graph_features = GraphFeatures()
        features = graph_features.generate_features(sample_data, node_col='user_id')
        
        # Check that features were generated
        assert len(features) > 0
        assert 'node_degree' in features
        assert 'node_centrality' in features
        
        # Check feature properties
        assert len(features['node_degree']) == len(sample_data)
        assert features['node_degree'].min() >= 0
        assert features['node_centrality'].min() >= 0
        assert features['node_centrality'].max() <= 1
    
    def test_auto_detection(self, sample_data):
        """Test automatic node column detection."""
        graph_features = GraphFeatures()
        features = graph_features.generate_features(sample_data)  # No node_col specified
        
        # Should auto-detect user_id as node column
        assert len(features) > 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        graph_features = GraphFeatures()
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        features = graph_features.generate_features(empty_df)
        assert len(features) == 0
        
        # No suitable node column
        no_id_df = pd.DataFrame({'value': [1, 2, 3]})
        features = graph_features.generate_features(no_id_df)
        assert len(features) == 0
        
        # Single node
        single_node_df = pd.DataFrame({'user_id': [1, 1, 1], 'value': [1, 2, 3]})
        features = graph_features.generate_features(single_node_df, node_col='user_id')
        assert features['node_degree'].iloc[0] == 3  # All rows have same user_id
    
    def test_function_interface(self, sample_data):
        """Test the convenience function interface."""
        features = get_graph_features(sample_data, node_col='user_id')
        assert len(features) > 0
        assert 'node_degree' in features
    
    def test_timing_integration(self, sample_data):
        """Test that timing information is recorded."""
        graph_features = GraphFeatures()
        features = graph_features.generate_features(sample_data, node_col='user_id')
        
        # Check that timing was recorded
        assert hasattr(graph_features, '_feature_timings')
        assert len(graph_features._feature_timings) > 0
        assert graph_features._class_total_time > 0
```

### Integration Testing

```python
# tests/test_features/test_integration/test_feature_pipeline.py
import pytest
from src.feature_space import FeatureSpace
from src.features.custom.ecommerce import CustomFeatureOperations


class TestFeaturePipelineIntegration:
    """Integration tests for feature pipeline."""
    
    @pytest.fixture
    def ecommerce_data(self):
        """Sample e-commerce dataset."""
        return pd.DataFrame({
            'customer_id': [1, 2, 3, 1, 2] * 20,
            'product_id': ['A', 'B', 'C', 'A', 'D'] * 20,
            'order_date': pd.date_range('2024-01-01', periods=100),
            'order_total': [50, 75, 100, 25, 150] * 20,
            'target': [0, 1, 1, 0, 1] * 20
        })
    
    def test_new_pipeline_integration(self, ecommerce_data):
        """Test new pipeline with custom domain."""
        config = {
            'feature_space': {
                'use_new_pipeline': True,
                'enabled_categories': ['statistical_aggregations', 'ecommerce_domain'],
                'check_signal': True
            }
        }
        
        feature_space = FeatureSpace(config)
        features_df = feature_space.generate_all_features_pipeline(
            ecommerce_data,
            dataset_name='test-ecommerce',
            target_column='target'
        )
        
        # Check that both generic and custom features were generated
        assert len(features_df.columns) > len(ecommerce_data.columns)
        
        # Check for specific feature types
        column_names = ' '.join(features_df.columns)
        assert 'mean_by' in column_names  # Statistical features
        assert any('customer' in col for col in features_df.columns)  # Custom features
    
    def test_signal_detection_integration(self, ecommerce_data):
        """Test signal detection in pipeline."""
        config = {
            'feature_space': {
                'use_new_pipeline': True,
                'check_signal': True,
                'min_signal_ratio': 0.01
            }
        }
        
        # Add a constant column (no signal)
        ecommerce_data['constant_col'] = 1
        
        feature_space = FeatureSpace(config)
        features_df = feature_space.generate_all_features_pipeline(
            ecommerce_data,
            dataset_name='test-ecommerce-signal'
        )
        
        # Constant-derived features should be filtered out
        constant_features = [col for col in features_df.columns if 'constant_col' in col]
        assert len(constant_features) == 1  # Only original constant column
```

## 📊 Performance Optimization

### Memory-Efficient Development

```python
def generate_memory_efficient_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Generate features with memory optimization."""
    features = {}
    
    # Process in chunks for large datasets
    chunk_size = kwargs.get('chunk_size', 50000)
    
    if len(df) > chunk_size:
        # Chunked processing
        feature_chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_features = self._process_chunk(chunk, **kwargs)
            feature_chunks.append(chunk_features)
        
        # Combine chunks
        for feature_name in feature_chunks[0].keys():
            features[feature_name] = pd.concat([
                chunk[feature_name] for chunk in feature_chunks
            ], ignore_index=True)
    else:
        features = self._process_chunk(df, **kwargs)
    
    return features

def _process_chunk(self, chunk: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Process a single chunk of data."""
    chunk_features = {}
    
    # Memory-efficient operations
    with self._time_feature('chunk_feature'):
        # Use in-place operations where possible
        if 'numeric_col' in chunk.columns:
            chunk_features['chunk_feature'] = chunk['numeric_col'].copy()
            chunk_features['chunk_feature'] *= 2  # In-place multiplication
    
    return chunk_features
```

### Parallel Processing Framework

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable

def generate_features_parallel(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Generate features using parallel processing."""
    features = {}
    
    # Define feature generation functions
    feature_functions = [
        self._generate_basic_features,
        self._generate_advanced_features,
        self._generate_interaction_features
    ]
    
    # Parallel execution
    max_workers = kwargs.get('max_workers', 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_function = {
            executor.submit(func, df, **kwargs): func.__name__
            for func in feature_functions
        }
        
        # Collect results
        for future in as_completed(future_to_function):
            function_name = future_to_function[future]
            try:
                func_features = future.result()
                features.update(func_features)
                logger.info(f"Completed {function_name}: {len(func_features)} features")
            except Exception as e:
                logger.error(f"Failed {function_name}: {e}")
    
    return features
```

---

*For operations catalog, see [FEATURES_OPERATIONS.md](FEATURES_OPERATIONS.md)*  
*For integration guides, see [FEATURES_INTEGRATION.md](FEATURES_INTEGRATION.md)*  
*For performance optimization, see [FEATURES_PERFORMANCE.md](FEATURES_PERFORMANCE.md)*