"""
Synthetic Data Generator for Mock Testing

Creates minimal synthetic datasets for fast MCTS testing without
loading/processing the full competition data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def generate_synthetic_features(config: Dict[str, Any], num_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic feature dataset for mock testing.
    
    Args:
        config: Configuration dictionary
        num_samples: Number of samples to generate
        
    Returns:
        pd.DataFrame: Synthetic feature dataset
    """
    logger.info(f"Generating {num_samples} synthetic samples for mock testing")
    
    np.random.seed(42)  # Reproducible for testing
    
    # Basic features
    features = {
        'id': range(num_samples),
        'Nitrogen': np.random.uniform(10, 80, num_samples),
        'Phosphorous': np.random.uniform(5, 50, num_samples), 
        'Potassium': np.random.uniform(10, 60, num_samples),
        'Temperature': np.random.uniform(15, 35, num_samples),
        'Humidity': np.random.uniform(20, 95, num_samples),
        'Moisture': np.random.uniform(3, 8, num_samples),
    }
    
    # Categorical features
    soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses']
    fertilizer_types = ['Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20', '10-26-26']
    
    features['Soil Type'] = np.random.choice(soil_types, num_samples)
    features['Crop Type'] = np.random.choice(crop_types, num_samples)
    features['Fertilizer Name'] = np.random.choice(fertilizer_types, num_samples)
    
    # Create base DataFrame
    df = pd.DataFrame(features)
    
    # Add some derived features to simulate feature engineering
    logger.info("Adding synthetic derived features...")
    
    # NPK ratios
    df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['PK_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
    
    # Environmental interactions
    df['temp_humidity'] = df['Temperature'] * df['Humidity'] / 100
    df['moisture_stress'] = np.where(df['Moisture'] < 4, 1, 0)
    
    # Soil-crop combinations
    df['soil_crop'] = df['Soil Type'] + '_' + df['Crop Type']
    
    # Nutrient balance
    df['npk_sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['npk_harmony'] = 3 / (1/df['Nitrogen'] + 1/(df['Phosphorous']+1e-6) + 1/(df['Potassium']+1e-6))
    
    # Feature binning
    df['nitrogen_bin'] = pd.cut(df['Nitrogen'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    df['temperature_bin'] = pd.cut(df['Temperature'], bins=3, labels=['cool', 'moderate', 'hot'])
    
    # Some "advanced" features to simulate complex feature engineering
    df['complex_feature_1'] = np.log1p(df['npk_sum']) * df['Temperature']
    df['complex_feature_2'] = df['Nitrogen'] ** 0.5 * df['Moisture']
    df['complex_feature_3'] = np.sin(df['Humidity'] / 100 * np.pi) * df['Potassium']
    
    # Statistical features (simulating groupby stats)
    soil_nitrogen_mean = df.groupby('Soil Type')['Nitrogen'].transform('mean')
    df['nitrogen_soil_deviation'] = df['Nitrogen'] - soil_nitrogen_mean
    
    crop_phosphorous_mean = df.groupby('Crop Type')['Phosphorous'].transform('mean')
    df['phosphorous_crop_deviation'] = df['Phosphorous'] - crop_phosphorous_mean
    
    # More synthetic features to reach reasonable count
    for i in range(10):
        df[f'synthetic_feature_{i}'] = np.random.uniform(-1, 1, num_samples)
        df[f'polynomial_feature_{i}'] = df['Nitrogen'] ** (i+1) / (10**(i+1))
    
    logger.info(f"Generated synthetic dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.debug(f"Features: {list(df.columns)}")
    
    return df

def augment_synthetic_features(base_df: pd.DataFrame, operation_name: str) -> pd.DataFrame:
    """
    Add new synthetic features to simulate MCTS feature generation.
    
    Args:
        base_df: Base feature DataFrame
        operation_name: Name of the operation being simulated
        
    Returns:
        pd.DataFrame: Augmented feature dataset
    """
    logger.debug(f"Augmenting synthetic features with operation: {operation_name}")
    
    df = base_df.copy()
    
    # Simulate different types of feature operations
    if 'npk' in operation_name.lower():
        # NPK interaction features
        df[f'npk_interaction_{len(df.columns)}'] = df['Nitrogen'] * df['Phosphorous'] * df['Potassium']
        df[f'npk_ratio_{len(df.columns)}'] = df['Nitrogen'] / (df['Phosphorous'] + df['Potassium'] + 1e-6)
    
    elif 'stress' in operation_name.lower():
        # Environmental stress features
        df[f'stress_indicator_{len(df.columns)}'] = (
            (df['Temperature'] > 30).astype(int) + 
            (df['Humidity'] < 30).astype(int) + 
            (df['Moisture'] < 4).astype(int)
        )
        df[f'optimal_conditions_{len(df.columns)}'] = (
            (df['Temperature'].between(20, 28)) & 
            (df['Humidity'].between(60, 80)) & 
            (df['Moisture'] > 5)
        ).astype(int)
    
    elif 'statistical' in operation_name.lower():
        # Statistical aggregation features
        for col in ['Nitrogen', 'Phosphorous', 'Potassium']:
            if col in df.columns:
                soil_stats = df.groupby('Soil Type')[col].transform(['mean', 'std'])
                df[f'{col}_soil_mean_{len(df.columns)}'] = soil_stats['mean']
                df[f'{col}_soil_std_{len(df.columns)}'] = soil_stats['std']
    
    elif 'polynomial' in operation_name.lower():
        # Polynomial features
        for col in ['Nitrogen', 'Phosphorous', 'Potassium']:
            if col in df.columns:
                df[f'{col}_squared_{len(df.columns)}'] = df[col] ** 2
                df[f'{col}_cubed_{len(df.columns)}'] = df[col] ** 3
    
    else:
        # Generic new features
        np.random.seed(hash(operation_name) % 2**32)  # Deterministic but different per operation
        for i in range(3):  # Add 3 random features
            df[f'{operation_name}_feature_{i}'] = np.random.uniform(-1, 1, len(df))
    
    logger.debug(f"Augmented to {df.shape[1]} columns (added {df.shape[1] - base_df.shape[1]} features)")
    
    return df