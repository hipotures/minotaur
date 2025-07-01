"""
Fertilizer S5E6 (Kaggle Playground Series) Custom Feature Operations

Domain-specific feature operations for fertilizer prediction dataset.
Refactored with timing support and modular structure.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base import BaseDomainFeatures

logger = logging.getLogger(__name__)


class CustomFeatureOperations(BaseDomainFeatures):
    """Custom feature operations for Fertilizer S5E6 dataset."""
    
    def __init__(self):
        """Initialize Fertilizer feature operations."""
        super().__init__('fertilizer_s5e6')
    
    def _register_operations(self):
        """Register all available operations for Fertilizer dataset."""
        self._operation_registry = {
            'environmental_stress_features': self.get_environmental_stress_features,
            'soil_crop_interaction_features': self.get_soil_crop_interaction_features,
            'nutrient_deficiency_features': self.get_nutrient_deficiency_features,
            'agricultural_recommendation_features': self.get_agricultural_recommendation_features,
        }
    
    def get_environmental_stress_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Environmental stress indicators based on weather and soil conditions."""
        features = {}
        
        # Temperature stress indicators
        if 'temperature' in df.columns:
            with self._time_feature('temp_stress_extreme_cold'):
                features['temp_stress_extreme_cold'] = (df['temperature'] < 15).astype(int)
            
            with self._time_feature('temp_stress_cold'):
                features['temp_stress_cold'] = df['temperature'].between(15, 20).astype(int)
            
            with self._time_feature('temp_stress_optimal'):
                features['temp_stress_optimal'] = df['temperature'].between(20, 30).astype(int)
            
            with self._time_feature('temp_stress_hot'):
                features['temp_stress_hot'] = df['temperature'].between(30, 35).astype(int)
            
            with self._time_feature('temp_stress_extreme_hot'):
                features['temp_stress_extreme_hot'] = (df['temperature'] > 35).astype(int)
        
        # Humidity stress
        if 'humidity' in df.columns:
            with self._time_feature('humidity_stress_low'):
                features['humidity_stress_low'] = (df['humidity'] < 40).astype(int)
            
            with self._time_feature('humidity_stress_optimal'):
                features['humidity_stress_optimal'] = df['humidity'].between(40, 70).astype(int)
            
            with self._time_feature('humidity_stress_high'):
                features['humidity_stress_high'] = (df['humidity'] > 70).astype(int)
        
        # Combined environmental stress score
        if all(col in df.columns for col in ['temperature', 'humidity']):
            with self._time_feature('environmental_stress_score'):
                temp_dev_from_optimal = np.abs(df['temperature'] - 25) / 10  # Normalized deviation
                humidity_dev_from_optimal = np.abs(df['humidity'] - 55) / 15
                features['environmental_stress_score'] = (temp_dev_from_optimal + humidity_dev_from_optimal) / 2
        
        # pH stress (soil acidity/alkalinity)
        if 'ph' in df.columns:
            with self._time_feature('ph_stress_acidic'):
                features['ph_stress_acidic'] = (df['ph'] < 5.5).astype(int)
            
            with self._time_feature('ph_stress_slightly_acidic'):
                features['ph_stress_slightly_acidic'] = df['ph'].between(5.5, 6.5).astype(int)
            
            with self._time_feature('ph_stress_neutral'):
                features['ph_stress_neutral'] = df['ph'].between(6.5, 7.5).astype(int)
            
            with self._time_feature('ph_stress_alkaline'):
                features['ph_stress_alkaline'] = (df['ph'] > 7.5).astype(int)
        
        # Water availability proxy (from rainfall)
        if 'rainfall' in df.columns:
            with self._time_feature('water_stress_severe'):
                features['water_stress_severe'] = (df['rainfall'] < 50).astype(int)
            
            with self._time_feature('water_stress_moderate'):
                features['water_stress_moderate'] = df['rainfall'].between(50, 100).astype(int)
            
            with self._time_feature('water_stress_none'):
                features['water_stress_none'] = (df['rainfall'] > 100).astype(int)
        
        return features
    
    def get_soil_crop_interaction_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Features capturing interaction between soil properties and crop type."""
        features = {}
        
        # Crop-specific pH preferences
        if all(col in df.columns for col in ['crop_name', 'ph']):
            # Rice prefers slightly acidic to neutral
            with self._time_feature('rice_ph_suitable'):
                rice_mask = df['crop_name'] == 'rice'
                features['rice_ph_suitable'] = (rice_mask & df['ph'].between(5.5, 7.0)).astype(int)
            
            # Wheat prefers neutral to slightly alkaline
            with self._time_feature('wheat_ph_suitable'):
                wheat_mask = df['crop_name'] == 'wheat'
                features['wheat_ph_suitable'] = (wheat_mask & df['ph'].between(6.0, 7.5)).astype(int)
            
            # Cotton tolerates wider pH range
            with self._time_feature('cotton_ph_suitable'):
                cotton_mask = df['crop_name'] == 'cotton'
                features['cotton_ph_suitable'] = (cotton_mask & df['ph'].between(5.8, 8.2)).astype(int)
        
        # Crop-specific temperature preferences
        if all(col in df.columns for col in ['crop_name', 'temperature']):
            with self._time_feature('crop_temp_suitability'):
                crop_temp_ranges = {
                    'rice': (20, 35),
                    'wheat': (12, 25),
                    'maize': (18, 32),
                    'cotton': (21, 32),
                    'jute': (24, 37),
                    'coffee': (18, 28),
                    'coconut': (20, 32),
                    'papaya': (21, 33),
                    'orange': (13, 35),
                    'apple': (21, 24),
                    'muskmelon': (25, 35),
                    'watermelon': (25, 35),
                    'grapes': (15, 35),
                    'mango': (24, 35),
                    'banana': (20, 35),
                    'pomegranate': (25, 35),
                    'lentil': (18, 30),
                    'blackgram': (25, 35),
                    'mungbean': (25, 35),
                    'mothbeans': (24, 32),
                    'pigeonpeas': (18, 35),
                    'kidneybeans': (10, 27),
                    'chickpea': (10, 35)
                }
                
                suitability = pd.Series(0, index=df.index)
                for crop, (min_temp, max_temp) in crop_temp_ranges.items():
                    mask = df['crop_name'] == crop
                    suitability[mask] = df.loc[mask, 'temperature'].between(min_temp, max_temp).astype(int)
                
                features['crop_temp_suitability'] = suitability
        
        # Crop water requirements vs rainfall
        if all(col in df.columns for col in ['crop_name', 'rainfall']):
            with self._time_feature('crop_water_match'):
                # High water requirement crops
                high_water_crops = ['rice', 'jute', 'coconut']
                medium_water_crops = ['maize', 'cotton', 'banana', 'papaya']
                low_water_crops = ['lentil', 'chickpea', 'mungbean', 'mothbeans']
                
                water_match = pd.Series(0, index=df.index)
                
                # High water crops need >200mm rainfall
                for crop in high_water_crops:
                    mask = df['crop_name'] == crop
                    water_match[mask] = (df.loc[mask, 'rainfall'] > 200).astype(int)
                
                # Medium water crops need 100-200mm
                for crop in medium_water_crops:
                    mask = df['crop_name'] == crop
                    water_match[mask] = df.loc[mask, 'rainfall'].between(100, 200).astype(int)
                
                # Low water crops need <100mm
                for crop in low_water_crops:
                    mask = df['crop_name'] == crop
                    water_match[mask] = (df.loc[mask, 'rainfall'] < 100).astype(int)
                
                features['crop_water_match'] = water_match
        
        return features
    
    def get_nutrient_deficiency_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Features indicating potential nutrient deficiencies."""
        features = {}
        
        # NPK balance indicators
        if all(col in df.columns for col in ['N', 'P', 'K']):
            with self._time_feature('npk_calculation'):
                total_npk = df['N'] + df['P'] + df['K']
                npk_values = df[['N', 'P', 'K']]
            
            # Low total nutrients
            with self._time_feature('low_total_nutrients'):
                features['low_total_nutrients'] = (total_npk < 150).astype(int)
            
            # Individual nutrient deficiencies
            with self._time_feature('low_nitrogen'):
                features['low_nitrogen'] = (df['N'] < 40).astype(int)
            
            with self._time_feature('low_phosphorus'):
                features['low_phosphorus'] = (df['P'] < 20).astype(int)
            
            with self._time_feature('low_potassium'):
                features['low_potassium'] = (df['K'] < 20).astype(int)
            
            # NPK ratios
            with self._time_feature('n_to_p_ratio'):
                features['n_to_p_ratio'] = self._safe_divide(df['N'], df['P'])
            
            with self._time_feature('n_to_k_ratio'):
                features['n_to_k_ratio'] = self._safe_divide(df['N'], df['K'])
            
            with self._time_feature('p_to_k_ratio'):
                features['p_to_k_ratio'] = self._safe_divide(df['P'], df['K'])
            
            # Nutrient balance score (coefficient of variation)
            with self._time_feature('nutrient_imbalance_score'):
                npk_std = npk_values.std(axis=1)
                npk_mean = npk_values.mean(axis=1)
                features['nutrient_imbalance_score'] = self._safe_divide(npk_std, npk_mean)
            
            # Predominant nutrient
            with self._time_feature('predominant_nutrient'):
                predominant = npk_values.idxmax(axis=1)
                features['predominant_nutrient'] = predominant.astype('category')
                
                # Binary indicators for predominant nutrient
                features['n_predominant'] = (predominant == 'N').astype(int)
                features['p_predominant'] = (predominant == 'P').astype(int)
                features['k_predominant'] = (predominant == 'K').astype(int)
            
            # Severely imbalanced nutrients
            with self._time_feature('severely_imbalanced'):
                # Check if any nutrient is >60% of total
                max_proportion = npk_values.div(total_npk + 1e-6, axis=0).max(axis=1)
                features['severely_imbalanced'] = (max_proportion > 0.6).astype(int)
        
        return features
    
    def get_agricultural_recommendation_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Features that combine multiple factors for agricultural recommendations."""
        features = {}
        
        # First, generate nutrient deficiency features if not already present
        if all(col in df.columns for col in ['N', 'P', 'K']):
            nutrient_features = self.get_nutrient_deficiency_features(df, **kwargs)
            # Add them to the features dict so they can be used
            features.update(nutrient_features)
        
        # Growing condition score
        if all(col in df.columns for col in ['temperature', 'humidity', 'ph', 'rainfall']):
            with self._time_feature('growing_condition_score'):
                # Normalize each factor to 0-1 where 1 is optimal
                temp_score = 1 - np.abs(df['temperature'] - 25) / 20  # Optimal around 25°C
                humidity_score = 1 - np.abs(df['humidity'] - 55) / 45  # Optimal around 55%
                ph_score = 1 - np.abs(df['ph'] - 7) / 3  # Optimal around neutral
                rainfall_score = np.clip(df['rainfall'] / 200, 0, 1)  # Up to 200mm is good
                
                features['growing_condition_score'] = (
                    temp_score + humidity_score + ph_score + rainfall_score
                ) / 4
        
        # Fertilizer need indicators
        if all(col in df.columns for col in ['N', 'P', 'K']):
            with self._time_feature('fertilizer_need_pattern'):
                # Create a pattern string indicating which nutrients are low
                n_low = df['N'] < 40
                p_low = df['P'] < 20
                k_low = df['K'] < 20
                
                pattern = []
                for idx in df.index:
                    row_pattern = []
                    if n_low[idx]: row_pattern.append('N')
                    if p_low[idx]: row_pattern.append('P')
                    if k_low[idx]: row_pattern.append('K')
                    
                    if not row_pattern:
                        pattern.append('balanced')
                    else:
                        pattern.append('_'.join(row_pattern) + '_deficient')
                
                features['fertilizer_need_pattern'] = pd.Series(pattern, index=df.index).astype('category')
        
        # Crop-environment match score
        if 'crop_name' in df.columns:
            match_score = pd.Series(0.0, index=df.index)
            
            # Add various suitability scores if they exist
            if 'crop_temp_suitability' in features:
                match_score += features['crop_temp_suitability']
            if 'crop_water_match' in features:
                match_score += features['crop_water_match']
            if any(f'{crop}_ph_suitable' in features for crop in ['rice', 'wheat', 'cotton']):
                for crop in ['rice', 'wheat', 'cotton']:
                    if f'{crop}_ph_suitable' in features:
                        match_score += features[f'{crop}_ph_suitable']
            
            with self._time_feature('crop_environment_match'):
                features['crop_environment_match'] = match_score / 3  # Normalize
        
        # NPK utilization efficiency proxy
        if all(col in df.columns for col in ['N', 'P', 'K']) and 'growing_condition_score' in features:
            with self._time_feature('npk_utilization_efficiency'):
                npk_values = df[['N', 'P', 'K']]
                total_npk = npk_values.sum(axis=1)
                
                # Higher efficiency when conditions are good and nutrients are balanced
                nutrient_balance = 1 - features.get('nutrient_imbalance_score', 0)
                features['npk_utilization_efficiency'] = (
                    features['growing_condition_score'] * nutrient_balance
                )
        
        # Stress factor count
        with self._time_feature('stress_factor_count'):
            stress_count = pd.Series(0, index=df.index)
            
            # Environmental stresses
            for col in ['temp_stress_extreme_cold', 'temp_stress_extreme_hot', 
                       'humidity_stress_low', 'humidity_stress_high',
                       'ph_stress_acidic', 'ph_stress_alkaline',
                       'water_stress_severe']:
                if col in features:
                    stress_count += features[col]
            
            # Nutrient stresses
            for col in ['low_nitrogen', 'low_phosphorus', 'low_potassium', 
                       'severely_imbalanced']:
                if col in features:
                    stress_count += features[col]
            
            features['stress_factor_count'] = stress_count
        
        # Overall recommendation complexity (more stress = more complex recommendation)
        if 'stress_factor_count' in features:
            with self._time_feature('recommendation_complexity'):
                features['recommendation_complexity'] = pd.cut(
                    features['stress_factor_count'],
                    bins=[-0.1, 1, 3, 5, 100],
                    labels=['simple', 'moderate', 'complex', 'very_complex']
                ).astype('category')
        
        # NPK balance patterns
        if all(col in df.columns for col in ['N', 'P', 'K']):
            with self._time_feature('npk_balance_pattern'):
                npk_values = df[['N', 'P', 'K']]
                total_npk = npk_values.sum(axis=1)
                
                # Calculate proportions
                n_prop = self._safe_divide(df['N'], total_npk)
                p_prop = self._safe_divide(df['P'], total_npk)
                k_prop = self._safe_divide(df['K'], total_npk)
                
                # Balanced vs unbalanced nutrients
                max_proportion = npk_values.div(total_npk, axis=0).max(axis=1)
                features['npk_balanced'] = (max_proportion < 0.5).astype(int)
                features['npk_highly_imbalanced'] = (max_proportion > 0.7).astype(int)
        
        return features