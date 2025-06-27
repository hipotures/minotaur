"""
Fertilizer S5E6 Domain-Specific Features

Agricultural domain knowledge for fertilizer prediction:
- Temperature and humidity stress indicators
- Moisture stress calculations
- Soil-crop interactions
- Agricultural deficiency indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FertilizerS5E6Operations:
    """Domain-specific features for fertilizer prediction."""
    
    @staticmethod
    def get_environmental_stress(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Environmental stress indicators."""
        features = {}
        
        if 'Temperature' in df.columns and 'Humidity' in df.columns:
            # Heat stress index
            features['heat_stress'] = np.where(
                (df['Temperature'] > 30) & (df['Humidity'] < 40),
                (df['Temperature'] - 30) * (40 - df['Humidity']) / 100,
                0
            )
            
            # Drought stress
            features['drought_stress'] = np.where(
                (df['Temperature'] > 25) & (df['Humidity'] < 50),
                df['Temperature'] * (50 - df['Humidity']) / 1000,
                0
            )
            
            # Optimal growing conditions
            features['optimal_temp_humidity'] = np.where(
                (df['Temperature'] >= 20) & (df['Temperature'] <= 30) & 
                (df['Humidity'] >= 50) & (df['Humidity'] <= 80),
                1, 0
            )
            
        if 'Moisture' in df.columns:
            # Moisture stress categories
            features['moisture_stress'] = pd.cut(
                df['Moisture'], 
                bins=[0, 30, 50, 70, 100], 
                labels=['severe', 'moderate', 'adequate', 'excess']
            )
            
            # Drought indicator
            features['drought_risk'] = (df['Moisture'] < 35).astype(int)
            
            # Waterlog indicator  
            features['waterlog_risk'] = (df['Moisture'] > 85).astype(int)
        
        return features
    
    @staticmethod
    def get_soil_crop_interactions(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Soil-crop specific interactions."""
        features = {}
        
        if 'Soil Type' in df.columns and 'Crop Type' in df.columns:
            # Soil-crop combination
            features['soil_crop'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
            features['soil_crop'] = features['soil_crop'].astype('category')
            
            # Sandy soil indicators (need more frequent fertilization)
            is_sandy = df['Soil Type'].astype(str).str.contains('Sandy|sandy', na=False)
            features['sandy_soil'] = is_sandy.astype(int)
            
            # Clay soil indicators (slower nutrient release)
            is_clay = df['Soil Type'].astype(str).str.contains('Clay|clay', na=False)
            features['clay_soil'] = is_clay.astype(int)
            
            # High-nutrient crops
            high_nutrient_crops = ['Corn', 'Wheat', 'Rice', 'Tomato']
            is_high_nutrient = df['Crop Type'].isin(high_nutrient_crops)
            features['high_nutrient_crop'] = is_high_nutrient.astype(int)
            
            # Legume crops (fix nitrogen)
            legume_crops = ['Soybean', 'Bean', 'Pea', 'Lentil']
            is_legume = df['Crop Type'].isin(legume_crops)
            features['legume_crop'] = is_legume.astype(int)
        
        return features
    
    @staticmethod
    def get_nutrient_deficiency_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Nutrient deficiency and adjustment features."""
        features = {}
        
        if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium']):
            # Deficiency thresholds (agricultural domain knowledge)
            features['low_Nitrogen'] = (df['Nitrogen'] < 50).astype(int)
            features['low_Phosphorous'] = (df['Phosphorous'] < 30).astype(int)
            features['low_Potassium'] = (df['Potassium'] < 40).astype(int)
            
            # Critical deficiency (multiple nutrients)
            features['multiple_deficiencies'] = (
                features['low_Nitrogen'] + 
                features['low_Phosphorous'] + 
                features['low_Potassium']
            )
            
            # Nutrient balance assessment
            features['nutrient_balance'] = np.where(
                (features['multiple_deficiencies'] == 0) & 
                (df['Nitrogen'] > 100) & (df['Phosphorous'] > 60) & (df['Potassium'] > 80),
                'excess',
                np.where(features['multiple_deficiencies'] >= 2, 'severe_deficit', 'moderate')
            ).astype('category')
            
        return features
    
    @staticmethod
    def get_seasonal_adjustments(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Seasonal and temporal feature adjustments."""
        features = {}
        
        # Temperature-based growing season
        if 'Temperature' in df.columns:
            features['growing_season'] = np.where(
                df['Temperature'] >= 15, 'active', 'dormant'
            ).astype('category')
            
            # Temperature stress categories
            features['temp_category'] = pd.cut(
                df['Temperature'],
                bins=[-np.inf, 10, 15, 25, 30, np.inf],
                labels=['cold', 'cool', 'optimal', 'warm', 'hot']
            )
        
        # Humidity-based irrigation needs
        if 'Humidity' in df.columns:
            features['irrigation_need'] = pd.cut(
                df['Humidity'],
                bins=[0, 40, 60, 80, 100],
                labels=['high', 'moderate', 'low', 'minimal']
            )
            
        return features
    
    @staticmethod
    def get_fertilizer_recommendations(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Agricultural fertilizer recommendation features."""
        features = {}
        
        if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium', 'Soil Type', 'Crop Type']):
            # NPK adjustment based on soil type
            soil_multipliers = {
                'Sandy': {'N': 1.2, 'P': 1.1, 'K': 1.3},  # Higher leaching
                'Clay': {'N': 0.9, 'P': 1.2, 'K': 0.8},   # Better retention
                'Loamy': {'N': 1.0, 'P': 1.0, 'K': 1.0}   # Balanced
            }
            
            # Default multiplier
            n_mult = p_mult = k_mult = 1.0
            
            for soil_type, multipliers in soil_multipliers.items():
                mask = df['Soil Type'].astype(str).str.contains(soil_type, na=False)
                if mask.any():
                    features[f'N_adjusted_{soil_type.lower()}'] = np.where(mask, df['Nitrogen'] * multipliers['N'], df['Nitrogen'])
                    features[f'P_adjusted_{soil_type.lower()}'] = np.where(mask, df['Phosphorous'] * multipliers['P'], df['Phosphorous'])
                    features[f'K_adjusted_{soil_type.lower()}'] = np.where(mask, df['Potassium'] * multipliers['K'], df['Potassium'])
            
            # Crop-specific nutrient requirements
            if 'Crop Type' in df.columns:
                high_n_crops = ['Corn', 'Wheat', 'Rice']
                high_p_crops = ['Tomato', 'Potato']
                high_k_crops = ['Fruit trees', 'Banana']
                
                features['high_N_crop_need'] = df['Crop Type'].isin(high_n_crops).astype(int)
                features['high_P_crop_need'] = df['Crop Type'].isin(high_p_crops).astype(int) 
                features['high_K_crop_need'] = df['Crop Type'].isin(high_k_crops).astype(int)
        
        return features
    
    @staticmethod
    def get_temperature_humidity_interactions(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Temperature-humidity interaction features (from synthetic_data)."""
        features = {}
        
        if all(col in df.columns for col in ['Temperature', 'Humidity']):
            # Temperature-humidity interaction
            features['temp_humidity'] = df['Temperature'] * df['Humidity'] / 100
            
        return features
    
    @staticmethod
    def get_soil_crop_combinations(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Soil-crop combination features (from synthetic_data)."""
        features = {}
        
        if all(col in df.columns for col in ['Soil Type', 'Crop Type']):
            # Soil-crop combination
            features['soil_crop'] = df['Soil Type'] + '_' + df['Crop Type']
            features['soil_crop'] = features['soil_crop'].astype('category')
            
        return features
    
    @staticmethod
    def get_nutrient_adequacy_ratios(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Nutrient adequacy ratios for crops."""
        features = {}
        
        if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium', 'Crop Type']):
            # Crop-specific optimal nutrient levels (simplified)
            crop_requirements = {
                'Corn': {'N': 150, 'P': 60, 'K': 120},
                'Wheat': {'N': 120, 'P': 50, 'K': 100},
                'Rice': {'N': 100, 'P': 40, 'K': 80},
                'Soybean': {'N': 50, 'P': 60, 'K': 120},  # Lower N due to fixation
                'Tomato': {'N': 140, 'P': 80, 'K': 160},
                'Potato': {'N': 120, 'P': 70, 'K': 180}
            }
            
            # Default requirements for unknown crops
            default_req = {'N': 100, 'P': 50, 'K': 100}
            
            # Calculate adequacy ratios for each nutrient
            n_adequacy = []
            p_adequacy = []
            k_adequacy = []
            
            for idx, row in df.iterrows():
                crop = row['Crop Type']
                req = crop_requirements.get(crop, default_req)
                
                n_adequacy.append(min(row['Nitrogen'] / req['N'], 2.0))  # Cap at 200%
                p_adequacy.append(min(row['Phosphorous'] / req['P'], 2.0))
                k_adequacy.append(min(row['Potassium'] / req['K'], 2.0))
            
            features['n_adequacy_ratio'] = pd.Series(n_adequacy, index=df.index)
            features['p_adequacy_ratio'] = pd.Series(p_adequacy, index=df.index)
            features['k_adequacy_ratio'] = pd.Series(k_adequacy, index=df.index)
            
            # Overall adequacy score
            features['overall_adequacy'] = (features['n_adequacy_ratio'] + 
                                           features['p_adequacy_ratio'] + 
                                           features['k_adequacy_ratio']) / 3
        
        return features
    
    @staticmethod
    def get_npk_dominance_patterns(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """NPK dominance and deficiency patterns."""
        features = {}
        
        if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium']):
            # Which nutrient is dominant (highest)
            npk_values = df[['Nitrogen', 'Phosphorous', 'Potassium']]
            dominant_nutrient = npk_values.idxmax(axis=1)
            
            features['n_dominant'] = (dominant_nutrient == 'Nitrogen').astype(int)
            features['p_dominant'] = (dominant_nutrient == 'Phosphorous').astype(int)
            features['k_dominant'] = (dominant_nutrient == 'Potassium').astype(int)
            
            # Deficiency pattern (which nutrients are low)
            if all(col in df.columns for col in ['low_Nitrogen', 'low_Phosphorous', 'low_Potassium']):
                # Create deficiency pattern string
                deficiency_patterns = []
                for idx, row in df.iterrows():
                    pattern = []
                    if row['low_Nitrogen']: pattern.append('N')
                    if row['low_Phosphorous']: pattern.append('P')
                    if row['low_Potassium']: pattern.append('K')
                    
                    if not pattern:
                        deficiency_patterns.append('none')
                    else:
                        deficiency_patterns.append('_'.join(pattern))
                
                features['npk_deficiency_pattern'] = pd.Series(deficiency_patterns, index=df.index).astype('category')
            
            # Nutrient ratios for dominance analysis
            total_npk = npk_values.sum(axis=1)
            features['n_proportion'] = df['Nitrogen'] / total_npk
            features['p_proportion'] = df['Phosphorous'] / total_npk
            features['k_proportion'] = df['Potassium'] / total_npk
            
            # Balanced vs unbalanced nutrients
            max_proportion = npk_values.div(total_npk, axis=0).max(axis=1)
            features['nutrient_balance_score'] = 1 - (max_proportion - 1/3) * 3  # 1 = perfect balance, 0 = extreme imbalance
        
        return features