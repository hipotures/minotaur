"""
Titanic Dataset Custom Feature Operations

Domain-specific feature operations for the Titanic survival prediction dataset.
Methods are automatically enabled when this module is loaded.
Comment out methods to disable specific feature operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class CustomFeatureOperations:
    """Custom feature operations for Titanic dataset."""
    
    @staticmethod
    def get_passenger_class_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Passenger class related features."""
        features = {}
        
        if 'Pclass' in df.columns:
            # Class indicators
            features['is_first_class'] = (df['Pclass'] == 1).astype(int)
            features['is_third_class'] = (df['Pclass'] == 3).astype(int)
            
            # Class as categorical
            features['pclass_cat'] = df['Pclass'].astype('category')
            
        if all(col in df.columns for col in ['Pclass', 'Fare']):
            # Fare per class (relative wealth indicator)
            class_fare_mean = df.groupby('Pclass')['Fare'].transform('mean')
            features['fare_vs_class_mean'] = df['Fare'] / (class_fare_mean + 1e-6)
            
        return features
    
    @staticmethod
    def get_family_size_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Family size and relationship features."""
        features = {}
        
        if all(col in df.columns for col in ['SibSp', 'Parch']):
            # Total family size
            features['family_size'] = df['SibSp'] + df['Parch'] + 1  # +1 for passenger themselves
            
            # Family size categories
            features['is_alone'] = (features['family_size'] == 1).astype(int)
            features['is_small_family'] = (features['family_size'].between(2, 4)).astype(int)
            features['is_large_family'] = (features['family_size'] > 4).astype(int)
            
            # Specific relationship indicators
            features['has_siblings_spouses'] = (df['SibSp'] > 0).astype(int)
            features['has_parents_children'] = (df['Parch'] > 0).astype(int)
            
            # Family size binned
            features['family_size_binned'] = pd.cut(
                features['family_size'], 
                bins=[0, 1, 4, 20], 
                labels=['alone', 'small', 'large']
            ).astype('category')
            
        return features
    
    @staticmethod
    def get_age_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Age-related features with missing value handling."""
        features = {}
        
        if 'Age' in df.columns:
            # Fill missing ages with median by passenger class
            age_filled = df['Age'].copy()
            if 'Pclass' in df.columns:
                age_filled = df.groupby('Pclass')['Age'].transform(
                    lambda x: x.fillna(x.median())
                )
            else:
                age_filled = age_filled.fillna(age_filled.median())
            
            # Age categories
            features['age_filled'] = age_filled
            features['is_child'] = (age_filled <= 16).astype(int)
            features['is_adult'] = (age_filled.between(17, 65)).astype(int)
            features['is_elderly'] = (age_filled > 65).astype(int)
            
            # Age groups
            features['age_group'] = pd.cut(
                age_filled,
                bins=[0, 16, 35, 60, 100],
                labels=['child', 'young_adult', 'middle_age', 'elderly']
            ).astype('category')
            
            # Missing age indicator
            features['age_was_missing'] = df['Age'].isna().astype(int)
            
        return features
    
    @staticmethod
    def get_fare_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Fare-related features."""
        features = {}
        
        if 'Fare' in df.columns:
            # Fill missing fares with median
            fare_filled = df['Fare'].fillna(df['Fare'].median())
            
            # Fare transformations
            features['fare_filled'] = fare_filled
            features['fare_log'] = np.log1p(fare_filled)  # log(1+fare) for skewed data
            
            # Fare categories
            features['fare_category'] = pd.cut(
                fare_filled,
                bins=[0, 7.91, 14.45, 31, 1000],  # Quartile-based bins
                labels=['low', 'medium', 'high', 'very_high']
            ).astype('category')
            
            # Fare indicators
            features['paid_no_fare'] = (fare_filled == 0).astype(int)
            features['expensive_ticket'] = (fare_filled > 50).astype(int)
            
        return features
    
    @staticmethod
    def get_name_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Name-based features (titles, name length)."""
        features = {}
        
        if 'Name' in df.columns:
            # Extract titles from names
            titles = df['Name'].str.extract(r',\s*([^.]*)\.')
            if not titles.empty:
                title_series = titles[0].str.strip()
                
                # Group rare titles
                title_mapping = {
                    'Mr': 'Mr',
                    'Miss': 'Miss',
                    'Mrs': 'Mrs',
                    'Master': 'Master',
                    'Dr': 'Rare',
                    'Rev': 'Rare',
                    'Col': 'Rare',
                    'Major': 'Rare',
                    'Mlle': 'Miss',
                    'Countess': 'Rare',
                    'Ms': 'Miss',
                    'Lady': 'Rare',
                    'Jonkheer': 'Rare',
                    'Don': 'Rare',
                    'Dona': 'Rare',
                    'Mme': 'Mrs',
                    'Capt': 'Rare',
                    'Sir': 'Rare'
                }
                
                features['title'] = title_series.map(title_mapping).fillna('Rare').astype('category')
                
                # Title indicators
                features['is_mr'] = (features['title'] == 'Mr').astype(int)
                features['is_mrs'] = (features['title'] == 'Mrs').astype(int)
                features['is_miss'] = (features['title'] == 'Miss').astype(int)
                features['is_master'] = (features['title'] == 'Master').astype(int)
                features['has_rare_title'] = (features['title'] == 'Rare').astype(int)
            
            # Name length
            features['name_length'] = df['Name'].str.len()
            
        return features
    
    @staticmethod
    def get_cabin_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Cabin-related features."""
        features = {}
        
        if 'Cabin' in df.columns:
            # Cabin deck (first letter)
            cabin_deck = df['Cabin'].str.extract(r'^([A-Z])')
            if not cabin_deck.empty:
                features['cabin_deck'] = cabin_deck[0].fillna('Unknown').astype('category')
                features['has_cabin'] = df['Cabin'].notna().astype(int)
                
                # Deck indicators for known decks
                for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                    features[f'deck_{deck}'] = (features['cabin_deck'] == deck).astype(int)
            
            # Number of cabins (multiple cabins indicates wealth)
            features['num_cabins'] = df['Cabin'].str.count(' ') + 1
            features['num_cabins'] = features['num_cabins'].fillna(0)
            features['multiple_cabins'] = (features['num_cabins'] > 1).astype(int)
            
        return features
    
    @staticmethod
    def get_embarkation_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Embarkation port features."""
        features = {}
        
        if 'Embarked' in df.columns:
            # Fill missing embarked with mode
            embarked_filled = df['Embarked'].fillna(df['Embarked'].mode()[0])
            features['embarked_filled'] = embarked_filled.astype('category')
            
            # Port indicators
            features['embarked_C'] = (embarked_filled == 'C').astype(int)  # Cherbourg
            features['embarked_Q'] = (embarked_filled == 'Q').astype(int)  # Queenstown
            features['embarked_S'] = (embarked_filled == 'S').astype(int)  # Southampton
            
        return features
    
    @staticmethod
    def get_interaction_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Interaction features between different variables."""
        features = {}
        
        # Sex and age interactions
        if all(col in df.columns for col in ['Sex', 'Age']):
            age_filled = df['Age'].fillna(df['Age'].median())
            features['sex_age_interaction'] = df['Sex'].astype(str) + '_' + pd.cut(
                age_filled, bins=[0, 16, 35, 100], labels=['child', 'adult', 'elderly']
            ).astype(str)
            
        # Class and family size interactions
        if 'Pclass' in df.columns and all(col in df.columns for col in ['SibSp', 'Parch']):
            family_size = df['SibSp'] + df['Parch'] + 1
            features['class_family_interaction'] = (
                df['Pclass'].astype(str) + '_' + 
                pd.cut(family_size, bins=[0, 1, 4, 20], labels=['alone', 'small', 'large']).astype(str)
            )
            
        # Sex and class interactions
        if all(col in df.columns for col in ['Sex', 'Pclass']):
            features['sex_class_interaction'] = df['Sex'].astype(str) + '_class' + df['Pclass'].astype(str)
            
        return features