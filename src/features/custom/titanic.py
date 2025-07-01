"""
Titanic Dataset Custom Feature Operations

Domain-specific feature operations for the Titanic survival prediction dataset.
Refactored with timing support and modular structure.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from .base import BaseDomainFeatures

logger = logging.getLogger(__name__)


class CustomFeatureOperations(BaseDomainFeatures):
    """Custom feature operations for Titanic dataset."""
    
    def __init__(self):
        """Initialize Titanic feature operations."""
        super().__init__('titanic')
    
    def _register_operations(self):
        """Register all available operations for Titanic dataset."""
        self._operation_registry = {
            'passenger_class_features': self.get_passenger_class_features,
            'family_size_features': self.get_family_size_features,
            'age_features': self.get_age_features,
            'fare_features': self.get_fare_features,
            'name_features': self.get_name_features,
            'cabin_features': self.get_cabin_features,
            'embarkation_features': self.get_embarkation_features,
            'interaction_features': self.get_interaction_features,
        }
    
    def get_passenger_class_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Passenger class related features."""
        features = {}
        
        if 'pclass' in df.columns:
            # Class indicators
            features['is_first_class'] = self._create_boolean_feature(
                df, df['pclass'] == 1, 'is_first_class'
            )
            features['is_third_class'] = self._create_boolean_feature(
                df, df['pclass'] == 3, 'is_third_class'
            )
            
            # Class as categorical
            with self._time_feature('pclass_cat'):
                features['pclass_cat'] = df['pclass'].astype('category')
            
        if all(col in df.columns for col in ['pclass', 'fare']):
            # Fare per class (relative wealth indicator)
            with self._time_feature('fare_vs_class_mean'):
                class_fare_mean = df.groupby('pclass')['fare'].transform('mean')
                features['fare_vs_class_mean'] = self._safe_divide(df['fare'], class_fare_mean + 1e-6)
            
        return features
    
    def get_family_size_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Family size and relationship features."""
        features = {}
        
        if all(col in df.columns for col in ['sibsp', 'parch']):
            # Total family size
            with self._time_feature('family_size'):
                features['family_size'] = df['sibsp'] + df['parch'] + 1  # +1 for passenger themselves
            
            # Family size categories
            features['is_alone'] = self._create_boolean_feature(
                df, features['family_size'] == 1, 'is_alone'
            )
            features['is_small_family'] = self._create_boolean_feature(
                df, features['family_size'].between(2, 4), 'is_small_family'
            )
            features['is_large_family'] = self._create_boolean_feature(
                df, features['family_size'] > 4, 'is_large_family'
            )
            
            # Specific relationship indicators
            features['has_siblings_spouses'] = self._create_boolean_feature(
                df, df['sibsp'] > 0, 'has_siblings_spouses'
            )
            features['has_parents_children'] = self._create_boolean_feature(
                df, df['parch'] > 0, 'has_parents_children'
            )
            
            # Family size binned
            with self._time_feature('family_size_binned'):
                features['family_size_binned'] = pd.cut(
                    features['family_size'], 
                    bins=[0, 1, 4, 20], 
                    labels=['alone', 'small', 'large']
                ).astype('category')
            
        return features
    
    def get_age_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Age-related features with missing value handling."""
        features = {}
        
        if 'age' in df.columns:
            # Fill missing ages with median by passenger class
            with self._time_feature('age_filled'):
                age_filled = df['age'].copy()
                if 'pclass' in df.columns:
                    age_filled = df.groupby('pclass')['age'].transform(
                        lambda x: x.fillna(x.median())
                    )
                else:
                    age_filled = age_filled.fillna(age_filled.median())
                features['age_filled'] = age_filled
            
            # Age categories
            features['is_child'] = self._create_boolean_feature(
                df, age_filled <= 16, 'is_child'
            )
            features['is_adult'] = self._create_boolean_feature(
                df, age_filled.between(17, 65), 'is_adult'
            )
            features['is_elderly'] = self._create_boolean_feature(
                df, age_filled > 65, 'is_elderly'
            )
            
            # Age groups
            with self._time_feature('age_group'):
                features['age_group'] = pd.cut(
                    age_filled,
                    bins=[0, 16, 35, 60, 100],
                    labels=['child', 'young_adult', 'middle_age', 'elderly']
                ).astype('category')
            
            # Missing age indicator
            features['age_was_missing'] = self._create_boolean_feature(
                df, df['age'].isna(), 'age_was_missing'
            )
            
        return features
    
    def get_fare_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Fare-related features."""
        features = {}
        
        if 'fare' in df.columns:
            # Fill missing fares with median
            with self._time_feature('fare_filled'):
                fare_filled = df['fare'].fillna(df['fare'].median())
                features['fare_filled'] = fare_filled
            
            # Fare transformations
            with self._time_feature('fare_log'):
                features['fare_log'] = np.log1p(fare_filled)  # log(1+fare) for skewed data
            
            # Fare categories
            with self._time_feature('fare_category'):
                features['fare_category'] = pd.cut(
                    fare_filled,
                    bins=[0, 7.91, 14.45, 31, 1000],  # Quartile-based bins
                    labels=['low', 'medium', 'high', 'very_high']
                ).astype('category')
            
            # Fare indicators
            features['paid_no_fare'] = self._create_boolean_feature(
                df, fare_filled == 0, 'paid_no_fare'
            )
            features['expensive_ticket'] = self._create_boolean_feature(
                df, fare_filled > 50, 'expensive_ticket'
            )
            
        return features
    
    def get_name_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Name-based features (titles, name length)."""
        features = {}
        
        if 'name' in df.columns:
            # Extract titles from names
            with self._time_feature('title_extraction', features):
                titles = df['name'].str.extract(r',\s*([^.]*)\.')
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
                    features['is_mr'] = self._create_boolean_feature(
                        df, features['title'] == 'Mr', 'is_mr'
                    )
                    features['is_mrs'] = self._create_boolean_feature(
                        df, features['title'] == 'Mrs', 'is_mrs'
                    )
                    features['is_miss'] = self._create_boolean_feature(
                        df, features['title'] == 'Miss', 'is_miss'
                    )
                    features['is_master'] = self._create_boolean_feature(
                        df, features['title'] == 'Master', 'is_master'
                    )
                    features['has_rare_title'] = self._create_boolean_feature(
                        df, features['title'] == 'Rare', 'has_rare_title'
                    )
            
            # Name length
            with self._time_feature('name_length', features):
                features['name_length'] = df['name'].str.len()
            
        return features
    
    def get_cabin_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Cabin-related features."""
        features = {}
        
        if 'cabin' in df.columns:
            # Cabin deck (first letter)
            with self._time_feature('cabin_deck_extraction', features):
                cabin_deck = df['cabin'].str.extract(r'^([A-Z])')
                if not cabin_deck.empty:
                    features['cabin_deck'] = cabin_deck[0].fillna('Unknown').astype('category')
                    features['has_cabin'] = self._create_boolean_feature(
                        df, df['cabin'].notna(), 'has_cabin'
                    )
                    
                    # Deck indicators for known decks
                    for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                        features[f'deck_{deck}'.lower()] = self._create_boolean_feature(
                            df, features['cabin_deck'] == deck, f'deck_{deck}'.lower()
                        )
            
            # Number of cabins (multiple cabins indicates wealth)
            with self._time_feature('num_cabins', features):
                features['num_cabins'] = df['cabin'].str.count(' ') + 1
                features['num_cabins'] = features['num_cabins'].fillna(0)
            
            features['multiple_cabins'] = self._create_boolean_feature(
                df, features['num_cabins'] > 1, 'multiple_cabins'
            )
            
        return features
    
    def get_embarkation_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Embarkation port features."""
        features = {}
        
        if 'embarked' in df.columns:
            # Fill missing embarked with mode
            with self._time_feature('embarked_filled', features):
                embarked_filled = df['embarked'].fillna(df['embarked'].mode()[0] if not df['embarked'].mode().empty else 'S')
                features['embarked_filled'] = embarked_filled.astype('category')
            
            # Port indicators
            features['embarked_c'] = self._create_boolean_feature(
                df, embarked_filled == 'C', 'embarked_c'
            )  # Cherbourg
            features['embarked_q'] = self._create_boolean_feature(
                df, embarked_filled == 'Q', 'embarked_q'
            )  # Queenstown
            features['embarked_s'] = self._create_boolean_feature(
                df, embarked_filled == 'S', 'embarked_s'
            )  # Southampton
            
        return features
    
    def get_interaction_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Interaction features between different variables."""
        features = {}
        
        # Sex and age interactions
        if all(col in df.columns for col in ['sex', 'age']):
            with self._time_feature('sex_age_interaction', features):
                age_filled = df['age'].fillna(df['age'].median())
                features['sex_age_interaction'] = df['sex'].astype(str) + '_' + pd.cut(
                    age_filled, bins=[0, 16, 35, 100], labels=['child', 'adult', 'elderly']
                ).astype(str)
            
        # Class and family size interactions
        if 'pclass' in df.columns and all(col in df.columns for col in ['sibsp', 'parch']):
            with self._time_feature('class_family_interaction', features):
                family_size = df['sibsp'] + df['parch'] + 1
                features['class_family_interaction'] = (
                    df['pclass'].astype(str) + '_' + 
                    pd.cut(family_size, bins=[0, 1, 4, 20], labels=['alone', 'small', 'large']).astype(str)
                )
            
        # Sex and class interactions
        if all(col in df.columns for col in ['sex', 'pclass']):
            with self._time_feature('sex_class_interaction', features):
                features['sex_class_interaction'] = df['sex'].astype(str) + '_class' + df['pclass'].astype(str)
            
        return features