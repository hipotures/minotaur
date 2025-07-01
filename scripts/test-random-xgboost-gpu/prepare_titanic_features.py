#!/usr/bin/env python3
"""
Titanic Data Preparation for Feature Benchmark

Loads raw Titanic CSV, applies basic feature engineering and encoding,
saves to parquet format for fast GPU evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class TitanicFeaturePreprocessor:
    """Preprocess Titanic dataset for feature evaluation benchmark."""
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize preprocessor.
        
        Args:
            input_path: Path to raw Titanic CSV file
            output_path: Path to save processed parquet file
        """
        self.input_path = input_path
        self.output_path = output_path
        self.data = None
        
    def load_data(self):
        """Load raw Titanic data."""
        print(f"📁 Loading data from: {self.input_path}")
        
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
        self.data = pd.read_csv(self.input_path)
        print(f"📊 Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
        print(f"📋 Columns: {list(self.data.columns)}")
        
        # Show basic info
        print(f"\n📈 Data info:")
        print(f"   Missing values: {self.data.isnull().sum().sum()}")
        print(f"   Target distribution: {self.data['Survived'].value_counts().to_dict()}")
        
    def clean_data(self):
        """Remove unwanted columns and handle basic cleaning."""
        print("\n🧹 Cleaning data...")
        
        # Remove text columns (names, ticket numbers)
        text_columns = ['Name', 'Ticket']
        available_text_cols = [col for col in text_columns if col in self.data.columns]
        
        if available_text_cols:
            print(f"   Removing text columns: {available_text_cols}")
            self.data = self.data.drop(columns=available_text_cols)
        
        # Keep PassengerId for potential debugging but rename for clarity
        if 'PassengerId' in self.data.columns:
            self.data = self.data.rename(columns={'PassengerId': 'passengerid'})
            
        # Rename target column to lowercase for consistency
        if 'Survived' in self.data.columns:
            self.data = self.data.rename(columns={'Survived': 'survived'})
            
        print(f"   Remaining columns: {list(self.data.columns)}")
        
    def feature_engineering(self):
        """Apply basic feature engineering."""
        print("\n🔧 Feature engineering...")
        
        # Age features
        if 'Age' in self.data.columns:
            print("   Creating age features...")
            self.data['age_filled'] = self.data['Age'].fillna(self.data['Age'].median())
            self.data['age_missing'] = self.data['Age'].isnull().astype(int)
            self.data['is_child'] = (self.data['age_filled'] < 16).astype(int)
            self.data['is_adult'] = ((self.data['age_filled'] >= 16) & (self.data['age_filled'] < 65)).astype(int)
            self.data['is_elderly'] = (self.data['age_filled'] >= 65).astype(int)
            
            # Age groups
            self.data['age_group'] = pd.cut(self.data['age_filled'], 
                                          bins=[0, 16, 30, 50, 100], 
                                          labels=['child', 'young', 'middle', 'old'])
        
        # Family features
        if 'SibSp' in self.data.columns and 'Parch' in self.data.columns:
            print("   Creating family features...")
            self.data['family_size'] = self.data['SibSp'] + self.data['Parch'] + 1
            self.data['is_alone'] = (self.data['family_size'] == 1).astype(int)
            self.data['has_siblings'] = (self.data['SibSp'] > 0).astype(int)
            self.data['has_parents_children'] = (self.data['Parch'] > 0).astype(int)
            self.data['large_family'] = (self.data['family_size'] > 4).astype(int)
            
        # Fare features
        if 'Fare' in self.data.columns:
            print("   Creating fare features...")
            self.data['fare_filled'] = self.data['Fare'].fillna(self.data['Fare'].median())
            self.data['fare_missing'] = self.data['Fare'].isnull().astype(int)
            self.data['fare_per_person'] = self.data['fare_filled'] / self.data['family_size']
            self.data['expensive_ticket'] = (self.data['fare_filled'] > self.data['fare_filled'].quantile(0.75)).astype(int)
            self.data['cheap_ticket'] = (self.data['fare_filled'] < self.data['fare_filled'].quantile(0.25)).astype(int)
            
            # Fare log (handle zeros)
            self.data['fare_log'] = np.log1p(self.data['fare_filled'])
            
            # Fare categories
            self.data['fare_category'] = pd.cut(self.data['fare_filled'], 
                                              bins=[0, 10, 25, 50, 100], 
                                              labels=['low', 'medium', 'high', 'premium'])
        
        # Cabin features
        if 'Cabin' in self.data.columns:
            print("   Creating cabin features...")
            self.data['has_cabin'] = (~self.data['Cabin'].isnull()).astype(int)
            
            # Extract deck from cabin (first letter)
            self.data['cabin_deck'] = self.data['Cabin'].str[0]
            self.data['cabin_deck'] = self.data['cabin_deck'].fillna('Unknown')
            
            # Number of cabins (count spaces + 1)
            self.data['num_cabins'] = self.data['Cabin'].str.count(' ') + 1
            self.data['num_cabins'] = self.data['num_cabins'].fillna(0)
            
        # Embarked features
        if 'Embarked' in self.data.columns:
            print("   Creating embarked features...")
            # Fill missing embarked with most common
            most_common_embarked = self.data['Embarked'].mode()[0]
            self.data['embarked_filled'] = self.data['Embarked'].fillna(most_common_embarked)
            self.data['embarked_missing'] = self.data['Embarked'].isnull().astype(int)
            
        # Passenger class features
        if 'Pclass' in self.data.columns:
            print("   Creating passenger class features...")
            self.data['is_first_class'] = (self.data['Pclass'] == 1).astype(int)
            self.data['is_second_class'] = (self.data['Pclass'] == 2).astype(int)
            self.data['is_third_class'] = (self.data['Pclass'] == 3).astype(int)
            
            # Class-based features
            if 'fare_filled' in self.data.columns:
                class_fare_mean = self.data.groupby('Pclass')['fare_filled'].mean()
                self.data['fare_vs_class_mean'] = self.data.apply(
                    lambda row: row['fare_filled'] / class_fare_mean[row['Pclass']], axis=1
                )
        
        print(f"   Total features after engineering: {len(self.data.columns)}")
        
    def encode_categorical(self):
        """Encode categorical variables."""
        print("\n🔤 Encoding categorical variables...")
        
        # Identify categorical columns
        categorical_columns = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object' and col not in ['passengerid', 'survived']:
                categorical_columns.append(col)
                
        print(f"   Categorical columns found: {categorical_columns}")
        
        # Apply label encoding
        for col in categorical_columns:
            print(f"   Encoding {col}...")
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            
        # One-hot encode some important categoricals
        if 'Sex' in self.data.columns:
            sex_dummies = pd.get_dummies(self.data['Sex'], prefix='sex')
            self.data = pd.concat([self.data, sex_dummies], axis=1)
            
        if 'embarked_filled' in self.data.columns:
            embarked_dummies = pd.get_dummies(self.data['embarked_filled'], prefix='embarked')
            self.data = pd.concat([self.data, embarked_dummies], axis=1)
            
        if 'cabin_deck' in self.data.columns:
            deck_dummies = pd.get_dummies(self.data['cabin_deck'], prefix='deck')
            self.data = pd.concat([self.data, deck_dummies], axis=1)
            
        print(f"   Total columns after encoding: {len(self.data.columns)}")
        
    def create_interaction_features(self):
        """Create some interaction features."""
        print("\n🔗 Creating interaction features...")
        
        # Age-Class interactions
        if 'age_filled' in self.data.columns and 'Pclass' in self.data.columns:
            self.data['age_class_interaction'] = self.data['age_filled'] * self.data['Pclass']
            
        # Fare-Family interactions  
        if 'fare_filled' in self.data.columns and 'family_size' in self.data.columns:
            self.data['fare_family_interaction'] = self.data['fare_filled'] * self.data['family_size']
            
        # Gender-Class interactions
        if 'Sex' in self.data.columns and 'Pclass' in self.data.columns:
            self.data['sex_class_interaction'] = self.data['Sex'] * self.data['Pclass']
            
        print(f"   Added interaction features. Total: {len(self.data.columns)}")
        
    def final_cleanup(self):
        """Final data cleanup and validation."""
        print("\n🧽 Final cleanup...")
        
        # Remove original columns that were transformed
        columns_to_remove = ['Age', 'Fare', 'Cabin', 'Embarked', 'SibSp', 'Parch', 'Sex', 'Pclass']
        available_to_remove = [col for col in columns_to_remove if col in self.data.columns]
        
        if available_to_remove:
            print(f"   Removing original columns: {available_to_remove}")
            self.data = self.data.drop(columns=available_to_remove)
            
        # Handle any remaining NaN values
        nan_counts = self.data.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"   Handling remaining NaN values:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"     {col}: {count} NaN values")
                if self.data[col].dtype in ['float64', 'int64']:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                else:
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                    
        # Convert all columns to numeric where possible
        for col in self.data.columns:
            if col not in ['passengerid', 'survived']:
                try:
                    self.data[col] = pd.to_numeric(self.data[col])
                except:
                    pass
                    
        # Ensure target is integer
        if 'survived' in self.data.columns:
            self.data['survived'] = self.data['survived'].astype(int)
            
        print(f"   Final dataset: {len(self.data)} rows, {len(self.data.columns)} columns")
        print(f"   Feature columns: {len([col for col in self.data.columns if col not in ['passengerid', 'survived']])}")
        
    def save_data(self):
        """Save processed data to parquet format."""
        print(f"\n💾 Saving to: {self.output_path}")
        
        # Create output directory if needed
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        self.data.to_parquet(self.output_path, index=False)
        
        # Show final summary
        print(f"✅ Saved {len(self.data)} rows, {len(self.data.columns)} columns")
        print(f"📊 Target distribution: {self.data['survived'].value_counts().to_dict()}")
        
        # Show sample features
        feature_cols = [col for col in self.data.columns if col not in ['passengerid', 'survived']]
        print(f"🎯 Sample features: {feature_cols[:10]}...")
        print(f"📈 Total features available: {len(feature_cols)}")
        
    def process(self):
        """Run complete preprocessing pipeline."""
        print("🚀 Starting Titanic data preprocessing...")
        
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        self.encode_categorical()
        self.create_interaction_features()
        self.final_cleanup()
        self.save_data()
        
        print("\n✅ Preprocessing completed!")
        return self.data

def main():
    """Main preprocessing execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Titanic Feature Preprocessing')
    parser.add_argument('--input', default='datasets/Titanic/train.csv', 
                       help='Input CSV file path')
    parser.add_argument('--output', default='cache/titanic/features.parquet',
                       help='Output parquet file path')
    
    args = parser.parse_args()
    
    print(f"🚢 Titanic Feature Preprocessing")
    print(f"📁 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    
    # Check input file
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Run preprocessing
    preprocessor = TitanicFeaturePreprocessor(args.input, args.output)
    data = preprocessor.process()
    
    print(f"\n🎯 Ready for benchmark!")
    print(f"   Run: python random_feature_benchmark.py --data {args.output}")

if __name__ == "__main__":
    main()