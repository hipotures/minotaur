#!/usr/bin/env python3
"""
Universal Feature Preparation Script

Handles data preprocessing, feature engineering, and categorical encoding
for any tabular dataset. Optimized for XGBoost compatibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class UniversalFeaturePreprocessor:
    """Universal preprocessor for tabular data with automatic feature engineering."""
    
    def __init__(self, target_column: str = None):
        """
        Initialize preprocessor.
        
        Args:
            target_column: Name of target column (if None, will try to detect)
        """
        self.target_column = target_column
        self.categorical_columns = []
        self.numeric_columns = []
        self.text_columns = []
        self.original_shape = None
        
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect column types.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with column type classifications
        """
        column_types = {
            'categorical': [],
            'numeric': [],
            'text': [],
            'datetime': [],
            'id': []
        }
        
        for col in df.columns:
            if col == self.target_column:
                continue
                
            # Check for ID columns (typically have 'id' in name and high cardinality)
            if 'id' in col.lower() and df[col].nunique() > len(df) * 0.8:
                column_types['id'].append(col)
                continue
            
            # Check for datetime
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                column_types['datetime'].append(col)
                continue
            
            # Check for text (object type with high cardinality or long strings)
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                avg_length = df[col].astype(str).str.len().mean()
                
                if unique_ratio > 0.8 or avg_length > 50:
                    column_types['text'].append(col)
                else:
                    column_types['categorical'].append(col)
                continue
            
            # Check for categorical (low cardinality numeric or already category)
            if df[col].dtype == 'category' or (df[col].dtype in ['int64', 'float64'] and df[col].nunique() < 20):
                column_types['categorical'].append(col)
                continue
            
            # Default to numeric
            column_types['numeric'].append(col)
        
        return column_types
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        print("🧹 Cleaning data...")
        df_clean = df.copy()
        
        # Remove completely empty columns
        empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
        if empty_cols:
            print(f"   Removing {len(empty_cols)} empty columns: {empty_cols}")
            df_clean = df_clean.drop(columns=empty_cols)
        
        # Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_rows:
            print(f"   Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Clean text columns (remove special characters, normalize case)
        text_cols = df_clean.select_dtypes(include=['object']).columns
        for col in text_cols:
            if col != self.target_column:
                # Strip whitespace and normalize case
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
                # Replace multiple spaces with single space
                df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
                # Replace empty strings with NaN
                df_clean[col] = df_clean[col].replace(['', 'nan', 'null', 'none'], np.nan)
        
        print(f"   Data shape after cleaning: {df_clean.shape}")
        return df_clean
    
    def feature_engineering(self, df: pd.DataFrame, column_types: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Apply feature engineering based on column types.
        
        Args:
            df: Input dataframe
            column_types: Column type classifications
            
        Returns:
            Dataframe with engineered features
        """
        print("⚙️  Engineering features...")
        df_features = df.copy()
        
        # Numeric feature engineering
        numeric_cols = column_types['numeric']
        if numeric_cols:
            print(f"   Processing {len(numeric_cols)} numeric columns")
            
            for col in numeric_cols:
                # Handle missing values with median
                if df_features[col].isnull().any():
                    median_val = df_features[col].median()
                    df_features[col] = df_features[col].fillna(median_val)
                    # Add missing indicator
                    df_features[f'{col}_was_missing'] = df[col].isnull().astype(int)
                
                # Create binned versions for high-cardinality numeric features
                if df_features[col].nunique() > 50:
                    try:
                        df_features[f'{col}_binned'] = pd.qcut(df_features[col], q=10, duplicates='drop', labels=False)
                    except:
                        df_features[f'{col}_binned'] = pd.cut(df_features[col], bins=10, duplicates='drop', labels=False)
                
                # Create log versions for skewed positive features
                if df_features[col].min() > 0:
                    skewness = df_features[col].skew()
                    if abs(skewness) > 1:
                        df_features[f'{col}_log'] = np.log1p(df_features[col])
                
                # Create squared versions for potentially non-linear relationships
                if df_features[col].nunique() > 10:
                    df_features[f'{col}_squared'] = df_features[col] ** 2
        
        # Categorical feature engineering
        categorical_cols = column_types['categorical']
        if categorical_cols:
            print(f"   Processing {len(categorical_cols)} categorical columns")
            
            for col in categorical_cols:
                # Handle missing values with mode or 'unknown'
                if df_features[col].isnull().any():
                    if df_features[col].dtype == 'object':
                        df_features[col] = df_features[col].fillna('unknown')
                    else:
                        mode_val = df_features[col].mode()
                        if len(mode_val) > 0:
                            df_features[col] = df_features[col].fillna(mode_val[0])
                        else:
                            df_features[col] = df_features[col].fillna(-1)
                
                # Create frequency encoding
                freq_map = df_features[col].value_counts().to_dict()
                df_features[f'{col}_frequency'] = df_features[col].map(freq_map)
                
                # Create rare category indicator
                rare_threshold = len(df_features) * 0.01  # 1% threshold
                rare_categories = [cat for cat, count in freq_map.items() if count < rare_threshold]
                df_features[f'{col}_is_rare'] = df_features[col].isin(rare_categories).astype(int)
        
        # DateTime feature engineering
        datetime_cols = column_types['datetime']
        if datetime_cols:
            print(f"   Processing {len(datetime_cols)} datetime columns")
            
            for col in datetime_cols:
                dt_col = pd.to_datetime(df_features[col], errors='coerce')
                
                # Extract temporal features
                df_features[f'{col}_year'] = dt_col.dt.year
                df_features[f'{col}_month'] = dt_col.dt.month
                df_features[f'{col}_day'] = dt_col.dt.day
                df_features[f'{col}_dayofweek'] = dt_col.dt.dayofweek
                df_features[f'{col}_quarter'] = dt_col.dt.quarter
                df_features[f'{col}_is_weekend'] = dt_col.dt.dayofweek.isin([5, 6]).astype(int)
                
                # Drop original datetime column
                df_features = df_features.drop(columns=[col])
        
        # Text feature engineering (basic)
        text_cols = column_types['text']
        if text_cols:
            print(f"   Processing {len(text_cols)} text columns")
            
            for col in text_cols:
                # Basic text statistics
                df_features[f'{col}_length'] = df_features[col].astype(str).str.len()
                df_features[f'{col}_word_count'] = df_features[col].astype(str).str.split().str.len()
                df_features[f'{col}_unique_words'] = df_features[col].astype(str).apply(lambda x: len(set(x.split())))
                
                # Drop original text column (too high dimensional for XGBoost)
                df_features = df_features.drop(columns=[col])
        
        # Remove ID columns
        id_cols = column_types['id']
        if id_cols:
            print(f"   Removing {len(id_cols)} ID columns: {id_cols}")
            df_features = df_features.drop(columns=id_cols)
        
        print(f"   Feature engineering complete: {df_features.shape}")
        return df_features
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables for XGBoost compatibility.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with encoded categorical variables
        """
        print("🔤 Encoding categorical variables...")
        df_encoded = df.copy()
        
        # Get categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        print(f"   Encoding {len(categorical_cols)} categorical columns")
        
        for col in categorical_cols:
            # Convert to string first to handle mixed types
            df_encoded[col] = df_encoded[col].astype(str)
            
            # Use pandas categorical for memory efficiency and XGBoost compatibility
            df_encoded[col] = pd.Categorical(df_encoded[col])
            
            # Convert to category dtype (XGBoost can handle this with enable_categorical=True)
            df_encoded[col] = df_encoded[col].astype('category')
        
        # Ensure all remaining columns are numeric or category
        for col in df_encoded.columns:
            if col != self.target_column and df_encoded[col].dtype == 'object':
                # Force convert any remaining object columns to category
                df_encoded[col] = pd.Categorical(df_encoded[col]).astype('category')
        
        print(f"   Categorical encoding complete")
        return df_encoded
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Args:
            df: Input dataframe
            
        Returns:
            Fully processed dataframe
        """
        print(f"🚀 Starting feature preprocessing pipeline")
        print(f"   Input shape: {df.shape}")
        self.original_shape = df.shape
        
        # Auto-detect target column if not specified
        if self.target_column is None:
            potential_targets = ['target', 'label', 'y', 'class', 'survived', 'prediction']
            for col in potential_targets:
                if col in df.columns:
                    self.target_column = col
                    print(f"   Auto-detected target column: {col}")
                    break
            
            if self.target_column is None:
                print("   ⚠️  No target column specified or detected")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Detect column types
        column_types = self.detect_column_types(df_clean)
        print(f"   Column types detected:")
        for col_type, cols in column_types.items():
            if cols:
                print(f"     {col_type}: {len(cols)} columns")
        
        # Step 3: Feature engineering
        df_features = self.feature_engineering(df_clean, column_types)
        
        # Step 4: Encode categorical variables
        df_final = self.encode_categorical(df_features)
        
        print(f"✅ Preprocessing complete!")
        print(f"   Final shape: {df_final.shape}")
        print(f"   Features added: {df_final.shape[1] - self.original_shape[1]}")
        
        # Final data type summary
        print(f"   Final data types:")
        print(f"     Numeric: {len(df_final.select_dtypes(include=[np.number]).columns)}")
        print(f"     Categorical: {len(df_final.select_dtypes(include=['category']).columns)}")
        print(f"     Other: {len(df_final.select_dtypes(exclude=[np.number, 'category']).columns)}")
        
        return df_final

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Universal Feature Preparation for ML')
    parser.add_argument('--input', required=True, help='Input CSV or parquet file')
    parser.add_argument('--output', required=True, help='Output parquet file')
    parser.add_argument('--target', help='Target column name (auto-detect if not specified)')
    parser.add_argument('--sep', default=',', help='CSV separator (default: comma)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    if args.target:
        print(f"🎯 Target: {args.target}")
    
    # Load data
    print("📖 Loading data...")
    if input_path.suffix.lower() == '.csv':
        df = pd.read_csv(args.input, sep=args.sep)
    elif input_path.suffix.lower() in ['.parquet', '.pq']:
        df = pd.read_parquet(args.input)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    print(f"   Loaded {len(df)} rows x {len(df.columns)} columns")
    
    # Process data
    preprocessor = UniversalFeaturePreprocessor(target_column=args.target)
    df_processed = preprocessor.process(df)
    
    # Save processed data
    print(f"💾 Saving to: {args.output}")
    df_processed.to_parquet(args.output, index=False)
    
    # Display sample
    print("\n📊 Sample of processed data:")
    print(df_processed.head())
    
    print("\n✅ Feature preparation completed successfully!")

if __name__ == "__main__":
    main()