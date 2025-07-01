#!/usr/bin/env python3
"""
Create Large Synthetic Dataset for GPU Benchmarking

Replicates Titanic features to create dataset sizes suitable for GPU testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def create_large_dataset(base_path: str, output_path: str, target_rows: int = 100000):
    """
    Create large synthetic dataset by replicating and augmenting base data.
    
    Args:
        base_path: Path to base parquet file (Titanic features)
        output_path: Path for output large dataset
        target_rows: Target number of rows
    """
    print(f"📚 Loading base dataset from: {base_path}")
    base_df = pd.read_parquet(base_path)
    base_rows = len(base_df)
    
    print(f"📊 Base dataset: {base_rows} rows x {len(base_df.columns)} cols")
    
    # Calculate replication factor
    replication_factor = target_rows // base_rows + 1
    print(f"🔄 Replication factor: {replication_factor}x")
    
    # Create replicated dataset
    large_dfs = []
    
    for i in range(replication_factor):
        df_copy = base_df.copy()
        
        # Add noise to numeric features to create variation
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'survived':  # Don't add noise to target
                # Add small random noise (5% of std dev)
                noise_std = df_copy[col].std() * 0.05
                if noise_std > 0:
                    noise = np.random.normal(0, noise_std, len(df_copy))
                    df_copy[col] = df_copy[col] + noise
        
        # Add batch identifier for tracking
        df_copy['batch_id'] = i
        
        large_dfs.append(df_copy)
        
        if (i + 1) % 100 == 0:
            print(f"⏳ Processed {i + 1}/{replication_factor} batches")
    
    # Combine all batches
    print("🔗 Combining batches...")
    large_df = pd.concat(large_dfs, ignore_index=True)
    
    # Trim to exact target size
    large_df = large_df.iloc[:target_rows]
    
    # Shuffle rows
    print("🔀 Shuffling dataset...")
    large_df = large_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"📊 Final dataset: {len(large_df)} rows x {len(large_df.columns)} cols")
    print(f"💾 Memory usage: {large_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Save to parquet
    print(f"💾 Saving to: {output_path}")
    large_df.to_parquet(output_path, index=False)
    
    print("✅ Large dataset created successfully!")
    
    # Display some stats
    print("\n📈 Dataset Statistics:")
    print(f"   Target distribution: {large_df['survived'].value_counts().to_dict()}")
    print(f"   Numeric features: {len(large_df.select_dtypes(include=[np.number]).columns)}")
    print(f"   Categorical features: {len(large_df.select_dtypes(include=['object', 'category']).columns)}")

def main():
    parser = argparse.ArgumentParser(description='Create large synthetic dataset for GPU benchmarking')
    parser.add_argument('--input', required=True, help='Input parquet file (base dataset)')
    parser.add_argument('--output', required=True, help='Output parquet file (large dataset)')
    parser.add_argument('--rows', type=int, default=100000, help='Target number of rows')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    create_large_dataset(args.input, args.output, args.rows)

if __name__ == "__main__":
    main()