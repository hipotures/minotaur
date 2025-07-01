#!/usr/bin/env python3
"""
AutoGluon Benchmark for S5E6 Dataset

Direct comparison with XGBoost GPU benchmark using the same dataset and evaluation methodology.
"""

import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
    print("✅ AutoGluon available")
except ImportError:
    AUTOGLUON_AVAILABLE = False
    raise ImportError("AutoGluon required for this benchmark")

class AutoGluonFeatureEvaluator:
    """AutoGluon feature subset evaluation for comparison with XGBoost."""
    
    def __init__(self, data_path: str, target_column: str = 'Fertilizer Name'):
        """
        Initialize evaluator with pre-processed data.
        
        Args:
            data_path: Path to parquet file with all features
            target_column: Name of target column
        """
        self.data_path = data_path
        self.target_column = target_column
        self.data = None
        self.target = None
        self.feature_columns = None
        self.label_encoder = None
        
        self._load_data()
        print(f"📊 Loaded {len(self.data)} rows, {len(self.feature_columns)} features")
        
    def _load_data(self):
        """Load data."""
        start_time = time.time()
        
        # Load to pandas
        self.data = pd.read_parquet(self.data_path)
        print("💻 Data loaded to CPU memory")
        
        # Separate target and features
        self.target = self.data[self.target_column]
        
        # Encode target if it's categorical (string)
        if self.target.dtype == 'object' or self.target.dtype.name == 'category':
            print("🔤 Encoding categorical target...")
            self.label_encoder = LabelEncoder()
            target_for_encoding = self.target.astype(str)
            self.target = self.label_encoder.fit_transform(target_for_encoding)
            print(f"   Target classes: {len(self.label_encoder.classes_)} ({list(self.label_encoder.classes_[:5])}{'...' if len(self.label_encoder.classes_) > 5 else ''})")
        else:
            self.label_encoder = None
            
        self.feature_columns = [col for col in self.data.columns 
                               if col != self.target_column and col != 'passengerid']
        
        # Remove target from feature data
        self.data = self.data[self.feature_columns]
        
        load_time = time.time() - start_time
        print(f"⏱️  Data loading time: {load_time:.3f}s")
        
    def evaluate_feature_subset(self, feature_subset: List[str], 
                               cv_folds: int = 3) -> Tuple[float, float]:
        """
        Evaluate a subset of features using AutoGluon.
        
        Args:
            feature_subset: List of feature column names
            cv_folds: Number of CV folds (not used for AutoGluon, kept for compatibility)
            
        Returns:
            Tuple of (mean_score, evaluation_time)
        """
        start_time = time.time()
        
        # Select feature subset
        X = self.data[feature_subset].copy()
        y = self.target.copy()
        
        # Create combined dataset for AutoGluon
        df = X.copy()
        df['target'] = y
        
        # Split for train/test
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=y)
        
        try:
            # Configure AutoGluon for speed
            predictor = TabularPredictor(
                label='target',
                eval_metric='accuracy',
                verbosity=0
            )
            
            # Fast training with minimal hyperparameter tuning
            predictor.fit(
                train_df,
                time_limit=30,  # 30 seconds max
                presets='medium_quality_faster_train',
                num_bag_folds=2,  # Reduced from default
                num_bag_sets=1,   # Reduced from default
                num_stack_levels=0,  # No stacking for speed
                hyperparameters={
                    'GBM': {'num_boost_round': 100},  # Limit boosting rounds
                    'XGB': {'n_estimators': 100},
                    'RF': {'n_estimators': 100},
                    'XT': {'n_estimators': 100},
                }
            )
            
            # Predict and calculate accuracy
            predictions = predictor.predict(test_df.drop('target', axis=1))
            accuracy = accuracy_score(test_df['target'], predictions)
            
            # Cleanup to save memory
            predictor.delete_models(models_to_keep=[], dry_run=False)
            del predictor
            
        except Exception as e:
            print(f"❌ AutoGluon evaluation failed for {len(feature_subset)} features: {e}")
            accuracy = 0.0
            
        eval_time = time.time() - start_time
        return accuracy, eval_time
        
    def random_feature_selection_benchmark(self, 
                                         n_iterations: int = 10,
                                         min_features: int = 5,
                                         max_features: int = None) -> Dict:
        """
        Run random feature selection benchmark using AutoGluon.
        
        Args:
            n_iterations: Number of random subsets to test
            min_features: Minimum number of features per subset
            max_features: Maximum number of features per subset (None = all)
            
        Returns:
            Dictionary with benchmark results
        """
        if max_features is None:
            max_features = len(self.feature_columns)
            
        max_features = min(max_features, len(self.feature_columns))
        
        print(f"\n🎲 Starting AutoGluon Feature Selection Benchmark")
        print(f"📊 {n_iterations} iterations, {min_features}-{max_features} features per subset")
        print(f"🎯 Target: {self.target_column}")
        print(f"💾 Using AutoGluon TabularPredictor")
        
        results = {
            'iterations': [],
            'feature_counts': [],
            'scores': [],
            'eval_times': [],
            'feature_subsets': [],
            'timestamp': datetime.now().isoformat(),
            'total_features_available': len(self.feature_columns),
            'model_type': 'autogluon'
        }
        
        best_score = 0.0
        best_features = []
        total_time = 0.0
        
        for i in range(n_iterations):
            # Random feature count
            n_features = random.randint(min_features, max_features)
            
            # Random feature selection
            feature_subset = random.sample(self.feature_columns, n_features)
            
            # Evaluate subset
            score, eval_time = self.evaluate_feature_subset(feature_subset)
            
            # Track results
            results['iterations'].append(i + 1)
            results['feature_counts'].append(n_features)
            results['scores'].append(score)
            results['eval_times'].append(eval_time)
            results['feature_subsets'].append(feature_subset)
            
            total_time += eval_time
            
            # Track best result
            if score > best_score:
                best_score = score
                best_features = feature_subset.copy()
                
            # Progress reporting
            print(f"📊 Iteration {i+1:3d}/{n_iterations}: "
                  f"features={n_features:2d}, score={score:.4f}, "
                  f"time={eval_time:.3f}s")
                
        # Summary statistics
        avg_eval_time = np.mean(results['eval_times'])
        avg_score = np.mean(results['scores'])
        std_score = np.std(results['scores'])
        
        summary = {
            'total_iterations': n_iterations,
            'total_time': total_time,
            'avg_evaluation_time': avg_eval_time,
            'avg_score': avg_score,
            'std_score': std_score,
            'best_score': best_score,
            'best_feature_count': len(best_features),
            'best_features': best_features,
            'evaluations_per_hour': 3600 / avg_eval_time if avg_eval_time > 0 else 0
        }
        
        results['summary'] = summary
        
        print(f"\n📋 AUTOGLUON BENCHMARK RESULTS:")
        print(f"⏱️  Average evaluation time: {avg_eval_time:.3f}s")
        print(f"📊 Average accuracy: {avg_score:.4f} ± {std_score:.4f}")
        print(f"🏆 Best accuracy: {best_score:.4f} ({len(best_features)} features)")
        print(f"🚀 Evaluations per hour: {summary['evaluations_per_hour']:.0f}")
        print(f"💡 Total time: {total_time:.1f}s for {n_iterations} evaluations")
        
        # Compare with XGBoost estimates
        print(f"\n🔄 COMPARISON WITH XGBOOST GPU:")
        xgb_eval_time = 5.114  # From previous benchmark
        xgb_evals_per_hour = 3600 / xgb_eval_time
        speedup = summary['evaluations_per_hour'] / xgb_evals_per_hour
        
        print(f"   AutoGluon: {summary['evaluations_per_hour']:.0f} evals/hour")
        print(f"   XGBoost GPU: {xgb_evals_per_hour:.0f} evals/hour")
        print(f"   AutoGluon speedup: {speedup:.1f}x")
        
        if speedup > 1:
            print("✅ AutoGluon is FASTER than XGBoost GPU!")
        elif speedup > 0.8:
            print("⚖️  AutoGluon and XGBoost GPU are COMPARABLE")
        else:
            print("❌ AutoGluon is SLOWER than XGBoost GPU")
            
        return results
        
    def save_results(self, results: Dict, output_path: str = None):
        """Save benchmark results to JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"autogluon_feature_benchmark_{timestamp}.json"
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Results saved to: {output_path}")

def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoGluon Feature Selection Benchmark')
    parser.add_argument('--data', required=True, help='Path to parquet data file')
    parser.add_argument('--target', default='Fertilizer Name', help='Target column name')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
    parser.add_argument('--min-features', type=int, default=5, help='Minimum features per subset')
    parser.add_argument('--max-features', type=int, default=15, help='Maximum features per subset')
    parser.add_argument('--output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Validate data file
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    print(f"🚀 AutoGluon Feature Selection Benchmark")
    print(f"📁 Data: {args.data}")
    print(f"🎯 Target: {args.target}")
    
    # Initialize evaluator
    evaluator = AutoGluonFeatureEvaluator(args.data, args.target)
    
    # Run benchmark
    results = evaluator.random_feature_selection_benchmark(
        n_iterations=args.iterations,
        min_features=args.min_features,
        max_features=args.max_features
    )
    
    # Save results
    evaluator.save_results(results, args.output)
    
    print("\n✅ AutoGluon benchmark completed!")

if __name__ == "__main__":
    main()