#!/usr/bin/env python3
"""
cuML vs AutoGluon Benchmark Comparison

Tests feasibility of using cuML RAPIDS as AutoGluon replacement for feature evaluation.
Compares performance of GPU-accelerated cuML algorithms vs AutoGluon TabularPredictor.
"""

import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.svm import SVC as cuSVC
    CUML_AVAILABLE = True
    print("✅ cuML available")
except ImportError:
    CUML_AVAILABLE = False
    print("❌ cuML not available - install with: conda install -c rapidsai cuml")

try:
    import cudf
    CUDF_AVAILABLE = True
    print("✅ cuDF available")
except ImportError:
    CUDF_AVAILABLE = False
    print("❌ cuDF not available")

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
    print("✅ AutoGluon available")
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("❌ AutoGluon not available")

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class cuMLFeatureEvaluator:
    """cuML-based feature subset evaluation for comparison with AutoGluon."""
    
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
        self.use_gpu = CUML_AVAILABLE and CUDF_AVAILABLE
        
        self._load_data()
        print(f"📊 Loaded {len(self.data)} rows, {len(self.feature_columns)} features")
        
    def _load_data(self):
        """Load data into appropriate format (GPU if available)."""
        start_time = time.time()
        
        if self.use_gpu:
            try:
                # Load to GPU memory
                self.data = cudf.read_parquet(self.data_path)
                print("🚀 Data loaded to GPU memory")
                
                # Check GPU memory usage
                import cupy as cp
                mempool = cp.get_default_memory_pool()
                gpu_mem_used = mempool.used_bytes() / 1024**3
                gpu_mem_total = cp.cuda.Device().mem_info[1] / 1024**3
                print(f"   GPU Memory: {gpu_mem_used:.2f}GB / {gpu_mem_total:.2f}GB used")
                
            except Exception as e:
                print(f"⚠️  GPU loading failed: {e}, falling back to CPU")
                self.use_gpu = False
                
        if not self.use_gpu:
            # Fallback to pandas
            self.data = pd.read_parquet(self.data_path)
            print("💻 Data loaded to CPU memory")
            
        # Separate target and features
        self.target = self.data[self.target_column]
        
        # Encode target if it's categorical
        if self.target.dtype == 'object' or self.target.dtype.name == 'category':
            print("🔤 Encoding categorical target...")
            self.label_encoder = LabelEncoder()
            
            # Convert to pandas for label encoding if needed
            if self.use_gpu and hasattr(self.target, 'to_pandas'):
                target_for_encoding = self.target.to_pandas().astype(str)
            else:
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
        
    def evaluate_feature_subset_cuml(self, feature_subset: List[str], 
                                   algorithm: str = 'rf') -> Tuple[float, float]:
        """
        Evaluate a subset of features using cuML algorithms.
        
        Args:
            feature_subset: List of feature column names
            algorithm: 'rf' (RandomForest), 'knn', or 'svc'
            
        Returns:
            Tuple of (accuracy, evaluation_time)
        """
        if not CUML_AVAILABLE:
            raise ImportError("cuML not available")
            
        start_time = time.time()
        
        # Select feature subset
        if self.use_gpu and CUDF_AVAILABLE:
            X = self.data[feature_subset]
            y = self.target
        else:
            # Convert to pandas if needed
            if hasattr(self.data, 'to_pandas'):
                X = self.data[feature_subset].to_pandas()
            else:
                X = self.data[feature_subset]
            y = self.target
            
        # Handle categorical features for cuML
        for col in X.columns:
            if hasattr(X, 'dtypes') and X[col].dtype == 'object':
                if self.use_gpu:
                    X[col] = X[col].astype('category').cat.codes
                else:
                    X[col] = pd.Categorical(X[col]).codes
                    
        # Split data
        if self.use_gpu:
            # cuML train_test_split
            from cuml.model_selection import train_test_split as cuml_train_test_split
            X_train, X_test, y_train, y_test = cuml_train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        try:
            # Create and train cuML model
            if algorithm == 'rf':
                model = cuRF(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            elif algorithm == 'knn':
                model = cuKNN(
                    n_neighbors=5
                )
            elif algorithm == 'svc':
                model = cuSVC(
                    kernel='rbf',
                    gamma='scale'
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
                
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and calculate accuracy
            y_pred = model.predict(X_test)
            
            # Convert to numpy if needed for accuracy calculation
            if hasattr(y_test, 'to_numpy'):
                y_test_np = y_test.to_numpy()
            else:
                y_test_np = y_test
                
            if hasattr(y_pred, 'to_numpy'):
                y_pred_np = y_pred.to_numpy()
            else:
                y_pred_np = y_pred
                
            accuracy = accuracy_score(y_test_np, y_pred_np)
            
        except Exception as e:
            print(f"❌ cuML evaluation failed for {len(feature_subset)} features: {e}")
            accuracy = 0.0
            
        eval_time = time.time() - start_time
        return accuracy, eval_time
        
    def evaluate_feature_subset_autogluon(self, feature_subset: List[str]) -> Tuple[float, float]:
        """
        Evaluate a subset of features using AutoGluon (for comparison).
        
        Args:
            feature_subset: List of feature column names
            
        Returns:
            Tuple of (accuracy, evaluation_time)
        """
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon not available")
            
        start_time = time.time()
        
        # Select feature subset and convert to pandas
        if hasattr(self.data, 'to_pandas'):
            X = self.data[feature_subset].to_pandas()
        else:
            X = self.data[feature_subset]
            
        y = self.target
        
        # Create combined dataset for AutoGluon
        df = X.copy()
        df['target'] = y
        
        # Split for train/test
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=y)
        
        try:
            # Configure AutoGluon for speed
            predictor = TabularPredictor(
                label='target',
                eval_metric='accuracy',
                verbosity=0
            )
            
            # Fast training
            predictor.fit(
                train_df,
                time_limit=30,
                presets='medium_quality_faster_train',
                num_bag_folds=2,
                num_bag_sets=1,
                num_stack_levels=0,
                hyperparameters={
                    'GBM': {'num_boost_round': 100},
                    'XGB': {'n_estimators': 100},
                    'RF': {'n_estimators': 100},
                }
            )
            
            # Predict and calculate accuracy
            predictions = predictor.predict(test_df.drop('target', axis=1))
            accuracy = accuracy_score(test_df['target'], predictions)
            
            # Cleanup
            predictor.delete_models(models_to_keep=[], dry_run=False)
            del predictor
            
        except Exception as e:
            print(f"❌ AutoGluon evaluation failed for {len(feature_subset)} features: {e}")
            accuracy = 0.0
            
        eval_time = time.time() - start_time
        return accuracy, eval_time
        
    def comparative_benchmark(self, n_iterations: int = 5,
                            min_features: int = 5,
                            max_features: int = 15) -> Dict:
        """
        Run comparative benchmark between cuML and AutoGluon.
        
        Args:
            n_iterations: Number of random subsets to test
            min_features: Minimum number of features per subset
            max_features: Maximum number of features per subset
            
        Returns:
            Dictionary with benchmark results
        """
        if max_features is None:
            max_features = len(self.feature_columns)
            
        max_features = min(max_features, len(self.feature_columns))
        
        print(f"\n🎲 Starting cuML vs AutoGluon Comparative Benchmark")
        print(f"📊 {n_iterations} iterations, {min_features}-{max_features} features per subset")
        print(f"🎯 Target: {self.target_column}")
        print(f"💾 Using {'GPU' if self.use_gpu else 'CPU'} acceleration")
        
        results = {
            'cuml': {
                'iterations': [],
                'feature_counts': [],
                'scores': [],
                'eval_times': [],
                'algorithm': 'RandomForest'
            },
            'autogluon': {
                'iterations': [],
                'feature_counts': [],
                'scores': [],
                'eval_times': []
            },
            'feature_subsets': [],
            'timestamp': datetime.now().isoformat(),
            'total_features_available': len(self.feature_columns),
            'gpu_accelerated': self.use_gpu
        }
        
        for i in range(n_iterations):
            # Random feature selection
            n_features = random.randint(min_features, max_features)
            feature_subset = random.sample(self.feature_columns, n_features)
            results['feature_subsets'].append(feature_subset)
            
            print(f"\n📊 Iteration {i+1}/{n_iterations}: {n_features} features")
            
            # cuML evaluation
            if CUML_AVAILABLE:
                cuml_score, cuml_time = self.evaluate_feature_subset_cuml(feature_subset)
                results['cuml']['iterations'].append(i + 1)
                results['cuml']['feature_counts'].append(n_features)
                results['cuml']['scores'].append(cuml_score)
                results['cuml']['eval_times'].append(cuml_time)
                print(f"   cuML: score={cuml_score:.4f}, time={cuml_time:.3f}s")
            
            # AutoGluon evaluation
            if AUTOGLUON_AVAILABLE:
                ag_score, ag_time = self.evaluate_feature_subset_autogluon(feature_subset)
                results['autogluon']['iterations'].append(i + 1)
                results['autogluon']['feature_counts'].append(n_features)
                results['autogluon']['scores'].append(ag_score)
                results['autogluon']['eval_times'].append(ag_time)
                print(f"   AutoGluon: score={ag_score:.4f}, time={ag_time:.3f}s")
                
        # Summary statistics
        if CUML_AVAILABLE and results['cuml']['eval_times']:
            cuml_avg_time = np.mean(results['cuml']['eval_times'])
            cuml_avg_score = np.mean(results['cuml']['scores'])
            cuml_evals_per_hour = 3600 / cuml_avg_time if cuml_avg_time > 0 else 0
            
            results['cuml']['summary'] = {
                'avg_evaluation_time': cuml_avg_time,
                'avg_score': cuml_avg_score,
                'evaluations_per_hour': cuml_evals_per_hour
            }
            
        if AUTOGLUON_AVAILABLE and results['autogluon']['eval_times']:
            ag_avg_time = np.mean(results['autogluon']['eval_times'])
            ag_avg_score = np.mean(results['autogluon']['scores'])
            ag_evals_per_hour = 3600 / ag_avg_time if ag_avg_time > 0 else 0
            
            results['autogluon']['summary'] = {
                'avg_evaluation_time': ag_avg_time,
                'avg_score': ag_avg_score,
                'evaluations_per_hour': ag_evals_per_hour
            }
            
        # Print comparison
        print(f"\n📋 COMPARATIVE BENCHMARK RESULTS:")
        
        if CUML_AVAILABLE and results['cuml']['eval_times']:
            print(f"🚀 cuML (GPU RandomForest):")
            print(f"   Average time: {cuml_avg_time:.3f}s")
            print(f"   Average accuracy: {cuml_avg_score:.4f}")
            print(f"   Evaluations/hour: {cuml_evals_per_hour:.0f}")
            
        if AUTOGLUON_AVAILABLE and results['autogluon']['eval_times']:
            print(f"🎯 AutoGluon:")
            print(f"   Average time: {ag_avg_time:.3f}s")
            print(f"   Average accuracy: {ag_avg_score:.4f}")
            print(f"   Evaluations/hour: {ag_evals_per_hour:.0f}")
            
        # Speed comparison
        if (CUML_AVAILABLE and AUTOGLUON_AVAILABLE and 
            results['cuml']['eval_times'] and results['autogluon']['eval_times']):
            speedup = ag_avg_time / cuml_avg_time
            print(f"\n⚡ PERFORMANCE COMPARISON:")
            print(f"   cuML is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than AutoGluon")
            
            if speedup > 5:
                print("✅ EXCELLENT: cuML provides significant speedup!")
            elif speedup > 2:
                print("✅ GOOD: cuML provides good speedup")
            elif speedup > 0.8:
                print("⚖️  COMPARABLE: Similar performance")
            else:
                print("❌ SLOWER: AutoGluon is faster")
                
        return results
        
    def save_results(self, results: Dict, output_path: str = None):
        """Save benchmark results to JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"cuml_vs_autogluon_benchmark_{timestamp}.json"
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Results saved to: {output_path}")

def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='cuML vs AutoGluon Benchmark')
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
        
    # Check dependencies
    if not CUML_AVAILABLE:
        print("⚠️  cuML not available. Install with: conda install -c rapidsai cuml")
        
    if not AUTOGLUON_AVAILABLE:
        print("⚠️  AutoGluon not available. Install with: pip install autogluon")
        
    if not CUML_AVAILABLE and not AUTOGLUON_AVAILABLE:
        print("❌ Neither cuML nor AutoGluon available. Cannot run benchmark.")
        return
    
    print(f"🚀 cuML vs AutoGluon Comparative Benchmark")
    print(f"📁 Data: {args.data}")
    print(f"🎯 Target: {args.target}")
    
    # Initialize evaluator
    evaluator = cuMLFeatureEvaluator(args.data, args.target)
    
    # Run benchmark
    results = evaluator.comparative_benchmark(
        n_iterations=args.iterations,
        min_features=args.min_features,
        max_features=args.max_features
    )
    
    # Save results
    evaluator.save_results(results, args.output)
    
    print("\n✅ Comparative benchmark completed!")

if __name__ == "__main__":
    main()