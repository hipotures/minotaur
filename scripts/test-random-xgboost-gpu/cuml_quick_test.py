#!/usr/bin/env python3
"""
Quick cuML GPU Performance Test

Tests basic cuML functionality and measures GPU utilization.
"""

import time
import numpy as np
import cudf
import cuml
import cupy as cp
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_cuml_gpu():
    """Test cuML GPU performance with simple dataset."""
    print("🚀 cuML GPU Performance Test")
    print(f"cuML version: {cuml.__version__}")
    print(f"cuDF version: {cudf.__version__}")
    
    # Create larger test dataset
    n_samples = 100000
    n_features = 10
    
    print(f"\n📊 Creating dataset: {n_samples} samples, {n_features} features")
    
    # Generate data on GPU
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = np.random.randint(0, 3, n_samples).astype(np.int32)
    
    # Convert to cuDF (GPU)
    start_time = time.time()
    X = cudf.DataFrame(X_np)
    y = cudf.Series(y_np)
    load_time = time.time() - start_time
    
    print(f"⏱️  Data loading to GPU: {load_time:.3f}s")
    
    # Check GPU memory usage
    mempool = cp.get_default_memory_pool()
    gpu_mem_before = mempool.used_bytes() / 1024**2
    print(f"💾 GPU Memory used: {gpu_mem_before:.1f}MB")
    
    # Split data on GPU
    print("\n🔄 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train cuML RandomForest on GPU
    print("🌲 Training cuML RandomForest...")
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"⏱️  Training time: {train_time:.3f}s")
    
    # Check GPU memory after training
    gpu_mem_after = mempool.used_bytes() / 1024**2
    print(f"💾 GPU Memory after training: {gpu_mem_after:.1f}MB")
    
    # Predict
    print("🎯 Making predictions...")
    start_time = time.time()
    y_pred = rf.predict(X_test)
    predict_time = time.time() - start_time
    
    print(f"⏱️  Prediction time: {predict_time:.3f}s")
    
    # Calculate accuracy
    if hasattr(y_test, 'to_numpy'):
        y_test_np = y_test.to_numpy()
    else:
        y_test_np = y_test
        
    if hasattr(y_pred, 'to_numpy'):
        y_pred_np = y_pred.to_numpy()
    else:
        y_pred_np = y_pred
        
    accuracy = accuracy_score(y_test_np, y_pred_np)
    
    print(f"\n📊 Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Training time: {train_time:.3f}s")
    print(f"   Prediction time: {predict_time:.3f}s")
    print(f"   Total time: {train_time + predict_time:.3f}s")
    print(f"   GPU Memory peak: {gpu_mem_after:.1f}MB")
    
    # Performance metrics
    samples_per_sec_train = len(X_train) / train_time
    samples_per_sec_predict = len(X_test) / predict_time
    
    print(f"\n⚡ Performance:")
    print(f"   Training: {samples_per_sec_train:.0f} samples/sec")
    print(f"   Prediction: {samples_per_sec_predict:.0f} samples/sec")
    
    # Estimate for benchmark comparison
    evaluation_time = train_time + predict_time
    evals_per_hour = 3600 / evaluation_time if evaluation_time > 0 else 0
    
    print(f"\n🎯 MCTS Evaluation Estimate:")
    print(f"   Per evaluation: {evaluation_time:.3f}s")
    print(f"   Evaluations/hour: {evals_per_hour:.0f}")
    
    if evals_per_hour > 1000:
        print("✅ EXCELLENT: Very fast for MCTS!")
    elif evals_per_hour > 500:
        print("✅ GOOD: Fast enough for MCTS")
    elif evals_per_hour > 100:
        print("⚠️  MODERATE: Acceptable for MCTS")
    else:
        print("❌ SLOW: May need optimization")

if __name__ == "__main__":
    test_cuml_gpu()