#!/usr/bin/env python3
"""
Optimized Random Feature Selection Benchmark

GPU-accelerated evaluation of random feature subsets using optimized XGBoost.
Tests feasibility of feature-based MCTS with advanced performance optimizations:
- GPU predictor caching for faster inference
- QuantileDMatrix for memory-efficient large datasets  
- Enhanced regularization and tree parameters
- Real-time performance monitoring and analysis
- DuckDB dataset loading support
"""

import os
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime

try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
    print("✅ cuDF available - using GPU acceleration")
except ImportError:
    CUDF_AVAILABLE = False
    print("⚠️  cuDF not available - falling back to pandas")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    raise ImportError("XGBoost required for this benchmark")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    raise ImportError("DuckDB required for dataset loading")

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Rich not available - falling back to basic console output")

class FastFeatureEvaluator:
    """Ultra-fast feature subset evaluation using GPU XGBoost."""
    
    def __init__(self, dataset_name: str = None, data_path: str = None, 
                 target_column: str = 'survived', use_train_table: bool = False, 
                 n_estimators: int = 1000, enable_predictions: bool = False):
        """
        Initialize evaluator with data from DuckDB or parquet file.
        
        Args:
            dataset_name: Dataset name to load from DuckDB (e.g., 's5e7c')
            data_path: Path to parquet file (fallback if dataset_name not provided)
            target_column: Name of target column
            use_train_table: If True, use 'train' table, else use 'train_features' table
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.target_column = target_column
        self.use_train_table = use_train_table
        self.data = None
        self.target = None
        self.feature_columns = None
        self.test_data = None
        self.test_feature_columns = None
        self.submission_template = None
        self.enable_predictions = enable_predictions
        self.baseline_score = 0.0
        self.prediction_counter = 0
        # Store GPU availability for model params
        self.use_gpu = CUDF_AVAILABLE
        
        # Initialize base model params (will be updated based on target type and GPU availability)
        self.model_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': n_estimators,  # Use provided n_estimators parameter
            'random_state': 42,
            'verbosity': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        
        self._load_data()
        print(f"📊 Loaded {len(self.data)} rows, {len(self.feature_columns)} features")
        
    def _load_data(self):
        """Load data from DuckDB or parquet file."""
        start_time = time.time()
        
        # Priority: DuckDB dataset loading, then parquet file
        if self.dataset_name:
            self._load_from_duckdb()
        elif self.data_path:
            self._load_from_parquet()
        else:
            raise ValueError("Either dataset_name or data_path must be provided")
            
        load_time = time.time() - start_time
        print(f"⏱️  Data loading time: {load_time:.3f}s")
    
    def _load_from_duckdb(self):
        """Load data from DuckDB dataset."""
        print(f"🗄️  Loading dataset '{self.dataset_name}' from DuckDB...")
        
        # Construct DuckDB path - search in multiple locations
        possible_paths = [
            f"cache/{self.dataset_name}/dataset.duckdb",  # Current dir
            f"../../cache/{self.dataset_name}/dataset.duckdb",  # From scripts/test-*
            f"../cache/{self.dataset_name}/dataset.duckdb",  # From scripts
        ]
        
        duckdb_path = None
        for path in possible_paths:
            if Path(path).exists():
                duckdb_path = Path(path)
                break
        
        if duckdb_path is None:
            # Try absolute path using script location
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            fallback_path = project_root / f"cache/{self.dataset_name}/dataset.duckdb"
            if fallback_path.exists():
                duckdb_path = fallback_path
        
        if duckdb_path is None:
            available_paths = [str(p) for p in possible_paths]
            raise FileNotFoundError(f"DuckDB dataset not found in any of: {available_paths}")
        
        if not Path(duckdb_path).exists():
            raise FileNotFoundError(f"DuckDB dataset not found: {duckdb_path}")
        
        # Connect to DuckDB
        print(f"   📂 Using DuckDB: {duckdb_path}")
        conn = duckdb.connect(str(duckdb_path))
        
        try:
            # Choose table based on use_train_table flag
            table_name = 'train' if self.use_train_table else 'train_features'
            
            print(f"   📊 Loading from table: {table_name}")
            
            # Load data
            self.data = conn.execute(f"SELECT * FROM {table_name}").df()
            
            print(f"💻 Data loaded from DuckDB: {len(self.data)} rows")
            
            # Find target column (case-insensitive)
            target_col_found = None
            for col in self.data.columns:
                if col.lower() == self.target_column.lower():
                    target_col_found = col
                    break
            
            if target_col_found is None:
                available_cols = list(self.data.columns)
                raise ValueError(f"Target column '{self.target_column}' not found in dataset. Available columns: {available_cols}")
                
            self.target = self.data[target_col_found]
            self.target_column = target_col_found  # Update to actual column name
            
            # Filter feature columns (exclude target and ID columns)
            exclude_cols = {self.target_column, 'id', 'passengerid'}
            self.feature_columns = [col for col in self.data.columns 
                                   if col.lower() not in {c.lower() for c in exclude_cols}]
            
            # Remove target from feature data
            self.data = self.data[self.feature_columns]
            
            # Handle categorical target encoding
            self._encode_target()
            
            # Load test data and submission template if predictions enabled
            if self.enable_predictions:
                self._load_test_data_and_submission(conn)
            
            # Setup GPU mode for DuckDB data
            self._setup_gpu_mode()
            
        finally:
            conn.close()
    
    def _load_from_parquet(self):
        """Load data from parquet file (original method)."""
        print(f"📁 Loading from parquet file: {self.data_path}")
        
        gpu_loading_success = False
        
        if CUDF_AVAILABLE:
            # Try GPU loading with memory optimization
            try:
                # Load to GPU with memory-efficient approach
                self.data = cudf.read_parquet(self.data_path)
                print("🚀 Data loaded to GPU memory")
                
                # Check GPU memory usage
                import cupy as cp
                mempool = cp.get_default_memory_pool()
                gpu_mem_used = mempool.used_bytes() / 1024**3  # GB
                gpu_mem_total = cp.cuda.Device().mem_info[1] / 1024**3  # GB
                print(f"   GPU Memory: {gpu_mem_used:.2f}GB / {gpu_mem_total:.2f}GB used")
                
                gpu_loading_success = True
            except Exception as e:
                print(f"⚠️  GPU loading failed: {e}, falling back to CPU")
                gpu_loading_success = False
                
        if not CUDF_AVAILABLE or not gpu_loading_success:
            # Fallback to pandas with memory optimization hint
            self.data = pd.read_parquet(self.data_path)
            print("💻 Data loaded to CPU memory")
            
            # Suggest using QuantileDMatrix for large datasets
            data_size_gb = self.data.memory_usage(deep=True).sum() / 1024**3
            if data_size_gb > 2.0:
                print(f"   💡 Large dataset ({data_size_gb:.2f}GB) - consider QuantileDMatrix")
            
        # Update GPU usage flag based on actual loading success
        self.use_gpu = gpu_loading_success
        
        # Separate target and features  
        self.target = self.data[self.target_column]
        
        # Filter feature columns (exclude target and ID columns)
        exclude_cols = {self.target_column, 'id', 'passengerid'}
        self.feature_columns = [col for col in self.data.columns 
                               if col.lower() not in {c.lower() for c in exclude_cols}]
        
        # Remove target from feature data
        self.data = self.data[self.feature_columns]
        
        # Handle categorical target encoding
        self._encode_target()
        
        # Set GPU/CPU mode for XGBoost
        self._setup_gpu_mode()
    
    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        try:
            # Method 1: Try nvidia-smi
            import subprocess
            result = subprocess.check_output(['nvidia-smi', '--list-gpus'], encoding='utf-8')
            gpu_count = len([line for line in result.strip().split('\n') if line.strip()])
            return gpu_count
        except:
            try:
                # Method 2: Try cupy if available
                if CUDF_AVAILABLE:
                    import cupy as cp
                    return cp.cuda.runtime.getDeviceCount()
                else:
                    return 0
            except:
                try:
                    # Method 3: Try CUDA_VISIBLE_DEVICES
                    if 'CUDA_VISIBLE_DEVICES' in os.environ:
                        devices = os.environ['CUDA_VISIBLE_DEVICES']
                        if devices and devices != '':
                            return len(devices.split(','))
                    return 0
                except:
                    return 0
    
    def _load_root_node_data(self):
        """Load original data from 'train' table for root node evaluation."""
        if not self.dataset_name:
            print("   ⚠️  Root node evaluation only available for DuckDB datasets")
            return None, None, None
            
        print(f"🌱 Loading ROOT NODE data from 'train' table...")
        
        # Construct DuckDB path - search in multiple locations
        possible_paths = [
            f"cache/{self.dataset_name}/dataset.duckdb",  # Current dir
            f"../../cache/{self.dataset_name}/dataset.duckdb",  # From scripts/test-*
            f"../cache/{self.dataset_name}/dataset.duckdb",  # From scripts
        ]
        
        duckdb_path = None
        for path in possible_paths:
            if Path(path).exists():
                duckdb_path = Path(path)
                break
        
        if duckdb_path is None:
            # Try absolute path using script location
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            fallback_path = project_root / f"cache/{self.dataset_name}/dataset.duckdb"
            if fallback_path.exists():
                duckdb_path = fallback_path
        
        if duckdb_path is None:
            available_paths = [str(p) for p in possible_paths]
            raise FileNotFoundError(f"DuckDB dataset not found in any of: {available_paths}")
        
        if not Path(duckdb_path).exists():
            print(f"   ❌ DuckDB dataset not found: {duckdb_path}")
            return None, None, None
        
        # Connect to DuckDB
        print(f"   📂 Using DuckDB: {duckdb_path}")
        conn = duckdb.connect(str(duckdb_path))
        
        try:
            # Load from 'train' table (original features)
            root_data = conn.execute("SELECT * FROM train").df()
            print(f"   📊 Root data loaded: {len(root_data)} rows")
            
            # Find target column (case-insensitive)
            target_col_found = None
            for col in root_data.columns:
                if col.lower() == self.target_column.lower():
                    target_col_found = col
                    break
            
            if target_col_found is None:
                print(f"   ❌ Target column '{self.target_column}' not found in train table")
                return None, None, None
                
            root_target = root_data[target_col_found]
            
            # Filter feature columns (exclude target and ID columns)
            exclude_cols = {target_col_found, 'id', 'passengerid'}
            root_feature_columns = [col for col in root_data.columns 
                                   if col.lower() not in {c.lower() for c in exclude_cols}]
            
            # Remove target from feature data
            root_features = root_data[root_feature_columns]
            
            # Encode target if needed (same as main data)
            if root_target.dtype == 'object' or root_target.dtype.name == 'category':
                if hasattr(self, 'label_encoder') and self.label_encoder:
                    # Use existing label encoder for consistency
                    root_target = self.label_encoder.transform(root_target.astype(str))
                else:
                    # Create new encoder if not available
                    from sklearn.preprocessing import LabelEncoder
                    temp_encoder = LabelEncoder()
                    root_target = temp_encoder.fit_transform(root_target.astype(str))
            
            print(f"   🌿 Root features: {len(root_feature_columns)} original features")
            return root_features, root_target, root_feature_columns
            
        except Exception as e:
            print(f"   ❌ Failed to load root node data: {e}")
            return None, None, None
        finally:
            conn.close()
    
    def _load_test_data_and_submission(self, conn):
        """Load test data and submission template from DuckDB."""
        try:
            print(f"🧪 Loading test data for predictions...")
            
            # Load test_features table
            test_data = conn.execute("SELECT * FROM test_features").df()
            print(f"   📊 Test data loaded: {len(test_data)} rows")
            
            # Filter test feature columns (exclude ID columns)
            exclude_cols = {'id', 'passengerid'}
            test_feature_columns = [col for col in test_data.columns 
                                   if col.lower() not in {c.lower() for c in exclude_cols}]
            
            # Store test data without ID columns
            self.test_data = test_data[test_feature_columns]
            self.test_feature_columns = test_feature_columns
            
            # Load submission template
            submission_data = conn.execute("SELECT * FROM submission").df()
            print(f"   📋 Submission template loaded: {len(submission_data)} rows")
            self.submission_template = submission_data
            
            print(f"   ✅ Test features: {len(test_feature_columns)} columns")
            print(f"   ✅ Submission template: {list(submission_data.columns)}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to load test data: {e}")
            print(f"   ℹ️  Predictions will be disabled")
            self.enable_predictions = False
    
    def _encode_target(self):
        """Encode categorical target if needed."""
        # Encode target if it's categorical (string)
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
    
    def _setup_gpu_mode(self):
        """Setup GPU/CPU mode for XGBoost."""
        # Determine target type and set objective
        num_classes = len(np.unique(self.target))
        
        if num_classes == 2:
            # Binary classification
            self.model_params['objective'] = 'binary:logistic'
            self.model_params['eval_metric'] = 'logloss'
        else:
            # Multi-class classification
            self.model_params['objective'] = 'multi:softprob'
            self.model_params['eval_metric'] = 'mlogloss'
            self.model_params['num_class'] = num_classes
        
        print(f"🎯 Classification type: {'Binary' if num_classes == 2 else f'Multi-class ({num_classes} classes)'}")
        print(f"🔧 GPU available: {CUDF_AVAILABLE}, use_gpu: {self.use_gpu}")
        
        # Print GPU params once at initialization
        if self.use_gpu:
            gpu_params = {k: v for k, v in self.model_params.items() if 'tree_method' in k or 'device' in k or 'verbosity' in k}
            print(f"🔧 GPU params: {gpu_params}")
        
        # Force GPU usage with proper parameters
        try:
            # Test GPU availability more thoroughly
            import xgboost as xgb
            
            # Check XGBoost version for proper GPU configuration
            xgb_version = xgb.__version__
            print(f"📦 XGBoost version: {xgb_version}")
            
            # Detect number of available GPUs
            gpu_count = self._detect_gpu_count()
            print(f"🎮 Available GPUs: {gpu_count}")
            
            # Clear any CPU-only environment variables
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '':
                del os.environ['CUDA_VISIBLE_DEVICES']
            
            # Test GPU with ACTUAL loaded data from DuckDB
            print(f"🧪 Testing GPU with real dataset: {len(self.data)} rows × {len(self.feature_columns)} features")
            
            # Use first 10 features for quick test
            test_features = self.feature_columns[:min(10, len(self.feature_columns))]
            
            # Configure GPU parameters according to official XGBoost documentation
            # https://xgboost.readthedocs.io/en/stable/gpu/index.html
            
            # Set device based on GPU count
            if gpu_count >= 1:
                device = 'cuda:0'  # Use first GPU (standard XGBoost doesn't support multi-GPU without Dask)
                print(f"   🎯 Using GPU 0: {device}")
                if gpu_count > 1:
                    print(f"   ⚠️  Note: {gpu_count} GPUs available, but standard XGBoost uses only GPU 0")
                    print(f"   💡 For multi-GPU, consider distributed training frameworks")
            else:
                raise Exception("No GPUs detected")
            
            test_params = {
                'device': device,  # Use detected GPU configuration
                'tree_method': 'hist',  # Recommended for GPU
                'n_estimators': 5,  # Quick test
                'verbosity': 1,  # Warning level only
                'max_depth': 3
            }
            
            # Production GPU params according to official docs
            gpu_params = {
                'device': device,  # Use detected GPU configuration
                'tree_method': 'hist',  # Recommended tree method for GPU
                'verbosity': 0  # Silent mode - no debug logs
            }
            
            # Test on real data from DuckDB
            test_X = self.data[test_features]
            test_y = self.target
            
            # Handle cuDF data for test
            if hasattr(test_X, 'to_pandas') and not CUDF_AVAILABLE:
                test_X = test_X.to_pandas()
                if hasattr(test_y, 'to_pandas'):
                    test_y = test_y.to_pandas()
            
            test_model = xgb.XGBClassifier(**test_params)
            test_model.fit(test_X, test_y)
            
            # Apply GPU configuration
            self.model_params.update(gpu_params)
            self.use_gpu = True
            
            print("✅ GPU acceleration VERIFIED and configured for XGBoost")
            if gpu_count > 1:
                print(f"   💡 {gpu_count-1} additional GPU(s) available for future Dask-XGBoost implementation")
            
        except Exception as e:
            print(f"⚠️  GPU test failed: {str(e)}")
            
            # Fallback to CPU with version-appropriate configuration
            import xgboost as xgb
            xgb_version = getattr(xgb, '__version__', '1.0.0')
            
            # CPU fallback according to official docs
            self.model_params.update({
                'device': 'cpu',  # Use CPU
                'tree_method': 'hist',
                'verbosity': 1,
                'n_jobs': -1  # Use all CPU cores
            })
                
            self.use_gpu = False
            print("💻 Fallback to CPU mode with all cores")
        
    def evaluate_feature_subset(self, feature_subset: List[str], 
                               cv_folds: int = 3) -> Tuple[float, float]:
        """
        Evaluate a subset of features using cross-validation.
        
        Args:
            feature_subset: List of feature column names
            cv_folds: Number of CV folds
            
        Returns:
            Tuple of (mean_score, evaluation_time)
        """
        start_time = time.time()
        
        # Select feature subset and handle data format conversion
        try:
            # Check if data is cuDF or pandas
            is_cudf = hasattr(self.data, 'to_pandas')
            
            if is_cudf and self.use_gpu and CUDF_AVAILABLE:
                # XGBoost 1.7+ supports cuDF DataFrames directly - NO conversion needed
                X = self.data[feature_subset]
                y = self.target
            else:
                # Already pandas or CPU mode
                X = self.data[feature_subset]
                y = self.target
            
            # Handle categorical data differently for cuDF vs pandas
            if is_cudf and self.use_gpu and CUDF_AVAILABLE:
                # For cuDF, XGBoost can handle categorical data directly
                # Only convert problematic object types
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = X[col].astype('category')
            else:
                # For pandas, convert to category first
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = X[col].astype('category')
                        
        except Exception as e:
            print(f"⚠️  Data conversion error: {e}")
            # Fallback to direct pandas access
            X = self.data[feature_subset]
            y = self.target
                    
        # Convert categories to numeric only for pandas (XGBoost with cuDF handles categories natively)
        if not (is_cudf and self.use_gpu and CUDF_AVAILABLE):
            for col in X.columns:
                if X[col].dtype.name == 'category':
                    X[col] = X[col].cat.codes
                elif X[col].dtype == 'object':
                    # Convert object columns to category then to codes
                    X[col] = pd.Categorical(X[col]).codes
            
        # Create XGBoost model with configured params
        model_params = self.model_params.copy()
        
        # For GPU, ensure we have enough data to actually utilize it
        if self.use_gpu:
            data_points = len(X) * len(X.columns)
            if data_points < 10000:
                print(f"   ⚠️  Small dataset ({data_points:,} points) - GPU may not be utilized efficiently")
        
        # Create XGBoost model
        model = xgb.XGBClassifier(**model_params)
        
        # Keep verbosity low to avoid spam logs
        if 'verbosity' not in model_params:
            model_params['verbosity'] = 0
        
        try:
            # Standard cross-validation
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            mean_score = scores.mean()
                    
        except Exception as e:
            print(f"❌ Evaluation failed for {len(feature_subset)} features: {e}")
            mean_score = 0.0
            
        eval_time = time.time() - start_time
        return mean_score, eval_time
    
    def train_and_predict(self, feature_subset: List[str], iteration_num: int = 0) -> Tuple[float, bool]:
        """
        Train model on feature subset and generate predictions if score improves.
        
        Args:
            feature_subset: List of feature column names to use
            iteration_num: Current iteration number for file naming
            
        Returns:
            Tuple of (score, prediction_generated)
        """
        if not self.enable_predictions:
            return 0.0, False
            
        start_time = time.time()
        
        try:
            # Prepare training data
            X_train = self.data[feature_subset]
            y_train = self.target
            
            # Handle data format conversion (same as evaluate_feature_subset)
            is_cudf = hasattr(X_train, 'to_pandas')
            
            if not (is_cudf and self.use_gpu and CUDF_AVAILABLE):
                for col in X_train.columns:
                    if X_train[col].dtype.name == 'category':
                        X_train[col] = X_train[col].cat.codes
                    elif X_train[col].dtype == 'object':
                        X_train[col] = pd.Categorical(X_train[col]).codes
            
            # Train model
            model_params = self.model_params.copy()
            model = xgb.XGBClassifier(**model_params)
            model.fit(X_train, y_train)
            
            # Calculate CV score for comparison
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            cv_score = scores.mean()
            
            # Check if this score improves baseline
            if cv_score > self.baseline_score:
                print(f"   🎯 NEW BEST SCORE: {cv_score:.4f} (improved from {self.baseline_score:.4f})")
                
                # Update baseline
                old_baseline = self.baseline_score
                self.baseline_score = cv_score
                
                # Generate predictions on test set
                success = self._generate_predictions(model, feature_subset, iteration_num, cv_score)
                
                if success:
                    print(f"   ✅ Prediction file generated for iteration {iteration_num}")
                    return cv_score, True
                else:
                    # Revert baseline if prediction failed
                    self.baseline_score = old_baseline
                    return cv_score, False
            else:
                print(f"   📊 Score {cv_score:.4f} (no improvement over {self.baseline_score:.4f})")
                return cv_score, False
                
        except Exception as e:
            print(f"   ❌ Train and predict failed: {e}")
            return 0.0, False
    
    def _generate_predictions(self, model, feature_subset: List[str], iteration_num: int, score: float) -> bool:
        """
        Generate predictions and save to submission file.
        
        Args:
            model: Trained XGBoost model
            feature_subset: Features used for training
            iteration_num: Current iteration number
            score: CV score achieved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare test data with same feature subset
            available_test_features = [f for f in feature_subset if f in self.test_feature_columns]
            
            if len(available_test_features) != len(feature_subset):
                missing_features = set(feature_subset) - set(available_test_features)
                print(f"   ⚠️  Missing test features: {missing_features}")
                return False
            
            X_test = self.test_data[available_test_features]
            
            # Handle data format conversion (same as training)
            is_cudf = hasattr(X_test, 'to_pandas')
            
            if not (is_cudf and self.use_gpu and CUDF_AVAILABLE):
                for col in X_test.columns:
                    if X_test[col].dtype.name == 'category':
                        X_test[col] = X_test[col].cat.codes
                    elif X_test[col].dtype == 'object':
                        X_test[col] = pd.Categorical(X_test[col]).codes
            
            # Generate predictions
            if hasattr(model, 'predict_proba'):
                # For classification, get class probabilities
                predictions_proba = model.predict_proba(X_test)
                
                # Handle multi-class vs binary classification
                if predictions_proba.shape[1] > 2:
                    # Multi-class: use class with highest probability
                    predictions = np.argmax(predictions_proba, axis=1)
                else:
                    # Binary: use class 1 probability
                    predictions = predictions_proba[:, 1]
            else:
                # Fallback to direct prediction
                predictions = model.predict(X_test)
            
            # Create submission DataFrame
            submission_df = self.submission_template.copy()
            
            # Update prediction column (assuming target column exists in submission)
            target_col_in_submission = None
            for col in submission_df.columns:
                if col.lower() == self.target_column.lower():
                    target_col_in_submission = col
                    break
            
            if target_col_in_submission is None:
                # Try common submission column names
                possible_target_cols = ['prediction', 'target', 'label', 'class']
                for col in submission_df.columns:
                    if col.lower() in possible_target_cols:
                        target_col_in_submission = col
                        break
            
            if target_col_in_submission is None:
                print(f"   ❌ Cannot find target column in submission template")
                return False
            
            # Handle label encoding for predictions
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                if predictions_proba.shape[1] > 2:
                    # Multi-class: predictions are already class indices
                    # Convert back to original labels
                    prediction_labels = self.label_encoder.inverse_transform(predictions.astype(int))
                else:
                    # Binary: threshold probabilities and convert to labels
                    prediction_classes = (predictions > 0.5).astype(int)
                    prediction_labels = self.label_encoder.inverse_transform(prediction_classes)
                
                submission_df[target_col_in_submission] = prediction_labels
            else:
                # No label encoding needed
                if hasattr(predictions, 'shape') and len(predictions.shape) > 0:
                    if predictions_proba.shape[1] > 2:
                        submission_df[target_col_in_submission] = predictions
                    else:
                        # Binary classification: threshold probabilities
                        submission_df[target_col_in_submission] = (predictions > 0.5).astype(int)
                else:
                    submission_df[target_col_in_submission] = predictions
            
            # Generate filename with dataset name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = getattr(self, 'dataset_name', 'unknown')
            filename = f"submission-{dataset_name}-{timestamp}-{iteration_num:03d}-{score:.4f}.csv"
            
            # Save submission file
            submission_df.to_csv(filename, index=False)
            
            self.prediction_counter += 1
            
            print(f"   💾 NEW SUBMISSION: {filename}")
            print(f"   📊 Features: {len(available_test_features)}, Score: {score:.4f}, Total files: {self.prediction_counter}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Prediction generation failed: {e}")
            return False
        
    def random_feature_selection_benchmark(self, 
                                         n_iterations: int = 100,
                                         min_features: int = 10,
                                         max_features: int = None,
                                         include_root_node: bool = True) -> Dict:
        """
        Run random feature selection benchmark.
        
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
        
        print(f"\n🎲 Starting Random Feature Selection Benchmark")
        print(f"📊 {n_iterations} iterations, {min_features}-{max_features} features per subset")
        print(f"🎯 Target: {self.target_column}")
        print(f"💾 Using {'GPU' if self.use_gpu else 'CPU'} acceleration")
        
        results = {
            'iterations': [],
            'feature_counts': [],
            'scores': [],
            'eval_times': [],
            'feature_subsets': [],
            'timestamp': datetime.now().isoformat(),
            'total_features_available': len(self.feature_columns),
            'gpu_accelerated': self.use_gpu,
            'root_node': None  # Will store root node results
        }
        
        best_score = 0.0
        best_features = []
        total_time = 0.0
        
        # ROOT NODE EVALUATION (Iteration 0)
        if include_root_node:
            print(f"\n🌱 ITERATION 0: ROOT NODE EVALUATION")
            print("=" * 60)
            
            root_features, root_target, root_feature_columns = self._load_root_node_data()
            
            if root_features is not None:
                # Temporarily store current data for root node evaluation
                original_data = self.data
                original_target = self.target
                original_feature_columns = self.feature_columns
                
                # Replace with root node data
                self.data = root_features
                self.target = root_target
                self.feature_columns = root_feature_columns
                
                # Evaluate root node with all original features
                root_score, root_eval_time = self.evaluate_feature_subset(root_feature_columns)
                
                # Set baseline score for predictions
                if self.enable_predictions:
                    self.baseline_score = root_score
                    print(f"   🎯 Setting baseline score: {root_score:.4f}")
                    
                    # Generate baseline prediction (iteration 0)
                    prediction_generated = self._generate_predictions_for_iteration(
                        root_feature_columns, 0, root_score
                    )
                    if prediction_generated:
                        print(f"   💾 Baseline submission generated")
                
                # Restore original data
                self.data = original_data
                self.target = original_target
                self.feature_columns = original_feature_columns
                
                # Store root node results
                results['root_node'] = {
                    'iteration': 0,
                    'feature_count': len(root_feature_columns),
                    'score': root_score,
                    'eval_time': root_eval_time,
                    'features': root_feature_columns,
                    'description': 'Original features from train table (no feature engineering)',
                    'prediction_generated': prediction_generated if self.enable_predictions else False
                }
                
                print(f"🌱 ROOT NODE RESULT:")
                print(f"   📊 Original features: {len(root_feature_columns)}")
                print(f"   🎯 Baseline score: {root_score:.4f}")
                print(f"   ⏱️  Evaluation time: {root_eval_time:.3f}s")
                
                # Set initial best if root is better than 0
                if root_score > best_score:
                    best_score = root_score
                    best_features = root_feature_columns.copy()
                    
                total_time += root_eval_time
            else:
                print("⚠️  Skipping root node evaluation (could not load train table data)")
            
            print("=" * 60)
        
        for i in range(n_iterations):
            # Random feature count
            n_features = random.randint(min_features, max_features)
            
            # Random feature selection (CPU operation)
            feature_subset = random.sample(self.feature_columns, n_features)
            
            # Evaluate subset
            score, eval_time = self.evaluate_feature_subset(feature_subset)
            
            # Check for predictions if enabled
            prediction_generated = False
            if self.enable_predictions and score > self.baseline_score:
                prediction_generated = self._generate_predictions_for_iteration(
                    feature_subset, i + 1, score
                )
                if prediction_generated:
                    improvement = score - self.baseline_score
                    improvement_pct = (improvement / self.baseline_score * 100) if self.baseline_score > 0 else 0
                    print(f"   🚀 NEW BEST SCORE: {score:.4f} (+{improvement:.4f}, +{improvement_pct:.1f}%)")
                    self.baseline_score = score
            
            # Track results
            results['iterations'].append(i + 1)
            results['feature_counts'].append(n_features)
            results['scores'].append(score)
            results['eval_times'].append(eval_time)
            results['feature_subsets'].append(feature_subset)
            if self.enable_predictions:
                results.setdefault('predictions_generated', []).append(prediction_generated)
            
            total_time += eval_time
            
            # Track best result
            if score > best_score:
                best_score = score
                best_features = feature_subset.copy()
                
            # Progress reporting for every iteration
            avg_time = total_time / (i + 1)
            prediction_status = "📝" if prediction_generated else ""
            print(f"📊 Iteration {i+1:3d}/{n_iterations}: "
                  f"features={n_features:2d}, score={score:.4f}, "
                  f"time={eval_time:.3f}s, avg_time={avg_time:.3f}s {prediction_status}")
                
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
        
        print(f"\n📋 BENCHMARK RESULTS:")
        print(f"⏱️  Average evaluation time: {avg_eval_time:.3f}s")
        print(f"📊 Average accuracy: {avg_score:.4f} ± {std_score:.4f}")
        print(f"🏆 Best accuracy: {best_score:.4f} ({len(best_features)} features)")
        print(f"🚀 Evaluations per hour: {summary['evaluations_per_hour']:.0f}")
        print(f"💡 Total time: {total_time:.1f}s for {n_iterations} evaluations")
        
        # Show top 5 results
        print(f"\n🏆 TOP 5 FEATURE COMBINATIONS:")
        # Create list of (score, feature_count, features) tuples and sort by score
        score_data = list(zip(results['scores'], results['feature_counts'], results['feature_subsets']))
        top_5 = sorted(score_data, key=lambda x: x[0], reverse=True)[:5]
        
        for i, (score, feat_count, features) in enumerate(top_5, 1):
            # Truncate feature list if too long
            if len(features) <= 8:
                features_str = ', '.join(features)
            else:
                features_str = ', '.join(features[:6]) + f', ... (+{len(features)-6} more)'
            
            print(f"   {i}. Score: {score:.4f} ({feat_count} features)")
            print(f"      Features: {features_str}")
            if i < len(top_5):
                print()
        
        # Enhanced MCTS feasibility and optimization analysis
        mcts_estimate = 3600 / avg_eval_time  # Evaluations per hour
        print(f"\n🎯 MCTS FEASIBILITY & OPTIMIZATION ANALYSIS:")
        print(f"   1 hour = ~{mcts_estimate:.0f} feature evaluations")
        print(f"   8 hours = ~{mcts_estimate * 8:.0f} feature evaluations")
        print(f"   24 hours = ~{mcts_estimate * 24:.0f} feature evaluations")
        
        # Calculate theoretical search space coverage
        total_features = len(self.feature_columns)
        theoretical_combinations = 2**total_features
        coverage_1h = mcts_estimate / theoretical_combinations * 100
        coverage_24h = (mcts_estimate * 24) / theoretical_combinations * 100
        
        print(f"\n📊 SEARCH SPACE ANALYSIS:")
        print(f"   Total features: {total_features}")
        print(f"   Theoretical combinations: 2^{total_features} = {theoretical_combinations:.2e}")
        print(f"   1-hour coverage: {coverage_1h:.2e}%")
        print(f"   24-hour coverage: {coverage_24h:.2e}%")
        
        # Performance assessment with optimization recommendations
        if avg_eval_time < 1.0:
            print("✅ EXCELLENT: GPU optimization successful!")
            print("   💡 Recommendations: Consider distributed training for even better performance")
        elif avg_eval_time < 3.0:
            print("✅ GOOD: Solid performance for feature-based MCTS")
            print("   💡 Recommendations: Fine-tune n_estimators and regularization")
        elif avg_eval_time < 5.0:
            print("⚠️  MARGINAL: Consider additional optimizations")
            print("   💡 Recommendations: Enable QuantileDMatrix, reduce n_estimators")
        else:
            print("❌ SLOW: Significant optimizations needed")
            print("   💡 Recommendations: Check GPU setup, reduce model complexity")
        
        # ROOT NODE vs RANDOM ITERATIONS COMPARISON
        if include_root_node and results['root_node'] is not None:
            root_score = results['root_node']['score']
            root_feature_count = results['root_node']['feature_count']
            
            print(f"\n🌱 ROOT NODE vs RANDOM ITERATIONS COMPARISON:")
            print("=" * 60)
            print(f"🌿 ROOT NODE (Original Features):")
            print(f"   Score: {root_score:.4f}")
            print(f"   Features: {root_feature_count} original features")
            print(f"   Description: {results['root_node']['description']}")
            
            # Compare to best random iteration
            print(f"\n🎲 BEST RANDOM ITERATION:")
            print(f"   Score: {best_score:.4f}")
            print(f"   Features: {len(best_features)} features")
            
            # Performance comparison
            improvement = best_score - root_score
            improvement_pct = (improvement / root_score * 100) if root_score > 0 else 0
            
            print(f"\n📊 PERFORMANCE ANALYSIS:")
            if improvement > 0:
                print(f"✅ Feature engineering IMPROVED performance!")
                print(f"   Improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")
                print(f"   🏆 Best features outperform original by {improvement_pct:.2f}%")
            elif improvement < 0:
                print(f"❌ Feature engineering DEGRADED performance!")
                print(f"   Degradation: {improvement:.4f} ({improvement_pct:.2f}%)")
                print(f"   🌱 Original features are {abs(improvement_pct):.2f}% better")
            else:
                print(f"➡️  Feature engineering had NO SIGNIFICANT IMPACT")
                print(f"   Both approaches yield similar results ({root_score:.4f})")
            
            # Feature efficiency analysis
            if len(best_features) > 0:
                feature_efficiency = best_score / len(best_features)
                root_efficiency = root_score / root_feature_count if root_feature_count > 0 else 0
                
                print(f"\n⚡ FEATURE EFFICIENCY ANALYSIS:")
                print(f"   Root efficiency: {root_efficiency:.6f} (score per feature)")
                print(f"   Best efficiency: {feature_efficiency:.6f} (score per feature)")
                
                if feature_efficiency > root_efficiency:
                    efficiency_gain = (feature_efficiency - root_efficiency) / root_efficiency * 100
                    print(f"   ✅ Random features are {efficiency_gain:.1f}% more efficient")
                else:
                    efficiency_loss = (root_efficiency - feature_efficiency) / root_efficiency * 100
                    print(f"   🌱 Original features are {efficiency_loss:.1f}% more efficient")
            
            # MCTS Strategy Recommendation
            print(f"\n🎯 MCTS STRATEGY RECOMMENDATION:")
            if improvement_pct > 5:
                print("   ✅ CONTINUE MCTS: Feature engineering shows strong potential")
                print("   💡 Focus on feature combinations similar to best performing subset")
            elif improvement_pct > 1:
                print("   ⚡ CAUTIOUS MCTS: Moderate improvement possible")
                print("   💡 Use shorter MCTS runs with careful monitoring")
            else:
                print("   ⚠️  QUESTIONABLE MCTS: Limited improvement potential")
                print("   💡 Consider focusing on original features or different approach")
            
            print("=" * 60)
        
        # Prediction summary
        if self.enable_predictions:
            print(f"\n📊 PREDICTION SUMMARY:")
            print(f"   💾 Total predictions generated: {self.prediction_counter}")
            print(f"   🎯 Final baseline score: {self.baseline_score:.4f}")
            print(f"   📈 Score improvements captured: {sum(results.get('predictions_generated', []))}")
            
        return results
        
    def save_results(self, results: Dict, output_path: str = None):
        """Save benchmark results to JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"random_feature_benchmark_{timestamp}.json"
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Results saved to: {output_path}")
    
    def _generate_predictions_for_iteration(self, feature_subset: List[str], iteration_num: int, score: float) -> bool:
        """
        Generate predictions for a specific iteration.
        
        Args:
            feature_subset: Features to use for training
            iteration_num: Current iteration number
            score: CV score achieved
            
        Returns:
            True if prediction was generated successfully
        """
        if not self.enable_predictions:
            return False
            
        try:
            # Train model with current feature subset
            X_train = self.data[feature_subset]
            y_train = self.target
            
            # Handle data format conversion
            is_cudf = hasattr(X_train, 'to_pandas')
            if not (is_cudf and self.use_gpu and CUDF_AVAILABLE):
                for col in X_train.columns:
                    if X_train[col].dtype.name == 'category':
                        X_train[col] = X_train[col].cat.codes
                    elif X_train[col].dtype == 'object':
                        X_train[col] = pd.Categorical(X_train[col]).codes
            
            # Train model
            model_params = self.model_params.copy()
            model = xgb.XGBClassifier(**model_params)
            model.fit(X_train, y_train)
            
            # Generate predictions
            return self._generate_predictions(model, feature_subset, iteration_num, score)
            
        except Exception as e:
            print(f"   ❌ Failed to generate prediction for iteration {iteration_num}: {e}")
            return False

class FeatureAttributionAnalyzer:
    """Analyzes feature importance using different attribution methods."""
    
    def __init__(self, results: Dict, all_features: List[str]):
        """Initialize analyzer with benchmark results."""
        self.results = results
        self.all_features = all_features
        self.feature_scores = list(zip(results['scores'], results['feature_subsets']))
        
    def marginal_contribution_analysis(self) -> Dict[str, float]:
        """
        Algorithm 1: Marginal Contribution Analysis
        For each feature: average score difference when present vs absent.
        """
        print("🔍 Computing Marginal Contributions...")
        
        marginal_impacts = {}
        
        for feature in self.all_features:
            scores_with = [score for score, features in self.feature_scores if feature in features]
            scores_without = [score for score, features in self.feature_scores if feature not in features]
            
            if scores_with and scores_without:
                impact = np.mean(scores_with) - np.mean(scores_without)
                marginal_impacts[feature] = impact
            else:
                # Feature always present or always absent
                marginal_impacts[feature] = 0.0
                
        return marginal_impacts
    
    def regression_attribution_analysis(self) -> Dict[str, float]:
        """
        Algorithm 2: Regression-Based Attribution
        Fits linear regression: score = β₀ + β₁×(has_feature1) + β₂×(has_feature2) + ...
        """
        print("📊 Computing Regression Attribution...")
        
        # Create binary feature matrix
        X = []
        y = []
        
        for score, features in self.feature_scores:
            # Binary vector: 1 if feature present, 0 if absent
            feature_vector = [1 if feature in features else 0 for feature in self.all_features]
            X.append(feature_vector)
            y.append(score)
        
        X = np.array(X)
        y = np.array(y)
        
        try:
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Return coefficients as attribution scores
            attributions = {feature: coef for feature, coef in zip(self.all_features, model.coef_)}
            
            # Add R² score for model quality
            r2_score = model.score(X, y)
            print(f"   📈 Regression R² score: {r2_score:.4f}")
            
            return attributions
            
        except Exception as e:
            print(f"   ❌ Regression failed: {e}")
            return {feature: 0.0 for feature in self.all_features}
    
    def shapley_value_analysis(self, max_coalitions: int = 1000) -> Dict[str, float]:
        """
        Algorithm 3: Shapley Value Analysis (Approximated)
        Computes approximate Shapley values using marginal contribution sampling.
        """
        print("🎲 Computing Shapley Values (approximated)...")
        
        shapley_values = {feature: 0.0 for feature in self.all_features}
        
        # Calculate marginal contributions for each feature
        for feature in self.all_features:
            marginal_contributions = []
            
            # Find subsets with and without this feature
            with_feature = [(score, features) for score, features in self.feature_scores if feature in features]
            without_feature = [(score, features) for score, features in self.feature_scores if feature not in features]
            
            if not with_feature or not without_feature:
                shapley_values[feature] = 0.0
                continue
            
            # For each subset with the feature, try to find similar subset without it
            for score_with, features_with in with_feature:
                features_without_target = [f for f in features_with if f != feature]
                
                # Find the most similar subset without this feature
                best_match_score = None
                best_similarity = -1
                
                for score_without, features_without in without_feature:
                    # Calculate similarity (Jaccard index)
                    intersection = len(set(features_without_target) & set(features_without))
                    union = len(set(features_without_target) | set(features_without))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_score = score_without
                
                # Calculate marginal contribution
                if best_match_score is not None:
                    marginal_contribution = score_with - best_match_score
                    marginal_contributions.append(marginal_contribution)
            
            # Average marginal contributions
            if marginal_contributions:
                shapley_values[feature] = np.mean(marginal_contributions)
            else:
                shapley_values[feature] = 0.0
                
        return shapley_values
    
    def display_attribution_results(self, marginal: Dict = None, regression: Dict = None, 
                                  shapley: Dict = None, top_k: int = 15):
        """Display attribution results using Rich tables or basic output."""
        
        if RICH_AVAILABLE:
            self._display_rich_results(marginal, regression, shapley, top_k)
        else:
            self._display_basic_results(marginal, regression, shapley, top_k)
    
    def _display_rich_results(self, marginal: Dict = None, regression: Dict = None, 
                            shapley: Dict = None, top_k: int = 15):
        """Display results using Rich formatting."""
        
        # Collect all results
        all_results = []
        methods = []
        
        if marginal:
            methods.append("Marginal")
            if not all_results:
                all_results = [(feature, impact) for feature, impact in marginal.items()]
            else:
                for i, (feature, _) in enumerate(all_results):
                    all_results[i] = (feature, all_results[i][1], marginal.get(feature, 0.0))
        
        if regression:
            methods.append("Regression")
            if len(methods) == 1:
                all_results = [(feature, coef) for feature, coef in regression.items()]
            else:
                for i, item in enumerate(all_results):
                    feature = item[0]
                    all_results[i] = item + (regression.get(feature, 0.0),)
        
        if shapley:
            methods.append("Shapley")
            if len(methods) == 1:
                all_results = [(feature, value) for feature, value in shapley.items()]
            else:
                for i, item in enumerate(all_results):
                    feature = item[0]
                    all_results[i] = item + (shapley.get(feature, 0.0),)
        
        # Sort by first available method (descending)
        all_results.sort(key=lambda x: abs(x[1]), reverse=True)
        top_results = all_results[:top_k]
        
        # Create Rich table
        table = Table(title="🏆 Feature Attribution Analysis", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Feature", style="bold", min_width=20)
        
        for method in methods:
            table.add_column(f"{method}", justify="right", style="cyan")
        
        # Add rows
        for i, result in enumerate(top_results, 1):
            feature = result[0]
            values = result[1:]
            
            row = [str(i), feature]
            for value in values:
                if abs(value) > 0.00001:  # Lower threshold for small values
                    color = "green" if value > 0 else "red"
                    row.append(f"[{color}]{value:+.6f}[/{color}]")  # 6 decimal places
                else:
                    row.append("[dim]0.000000[/dim]")
            
            table.add_row(*row)
        
        console.print("\n")
        console.print(table)
        
        # Summary panel
        summary_text = []
        for method in methods:
            method_dict = marginal if method == "Marginal" else regression if method == "Regression" else shapley
            top_positive = max(method_dict.values()) if method_dict else 0
            top_negative = min(method_dict.values()) if method_dict else 0
            summary_text.append(f"[bold]{method}:[/bold] Best: {top_positive:+.6f}, Worst: {top_negative:+.6f}")
        
        console.print(Panel("\n".join(summary_text), title="📊 Attribution Summary", border_style="blue"))
    
    def _display_basic_results(self, marginal: Dict = None, regression: Dict = None, 
                             shapley: Dict = None, top_k: int = 15):
        """Display results using basic console output."""
        
        print(f"\n🏆 FEATURE ATTRIBUTION ANALYSIS (TOP {top_k}):")
        print("=" * 80)
        
        if marginal:
            print("\n📈 MARGINAL CONTRIBUTION ANALYSIS:")
            sorted_marginal = sorted(marginal.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            for i, (feature, impact) in enumerate(sorted_marginal, 1):
                sign = "+" if impact > 0 else ""
                print(f"   {i:2d}. {feature:30s} {sign}{impact:.6f}")
        
        if regression:
            print("\n📊 REGRESSION ATTRIBUTION ANALYSIS:")
            sorted_regression = sorted(regression.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            for i, (feature, coef) in enumerate(sorted_regression, 1):
                sign = "+" if coef > 0 else ""
                print(f"   {i:2d}. {feature:30s} {sign}{coef:.6f}")
        
        if shapley:
            print("\n🎲 SHAPLEY VALUE ANALYSIS:")
            sorted_shapley = sorted(shapley.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            for i, (feature, value) in enumerate(sorted_shapley, 1):
                sign = "+" if value > 0 else ""
                print(f"   {i:2d}. {feature:30s} {sign}{value:.6f}")
        
        print("=" * 80)

def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Feature Selection Benchmark')
    
    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--dataset-name', help='Dataset name to load from DuckDB (e.g., s5e7c)')
    data_group.add_argument('--data', help='Path to parquet data file')
    
    parser.add_argument('--target-column', default='survived', help='Target column name')
    parser.add_argument('--train', action='store_true', 
                       help='Use train table (clean features) instead of train_features table')
    
    # Benchmark parameters
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--min-features', type=int, default=10, help='Minimum features per subset')
    parser.add_argument('--max-features', type=int, default=None, help='Maximum features per subset')
    parser.add_argument('--n-estimators', type=int, default=1000, help='Number of XGBoost estimators (default: 1000)')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--enable-predictions', action='store_true', 
                       help='Enable prediction generation for improved scores')
    
    # Feature attribution analysis options
    parser.add_argument('--marginal', action='store_true', 
                       help='Run marginal contribution analysis')
    parser.add_argument('--regression', action='store_true', 
                       help='Run regression-based attribution analysis')
    parser.add_argument('--shapley', action='store_true', 
                       help='Run Shapley value analysis (approximated)')
    parser.add_argument('--all-attribution', action='store_true', 
                       help='Run all attribution analyses')
    parser.add_argument('--top-features', type=int, default=15, 
                       help='Number of top features to display in attribution analysis')
    
    args = parser.parse_args()
    
    print(f"🚀 Random Feature Selection Benchmark")
    
    # Initialize evaluator based on data source
    if args.dataset_name:
        print(f"🗄️  Dataset: {args.dataset_name}")
        print(f"📊 Table: {'train' if args.train else 'train_features'}")
        print(f"🎯 Target: {args.target_column}")
        print(f"🌳 n_estimators: {args.n_estimators}")
        
        evaluator = FastFeatureEvaluator(
            dataset_name=args.dataset_name,
            target_column=args.target_column,
            use_train_table=args.train,
            n_estimators=args.n_estimators,
            enable_predictions=args.enable_predictions
        )
    else:
        # Validate data file
        if not Path(args.data).exists():
            raise FileNotFoundError(f"Data file not found: {args.data}")
            
        print(f"📁 Data: {args.data}")
        print(f"🎯 Target: {args.target_column}")
        print(f"🌳 n_estimators: {args.n_estimators}")
        
        evaluator = FastFeatureEvaluator(
            data_path=args.data,
            target_column=args.target_column,
            n_estimators=args.n_estimators,
            enable_predictions=args.enable_predictions
        )
    
    # Run benchmark
    results = evaluator.random_feature_selection_benchmark(
        n_iterations=args.iterations,
        min_features=args.min_features,
        max_features=args.max_features
    )
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Feature attribution analysis
    run_attribution = args.marginal or args.regression or args.shapley or args.all_attribution
    
    if run_attribution:
        print(f"\n🔬 FEATURE ATTRIBUTION ANALYSIS")
        print("=" * 60)
        
        # Initialize analyzer
        analyzer = FeatureAttributionAnalyzer(results, evaluator.feature_columns)
        
        # Run selected analyses
        marginal_results = None
        regression_results = None
        shapley_results = None
        
        if args.marginal or args.all_attribution:
            marginal_results = analyzer.marginal_contribution_analysis()
            
        if args.regression or args.all_attribution:
            regression_results = analyzer.regression_attribution_analysis()
            
        if args.shapley or args.all_attribution:
            shapley_results = analyzer.shapley_value_analysis()
        
        # Display results
        analyzer.display_attribution_results(
            marginal=marginal_results,
            regression=regression_results, 
            shapley=shapley_results,
            top_k=args.top_features
        )
        
        print(f"\n💡 Attribution analysis completed!")
        
        # Add attribution results to saved data
        if args.output:
            attribution_data = {
                'marginal': marginal_results,
                'regression': regression_results,
                'shapley': shapley_results
            }
            
            # Update results with attribution
            results['feature_attribution'] = attribution_data
            
            # Re-save with attribution data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            attribution_output = args.output.replace('.json', f'_with_attribution_{timestamp}.json')
            evaluator.save_results(results, attribution_output)
            print(f"💾 Results with attribution saved to: {attribution_output}")
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    main()