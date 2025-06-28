import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
import rtdl_num_embeddings
import scipy.special
import warnings
import os
import math
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import TabM
import sys
sys.path.append('../tabm')
from tabm_reference import Model, make_parameter_groups

# Import shared feature engineering
from scripts.feature_engineering import load_or_create_features

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
CLASS_THRESHOLD = 0.2
DATA_FRACTION = 1.0  # Fraction of training data to use (1.0 = full dataset)
N_EPOCHS = 1000
PATIENCE = 16
BATCH_SIZE = 2048  # Optimized for 2x RTX 4090 (48GB total VRAM)
SELECT_K_FEATURES = 80  # Select top K features from ~120 total

def map3(y_true, y_pred_proba):
    """Calculate MAP@3 metric for multi-class classification"""
    top3_preds = []
    for probs in y_pred_proba:
        top3 = np.argsort(probs)[::-1][:3]
        top3_preds.append(top3)
    
    map_score = 0.0
    for true, pred in zip(y_true, top3_preds):
        score = 0.0
        for i, p in enumerate(pred):
            if p == true:
                score = 1.0 / (i + 1)
                break
        map_score += score
    return map_score / len(y_true)


def select_best_features(X_train, y_train, X_test, feature_names, k=SELECT_K_FEATURES):
    """Select top K features using ANOVA F-test"""
    print(f"Selecting top {k} features from {len(feature_names)} total features...")
    
    # Only select from numerical features for ANOVA F-test (exclude boolean and object)
    numerical_mask = (X_train.dtypes != 'object') & (X_train.dtypes != 'bool') & (X_train.dtypes != 'int8')
    X_train_num = X_train.loc[:, numerical_mask].copy()
    X_test_num = X_test.loc[:, numerical_mask].copy()
    numerical_features = X_train_num.columns.tolist()
    
    print(f"Found {len(numerical_features)} truly numerical features for selection")
    
    # Clean infinite and extreme values before feature selection
    print("Cleaning infinite and extreme values...")
    for col in numerical_features:
        # Convert to float64 to ensure proper handling
        X_train_num[col] = X_train_num[col].astype(np.float64)
        X_test_num[col] = X_test_num[col].astype(np.float64)
        
        # Replace infinities and extreme values
        X_train_num[col] = np.nan_to_num(X_train_num[col], nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_num[col] = np.nan_to_num(X_test_num[col], nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Only clip if we have valid numerical data
        if len(X_train_num[col].dropna()) > 0:
            q99 = np.percentile(X_train_num[col].dropna(), 99)
            q01 = np.percentile(X_train_num[col].dropna(), 1)
            X_train_num[col] = np.clip(X_train_num[col], q01, q99)
            X_test_num[col] = np.clip(X_test_num[col], q01, q99)
    
    # Skip feature selection if we have fewer features than requested
    if len(numerical_features) <= k:
        print(f"Only {len(numerical_features)} numerical features available, using all")
        selected_features = numerical_features
        X_train_selected = X_train_num
        X_test_selected = X_test_num
    else:
        # Select best features
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train_num, y_train)
        X_test_selected = selector.transform(X_test_num)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [numerical_features[i] for i in selected_indices]
        
        # Convert back to DataFrame
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train_num.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test_num.index)
    
    # Add back non-numerical features (categorical, boolean, int8)
    non_numerical_features = X_train.loc[:, ~numerical_mask].columns.tolist()
    if non_numerical_features:
        X_train_selected = pd.concat([X_train_selected, X_train[non_numerical_features]], axis=1)
        X_test_selected = pd.concat([X_test_selected, X_test[non_numerical_features]], axis=1)
    
    final_features = selected_features + non_numerical_features
    
    print(f"Selected {len(selected_features)} numerical + {len(non_numerical_features)} non-numerical = {len(final_features)} total features")
    
    return X_train_selected, X_test_selected, final_features

def prepare_data_for_tabm(train, test, features, target_col, n_classes):
    """Prepare data in TabM format"""
    
    # Separate numerical and categorical features
    numerical_features = train[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train[features].select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Prepare numerical data
    X_num_train = train[numerical_features].values.astype(np.float32)
    X_num_test = test[numerical_features].values.astype(np.float32)
    
    # Handle infinite and extreme values
    X_num_train = np.nan_to_num(X_num_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_num_test = np.nan_to_num(X_num_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Quantile transform for numerical features
    print("Applying QuantileTransformer...")
    noise = np.random.normal(0.0, 1e-5, X_num_train.shape).astype(np.float32)
    qt = QuantileTransformer(
        n_quantiles=max(min(len(X_num_train) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9
    )
    X_num_train = qt.fit_transform(X_num_train + noise)
    X_num_test = qt.transform(X_num_test)
    
    # Prepare categorical data
    X_cat_train = X_cat_test = cat_cardinalities = None
    if categorical_features:
        print("Encoding categorical features...")
        cat_encoders = {}
        cat_cardinalities = []
        X_cat_data = []
        
        for col in categorical_features:
            le = LabelEncoder()
            train_encoded = le.fit_transform(train[col].astype(str))
            test_encoded = le.transform(test[col].astype(str))
            
            cat_encoders[col] = le
            cat_cardinalities.append(len(le.classes_))
            X_cat_data.append((train_encoded, test_encoded))
        
        X_cat_train = np.column_stack([x[0] for x in X_cat_data])
        X_cat_test = np.column_stack([x[1] for x in X_cat_data])
    
    # Prepare target
    y_train = train[target_col].values
    
    return {
        'X_num_train': X_num_train,
        'X_num_test': X_num_test, 
        'X_cat_train': X_cat_train,
        'X_cat_test': X_cat_test,
        'y_train': y_train,
        'cat_cardinalities': cat_cardinalities,
        'n_num_features': len(numerical_features)
    }

def train_tabm_model(X_num, X_cat, y, cat_cardinalities, n_num_features, n_classes, device, multi_gpu=True):
    """Train TabM model with multi-GPU support"""
    
    # Create model
    model = Model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities or [],
        n_classes=n_classes,
        backbone={
            'type': 'MLP',
            'n_blocks': 3,
            'd_block': 512,
            'dropout': 0.1,
        },
        bins=None,
        num_embeddings=None,
        arch_type='tabm',
        k=32,
        share_training_batches=True,
    ).to(device)
    
    # Enable multi-GPU training if available
    is_multi_gpu = multi_gpu and torch.cuda.device_count() > 1
    if is_multi_gpu:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=2e-3, weight_decay=3e-4)
    
    def apply_model(x_num, x_cat=None):
        return model(x_num, x_cat).squeeze(-1).float()
    
    def loss_fn(y_pred, y_true):
        """TabM produces k predictions. Each must be trained separately."""
        k = y_pred.shape[-1 if len(y_pred.shape) == 2 else -2]
        # Access original model attributes through .module if using DataParallel
        original_model = model.module if is_multi_gpu else model
        return F.cross_entropy(
            y_pred.flatten(0, 1),
            y_true.repeat_interleave(k) if original_model.share_training_batches else y_true,
        )
    
    # Convert to tensors
    X_num = torch.as_tensor(X_num, device=device, dtype=torch.float32)
    if X_cat is not None:
        X_cat = torch.as_tensor(X_cat, device=device, dtype=torch.long)
    y = torch.as_tensor(y, device=device, dtype=torch.long)
    
    train_size = len(y)
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"Training TabM model with {len(y)} samples...")
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        
        # Create batches
        indices = torch.randperm(train_size, device=device)
        
        for i in range(0, train_size, BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            
            optimizer.zero_grad()
            
            x_num_batch = X_num[batch_indices]
            x_cat_batch = X_cat[batch_indices] if X_cat is not None else None
            y_batch = y[batch_indices]
            
            y_pred = apply_model(x_num_batch, x_cat_batch)
            loss = loss_fn(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model

@torch.inference_mode()
def predict_tabm(model, X_num, X_cat, device):
    """Make predictions with TabM model"""
    model.eval()
    
    X_num = torch.as_tensor(X_num, device=device, dtype=torch.float32)
    if X_cat is not None:
        X_cat = torch.as_tensor(X_cat, device=device, dtype=torch.long)
    
    batch_size = 1024
    predictions = []
    
    for i in range(0, len(X_num), batch_size):
        x_num_batch = X_num[i:i+batch_size]
        x_cat_batch = X_cat[i:i+batch_size] if X_cat is not None else None
        
        y_pred = model(x_num_batch, x_cat_batch).float()
        
        # Convert to probabilities and average over k predictions
        y_pred = F.softmax(y_pred, dim=-1)
        y_pred = y_pred.mean(dim=1)  # Average over k predictions
        
        predictions.append(y_pred.cpu().numpy())
    
    return np.vstack(predictions)

def main():
    try:
        # Get script name for dynamic file naming
        import os
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        
        # Set random seeds
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        
        # Device setup
        if torch.cuda.is_available():
            device = torch.device('cuda')
            n_gpus = torch.cuda.device_count()
            print(f"Found {n_gpus} GPU(s), using all available")
            for i in range(n_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            device = torch.device('cpu')
            n_gpus = 0
            print("Using CPU")
        
        print(f"Primary device: {device}")
        
        # Load data
        train_path = '/mnt/ml/competitions/2025/playground-series-s5e6/train.csv'
        test_path = '/mnt/ml/competitions/2025/playground-series-s5e6/test.csv'
        
        print("Loading data...")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")
        
        # Sample training data if DATA_FRACTION < 1.0
        if DATA_FRACTION < 1.0:
            sample_size = int(len(train) * DATA_FRACTION)
            print(f"Sampling {DATA_FRACTION*100:.1f}% of training data: {sample_size:,} rows")
            train = train.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
            print(f"Sampled train shape: {train.shape}")
        
        # Save test IDs for submission
        test_ids = test['id']
        
        # Feature engineering using shared module
        train = load_or_create_features(train_path, 'train', 'basic')
        test = load_or_create_features(test_path, 'test', 'basic')
        
        # Encode target
        le = LabelEncoder()
        train['fertilizer_encoded'] = le.fit_transform(train['Fertilizer Name'])
        target = 'fertilizer_encoded'
        n_classes = len(le.classes_)
        
        print(f"Number of classes: {n_classes}")
        
        # Prepare features
        features = [col for col in train.columns if col not in ['id', 'Fertilizer Name', 'fertilizer_encoded']]
        print(f"Total engineered features: {len(features)}")
        
        # Feature selection
        X_train_selected, X_test_selected, selected_features = select_best_features(
            train[features], train[target], test[features], features, k=SELECT_K_FEATURES
        )
        
        print(f"Using {len(selected_features)} selected features")
        
        # Replace train and test data with selected features
        train_selected = train[['id', 'Fertilizer Name', 'fertilizer_encoded']].copy()
        test_selected = test[['id']].copy()
        
        # Add selected features
        for col in selected_features:
            train_selected[col] = X_train_selected[col]
            test_selected[col] = X_test_selected[col]
        
        # Prepare data for TabM
        data_dict = prepare_data_for_tabm(train_selected, test_selected, selected_features, target, n_classes)
        
        # Cross-validation
        oof_preds = np.zeros((len(train_selected), n_classes))
        test_preds = np.zeros((len(test_selected), n_classes))
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        print(f"\n🚀 Starting TabM training with {N_FOLDS}-fold CV and feature selection...")
        
        for fold, (train_idx, valid_idx) in enumerate(folds.split(data_dict['X_num_train'], data_dict['y_train'])):
            print(f'\n=== Fold {fold + 1}/{N_FOLDS} ===')
            
            # Split data
            X_num_fold = data_dict['X_num_train'][train_idx]
            X_cat_fold = data_dict['X_cat_train'][train_idx] if data_dict['X_cat_train'] is not None else None
            y_fold = data_dict['y_train'][train_idx]
            
            X_num_val = data_dict['X_num_train'][valid_idx]
            X_cat_val = data_dict['X_cat_train'][valid_idx] if data_dict['X_cat_train'] is not None else None
            y_val = data_dict['y_train'][valid_idx]
            
            # Train model
            model = train_tabm_model(
                X_num_fold, X_cat_fold, y_fold,
                data_dict['cat_cardinalities'],
                data_dict['n_num_features'],
                n_classes,
                device,
                multi_gpu=True
            )
            
            # Validation predictions
            val_preds = predict_tabm(model, X_num_val, X_cat_val, device)
            oof_preds[valid_idx] = val_preds
            
            # Test predictions
            test_fold_preds = predict_tabm(model, data_dict['X_num_test'], data_dict['X_cat_test'], device)
            test_preds += test_fold_preds / N_FOLDS
            
            # Fold MAP@3 score
            fold_map3 = map3(y_val, val_preds)
            print(f'Fold {fold + 1} MAP@3: {fold_map3:.5f}')
        
        # Overall OOF MAP@3
        oof_map3 = map3(data_dict['y_train'], oof_preds)
        print(f'\n🎯 Overall OOF MAP@3: {oof_map3:.5f}')
        
        # Generate submission
        print("\nGenerating TabM optimized submission...")
        top3_preds = []
        for probs in test_preds:
            top3 = np.argsort(probs)[::-1][:3]
            top3_preds.append(' '.join(le.inverse_transform(top3)))
        
        submission = pd.DataFrame({
            'id': test_ids,
            'Fertilizer Name': top3_preds
        })
        
        submission_file = f'{script_name}_submission.csv'
        submission.to_csv(submission_file, index=False)
        print(f'✅ TabM optimized submission saved to {submission_file}')
        print(f"Submission shape: {submission.shape}")
        print("\nSample predictions:")
        print(submission.head(10))
        
        # Save selected features for reference
        features_file = f'{script_name}_selected_features.csv'
        pd.DataFrame({'selected_features': selected_features}).to_csv(features_file, index=False)
        print(f"✅ Selected features saved to {features_file}")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()