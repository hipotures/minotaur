import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5  # Full CV for better validation
CLASS_THRESHOLD = 0.2

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

def feature_engineering(df):
    """Create agricultural feature interactions and deficiency flags"""
    df = df.copy()
    
    # Fix column name typo
    if 'Temparature' in df.columns:
        df['Temperature'] = df['Temparature']
        df = df.drop('Temparature', axis=1)
    
    # Nutrient ratios
    df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df['PK_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    
    # Deficiency flags
    for nutrient in ['Nitrogen', 'Phosphorous', 'Potassium']:
        df[f'low_{nutrient}'] = (df[nutrient] < df[nutrient].quantile(CLASS_THRESHOLD)).astype(int)
    
    # Combined features
    df['soil_crop'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
    df['nutrient_balance'] = df['Nitrogen'] + df['Phosphorous'] * 1.2 + df['Potassium'] * 0.8
    
    # Moisture stress index
    df['moisture_stress'] = df['Temperature'] * df['Humidity'] / (df['Moisture'] + 1e-5)
    
    return df

def main():
    try:
        # Load data
        train_path = '/mnt/ml/competitions/2025/playground-series-s5e6/train.csv'
        test_path = '/mnt/ml/competitions/2025/playground-series-s5e6/test.csv'
        
        print("Loading data...")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        # Use full dataset for final training
        print("Using 100% of training data and full test set")
        
        # DO NOT SAMPLE TEST DATA - Kaggle expects all IDs in order!
        # test = test.sample(...) # REMOVED - keep original order
        
        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")
        
        # Save test IDs for submission
        test_ids = test['id']
        
        # Feature engineering
        print("Feature engineering...")
        train = feature_engineering(train)
        test = feature_engineering(test)
        
        # Encode target
        le = LabelEncoder()
        train['fertilizer_encoded'] = le.fit_transform(train['Fertilizer Name'])
        target = 'fertilizer_encoded'
        
        print(f"Number of classes: {len(le.classes_)}")
        
        # Prepare features
        features = [col for col in train.columns if col not in ['id', 'Fertilizer Name', 'fertilizer_encoded']]
        X = train[features]
        y = train[target]
        X_test = test[features]
        
        # Identify categorical features
        categoricals = [col for col in features if X[col].dtype == 'object' or 'soil_crop' in col]
        for col in categoricals:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
        
        print(f"Using {len(features)} features")
        
        # GPU-optimized LightGBM parameters
        params = {
            'objective': 'multiclass',
            'num_class': len(le.classes_),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'device': 'gpu',  # Enable GPU acceleration
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'num_leaves': 127,  # Increased for GPU
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'min_data_in_leaf': 50,  # Increased for full dataset
            'max_depth': -1,
            'seed': RANDOM_STATE,
            'verbosity': 1,  # Show progress
            'force_col_wise': True,  # Better for GPU
            'num_threads': 8
        }
        
        # Cross-validation setup
        oof_preds = np.zeros((len(train), len(le.classes_)))
        test_preds = np.zeros((len(test), len(le.classes_)))
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        # Cross-validation loop
        print("Starting GPU-accelerated training...")
        for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            print(f'\n=== Fold {fold + 1}/{N_FOLDS} ===')
            
            # Data splits
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            
            # LightGBM dataset
            train_set = lgb.Dataset(
                X_train, 
                label=y_train,
                categorical_feature=categoricals
            )
            valid_set = lgb.Dataset(
                X_valid, 
                label=y_valid,
                reference=train_set
            )
            
            # Train model on GPU with full data
            model = lgb.train(
                params,
                train_set,
                num_boost_round=2000,  # Increased for full dataset
                valid_sets=[valid_set],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(100)  # Log every 100 iterations
                ]
            )
            
            # Predictions
            oof_preds[valid_idx] = model.predict(X_valid)
            test_preds += model.predict(X_test) / N_FOLDS
            
            # Fold MAP@3 score
            fold_map3 = map3(y_valid, oof_preds[valid_idx])
            print(f'Fold {fold + 1} MAP@3: {fold_map3:.5f}')
        
        # Overall OOF MAP@3
        oof_map3 = map3(y, oof_preds)
        print(f'\n🎯 Overall OOF MAP@3: {oof_map3:.5f}')
        
        # Generate submission
        print("\nGenerating submission...")
        top3_preds = []
        for probs in test_preds:
            top3 = np.argsort(probs)[::-1][:3]
            top3_preds.append(' '.join(le.inverse_transform(top3)))
        
        submission = pd.DataFrame({
            'id': test_ids,
            'Fertilizer Name': top3_preds
        })
        
        submission.to_csv('submission_gpu_full.csv', index=False)
        print('✅ Full dataset submission saved to submission_gpu_full.csv')
        print(f"Submission shape: {submission.shape}")
        print("\nSample predictions:")
        print(submission.head(10))
        
        # Feature importance
        print(f"\n📊 Top 10 Feature Importance:")
        feature_imp = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        print(feature_imp.head(10))
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()