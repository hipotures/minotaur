import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
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

def advanced_npk_features(df):
    """Advanced NPK interaction features"""
    print("  -> Adding advanced NPK features...")
    
    # Complex NPK interactions
    df['npk_harmony'] = 1 / (1 + np.abs(df['Nitrogen']/10 - df['Phosphorous']/26 - df['Potassium']/26))
    df['npk_distance'] = np.sqrt((df['Nitrogen']-20)**2 + (df['Phosphorous']-20)**2 + (df['Potassium']-20)**2)
    df['npk_sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['npk_product'] = df['Nitrogen'] * df['Phosphorous'] * df['Potassium']
    
    # Fertilizer balance indicators
    df['is_balanced'] = ((df['Nitrogen'] / (df['Phosphorous'] + 1e-5)).between(0.8, 1.2) & 
                        (df['Phosphorous'] / (df['Potassium'] + 1e-5)).between(0.8, 1.2)).astype(int)
    
    # NPK dominance patterns
    df['n_dominant'] = (df['Nitrogen'] > df['Phosphorous']) & (df['Nitrogen'] > df['Potassium'])
    df['p_dominant'] = (df['Phosphorous'] > df['Nitrogen']) & (df['Phosphorous'] > df['Potassium'])
    df['k_dominant'] = (df['Potassium'] > df['Nitrogen']) & (df['Potassium'] > df['Phosphorous'])
    
    # NPK deficiency combinations
    df['npk_deficiency_pattern'] = (df['low_Nitrogen'].astype(str) + 
                                   df['low_Phosphorous'].astype(str) + 
                                   df['low_Potassium'].astype(str))
    
    # Advanced ratios
    df['np_pk_ratio'] = (df['Nitrogen'] / (df['Phosphorous'] + 1e-5)) / (df['Phosphorous'] / (df['Potassium'] + 1e-5))
    df['npk_variance'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].var(axis=1)
    df['npk_cv'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1) / (df[['Nitrogen', 'Phosphorous', 'Potassium']].mean(axis=1) + 1e-5)
    
    return df

def environmental_features(df):
    """Environmental interaction features"""
    print("  -> Adding environmental features...")
    
    # Temperature-humidity stress
    df['heat_stress'] = np.where(df['Temperature'] > 35, 
                                (df['Temperature'] - 35) * (100 - df['Humidity']) / 100, 0)
    df['cold_stress'] = np.where(df['Temperature'] < 20, 
                                (20 - df['Temperature']) * df['Humidity'] / 100, 0)
    
    # Water stress indicators
    df['drought_stress'] = np.where(df['Moisture'] < 30, 
                                   (30 - df['Moisture']) * (df['Temperature'] / 30), 0)
    df['flood_stress'] = np.where(df['Moisture'] > 70, 
                                 (df['Moisture'] - 70) * (df['Humidity'] / 100), 0)
    df['water_stress'] = df['drought_stress'] + df['flood_stress']
    
    # Optimal growing conditions
    df['optimal_temp'] = ((df['Temperature'] >= 25) & (df['Temperature'] <= 32)).astype(int)
    df['optimal_humidity'] = ((df['Humidity'] >= 60) & (df['Humidity'] <= 75)).astype(int)
    df['optimal_moisture'] = ((df['Moisture'] >= 40) & (df['Moisture'] <= 60)).astype(int)
    df['optimal_conditions'] = df['optimal_temp'] * df['optimal_humidity'] * df['optimal_moisture']
    
    # Environmental interactions
    df['temp_humidity_interaction'] = df['Temperature'] * df['Humidity'] / 100
    df['temp_moisture_interaction'] = df['Temperature'] * df['Moisture'] / 100
    df['humidity_moisture_interaction'] = df['Humidity'] * df['Moisture'] / 100
    
    # Climate zones simulation
    df['hot_humid'] = ((df['Temperature'] > 30) & (df['Humidity'] > 70)).astype(int)
    df['hot_dry'] = ((df['Temperature'] > 30) & (df['Humidity'] < 50)).astype(int)
    df['cool_wet'] = ((df['Temperature'] < 25) & (df['Moisture'] > 60)).astype(int)
    
    return df

def agricultural_features(df):
    """Agricultural domain knowledge features"""
    print("  -> Adding agricultural domain features...")
    
    # Crop-specific nutrient needs (approximate values)
    crop_n_needs = {
        'Wheat': 120, 'Rice': 100, 'Corn': 150, 'Sugarcane': 200, 'Cotton': 120,
        'Barley': 90, 'Millets': 80, 'Paddy': 100, 'Pulses': 60, 'Tobacco': 110,
        'Ground Nuts': 70, 'Maize': 150
    }
    crop_p_needs = {
        'Wheat': 40, 'Rice': 30, 'Corn': 50, 'Sugarcane': 60, 'Cotton': 40,
        'Barley': 35, 'Millets': 25, 'Paddy': 30, 'Pulses': 25, 'Tobacco': 45,
        'Ground Nuts': 30, 'Maize': 50
    }
    crop_k_needs = {
        'Wheat': 40, 'Rice': 35, 'Corn': 40, 'Sugarcane': 80, 'Cotton': 45,
        'Barley': 35, 'Millets': 30, 'Paddy': 35, 'Pulses': 40, 'Tobacco': 60,
        'Ground Nuts': 40, 'Maize': 40
    }
    
    # Nutrient deficits for each crop
    df['n_deficit_crop'] = df.apply(lambda x: max(0, crop_n_needs.get(x['Crop Type'], 100) - x['Nitrogen']), axis=1)
    df['p_deficit_crop'] = df.apply(lambda x: max(0, crop_p_needs.get(x['Crop Type'], 35) - x['Phosphorous']), axis=1)
    df['k_deficit_crop'] = df.apply(lambda x: max(0, crop_k_needs.get(x['Crop Type'], 40) - x['Potassium']), axis=1)
    df['total_deficit'] = df['n_deficit_crop'] + df['p_deficit_crop'] + df['k_deficit_crop']
    
    # Nutrient adequacy ratios
    df['n_adequacy'] = df.apply(lambda x: x['Nitrogen'] / crop_n_needs.get(x['Crop Type'], 100), axis=1)
    df['p_adequacy'] = df.apply(lambda x: x['Phosphorous'] / crop_p_needs.get(x['Crop Type'], 35), axis=1)
    df['k_adequacy'] = df.apply(lambda x: x['Potassium'] / crop_k_needs.get(x['Crop Type'], 40), axis=1)
    
    # Soil-specific adjustments
    soil_factors = {'Sandy': 1.2, 'Clayey': 0.8, 'Loamy': 1.0, 'Red': 1.1, 'Black': 0.9}
    df['soil_nutrient_factor'] = df['Soil Type'].map(soil_factors).fillna(1.0)
    df['adjusted_n'] = df['Nitrogen'] * df['soil_nutrient_factor']
    df['adjusted_p'] = df['Phosphorous'] * df['soil_nutrient_factor']
    df['adjusted_k'] = df['Potassium'] * df['soil_nutrient_factor']
    
    # Season simulation (based on moisture/temp patterns)
    conditions = [
        (df['Temperature'] > 30) & (df['Moisture'] < 40),  # dry season
        (df['Temperature'] < 25) & (df['Moisture'] > 60),  # wet season
        True  # normal season
    ]
    choices = ['dry', 'wet', 'normal']
    df['season_indicator'] = np.select(conditions, choices, default='normal')
    
    # Crop-soil compatibility
    compatible_combinations = {
        'Sandy_Millets': 1, 'Sandy_Ground Nuts': 1, 'Clayey_Paddy': 1, 'Clayey_Sugarcane': 1,
        'Red_Pulses': 1, 'Red_Cotton': 1, 'Loamy_Wheat': 1, 'Loamy_Maize': 1, 'Black_Cotton': 1
    }
    df['crop_soil_compatibility'] = (df['Soil Type'] + '_' + df['Crop Type']).map(compatible_combinations).fillna(0)
    
    return df

def fertilizer_pattern_features(df):
    """Fertilizer pattern recognition features"""
    print("  -> Adding fertilizer pattern features...")
    
    # Common fertilizer signatures (NPK patterns)
    # Urea: 46-0-0
    df['urea_signature'] = np.abs(df['Nitrogen'] - 46) + df['Phosphorous'] + df['Potassium']
    df['urea_affinity'] = np.exp(-df['urea_signature'] / 20)  # Closer to 0 = higher affinity
    
    # DAP: 18-46-0  
    df['dap_signature'] = np.abs(df['Nitrogen'] - 18) + np.abs(df['Phosphorous'] - 46) + df['Potassium']
    df['dap_affinity'] = np.exp(-df['dap_signature'] / 20)
    
    # Complex fertilizers (balanced NPK)
    df['complex_signature'] = np.abs(df['Nitrogen'] - df['Phosphorous']) + np.abs(df['Phosphorous'] - df['Potassium'])
    df['complex_affinity'] = np.exp(-df['complex_signature'] / 10)
    
    # 20-20-0 pattern
    df['2020_signature'] = np.abs(df['Nitrogen'] - 20) + np.abs(df['Phosphorous'] - 20) + df['Potassium']
    df['2020_affinity'] = np.exp(-df['2020_signature'] / 15)
    
    # High P fertilizers (like 10-26-26)
    df['high_p_signature'] = np.abs(df['Phosphorous'] - 26) + np.abs(df['Potassium'] - 26)
    df['high_p_affinity'] = np.exp(-df['high_p_signature'] / 15)
    
    # Fertilizer need indicators
    df['needs_nitrogen'] = (df['Nitrogen'] < 25).astype(int)
    df['needs_phosphorus'] = (df['Phosphorous'] < 20).astype(int)
    df['needs_potassium'] = (df['Potassium'] < 15).astype(int)
    df['fertilizer_urgency'] = df['needs_nitrogen'] + df['needs_phosphorus'] + df['needs_potassium']
    
    return df

def statistical_features(df):
    """Statistical aggregation features"""
    print("  -> Adding statistical features...")
    
    # Group statistics by soil type
    for col in ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture']:
        df[f'{col}_soil_mean'] = df.groupby('Soil Type')[col].transform('mean')
        df[f'{col}_soil_std'] = df.groupby('Soil Type')[col].transform('std')
        df[f'{col}_soil_deviation'] = df[col] - df[f'{col}_soil_mean']
        df[f'{col}_soil_zscore'] = df[f'{col}_soil_deviation'] / (df[f'{col}_soil_std'] + 1e-5)
    
    # Group statistics by crop type  
    for col in ['Nitrogen', 'Phosphorous', 'Potassium']:
        df[f'{col}_crop_mean'] = df.groupby('Crop Type')[col].transform('mean')
        df[f'{col}_crop_std'] = df.groupby('Crop Type')[col].transform('std')
        df[f'{col}_crop_deviation'] = df[col] - df[f'{col}_crop_mean']
        df[f'{col}_crop_zscore'] = df[f'{col}_crop_deviation'] / (df[f'{col}_crop_std'] + 1e-5)
    
    # Soil-Crop combinations
    for col in ['Nitrogen', 'Phosphorous', 'Potassium']:
        df[f'{col}_soilcrop_mean'] = df.groupby(['Soil Type', 'Crop Type'])[col].transform('mean')
        df[f'{col}_soilcrop_deviation'] = df[col] - df[f'{col}_soilcrop_mean']
    
    # Rank features
    df['nitrogen_rank'] = df['Nitrogen'].rank(pct=True)
    df['phosphorous_rank'] = df['Phosphorous'].rank(pct=True)
    df['potassium_rank'] = df['Potassium'].rank(pct=True)
    
    return df

def feature_engineering(df):
    """Complete feature engineering pipeline"""
    print("Starting feature engineering...")
    df = df.copy()
    
    # Fix column name typo
    if 'Temparature' in df.columns:
        df['Temperature'] = df['Temparature']
        df = df.drop('Temparature', axis=1)
    
    # Basic features (from original)
    df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df['PK_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    
    # Deficiency flags
    for nutrient in ['Nitrogen', 'Phosphorous', 'Potassium']:
        df[f'low_{nutrient}'] = (df[nutrient] < df[nutrient].quantile(CLASS_THRESHOLD)).astype(int)
    
    # Basic combined features
    df['soil_crop'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
    df['nutrient_balance'] = df['Nitrogen'] + df['Phosphorous'] * 1.2 + df['Potassium'] * 0.8
    df['moisture_stress'] = df['Temperature'] * df['Humidity'] / (df['Moisture'] + 1e-5)
    
    # Add all advanced features
    df = advanced_npk_features(df)
    df = environmental_features(df)
    df = agricultural_features(df)
    df = fertilizer_pattern_features(df)
    df = statistical_features(df)
    
    print(f"Feature engineering completed. Total features: {df.shape[1]}")
    return df

def main():
    try:
        # Load data
        train_path = 'datasets/playground-series-s5e6/train.csv'
        test_path = 'datasets/playground-series-s5e6/test.csv'
        
        print("Loading data...")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")
        
        # Save test IDs for submission
        test_ids = test['id']
        
        # Feature engineering
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
        
        # Handle inf and extreme values for XGBoost
        print("Cleaning infinite and extreme values...")
        X = X.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for numerical columns
        for col in X.select_dtypes(include=[np.number]).columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
        
        # Clip extreme values to prevent overflow
        for col in X.select_dtypes(include=[np.number]).columns:
            q99 = X[col].quantile(0.99)
            q01 = X[col].quantile(0.01)
            X[col] = np.clip(X[col], q01, q99)
            X_test[col] = np.clip(X_test[col], q01, q99)
        
        print(f"Using {len(features)} features")
        
        # Handle categorical features for XGBoost
        categoricals = [col for col in features if X[col].dtype == 'object']
        categorical_indices = []
        
        # Label encode categorical features
        for col in categoricals:
            le_cat = LabelEncoder()
            X[col] = le_cat.fit_transform(X[col].astype(str))
            X_test[col] = le_cat.transform(X_test[col].astype(str))
            categorical_indices.append(features.index(col))
        
        print(f"Categorical features: {len(categoricals)}")
        
        # XGBoost GPU parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(le.classes_),
            'eval_metric': 'mlogloss',
            'tree_method': 'gpu_hist',  # GPU acceleration
            'gpu_id': 0,
            'predictor': 'gpu_predictor',  # GPU prediction
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'min_child_weight': 50,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 0.1,  # L2 regularization
            'random_state': RANDOM_STATE,
            'verbosity': 1,
            'n_jobs': -1
        }
        
        # Cross-validation setup
        oof_preds = np.zeros((len(train), len(le.classes_)))
        test_preds = np.zeros((len(test), len(le.classes_)))
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        # Cross-validation loop
        print("\nStarting XGBoost GPU training...")
        for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            print(f'\n=== Fold {fold + 1}/{N_FOLDS} ===')
            
            # Data splits
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            
            # XGBoost DMatrix
            dtrain = xgb.DMatrix(
                X_train, 
                label=y_train,
                enable_categorical=False  # We pre-encoded categoricals
            )
            dvalid = xgb.DMatrix(
                X_valid, 
                label=y_valid,
                enable_categorical=False
            )
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=3000,
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=150,
                verbose_eval=100
            )
            
            # Predictions
            dvalid_pred = xgb.DMatrix(X_valid, enable_categorical=False)
            dtest_pred = xgb.DMatrix(X_test, enable_categorical=False)
            
            oof_preds[valid_idx] = model.predict(dvalid_pred)
            test_preds += model.predict(dtest_pred) / N_FOLDS
            
            # Fold MAP@3 score
            fold_map3 = map3(y_valid, oof_preds[valid_idx])
            print(f'Fold {fold + 1} MAP@3: {fold_map3:.5f}')
        
        # Overall OOF MAP@3
        oof_map3 = map3(y, oof_preds)
        print(f'\n🎯 Overall OOF MAP@3: {oof_map3:.5f}')
        
        # Generate submission
        print("\nGenerating XGBoost submission...")
        top3_preds = []
        for probs in test_preds:
            top3 = np.argsort(probs)[::-1][:3]
            top3_preds.append(' '.join(le.inverse_transform(top3)))
        
        submission = pd.DataFrame({
            'id': test_ids,
            'Fertilizer Name': top3_preds
        })
        
        submission.to_csv('submission_xgboost_gpu.csv', index=False)
        print('✅ XGBoost GPU submission saved to submission_xgboost_gpu.csv')
        print(f"Submission shape: {submission.shape}")
        print("\nSample predictions:")
        print(submission.head(10))
        
        # Feature importance
        print(f"\n📊 Top 20 Feature Importance:")
        feature_imp = pd.DataFrame({
            'feature': features,
            'importance': model.get_score(importance_type='gain')
        })
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        print(feature_imp.head(20))
        
        # Save feature importance
        feature_imp.to_csv('feature_importance_xgboost.csv', index=False)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()