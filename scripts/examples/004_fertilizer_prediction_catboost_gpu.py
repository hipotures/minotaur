import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
import warnings
import os
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

def load_or_compute_feature(df, feature_name, computation_func, data_type='train'):
    """Load feature from cache or compute and save"""
    features_dir = f'features/{data_type}'
    os.makedirs(features_dir, exist_ok=True)
    feature_path = f'{features_dir}/{feature_name}.csv'
    
    if os.path.exists(feature_path):
        print(f"    Loading cached {feature_name}...")
        feature_data = pd.read_csv(feature_path, index_col=0)
        return feature_data[feature_name]
    else:
        print(f"    Computing {feature_name}...")
        feature_values = computation_func(df)
        feature_df = pd.DataFrame({feature_name: feature_values}, index=df.index)
        feature_df.to_csv(feature_path)
        return feature_values

def advanced_npk_features(df, data_type='train'):
    """Advanced NPK interaction features"""
    print("  -> Adding advanced NPK features...")
    
    # Complex NPK interactions
    df['npk_harmony'] = load_or_compute_feature(df, 'npk_harmony', 
        lambda x: 1 / (1 + np.abs(x['Nitrogen']/10 - x['Phosphorous']/26 - x['Potassium']/26)), data_type)
    df['npk_distance'] = load_or_compute_feature(df, 'npk_distance',
        lambda x: np.sqrt((x['Nitrogen']-20)**2 + (x['Phosphorous']-20)**2 + (x['Potassium']-20)**2), data_type)
    df['npk_sum'] = load_or_compute_feature(df, 'npk_sum',
        lambda x: x['Nitrogen'] + x['Phosphorous'] + x['Potassium'], data_type)
    df['npk_product'] = load_or_compute_feature(df, 'npk_product',
        lambda x: x['Nitrogen'] * x['Phosphorous'] * x['Potassium'], data_type)
    
    # Fertilizer balance indicators
    df['is_balanced'] = load_or_compute_feature(df, 'is_balanced',
        lambda x: ((x['Nitrogen'] / (x['Phosphorous'] + 1e-5)).between(0.8, 1.2) & 
                   (x['Phosphorous'] / (x['Potassium'] + 1e-5)).between(0.8, 1.2)).astype(int), data_type)
    
    # NPK dominance patterns
    df['n_dominant'] = load_or_compute_feature(df, 'n_dominant',
        lambda x: (x['Nitrogen'] > x['Phosphorous']) & (x['Nitrogen'] > x['Potassium']), data_type)
    df['p_dominant'] = load_or_compute_feature(df, 'p_dominant',
        lambda x: (x['Phosphorous'] > x['Nitrogen']) & (x['Phosphorous'] > x['Potassium']), data_type)
    df['k_dominant'] = load_or_compute_feature(df, 'k_dominant',
        lambda x: (x['Potassium'] > x['Nitrogen']) & (x['Potassium'] > x['Phosphorous']), data_type)
    
    # NPK deficiency combinations
    df['npk_deficiency_pattern'] = load_or_compute_feature(df, 'npk_deficiency_pattern',
        lambda x: (x['low_Nitrogen'].astype(str) + x['low_Phosphorous'].astype(str) + x['low_Potassium'].astype(str)), data_type)
    
    # Advanced ratios
    df['np_pk_ratio'] = load_or_compute_feature(df, 'np_pk_ratio',
        lambda x: (x['Nitrogen'] / (x['Phosphorous'] + 1e-5)) / (x['Phosphorous'] / (x['Potassium'] + 1e-5)), data_type)
    df['npk_variance'] = load_or_compute_feature(df, 'npk_variance',
        lambda x: x[['Nitrogen', 'Phosphorous', 'Potassium']].var(axis=1), data_type)
    df['npk_cv'] = load_or_compute_feature(df, 'npk_cv',
        lambda x: x[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1) / (x[['Nitrogen', 'Phosphorous', 'Potassium']].mean(axis=1) + 1e-5), data_type)
    
    return df

def environmental_features(df, data_type='train'):
    """Environmental interaction features"""
    print("  -> Adding environmental features...")
    
    # Temperature-humidity stress
    df['heat_stress'] = load_or_compute_feature(df, 'heat_stress',
        lambda x: np.where(x['Temperature'] > 35, (x['Temperature'] - 35) * (100 - x['Humidity']) / 100, 0), data_type)
    df['cold_stress'] = load_or_compute_feature(df, 'cold_stress',
        lambda x: np.where(x['Temperature'] < 20, (20 - x['Temperature']) * x['Humidity'] / 100, 0), data_type)
    
    # Water stress indicators
    df['drought_stress'] = load_or_compute_feature(df, 'drought_stress',
        lambda x: np.where(x['Moisture'] < 30, (30 - x['Moisture']) * (x['Temperature'] / 30), 0), data_type)
    df['flood_stress'] = load_or_compute_feature(df, 'flood_stress',
        lambda x: np.where(x['Moisture'] > 70, (x['Moisture'] - 70) * (x['Humidity'] / 100), 0), data_type)
    df['water_stress'] = load_or_compute_feature(df, 'water_stress',
        lambda x: x['drought_stress'] + x['flood_stress'], data_type)
    
    # Optimal growing conditions
    df['optimal_temp'] = load_or_compute_feature(df, 'optimal_temp',
        lambda x: ((x['Temperature'] >= 25) & (x['Temperature'] <= 32)).astype(int), data_type)
    df['optimal_humidity'] = load_or_compute_feature(df, 'optimal_humidity',
        lambda x: ((x['Humidity'] >= 60) & (x['Humidity'] <= 75)).astype(int), data_type)
    df['optimal_moisture'] = load_or_compute_feature(df, 'optimal_moisture',
        lambda x: ((x['Moisture'] >= 40) & (x['Moisture'] <= 60)).astype(int), data_type)
    df['optimal_conditions'] = load_or_compute_feature(df, 'optimal_conditions',
        lambda x: x['optimal_temp'] * x['optimal_humidity'] * x['optimal_moisture'], data_type)
    
    # Environmental interactions
    df['temp_humidity_interaction'] = load_or_compute_feature(df, 'temp_humidity_interaction',
        lambda x: x['Temperature'] * x['Humidity'] / 100, data_type)
    df['temp_moisture_interaction'] = load_or_compute_feature(df, 'temp_moisture_interaction',
        lambda x: x['Temperature'] * x['Moisture'] / 100, data_type)
    df['humidity_moisture_interaction'] = load_or_compute_feature(df, 'humidity_moisture_interaction',
        lambda x: x['Humidity'] * x['Moisture'] / 100, data_type)
    
    # Climate zones simulation
    df['hot_humid'] = load_or_compute_feature(df, 'hot_humid',
        lambda x: ((x['Temperature'] > 30) & (x['Humidity'] > 70)).astype(int), data_type)
    df['hot_dry'] = load_or_compute_feature(df, 'hot_dry',
        lambda x: ((x['Temperature'] > 30) & (x['Humidity'] < 50)).astype(int), data_type)
    df['cool_wet'] = load_or_compute_feature(df, 'cool_wet',
        lambda x: ((x['Temperature'] < 25) & (x['Moisture'] > 60)).astype(int), data_type)
    
    return df

def agricultural_features(df, data_type='train'):
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

def fertilizer_pattern_features(df, data_type='train'):
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

def statistical_features(df, data_type='train'):
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

def feature_engineering(df, data_type='train'):
    """Complete feature engineering pipeline"""
    print(f"Starting feature engineering for {data_type}...")
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
    df = advanced_npk_features(df, data_type)
    df = environmental_features(df, data_type)
    df = agricultural_features(df, data_type)
    df = fertilizer_pattern_features(df, data_type)
    df = statistical_features(df, data_type)
    
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
        train = feature_engineering(train, 'train')
        test = feature_engineering(test, 'test')
        
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
        
        print(f"Using {len(features)} features")
        
        # Identify categorical features for CatBoost
        categorical_features = [col for col in features if X[col].dtype == 'object']
        categorical_indices = [i for i, col in enumerate(features) if col in categorical_features]
        
        print(f"Categorical features: {len(categorical_features)}")
        
        # Cross-validation setup
        oof_preds = np.zeros((len(train), len(le.classes_)))
        test_preds = np.zeros((len(test), len(le.classes_)))
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        # Cross-validation loop
        print("\nStarting CatBoost GPU training...")
        for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            print(f'\n=== Fold {fold + 1}/{N_FOLDS} ===')
            
            # Data splits
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            
            # CatBoost Pool objects
            train_pool = Pool(
                X_train, 
                y_train, 
                cat_features=categorical_indices
            )
            valid_pool = Pool(
                X_valid, 
                y_valid, 
                cat_features=categorical_indices
            )
            
            # CatBoost model with GPU
            model = CatBoostClassifier(
                iterations=3000,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=0.1,
                bootstrap_type='Bernoulli',  # Enable subsample
                subsample=0.8,
                # colsample_bylevel=0.7,  # rsm not supported on GPU for multiclass
                min_data_in_leaf=50,
                task_type='GPU',  # GPU acceleration
                devices='0',
                objective='MultiClass',
                eval_metric='MultiClass',
                random_seed=RANDOM_STATE,
                verbose=100,
                early_stopping_rounds=150,
                use_best_model=True
            )
            
            # Train model
            model.fit(
                train_pool,
                eval_set=valid_pool,
                plot=False,
                verbose_eval=100
            )
            
            # Predictions
            oof_preds[valid_idx] = model.predict_proba(X_valid)
            test_preds += model.predict_proba(X_test) / N_FOLDS
            
            # Fold MAP@3 score
            fold_map3 = map3(y_valid, oof_preds[valid_idx])
            print(f'Fold {fold + 1} MAP@3: {fold_map3:.5f}')
        
        # Overall OOF MAP@3
        oof_map3 = map3(y, oof_preds)
        print(f'\n🎯 Overall OOF MAP@3: {oof_map3:.5f}')
        
        # Generate submission
        print("\nGenerating CatBoost submission...")
        top3_preds = []
        for probs in test_preds:
            top3 = np.argsort(probs)[::-1][:3]
            top3_preds.append(' '.join(le.inverse_transform(top3)))
        
        submission = pd.DataFrame({
            'id': test_ids,
            'Fertilizer Name': top3_preds
        })
        
        submission.to_csv('submission_catboost_gpu.csv', index=False)
        print('✅ CatBoost GPU submission saved to submission_catboost_gpu.csv')
        print(f"Submission shape: {submission.shape}")
        print("\nSample predictions:")
        print(submission.head(10))
        
        # Feature importance
        print(f"\n📊 Top 20 Feature Importance:")
        feature_imp = pd.DataFrame({
            'feature': features,
            'importance': model.get_feature_importance()
        }).sort_values('importance', ascending=False)
        print(feature_imp.head(20))
        
        # Save feature importance
        feature_imp.to_csv('feature_importance_catboost.csv', index=False)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()