import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import optuna
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
CLASS_THRESHOLD = 0.2
N_TRIALS = 10  # Optuna optimization trials
DATA_FRACTION = 1.0  # Fraction of training data to use (1.0 = full dataset)

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
    df['n_deficit_crop'] = load_or_compute_feature(df, 'n_deficit_crop',
        lambda x: x.apply(lambda row: max(0, crop_n_needs.get(row['Crop Type'], 100) - row['Nitrogen']), axis=1), data_type)
    df['p_deficit_crop'] = load_or_compute_feature(df, 'p_deficit_crop',
        lambda x: x.apply(lambda row: max(0, crop_p_needs.get(row['Crop Type'], 35) - row['Phosphorous']), axis=1), data_type)
    df['k_deficit_crop'] = load_or_compute_feature(df, 'k_deficit_crop',
        lambda x: x.apply(lambda row: max(0, crop_k_needs.get(row['Crop Type'], 40) - row['Potassium']), axis=1), data_type)
    df['total_deficit'] = load_or_compute_feature(df, 'total_deficit',
        lambda x: x['n_deficit_crop'] + x['p_deficit_crop'] + x['k_deficit_crop'], data_type)
    
    # Nutrient adequacy ratios
    df['n_adequacy'] = load_or_compute_feature(df, 'n_adequacy',
        lambda x: x.apply(lambda row: row['Nitrogen'] / crop_n_needs.get(row['Crop Type'], 100), axis=1), data_type)
    df['p_adequacy'] = load_or_compute_feature(df, 'p_adequacy',
        lambda x: x.apply(lambda row: row['Phosphorous'] / crop_p_needs.get(row['Crop Type'], 35), axis=1), data_type)
    df['k_adequacy'] = load_or_compute_feature(df, 'k_adequacy',
        lambda x: x.apply(lambda row: row['Potassium'] / crop_k_needs.get(row['Crop Type'], 40), axis=1), data_type)
    
    # Soil-specific adjustments
    soil_factors = {'Sandy': 1.2, 'Clayey': 0.8, 'Loamy': 1.0, 'Red': 1.1, 'Black': 0.9}
    df['soil_nutrient_factor'] = load_or_compute_feature(df, 'soil_nutrient_factor',
        lambda x: x['Soil Type'].map(soil_factors).fillna(1.0), data_type)
    df['adjusted_n'] = load_or_compute_feature(df, 'adjusted_n',
        lambda x: x['Nitrogen'] * x['soil_nutrient_factor'], data_type)
    df['adjusted_p'] = load_or_compute_feature(df, 'adjusted_p',
        lambda x: x['Phosphorous'] * x['soil_nutrient_factor'], data_type)
    df['adjusted_k'] = load_or_compute_feature(df, 'adjusted_k',
        lambda x: x['Potassium'] * x['soil_nutrient_factor'], data_type)
    
    # Season simulation (based on moisture/temp patterns)
    df['season_indicator'] = load_or_compute_feature(df, 'season_indicator',
        lambda x: np.select([
            (x['Temperature'] > 30) & (x['Moisture'] < 40),  # dry season
            (x['Temperature'] < 25) & (x['Moisture'] > 60),  # wet season
            True  # normal season
        ], ['dry', 'wet', 'normal'], default='normal'), data_type)
    
    # Crop-soil compatibility
    compatible_combinations = {
        'Sandy_Millets': 1, 'Sandy_Ground Nuts': 1, 'Clayey_Paddy': 1, 'Clayey_Sugarcane': 1,
        'Red_Pulses': 1, 'Red_Cotton': 1, 'Loamy_Wheat': 1, 'Loamy_Maize': 1, 'Black_Cotton': 1
    }
    df['crop_soil_compatibility'] = load_or_compute_feature(df, 'crop_soil_compatibility',
        lambda x: (x['Soil Type'] + '_' + x['Crop Type']).map(compatible_combinations).fillna(0), data_type)
    
    return df

def fertilizer_pattern_features(df, data_type='train'):
    """Fertilizer pattern recognition features"""
    print("  -> Adding fertilizer pattern features...")
    
    # Common fertilizer signatures (NPK patterns)
    # Urea: 46-0-0
    df['urea_signature'] = load_or_compute_feature(df, 'urea_signature',
        lambda x: np.abs(x['Nitrogen'] - 46) + x['Phosphorous'] + x['Potassium'], data_type)
    df['urea_affinity'] = load_or_compute_feature(df, 'urea_affinity',
        lambda x: np.exp(-x['urea_signature'] / 20), data_type)  # Closer to 0 = higher affinity
    
    # DAP: 18-46-0  
    df['dap_signature'] = load_or_compute_feature(df, 'dap_signature',
        lambda x: np.abs(x['Nitrogen'] - 18) + np.abs(x['Phosphorous'] - 46) + x['Potassium'], data_type)
    df['dap_affinity'] = load_or_compute_feature(df, 'dap_affinity',
        lambda x: np.exp(-x['dap_signature'] / 20), data_type)
    
    # Complex fertilizers (balanced NPK)
    df['complex_signature'] = load_or_compute_feature(df, 'complex_signature',
        lambda x: np.abs(x['Nitrogen'] - x['Phosphorous']) + np.abs(x['Phosphorous'] - x['Potassium']), data_type)
    df['complex_affinity'] = load_or_compute_feature(df, 'complex_affinity',
        lambda x: np.exp(-x['complex_signature'] / 10), data_type)
    
    # 20-20-0 pattern
    df['2020_signature'] = load_or_compute_feature(df, '2020_signature',
        lambda x: np.abs(x['Nitrogen'] - 20) + np.abs(x['Phosphorous'] - 20) + x['Potassium'], data_type)
    df['2020_affinity'] = load_or_compute_feature(df, '2020_affinity',
        lambda x: np.exp(-x['2020_signature'] / 15), data_type)
    
    # High P fertilizers (like 10-26-26)
    df['high_p_signature'] = load_or_compute_feature(df, 'high_p_signature',
        lambda x: np.abs(x['Phosphorous'] - 26) + np.abs(x['Potassium'] - 26), data_type)
    df['high_p_affinity'] = load_or_compute_feature(df, 'high_p_affinity',
        lambda x: np.exp(-x['high_p_signature'] / 15), data_type)
    
    # Fertilizer need indicators
    df['needs_nitrogen'] = load_or_compute_feature(df, 'needs_nitrogen',
        lambda x: (x['Nitrogen'] < 25).astype(int), data_type)
    df['needs_phosphorus'] = load_or_compute_feature(df, 'needs_phosphorus',
        lambda x: (x['Phosphorous'] < 20).astype(int), data_type)
    df['needs_potassium'] = load_or_compute_feature(df, 'needs_potassium',
        lambda x: (x['Potassium'] < 15).astype(int), data_type)
    df['fertilizer_urgency'] = load_or_compute_feature(df, 'fertilizer_urgency',
        lambda x: x['needs_nitrogen'] + x['needs_phosphorus'] + x['needs_potassium'], data_type)
    
    return df

def statistical_features(df, data_type='train'):
    """Statistical aggregation features"""
    print("  -> Adding statistical features...")
    
    # Group statistics by soil type
    for col in ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture']:
        df[f'{col}_soil_mean'] = load_or_compute_feature(df, f'{col}_soil_mean',
            lambda x, column=col: x.groupby('Soil Type')[column].transform('mean'), data_type)
        df[f'{col}_soil_std'] = load_or_compute_feature(df, f'{col}_soil_std',
            lambda x, column=col: x.groupby('Soil Type')[column].transform('std'), data_type)
        df[f'{col}_soil_deviation'] = load_or_compute_feature(df, f'{col}_soil_deviation',
            lambda x, column=col: x[column] - x[f'{column}_soil_mean'], data_type)
        df[f'{col}_soil_zscore'] = load_or_compute_feature(df, f'{col}_soil_zscore',
            lambda x, column=col: x[f'{column}_soil_deviation'] / (x[f'{column}_soil_std'] + 1e-5), data_type)
    
    # Group statistics by crop type  
    for col in ['Nitrogen', 'Phosphorous', 'Potassium']:
        df[f'{col}_crop_mean'] = load_or_compute_feature(df, f'{col}_crop_mean',
            lambda x, column=col: x.groupby('Crop Type')[column].transform('mean'), data_type)
        df[f'{col}_crop_std'] = load_or_compute_feature(df, f'{col}_crop_std',
            lambda x, column=col: x.groupby('Crop Type')[column].transform('std'), data_type)
        df[f'{col}_crop_deviation'] = load_or_compute_feature(df, f'{col}_crop_deviation',
            lambda x, column=col: x[column] - x[f'{column}_crop_mean'], data_type)
        df[f'{col}_crop_zscore'] = load_or_compute_feature(df, f'{col}_crop_zscore',
            lambda x, column=col: x[f'{column}_crop_deviation'] / (x[f'{column}_crop_std'] + 1e-5), data_type)
    
    # Soil-Crop combinations
    for col in ['Nitrogen', 'Phosphorous', 'Potassium']:
        df[f'{col}_soilcrop_mean'] = load_or_compute_feature(df, f'{col}_soilcrop_mean',
            lambda x, column=col: x.groupby(['Soil Type', 'Crop Type'])[column].transform('mean'), data_type)
        df[f'{col}_soilcrop_deviation'] = load_or_compute_feature(df, f'{col}_soilcrop_deviation',
            lambda x, column=col: x[column] - x[f'{column}_soilcrop_mean'], data_type)
    
    # Rank features
    df['nitrogen_rank'] = load_or_compute_feature(df, 'nitrogen_rank',
        lambda x: x['Nitrogen'].rank(pct=True), data_type)
    df['phosphorous_rank'] = load_or_compute_feature(df, 'phosphorous_rank',
        lambda x: x['Phosphorous'].rank(pct=True), data_type)
    df['potassium_rank'] = load_or_compute_feature(df, 'potassium_rank',
        lambda x: x['Potassium'].rank(pct=True), data_type)
    
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
    
    # Add all advanced features with caching
    df = advanced_npk_features(df, data_type)
    df = environmental_features(df, data_type)
    df = agricultural_features(df, data_type)
    df = fertilizer_pattern_features(df, data_type)
    df = statistical_features(df, data_type)
    
    # Add binning for numerical features
    print("  -> Adding binned features...")
    numerical_features = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture']
    for col in numerical_features:
        if col in df.columns:
            df[f'{col}_Binned'] = df[col].astype(str).astype('category')
    
    # Convert integer columns to int8 for memory efficiency
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int8')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float16')
    
    print(f"Feature engineering completed. Total features: {df.shape[1]}")
    return df

def objective(trial, X, y, n_classes):
    """Optuna objective function for hyperparameter optimization"""
    
    # Hyperparameter search space
    params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'predictor': 'gpu_predictor',
        'random_state': RANDOM_STATE,
        'verbosity': 0,
        'n_jobs': -1,
        
        # Tunable parameters
        'max_depth': trial.suggest_int('max_depth', 6, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
    }
    
    # Cross-validation
    scores = []
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # Reduced folds for speed
    
    for train_idx, valid_idx in folds.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        # XGBoost DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=False)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,  # Reduced for speed
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Predictions
        dvalid_pred = xgb.DMatrix(X_valid, enable_categorical=False)
        preds = model.predict(dvalid_pred)
        
        # MAP@3 score
        score = map3(y_valid, preds)
        scores.append(score)
    
    return np.mean(scores)

def main():
    try:
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
        
        # Hyperparameter optimization with Optuna
        print(f"\n🔍 Starting Optuna hyperparameter optimization ({N_TRIALS} trials)...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(lambda trial: objective(trial, X, y, len(le.classes_)), n_trials=N_TRIALS)
        
        print(f"✅ Best MAP@3 score: {study.best_value:.5f}")
        print(f"🎯 Best parameters: {study.best_params}")
        
        # Train final model with best parameters
        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'multi:softprob',
            'num_class': len(le.classes_),
            'eval_metric': 'mlogloss',
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'predictor': 'gpu_predictor',
            'random_state': RANDOM_STATE,
            'verbosity': 1,
            'n_jobs': -1
        })
        
        # Cross-validation with best parameters
        oof_preds = np.zeros((len(train), len(le.classes_)))
        test_preds = np.zeros((len(test), len(le.classes_)))
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        print(f"\n🚀 Training final model with optimized parameters...")
        for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            print(f'\n=== Fold {fold + 1}/{N_FOLDS} ===')
            
            # Data splits
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            
            # XGBoost DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
            dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=False)
            
            # Train model
            model = xgb.train(
                best_params,
                dtrain,
                num_boost_round=5000,
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=200,
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
        print("\nGenerating optimized XGBoost submission...")
        top3_preds = []
        for probs in test_preds:
            top3 = np.argsort(probs)[::-1][:3]
            top3_preds.append(' '.join(le.inverse_transform(top3)))
        
        submission = pd.DataFrame({
            'id': test_ids,
            'Fertilizer Name': top3_preds
        })
        
        submission.to_csv('005_fertilizer_prediction_xgboost_optuna_submission.csv', index=False)
        print('✅ Optimized XGBoost submission saved to 005_fertilizer_prediction_xgboost_optuna_submission.csv')
        print(f"Submission shape: {submission.shape}")
        print("\nSample predictions:")
        print(submission.head(10))
        
        # Feature importance
        print(f"\n📊 Top 20 Feature Importance:")
        importance_dict = model.get_score(importance_type='gain')
        feature_imp = pd.DataFrame([
            {'feature': fname, 'importance': importance_dict.get(fname, 0)}
            for fname in features
        ])
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        print(feature_imp.head(20))
        
        # Save feature importance and optimization results
        feature_imp.to_csv('005_fertilizer_prediction_xgboost_optuna_feature_importance.csv', index=False)
        
        # Save optimization results
        optuna_results = pd.DataFrame({
            'trial': range(len(study.trials)),
            'value': [trial.value for trial in study.trials],
            'params': [str(trial.params) for trial in study.trials]
        })
        optuna_results.to_csv('005_fertilizer_prediction_xgboost_optuna_results.csv', index=False)
        
        print(f"\n📈 Optimization results saved:")
        print(f"  - Feature importance: 005_fertilizer_prediction_xgboost_optuna_feature_importance.csv")
        print(f"  - Optuna trials: 005_fertilizer_prediction_xgboost_optuna_results.csv")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()