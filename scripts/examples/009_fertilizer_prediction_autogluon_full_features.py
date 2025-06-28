import numpy as np
import pandas as pd
import warnings
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from autogluon.tabular import TabularDataset, TabularPredictor
from scripts.feature_engineering import load_or_create_features
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 3
CLASS_THRESHOLD = 0.2
DATA_FRACTION = 1.0  # Fraction of training data to use (1.0 = full dataset)
TIME_LIMIT = 8*3600  # AutoGluon training time limit in seconds (1 hour)
PRESETS = 'best_quality'  # 'medium_quality_faster_train', 'medium_quality', 'best_quality'

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
    """Environmental interaction features (excluding zero-importance features)"""
    print("  -> Adding environmental features...")
    
    # Temperature-humidity stress (but skip cold_stress - zero importance)
    df['heat_stress'] = load_or_compute_feature(df, 'heat_stress',
        lambda x: np.where(x['Temperature'] > 35, (x['Temperature'] - 35) * (100 - x['Humidity']) / 100, 0), data_type)
    
    # Water stress indicators (but skip flood_stress - zero importance)
    df['drought_stress'] = load_or_compute_feature(df, 'drought_stress',
        lambda x: np.where(x['Moisture'] < 30, (30 - x['Moisture']) * (x['Temperature'] / 30), 0), data_type)
    df['water_stress'] = load_or_compute_feature(df, 'water_stress',
        lambda x: x['drought_stress'], data_type)  # Only drought stress now
    
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
    
    # Climate zones simulation (excluding cool_wet and hot_dry - zero importance)
    df['hot_humid'] = load_or_compute_feature(df, 'hot_humid',
        lambda x: ((x['Temperature'] > 30) & (x['Humidity'] > 70)).astype(int), data_type)
    
    return df

def agricultural_features(df, data_type='train'):
    """Agricultural domain-specific features"""
    print("  -> Adding agricultural features...")
    
    # Crop-specific nutrient requirements (from agricultural knowledge)
    crop_n_needs = {
        'Wheat': 80, 'Rice': 60, 'Corn': 100, 'Sugarcane': 120, 'Cotton': 80,
        'Barley': 70, 'Millets': 40, 'Paddy': 60, 'Pulses': 20, 'Tobacco': 100,
        'Ground Nuts': 40, 'Maize': 100
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
        'Loamy_Wheat': 1, 'Loamy_Corn': 1, 'Red_Cotton': 1, 'Black_Cotton': 1
    }
    df['crop_soil_compatibility'] = load_or_compute_feature(df, 'crop_soil_compatibility',
        lambda x: x['soil_crop'].map(compatible_combinations).fillna(0), data_type)
    
    # Nutrient needs flags
    df['needs_nitrogen'] = load_or_compute_feature(df, 'needs_nitrogen',
        lambda x: (x['n_deficit_crop'] > 20).astype(int), data_type)
    df['needs_phosphorus'] = load_or_compute_feature(df, 'needs_phosphorus',
        lambda x: (x['p_deficit_crop'] > 10).astype(int), data_type)
    df['needs_potassium'] = load_or_compute_feature(df, 'needs_potassium',
        lambda x: (x['k_deficit_crop'] > 15).astype(int), data_type)
    
    # Fertilizer urgency
    df['fertilizer_urgency'] = load_or_compute_feature(df, 'fertilizer_urgency',
        lambda x: x['needs_nitrogen'] + x['needs_phosphorus'] + x['needs_potassium'], data_type)
    
    return df

def fertilizer_pattern_features(df, data_type='train'):
    """Fertilizer-specific pattern features"""
    print("  -> Adding fertilizer pattern features...")
    
    # Urea patterns (high nitrogen)
    df['urea_signature'] = load_or_compute_feature(df, 'urea_signature',
        lambda x: ((x['Nitrogen'] > 40) & (x['Phosphorous'] < 20) & (x['Potassium'] < 20)).astype(int), data_type)
    df['urea_affinity'] = load_or_compute_feature(df, 'urea_affinity',
        lambda x: x['Nitrogen'] / (x['Phosphorous'] + x['Potassium'] + 1e-5), data_type)
    
    # DAP patterns (high phosphorus)
    df['dap_signature'] = load_or_compute_feature(df, 'dap_signature',
        lambda x: ((x['Phosphorous'] > 30) & (x['Nitrogen'] > 15) & (x['Potassium'] < 20)).astype(int), data_type)
    df['dap_affinity'] = load_or_compute_feature(df, 'dap_affinity',
        lambda x: (x['Nitrogen'] + x['Phosphorous']) / (x['Potassium'] + 1e-5), data_type)
    
    # Complex fertilizer patterns (balanced NPK)
    df['complex_signature'] = load_or_compute_feature(df, 'complex_signature',
        lambda x: ((x['Nitrogen'] > 15) & (x['Phosphorous'] > 15) & (x['Potassium'] > 15)).astype(int), data_type)
    df['complex_affinity'] = load_or_compute_feature(df, 'complex_affinity',
        lambda x: 1 / (1 + np.abs(x['Nitrogen'] - x['Phosphorous']) + np.abs(x['Phosphorous'] - x['Potassium'])), data_type)
    
    # 20-20 patterns
    df['2020_signature'] = load_or_compute_feature(df, '2020_signature',
        lambda x: ((x['Nitrogen'].between(18, 25)) & (x['Phosphorous'].between(18, 25))).astype(int), data_type)
    df['2020_affinity'] = load_or_compute_feature(df, '2020_affinity',
        lambda x: 1 / (1 + np.abs(x['Nitrogen'] - 20) + np.abs(x['Phosphorous'] - 20)), data_type)
    
    # High-P signature fertilizers
    df['high_p_signature'] = load_or_compute_feature(df, 'high_p_signature',
        lambda x: (x['Phosphorous'] > 30).astype(int), data_type)
    df['high_p_affinity'] = load_or_compute_feature(df, 'high_p_affinity',
        lambda x: x['Phosphorous'] / (x['Nitrogen'] + x['Potassium'] + 1e-5), data_type)
    
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
            lambda x, column=col: x.groupby('soil_crop')[column].transform('mean'), data_type)
        df[f'{col}_soilcrop_deviation'] = load_or_compute_feature(df, f'{col}_soilcrop_deviation',
            lambda x, column=col: x[column] - x[f'{column}_soilcrop_mean'], data_type)
    
    # Nutrient rankings within groups
    for col in ['Nitrogen', 'Phosphorous', 'Potassium']:
        df[f'{col.lower()}_rank'] = load_or_compute_feature(df, f'{col.lower()}_rank',
            lambda x, column=col: x.groupby('soil_crop')[column].rank(pct=True), data_type)
    
    return df

def feature_engineering(df, data_type='train'):
    """Complete feature engineering pipeline with all features (like 007)"""
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
    
    # Add all advanced features with caching (like 005/007)
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
            df[f'{col}_Binned'] = pd.cut(df[col], bins=10, labels=False, duplicates='drop')
    
    # Clean infinite and extreme values
    print("  -> Cleaning data...")
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = np.nan_to_num(df[col], nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Convert integer columns to int8 for memory efficiency (AutoGluon can handle this)
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int8')
        elif df[col].dtype == 'float64':
            # Keep float64 for AutoGluon - it handles precision better
            pass
    
    print(f"Feature engineering completed. Total features: {df.shape[1]}")
    return df

def main():
    try:
        # Get script name for dynamic file naming
        import os
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        
        # Set random seed
        np.random.seed(RANDOM_STATE)
        
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
        
        # Feature engineering with ALL features using shared module
        train = load_or_create_features(train_path, 'train', 'full')
        test = load_or_create_features(test_path, 'test', 'full')
        
        # Prepare target
        target = 'Fertilizer Name'
        n_classes = train[target].nunique()
        print(f"Number of classes: {n_classes}")
        print(f"Classes: {sorted(train[target].unique())}")
        
        # Prepare features (remove id column)
        features = [col for col in train.columns if col not in ['id', target]]
        print(f"Total features: {len(features)}")
        
        # Cross-validation setup
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof_preds = []
        test_preds = []
        oof_labels = []
        
        print(f"\n🚀 Starting AutoGluon training with {N_FOLDS}-fold CV and FULL feature set...")
        print(f"Time limit per fold: {TIME_LIMIT} seconds")
        print(f"Preset: {PRESETS}")
        
        for fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train[target])):
            print(f'\n=== Fold {fold + 1}/{N_FOLDS} ===')
            
            # Split data
            train_fold = train.iloc[train_idx][features + [target]].copy()
            valid_fold = train.iloc[valid_idx][features + [target]].copy()
            
            print(f"Train fold shape: {train_fold.shape}")
            print(f"Valid fold shape: {valid_fold.shape}")
            
            # Convert to TabularDataset
            train_data = TabularDataset(train_fold)
            valid_data = TabularDataset(valid_fold)
            
            # AutoGluon model directory
            model_dir = f'autogluon_models_full/fold_{fold}'
            
            # Train AutoGluon predictor
            print(f"Training AutoGluon predictor for fold {fold + 1}...")
            predictor = TabularPredictor(
                label=target,
                path=model_dir,
                eval_metric='log_loss',  # Better for probability ranking (closer to MAP@3)
                verbosity=2
            )
            
            # Combine train and validation data (AutoGluon 1.3+ best practice)
            combined_data = pd.concat([train_fold, valid_fold], ignore_index=True)
            combined_data = TabularDataset(combined_data)
            
            predictor.fit(
                combined_data,
                time_limit=TIME_LIMIT,
                presets=PRESETS,
                holdout_frac=0.2,  # AutoGluon will automatically hold out 20% for validation
                num_bag_folds=5,   # Enable bagging with 5 folds
                excluded_model_types=['KNN', 'NN_TORCH', 'FASTAI'],  # Exclude slow models for speed
            )
            
            # Get validation predictions using original validation fold
            print("Making validation predictions...")
            valid_features = valid_fold.drop(columns=[target])
            val_pred_proba = predictor.predict_proba(valid_features)
            
            # Store OOF predictions and labels
            oof_preds.append(val_pred_proba.values)
            oof_labels.extend(valid_fold[target].values)
            
            # Test predictions for this fold
            print("Making test predictions...")
            test_pred_proba = predictor.predict_proba(test[features])
            test_preds.append(test_pred_proba.values)
            
            # Calculate fold MAP@3 score
            fold_map3 = map3(valid_fold[target].values, val_pred_proba.values)
            print(f'Fold {fold + 1} MAP@3: {fold_map3:.5f}')
            
            # Print leaderboard for this fold
            print(f"\nFold {fold + 1} Leaderboard:")
            try:
                leaderboard = predictor.leaderboard(silent=True)
                print(leaderboard.head(10))
            except Exception as e:
                print(f"Could not display leaderboard: {e}")
        
        # Combine OOF predictions
        oof_pred_proba = np.vstack(oof_preds)
        
        # Calculate overall OOF MAP@3
        oof_map3 = map3(oof_labels, oof_pred_proba)
        print(f'\n🎯 Overall OOF MAP@3: {oof_map3:.5f}')
        
        # Average test predictions across folds
        test_pred_proba_avg = np.mean(test_preds, axis=0)
        
        # Get class names for submission
        class_names = predictor.class_labels
        
        # Generate submission
        print("\nGenerating AutoGluon FULL features submission...")
        top3_preds = []
        for probs in test_pred_proba_avg:
            top3_indices = np.argsort(probs)[::-1][:3]
            top3_classes = [class_names[i] for i in top3_indices]
            top3_preds.append(' '.join(top3_classes))
        
        submission = pd.DataFrame({
            'id': test_ids,
            'Fertilizer Name': top3_preds
        })
        
        submission_file = f'{script_name}_submission.csv'
        submission.to_csv(submission_file, index=False)
        print(f'✅ AutoGluon FULL features submission saved to {submission_file}')
        print(f"Submission shape: {submission.shape}")
        print("\nSample predictions:")
        print(submission.head(10))
        
        # Save feature importance from last fold
        print("\nSaving feature importance...")
        try:
            feature_importance = predictor.feature_importance(test[features])
            importance_file = f'{script_name}_feature_importance.csv'
            feature_importance.to_csv(importance_file)
            print(f"✅ Feature importance saved to {importance_file}")
            print(f"Top 10 most important features:")
            print(feature_importance.head(10))
        except Exception as e:
            print(f"Could not save feature importance: {e}")
        
        # Save model summary
        print("\nFinal Model Summary:")
        print(f"Used FULL feature set: {len(features)} features")
        print(f"Overall OOF MAP@3: {oof_map3:.5f}")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
