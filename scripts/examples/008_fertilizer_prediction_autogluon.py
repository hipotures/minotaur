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
N_FOLDS = 5
CLASS_THRESHOLD = 0.2
DATA_FRACTION = 1.0  # Fraction of training data to use (1.0 = full dataset)
TIME_LIMIT = 3600  # AutoGluon training time limit in seconds (1 hour)
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


def custom_map3_scorer(y_true, y_pred_proba):
    """Custom MAP@3 scorer for AutoGluon"""
    return map3(y_true, y_pred_proba)

def main():
    try:
        # Get script name for dynamic file naming
        import os
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        
        # Set random seed
        np.random.seed(RANDOM_STATE)
        
        # Load data
        train_path = 'datasets/playground-series-s5e6/train.csv'
        test_path = 'datasets/playground-series-s5e6/test.csv'
        
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
        
        print(f"\n🚀 Starting AutoGluon training with {N_FOLDS}-fold CV...")
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
            model_dir = f'autogluon_models/fold_{fold}'
            
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
        print("\nGenerating AutoGluon submission...")
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
        print(f'✅ AutoGluon submission saved to {submission_file}')
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
        except Exception as e:
            print(f"Could not save feature importance: {e}")
        
        # Save model summary
        print("\nFinal Model Summary:")
        print(f"Best models across folds used various algorithms")
        print(f"Overall OOF MAP@3: {oof_map3:.5f}")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()