# Fertilizer Prediction Models - Evolution

This directory contains the evolution of MLAgents-generated fertilizer prediction models for Kaggle Playground Series S5E6.

## Model Versions

### 001_fertilizer_prediction_gpu.py
**Commit:** `460fa15` → `acf69ab`  
**Features:** 15 basic features  
**Performance:**
- OOF MAP@3: 0.32065
- Kaggle Public Score: 0.32833  
- Ranking: ~1300th place

**Key Features:**
- Basic NPK ratios (NP_ratio, NK_ratio, PK_ratio)
- Soil-crop interactions (soil_crop)
- Simple deficiency flags
- Environmental moisture stress
- GPU-accelerated LightGBM training

**Notes:** Solid baseline with clean feature engineering. Good generalization (public > validation).

---

### 002_fertilizer_prediction_optimized.py  
**Commit:** `3eaf876`  
**Features:** 116 engineered features  
**Performance:**
- OOF MAP@3: 0.32361 (+0.9%)
- Kaggle Public Score: 0.32756 (-0.2%)
- Ranking: TBD

**Key Features:**
- Advanced NPK interactions (harmony, distance, variance, dominance patterns)
- Environmental stress indicators (heat, drought, flood, climate zones)
- Agricultural domain knowledge (crop-specific nutrient needs, soil factors)
- Statistical group features (deviations, z-scores, rankings by soil/crop)
- Fertilizer pattern matching (Urea, DAP, complex signatures)
- Enhanced GPU optimization with L1/L2 regularization

**Notes:** Over-engineered model showing validation improvement but public regression. Classic overfitting pattern - more features don't always help generalization.

---

### 003_fertilizer_prediction_xgboost_gpu.py
**Commit:** `b32276e`  
**Features:** 116 engineered features (same as 002)  
**Performance:**
- OOF MAP@3: 0.33311 (+3.7% vs baseline)
- Kaggle Public Score: 0.33053 (+0.7% vs baseline)

**Key Features:**
- XGBoost GPU acceleration with `gpu_hist` tree method
- Same comprehensive feature set as optimized LightGBM model
- GPU predictor for faster inference
- Enhanced regularization (L1/L2) for overfitting control
- Data cleaning for inf/extreme values (XGBoost sensitivity)
- Tree-based feature interactions vs gradient boosting differences

**Notes:** Alternative gradient boosting implementation with robust data preprocessing for XGBoost compatibility.

---

### 004_fertilizer_prediction_catboost_gpu.py
**Commit:** `d26d149`  
**Features:** 116 engineered features (same as 002)  
**Performance:**
- OOF MAP@3: 0.32477 (+1.3% vs baseline)
- Kaggle Public Score: 0.32364 (-1.4% vs baseline)

**Key Features:**
- CatBoost GPU acceleration with automatic categorical handling
- Native categorical feature support (no manual encoding needed)
- Same comprehensive feature set as optimized models
- Built-in overfitting protection and regularization
- Symmetric tree structure for better generalization

**Notes:** Third gradient boosting alternative with superior categorical feature handling and built-in robustness.

---

## MLAgents Pipeline Evolution

1. **Initial Challenge:** Generated code with matplotlib hallucinations and duplicate parameters
2. **Prompt Optimization:** Removed visualization dependencies, added NO PLOTTING constraints
3. **Feature Engineering:** From basic 15 features to comprehensive 116 features
4. **Performance Plateau:** Diminishing returns from feature complexity

## Lessons Learned

- ✅ **MLAgents can generate production-ready ML code** when properly constrained
- ✅ **Domain knowledge helps** but can be over-applied
- ✅ **GPU acceleration** significantly reduces training time
- ❌ **Feature explosion** doesn't guarantee better performance
- ❌ **Validation improvements** may not translate to public leaderboard

## Requirements

See `requirements.txt` for dependencies.

## Usage

```bash
# Run basic model
python 001_fertilizer_prediction_gpu.py

# Run optimized LightGBM model  
python 002_fertilizer_prediction_optimized.py

# Run XGBoost GPU model
python 003_fertilizer_prediction_xgboost_gpu.py

# Run CatBoost GPU model
python 004_fertilizer_prediction_catboost_gpu.py
```

All models generate Kaggle-ready submissions and feature importance analysis.