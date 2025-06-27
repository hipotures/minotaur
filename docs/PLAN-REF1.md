# PLAN-REF1: Refaktoryzacja systemu MCTS z cache'owaniem

## 1. COMMIT obecnych zmian
```bash
git add .
git commit -m "Fix paths and colorized logging

🎯 Fixed duplicate logging from mock_evaluator
📊 Added colorized logs with emojis  
🔧 Fixed thorough_eval configuration
📁 Corrected relative paths in fast_real config

🤖 Generated with Claude Code"
```

## 2. SYSTEM CACHE'OWANIA na bazie MD5

### A. Nowy moduł `src/feature_cache.py`:
```python
import hashlib
import logging
from pathlib import Path
from typing import Union
import pandas as pd
import time

logger = logging.getLogger(__name__)

class FeatureCacheManager:
    def __init__(self, data_path: str):
        self.data_path = data_path.rstrip('/')
        self.path_hash = hashlib.md5(self.data_path.encode()).hexdigest()
        self.cache_dir = Path(f"data/{self.path_hash}")
        self.features_dir = self.cache_dir / "features"
        
    def get_dataset_cache_dir(self) -> Path:
        return self.cache_dir
        
    def ensure_base_datasets(self, train_path: str, test_path: str):
        """Sprawdz/wygeneruj podstawowe pliki parquet."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Train parquet
        train_parquet = self.cache_dir / "train.parquet"
        if not train_parquet.exists():
            logger.info(f"💾 Converting {train_path} to parquet...")
            df = pd.read_csv(train_path)
            df.to_parquet(train_parquet, index=False)
            logger.info(f"✅ Created train.parquet cache ({len(df)} rows)")
        else:
            logger.info(f"📂 Using cached train.parquet")
            
        # Test parquet  
        test_parquet = self.cache_dir / "test.parquet"
        if not test_parquet.exists():
            logger.info(f"💾 Converting {test_path} to parquet...")
            df = pd.read_csv(test_path)
            df.to_parquet(test_parquet, index=False)
            logger.info(f"✅ Created test.parquet cache ({len(df)} rows)")
        else:
            logger.info(f"📂 Using cached test.parquet")
            
    def get_feature_path(self, feature_name: str, data_type: str) -> Path:
        return self.features_dir / data_type / f"{feature_name}.parquet"
        
    def is_feature_cached(self, feature_name: str, data_type: str) -> bool:
        return self.get_feature_path(feature_name, data_type).exists()
        
    def load_feature(self, feature_name: str, data_type: str) -> pd.Series:
        path = self.get_feature_path(feature_name, data_type)
        logger.debug(f"📂 Loading cached feature: {feature_name} ({data_type})")
        return pd.read_parquet(path).iloc[:, 0]
        
    def save_feature(self, feature_name: str, data_type: str, feature_data: pd.Series):
        path = self.get_feature_path(feature_name, data_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({feature_name: feature_data})
        df.to_parquet(path, index=False)
        logger.info(f"💾 Cached new feature: {feature_name} ({data_type})")
```

### B. Modyfikacja `src/data_utils.py`:
- Zintegrować FeatureCacheManager
- Dodać metodę `prepare_training_data()` z obsługą train_size

### C. Modyfikacja `src/feature_space.py`:
- Dodać szczegółowe logowanie cache hit/miss
- Integracja z FeatureCacheManager

## 3. UOGÓLNIENIE systemu (usunięcie s5e6)

### A. `config/mcts_config.yaml`:
```yaml
autogluon:
  train_path: null
  test_path: null
  target_metric: null
```

### B. Zmiana nazw konfiguracji:
- `config/mcts_config_fast_real.yaml` → `config/mcts_config_s5e6_fast_real.yaml`
- `config/mcts_config_fast_test.yaml` → `config/mcts_config_s5e6_fast_test.yaml`
- Nowy: `config/mcts_config_s5e6_production.yaml`

### C. Wszystkie lokalne konfigi z:
```yaml
autogluon:
  train_path: "/mnt/ml/competitions/2025/playground-series-s5e6/train.csv"
  test_path: "/mnt/ml/competitions/2025/playground-series-s5e6/test.csv"
  target_metric: 'MAP@3'
```

### D. Usunąć hardcoded ścieżki z examples/

## 4. UPROSZCZENIE parametrów testowych

### A. Usunąć z wszystkich konfigów:
- Sekcję `testing:`
- `use_small_dataset`
- `small_dataset_size`

### B. Zachować tylko `--test-mode` dla mock evaluatora

### C. Dodać pole `is_test_mode` do tabeli sessions w bazie

## 5. LOGIKA train_size

### A. W `src/autogluon_evaluator.py`:
```python
def prepare_training_data(self, df: pd.DataFrame, train_size: Union[int, float]) -> pd.DataFrame:
    if isinstance(train_size, float) and 0 <= train_size <= 1:
        n_samples = int(len(df) * train_size)
        logger.info(f"📊 Using {train_size*100:.1f}% of data: {n_samples}/{len(df)} samples")
    else:
        n_samples = min(int(train_size), len(df))
        logger.info(f"📊 Using fixed sample size: {n_samples}/{len(df)} samples")
    
    if n_samples < len(df):
        sampled_df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        logger.info(f"✅ Sampled training data: {len(sampled_df)} rows")
        return sampled_df
    else:
        logger.info(f"📊 Using full dataset: {len(df)} rows")
        return df
```

## 6. DOMAIN-SPECIFIC features

### A. Struktura:
```
src/domains/
├── __init__.py
├── generic.py          # NPK ratios, statistical, polynomial
└── fertilizer_s5e6.py  # temp_humidity, moisture_stress, soil interactions
```

### B. Konfiguracja:
```yaml
feature_space:
  domain_module: 'domains.fertilizer_s5e6'
  enabled_categories:
    - 'npk_interactions'
    - 'fertilizer_domain'
```

## 7. CZYSZCZENIE synthetic_data.py

### A. Usunąć:
- Losowe kolumny (`synthetic_feature_*`, `polynomial_feature_*`)
- `augment_synthetic_features()`

### B. Zostaw tylko sensowne agricultural data w `generate_synthetic_features()`

## 8. README.md

### A. Dodać ostrzeżenie:
```markdown
## ⚠️ UWAGA: config/mcts_config.yaml
**NIE MODYFIKUJ** głównego pliku `config/mcts_config.yaml`!
Twórz własne konfiguracje dziedziczące z tego pliku.
```

## 9. TESTOWANIE

### A. Konfiguracje testowe:
- `mcts_config_s5e6_fast_test.yaml`: train_size=100, 3 iteracje
- `mcts_config_s5e6_fast_real.yaml`: train_size=0.05, XGB only
- `mcts_config_s5e6_production.yaml`: train_size=0.8, full features

### B. Test cache'owania:
```bash
python run_feature_discovery.py --config config/mcts_config_s5e6_fast_test.yaml --test-mode
```

## REZULTAT:
✅ Cache per-dataset na bazie MD5
✅ Features pre-computed na 100% danych  
✅ Szczegółowe logowanie cache operations
✅ train_size: liczba lub procent
✅ Domain-specific features w modułach
✅ Kod uniwersalny (bez s5e6 dependencies)
✅ Zachowany --test-mode dla mock evaluatora

---

**INSTRUKCJA**: Plan zapisany - gotowy do realizacji po wyczyszczeniu cache'u.