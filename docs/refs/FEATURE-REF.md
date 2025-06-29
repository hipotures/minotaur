# Feature Generation Refactoring Plan

## Problem Description

System MCTS Minotaur generuje features z data leakage, powodując nieprawidłowe wyniki AutoGluon (perfekcyjne scores 1.0).

### Główne problemy:
1. **Data leakage**: Features budowane na kolumnie target (`Survived_mean_by_Cabin`)
2. **Brak filtrowania**: Generic operations używają target/ID/ignored columns 
3. **Zahardkodowane nazwy**: Możliwe wprost wpisane nazwy target w kodzie
4. **No-signal features**: Features bez sygnału (wszystkie wartości identyczne)
5. **Desynchronizacja**: Różne kolumny w train_features vs test_features

### Przykład data leakage:
```
Columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', ..., 
         'Survived_mean_by_Name', 'Survived_std_by_Name', 'Survived_mean_by_Cabin', ...]
```

`Survived_mean_by_Cabin` = średni survival rate dla danej kabiny = informacja z przyszłości!

## Root Cause Analysis

### Lokalizacja problemu:
**Plik**: `/home/xai/DEV/minotaur/src/feature_space.py` linie ~1053, 1164

```python
# BŁĘDNY KOD:
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # zawiera target!
stat_features = GenericFeatureOperations.get_statistical_aggregations(
    df, categorical_cols[:5], numeric_cols[:10]  # target w numeric_cols!
)
```

### Brak filtrowania:
- `numeric_cols` zawiera target column (`Survived`)
- `categorical_cols` może zawierać ID column 
- Brak sprawdzania `ignore_columns`
- Generic operations ślepo używają wszystkich kolumn

### Konsekwencje:
- AutoGluon otrzymuje features z targetu → perfekcyjne wyniki
- MCTS myśli że znalazł świetne features → nieprawidłowa optymalizacja
- Model nie jest predykcyjny → brak wartości biznesowej

## Implementation Plan

### PUNKT 0: Dokumentacja ✅
Plik: `/home/xai/DEV/minotaur/FEATURE-REF.md` (ten dokument)

### CZĘŚĆ 1: Filtrowanie w feature generation

#### 1A. Generic features
**Plik**: `/home/xai/DEV/minotaur/src/feature_space.py`
**Linie**: ~1053, 1164 (wywołania `get_statistical_aggregations`)

**Zmiana**:
```python
# PRZED:
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# PO:
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Filtruj target/ID/ignored
forbidden_cols = {target_column, id_column} | set(ignore_columns or [])
numeric_cols = [col for col in numeric_cols if col not in forbidden_cols]
categorical_cols = [col for col in categorical_cols if col not in forbidden_cols]

logger.info(f"Filtered out forbidden columns from generic features: {forbidden_cols}")
```

#### 1B. Custom features
**Lokalizacja**: `/home/xai/DEV/minotaur/src/domains/` (wszystkie pliki)

**Logika**:
- **Jeśli custom feature używa target/ID**: 
  - `logger.error(f"CRITICAL: Custom feature {name} uses target/ID - coding error!")`
  - `raise ValueError("Custom features cannot use target/ID columns")`
- **Jeśli custom feature używa ignored**: 
  - `logger.warning(f"Skipping custom feature {name} - uses ignored column")`
  - Pominąć (nie budować)

### CZĘŚĆ 2: AutoGluon konfiguracja

#### 2A. Dane dla AutoGluon
**Zasada**: AutoGluon otrzymuje **wszystkie** kolumny (włącznie z target/ID/ignored)
**Lokalizacja**: Bez zmian w `_prepare_autogluon_data()`

#### 2B. Konfiguracja AutoGluon
**Plik**: `/home/xai/DEV/minotaur/src/autogluon_evaluator.py`
**Metoda**: `_train_and_evaluate()` linia ~400

**Zmiana**:
```python
# PRZED:
predictor = TabularPredictor(
    label=self.target_column,
    path=model_dir,
    eval_metric=eval_metric,
    verbosity=eval_config.get('verbosity', 0)
)

# PO:
ignored_columns = []
if self.id_column:
    ignored_columns.append(self.id_column)
if self.ignore_columns:
    ignored_columns.extend(self.ignore_columns)

predictor = TabularPredictor(
    label=self.target_column,
    path=model_dir,
    eval_metric=eval_metric,
    learner_kwargs={'ignored_columns': ignored_columns} if ignored_columns else {},
    verbosity=eval_config.get('verbosity', 0)
)
```

### CZĘŚĆ 3: Logowanie konfiguracji

**Plik**: `/home/xai/DEV/minotaur/src/autogluon_evaluator.py`
**Lokalizacja**: Przed "Starting AutoGluon training and evaluation..."

**Dodać**:
```python
logger.info(f"AutoGluon configuration:")
logger.info(f"  Target column: {self.target_column}")
logger.info(f"  ID column: {self.id_column}")
logger.info(f"  Ignored columns: {self.ignore_columns}")
logger.info(f"  Total ignored for AutoGluon: {ignored_columns}")
```

### CZĘŚĆ 4: Sprawdzenie zahardkodowanych nazw

**Lokalizacje do sprawdzenia**:
- `/home/xai/DEV/minotaur/src/domains/` (wszystkie pliki)
- `/home/xai/DEV/minotaur/src/domains/generic.py`
- `/home/xai/DEV/minotaur/src/feature_space.py`

**Szukać pattern'ów**:
- `'Survived'`
- `'target'` 
- `df['Survived']`
- `== 'Survived'`
- Inne konkretne nazwy target

**Jeśli znajdę**:
```python
logger.error(f"CRITICAL: Hardcoded target column '{name}' found in {file}:{line}")
raise ValueError("Hardcoded target columns are not allowed - use dynamic target_column")
```

### CZĘŚĆ 5: Filtrowanie no-signal features

#### 5A. Metoda sprawdzenia sygnału
**Lokalizacja**: Nowa metoda w odpowiednim miejscu

```python
def has_signal(feature_series: pd.Series) -> bool:
    """
    Sprawdź czy feature ma sygnał (różne wartości).
    
    Returns:
        True jeśli feature ma sygnał (nunique > 1)
        False jeśli brak sygnału (wszystkie wartości identyczne)
    """
    # Usuń NaN i sprawdź liczbę unikalnych wartości
    unique_count = feature_series.dropna().nunique()
    return unique_count > 1
```

#### 5B. Filtrowanie przed zapisem
**Lokalizacja**: Podczas budowy features (przed zapisem do tabel)

```python
# Po zbudowaniu każdego feature:
if not has_signal(feature_data):
    logger.info(f"Skipping no-signal feature '{feature_name}' - all values identical")
    continue  # Nie zapisuj do tabeli

# Feature ma sygnał - zapisz normalnie
logger.debug(f"Feature '{feature_name}' has signal - saving to table")
```

#### 5C. Rejestracja kolumn w bazie
**Podczas rejestracji datasetu zapisać**:
- `source_columns`: Lista kolumn źródłowych (z CSV)
- `custom_features_with_signal`: Lista custom features z sygnałem  
- `generic_features_with_signal`: Lista generic features z sygnałem

#### 5D. Synchronizacja train/test
**Zasada**: `test_features.columns == train_features.columns - {target_column}`

**Implementacja**:
1. Po budowie train features → zapisać listę w bazie/cache
2. Dla test features → budować **TYLKO** te które miały sygnał w train
3. **Walidacja na końcu**:

```python
def validate_feature_columns(train_features: pd.DataFrame, test_features: pd.DataFrame, target_column: str):
    """Sprawdź czy kolumny train i test są zsynchronizowane."""
    
    train_cols = set(train_features.columns) - {target_column}
    test_cols = set(test_features.columns)
    
    if train_cols != test_cols:
        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        
        logger.error(f"Feature column mismatch!")
        logger.error(f"  Missing in test: {missing_in_test}")
        logger.error(f"  Extra in test: {extra_in_test}")
        logger.error(f"  Train columns (without target): {len(train_cols)}")
        logger.error(f"  Test columns: {len(test_cols)}")
        
        raise ValueError("Feature columns must be identical between train and test!")
    
    logger.info(f"✅ Feature columns synchronized: {len(train_cols)} columns in both train and test")
```

## Technical Implementation Details

### Pliki do modyfikacji:
1. `/home/xai/DEV/minotaur/src/feature_space.py` - filtrowanie generic
2. `/home/xai/DEV/minotaur/src/domains/` - sprawdzenie custom  
3. `/home/xai/DEV/minotaur/src/autogluon_evaluator.py` - konfiguracja AutoGluon
4. `/home/xai/DEV/minotaur/src/dataset_importer.py` - ewentualnie dla sync

### Nowe metody:
- `has_signal(feature_series)` - sprawdzenie sygnału
- `validate_feature_columns()` - walidacja sync
- `filter_forbidden_columns()` - helper do filtrowania

### Kolejność implementacji:
1. **CZĘŚĆ 4** - sprawdzenie zahardkodowanych (może zatrzymać wszystko)
2. **CZĘŚĆ 1** - filtrowanie generic/custom  
3. **CZĘŚĆ 2** - AutoGluon config
4. **CZĘŚĆ 3** - logowanie
5. **CZĘŚĆ 5** - no-signal filtering i sync

## Validation & Testing

### Jak sprawdzić że fix działa:

#### Test 1: Brak data leakage features
```bash
./manager.py datasets --register --dataset-name titanic --dataset-path datasets/Titanic --target-column Survived --force-update
```

**Oczekiwany rezultat**: Brak features typu `Survived_mean_by_*`

#### Test 2: AutoGluon wyniki
```bash
python mcts.py --config config/mcts_config_titanic_test.yaml
```

**Oczekiwany rezultat**: 
- Wyniki AutoGluon ~0.8-0.85 (nie 1.0)
- Logi pokazują ignored_columns
- Features bez target/ID

#### Test 3: Synchronizacja kolumn
**Sprawdzić**: `train_features.columns - {target} == test_features.columns`

### Metryki sukcesu:
- ❌ Brak features z target w nazwie (`Survived_*`)
- ❌ AutoGluon score < 0.95 (nie perfekcyjne)
- ❌ Logi pokazują filtrowanie forbidden columns  
- ❌ Walidacja synchronizacji przechodzi
- ❌ Brak ERROR logów o hardcoded target

## Rollback Plan

**Jeśli refactoring się nie powiedzie**:
1. Przywrócić backup plików
2. Zarejestrować dataset ponownie
3. Sprawdzić czy stary kod działa
4. Analiza błędów w logach
5. Iteracyjne naprawienia

---

**Utworzono**: 2025-06-29  
**Status**: W implementacji  
**Następny krok**: CZĘŚĆ 4 - sprawdzenie zahardkodowanych nazw target