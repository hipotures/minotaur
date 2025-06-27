# ZERO.md - Historia migracji projektu Minotaur

**WAŻNE**: Przeczytaj ten plik po przełączeniu na nowy projekt minotaur!

## 🎯 Co to jest za projekt

**Minotaur** to niezależny system MCTS-driven feature discovery dla przewidywania nawozów rolniczych. Powstał przez migrację z projektu `aideml-alt/fertilizer_models/`.

## 📋 Historia migracji (27.06.2025)

### Skąd powstał
- **Źródło**: `/home/xai/DEV/aideml-alt/fertilizer_models/` 
- **Problem**: fertilizer_models był częścią większego systemu aideml-alt, ale powinien być niezależny
- **Rozwiązanie**: Wydzielenie do standalone projektu `~/DEV/minotaur`

### Co zostało zmigrowane
1. **Cała struktura katalogów** - skopiowana i zorganizowana logicznie
2. **Wszystkie skrypty ML** (001-009) → przeniesione do `examples/`
3. **System MCTS** (`mcts_feature_discovery/`) → bez zmian
4. **Konfiguracje** → zaktualizowane ścieżki danych
5. **Dokumentacja** → skopiowana i zaktualizowana

### 🔧 Kluczowe zmiany w migracji

#### Ścieżki danych
- **STARE**: `../competitions/playground-series-s5e6/train.csv`
- **NOWE**: `/mnt/ml/competitions/2025/playground-series-s5e6/train.csv` (READ-ONLY)
- **Cache lokalny**: `data/train_data.parquet` (auto-generowany, 4.2x szybszy)

#### Struktura katalogów
```
PRZED (aideml-alt/fertilizer_models/):
├── 001-009_*.py                    # Modele w głównym katalogu
├── feature_engineering.py          # W głównym katalogu  
├── run_feature_discovery.py        # W scripts/
└── mcts_config*.yaml              # W głównym katalogu

PO MIGRACJI (~/DEV/minotaur/):
├── run_feature_discovery.py        # GŁÓWNY SKRYPT w root
├── examples/001-009_*.py           # Przykłady ML w examples/
├── scripts/feature_engineering.py  # Narzędzia w scripts/
├── config/mcts_config*.yaml       # Konfigi w config/
└── CLAUDE.md, README.md           # Dokumentacja w root
```

#### Importy (POPRAWIONE)
- **Stare**: `from feature_engineering import`
- **Nowe**: `from scripts.feature_engineering import`
- **Poprawione w**: `examples/0*.py` + `mcts_feature_discovery/feature_space.py`

## 🚨 STAN PRZED TESTAMI - PROBLEMY DO ROZWIĄZANIA

### Environment
- **Aktualny venv**: `/home/xai/DEV/aideml-alt/.venv/` (226 pakietów)
- **Nowy projekt**: Będzie pusty venv, trzeba zainstalować wszystko od zera
- **requirements.txt**: Zaktualizowany z 12 → 25 pakietów (wszystkie niezbędne)

### Potencjalne problemy po przełączeniu:

#### 1. **Brakujące pakiety**
```bash
# Trzeba będzie zainstalować:
pip install -r requirements.txt
# Może nie działać od razu - sprawdzić każdy pakiet osobno
```

#### 2. **Ścieżki mogą nie działać**
- RO data source: `/mnt/ml/competitions/2025/playground-series-s5e6/` - sprawdzić czy istnieje
- Relative paths w konfigach - sprawdzić czy działają z nowego PWD

#### 3. **Importy mogą się sypać**
- `from scripts.feature_engineering import` - sprawdzić czy PYTHONPATH OK
- Moduły `mcts_feature_discovery` - sprawdzić czy importują się prawidłowo

#### 4. **Katalogi nie utworzone**
```bash
# Może trzeba będzie utworzyć:
mkdir -p data outputs logs models features
```

## 🔍 Jak przetestować po migracji

### Test 1: Environment
```bash
cd ~/DEV/minotaur
python -c "import numpy, pandas, sklearn, xgboost, lightgbm, catboost, autogluon; print('✅ Core ML packages OK')"
```

### Test 2: Konfiguracja
```bash
python -c "
from run_feature_discovery import load_config_with_overrides
config = load_config_with_overrides('config/mcts_config.yaml')
print(f'✅ Config OK: {config[\"autogluon\"][\"train_path\"]}')
"
```

### Test 3: Data loading
```bash
python -c "
from scripts.feature_engineering import _load_data_with_cache
df = _load_data_with_cache('/mnt/ml/competitions/2025/playground-series-s5e6/train.csv', 'train')
print(f'✅ Data loading OK: {df.shape}')
"
```

### Test 4: MCTS (ultra szybki)
```bash
python run_feature_discovery.py --config config/mcts_config_fast_test.yaml --test-mode
```

## 📊 Stan końcowy migracji

### Git commits w minotaur:
1. **12bd264** - Initial commit (cały system)
2. **fe11fcf** - Fix import paths after reorganization  
3. **e3d1d7e** - Update requirements.txt (25 pakietów)
4. **a527de2** - Add documentation (CLAUDE.md, PLAN.md)

### Pliki kluczowe:
- **run_feature_discovery.py** - główny skrypt MCTS
- **config/mcts_config*.yaml** - 3 tryby (test/fast/production)
- **scripts/feature_engineering.py** - współdzielone features z cache parquet
- **examples/005_fertilizer_prediction_xgboost_optuna.py** - najlepszy model (MAP@3: 0.33453)
- **CLAUDE.md** - instrukcje dla Claude (PRZECZYTAJ!)

### Requirements.txt zawiera:
- Core ML: lightgbm, xgboost, catboost, autogluon, torch, sklearn
- Data: numpy, pandas, scipy  
- Optymalizacja: optuna, hyperopt
- Wizualizacja: matplotlib, seaborn, plotly
- Utils: PyYAML, joblib, cloudpickle, requests
- Neural: rtdl_num_embeddings (TabM)

## 🎯 Następne kroki po przełączeniu

1. **Przeczytaj CLAUDE.md** - tam są wszystkie instrukcje
2. **Zainstaluj dependencies**: `pip install -r requirements.txt`
3. **Przetestuj environment** powyższymi testami
4. **Uruchom szybki test**: `python run_feature_discovery.py --test-mode`
5. **Jeśli coś nie działa** - sprawdź importy, ścieżki, brakujące katalogi

## 💡 Dlaczego ten plik

Ten plik istnieje, bo **po przełączeniu na nowy projekt stracisz całą historię rozmowy**. Będziesz miał tylko pliki projektu bez kontekstu jak powstały. ZERO.md daje ci kompletny obraz tego, co zostało zrobione i co może pójść nie tak.

**POWODZENIA W TESTACH!** 🚀

---
*Utworzone: 27.06.2025 podczas migracji z aideml-alt*
*Autor migracji: Claude Code*