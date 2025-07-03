# UX/UI Design Document - Final Version
## Kompleksowa specyfikacja interfejsu użytkownika z implementacyjnymi detalami

### Filozofia projektowania interfejsu

System został zaprojektowany z myślą o zaawansowanych użytkownikach Data Science, którzy cenią sobie efektywność i automatyzację. Interfejs CLI priorytetyzuje szybkość działania i skryptowalność, jednocześnie oferując pomocne wskazówki dla nowych użytkowników.

**Kluczowe zasady projektowe**:
- **Productivity First**: Minimalna liczba kroków do osiągnięcia celu
- **Fail Fast, Fail Clear**: Natychmiastowa walidacja z konkretnymi wskazówkami
- **Progressive Disclosure**: Zaawansowane opcje dostępne, ale nie przytłaczające
- **Automation Friendly**: Każda operacja możliwa do zautomatyzowania
- **Context Aware**: Inteligentne domyślne wartości bazujące na kontekście

### Architektura interfejsu użytkownika

#### Warstwy interfejsu

**1. Command Line Parser Layer**
```
Odpowiedzialności:
- Parsowanie argumentów z wykorzystaniem hierarchicznej struktury
- Walidacja typów i zakresów wartości
- Generowanie automatycznych help messages
- Obsługa aliasów i skrótów komend

Komponenty:
- Main argument parser z subcommands
- Validators dla custom types (paths, dates, ranges)
- Error formatter z sugestiami podobnych komend
- Tab completion generator
```

**2. Interactive Shell Layer**
```
Funkcjonalności:
- REPL mode dla interaktywnej eksploracji
- Kontekstowa historia komend
- Syntax highlighting dla wyników
- Auto-suggest based on command history
- Inline documentation viewer

Zachowania:
- Persystencja sesji między uruchomieniami
- Intelligent command abbreviation
- Multi-line input dla złożonych komend
- Undo/redo dla ostatnich operacji
```

**3. Output Formatting Layer**
```
Możliwości:
- Strukturalne formatowanie (tables, trees, lists)
- Kolorowanie według kontekstu i ważności
- Progress indicators z ETA
- Streaming output dla długich operacji
- Export do różnych formatów

Adaptacyjność:
- Detekcja szerokości terminala
- Fallback dla non-TTY environments
- Respektowanie NO_COLOR i FORCE_COLOR
- Różne poziomy szczegółowości
```

### Szczegółowa specyfikacja komend

#### Primary Orchestrator - MCTS Discovery

**Komenda uruchomienia nowej sesji**
```
Składnia pełna:
mcts run --config PATH [OPTIONS]

Składnia skrócona:
mcts r -c PATH

Parametry wymagane:
--config, -c PATH    Ścieżka do pliku konfiguracyjnego YAML

Parametry opcjonalne:
--name, -n TEXT      Nazwa sesji (auto: mcts_YYYYMMDD_HHMMSS)
--dataset, -d TEXT   Nazwa zarejestrowanego datasetu
--iterations INT     Liczba iteracji (nadpisuje config)
--time-limit HOURS   Limit czasu w godzinach
--sample-size INT    Rozmiar próbki dla ewaluacji
--random-seed INT    Seed dla reprodukowalności
--gpu / --no-gpu     Włącz/wyłącz akcelerację GPU
--memory-limit GB    Limit pamięci RAM
--checkpoint INT     Checkpoint co N iteracji
--export-every INT   Eksportuj wyniki co N iteracji
--test-mode         Mock evaluator dla testów
--validate-only     Tylko waliduj config bez uruchamiania
--profile           Włącz profiling wydajności
--debug             Szczegółowe logi debugowania
--quiet, -q         Minimalne wyjście
--verbose, -v       Szczegółowe wyjście
--yes, -y           Automatyczne "yes" na pytania

Zmienne środowiskowe:
MCTS_CONFIG_PATH     Domyślna ścieżka do konfiguracji
MCTS_GPU_ID          ID karty GPU do użycia
MCTS_LOG_LEVEL       Poziom logowania (DEBUG/INFO/WARNING/ERROR)
MCTS_CACHE_DIR       Katalog cache dla wyników
```

**Wznawianie sesji**
```
Składnia:
mcts resume [SESSION_ID] [OPTIONS]

Zachowania specjalne:
- Brak SESSION_ID: wznawia ostatnią sesję
- --last: explicite wznawia ostatnią
- --failed: wznawia ostatnią failed session

Parametry dodatkowe:
--additional, -a INT    Dodaj N iteracji do limitu
--extend-time HOURS     Przedłuż limit czasu
--from-iteration INT    Wznów od konkretnej iteracji
--reset-gpu            Reset GPU state przed wznowieniem
--force                Wznów mimo warnings
--repair               Próbuj naprawić uszkodzoną sesję

Weryfikacje przed wznowieniem:
- Sprawdzenie integralności checkpointu
- Walidacja zgodności wersji
- Dostępność zasobów (GPU, pamięć)
- Lock file aby uniknąć podwójnego uruchomienia
```

**Monitorowanie statusu**
```
Składnia:
mcts status [SESSION_ID] [OPTIONS]

Informacje wyświetlane:
- Progress: aktualna/całkowita liczba iteracji
- Performance: iteracje/godzinę, ETA
- Resources: CPU, RAM, GPU usage
- Best score: najlepszy wynik i feature set
- Recent discoveries: ostatnie 5 odkryć

Parametry:
--watch, -w         Odświeżaj co sekundę
--interval SEC      Custom interval odświeżania
--metrics           Pokaż szczegółowe metryki
--tree              Wizualizuj drzewo MCTS
--depth INT         Głębokość drzewa do pokazania
--format FORMAT     json|yaml|table|tree

Real-time monitoring mode:
- CPU/GPU utilization graphs
- Memory pressure indicators
- Iteration speed trending
- Score improvement chart
```

**Eksport wyników**
```
Składnia:
mcts export --session SESSION_ID [OPTIONS]

Formaty eksportu:
--format, -f TYPE    python|r|julia|sql|parquet|pickle
--output, -o PATH    Ścieżka wyjściowa

Selekcja features:
--top N              Eksportuj top N features
--min-score FLOAT    Minimalna ważność
--category CAT       Tylko features z kategorii
--stable-only        Tylko stabilne features

Opcje eksportu:
--include-metadata   Dodaj metadane o features
--include-code       Generuj kod transformacji
--include-tests      Generuj testy jednostkowe
--docstring          Dodaj dokumentację
--optimize           Optymalizuj generated code
--validate           Sprawdź poprawność eksportu

Przykład wygenerowanego kodu:
- Class-based transformer
- Fit/transform interface
- Serialization support
- Type hints i docstrings
```

#### System Manager - Administracja

**Dataset registration z auto-detection**
```
Składnia:
manager datasets register --name NAME --path PATH [OPTIONS]

Auto-detection wykrywa:
- Format pliku (CSV, Parquet, JSON, Excel)
- Encoding (UTF-8, Latin-1, etc.)
- Delimiter i quote character
- Header presence
- Column types
- Missing value patterns
- Target column (heurystyki)

Parametry szczegółowe:
--description TEXT      Opis datasetu
--target COL           Nazwa kolumny target
--index-col COL        Kolumna indeksu
--datetime-cols LIST   Kolumny datetime
--categorical LIST     Explicite categorical
--exclude-cols LIST    Kolumny do pominięcia
--test-split FLOAT     Procent na test set
--stratify BOOL        Stratified split
--compression TYPE     Kompresja (auto-detect)
--chunksize INT        Dla dużych plików
--encoding TEXT        Force encoding
--na-values LIST       Custom NA values
--parse-dates BOOL     Parse dates automatically

Walidacje podczas rejestracji:
- Rozmiar datasetu vs dostępna pamięć
- Duplikaty w danych
- Typy kolumn i zgodność
- Balans klas (dla klasyfikacji)
- Outliers i anomalie
- Data leakage indicators
```

**Feature catalog operations**
```
Składnia podstawowa:
manager features COMMAND [OPTIONS]

Komendy:
list        Wylistuj features z filtrami
search      Szukaj po pattern/regex
show        Pokaż szczegóły feature
compare     Porównaj features między sesjami
export      Eksportuj katalog features
import      Importuj features z innej sesji
validate    Sprawdź poprawność features
optimize    Sugestie optymalizacji

Przykład list z filtrami:
manager features list \
  --dataset sales_2024 \
  --session mcts_20240702_* \
  --category statistical \
  --min-importance 0.05 \
  --max-correlation 0.95 \
  --stable-across 3 \
  --sort-by importance \
  --format rich-table

Przykład search:
manager features search "price.*mean" \
  --regex \
  --in-name \
  --in-description \
  --highlight-matches

Feature comparison:
manager features compare SESSION1 SESSION2 \
  --metric correlation \
  --threshold 0.8 \
  --show-unique \
  --show-common \
  --export-diff diff.csv
```

**Analytics i reporting**
```
Składnia:
manager analytics COMMAND [OPTIONS]

Dostępne raporty:
summary         Podsumowanie dla sesji/datasetu
trends          Analiza trendów w czasie
efficiency      Metryki wydajności
convergence     Analiza zbieżności MCTS
importance      Feature importance analysis
correlation     Correlation matrix i clusters
leaderboard     Ranking sesji/modeli

Parametry uniwersalne:
--session TEXT      Filtr po sesji
--dataset TEXT      Filtr po datasecie  
--date-from DATE    Od daty
--date-to DATE      Do daty
--output PATH       Zapisz do pliku
--format FORMAT     html|pdf|json|excel
--template TMPL     Custom template

Przykład kompleksowego raportu:
manager analytics summary \
  --session mcts_20240702_143022 \
  --include-visualizations \
  --include-recommendations \
  --format html \
  --output report.html \
  --open-browser
```

### Interaktywne wzorce użytkowania

#### Intelligent prompting system
```
Przykład 1 - Walidacja z sugestiami:
> mcts run --config configs/test.yml
⚠️  Configuration warning:
    - 'max_iterations' not specified (using default: 100)
    - 'autogluon.time_limit' might be too low for quality results

Suggested configuration:
  max_iterations: 500
  autogluon:
    time_limit: 300
    presets: 'medium_quality'

Proceed with current config? [y/N/edit]:

Przykład 2 - Interaktywna naprawa:
> manager datasets register --name sales --path data.csv
❌ Validation issues found:

1. Missing values detected:
   - Column 'age': 234 missing (5.2%)
   - Column 'income': 56 missing (1.2%)

2. Suspicious values:
   - Column 'price': 3 negative values
   - Column 'date': 12 future dates

How to proceed?
  [1] Fix automatically (recommended)
  [2] Fix interactively 
  [3] Skip validation
  [4] Cancel operation

Your choice [1-4]: 2

Fixing 'age' column (234 missing):
  [1] Fill with median (32.5)
  [2] Fill with mean (33.2)
  [3] Forward fill
  [4] Drop rows
  [5] Keep as-is

Your choice [1-5]:
```

#### Rich progress indication
```
MCTS Feature Discovery Session: quantum_sales_analysis
═══════════════════════════════════════════════════════════════

Configuration: config/production.yaml
Dataset: sales_2024_q1 (1.2M rows × 47 columns)
Target: revenue (regression, MAE metric)

Progress Overview
────────────────────────────────────────────────────────────────
Iteration    [████████████░░░░░░░░░░░░░] 126/500 (25.2%)
Time Elapsed [████████████████░░░░░░░░░]  2h 34m / ~4h 12m
Best Score   0.8934 → 0.9156 → 0.9234 (↑ improving)

Current Status
────────────────────────────────────────────────────────────────
├─ 🔍 Exploring: price_volatility_by_category_season
├─ 📊 Features evaluated: 1,247
├─ 🌳 Tree nodes: 423 (depth: 7)
├─ 💾 Cache hit rate: 78.3%
├─ 🔥 GPU utilization: 84%
└─ 💻 Memory: 12.4 GB / 32.0 GB

Latest Discoveries
────────────────────────────────────────────────────────────────
[12:34:22] ✓ customer_ltv_percentile_rank       (+0.0156)
[12:31:18] ✓ seasonal_price_elasticity          (+0.0089)
[12:28:45] ✓ cross_category_purchase_frequency  (+0.0234)

Performance Metrics
────────────────────────────────────────────────────────────────
Iterations/hour: 48.2 (↑ 12% vs last hour)
Avg evaluation time: 4.2s (target: <5s) ✓
Queue efficiency: 92% (excellent)
```

#### Error handling with recovery
```
Przykład 1 - Graceful interruption:
^C
🛑 Interrupt received - Gracefully shutting down...

✓ Current iteration completed
✓ Results saved to checkpoint
✓ Database synchronized
✓ GPU memory released
✓ Temporary files cleaned

Session: mcts_20240702_152419
Status: PAUSED at iteration 234/500
Resume: mcts resume mcts_20240702_152419

💡 Tip: Use --checkpoint more frequently for long runs

Przykład 2 - Error z recovery options:
❌ ERROR: AutoGluon training failed

Details:
  Error type: OutOfMemoryError
  Component: LightGBM model
  Memory requested: 18.2 GB
  Memory available: 16.0 GB

Recovery options:
  1. Retry with reduced sample size (recommended)
     └─ Current: 50000 → Suggested: 30000
  
  2. Disable memory-intensive models
     └─ Skip: NeuralNetwork, CatBoost
  
  3. Increase memory limit
     └─ docker run --memory=32g
  
  4. Save progress and exit
     └─ Resume later with more resources

Choose option [1-4] or 'details' for more info:
```

### Wizualne elementy interfejsu

#### Adaptive table formatting
```
Przykład prostej tabeli:
┌─────────────────┬──────────┬────────────┬───────────┐
│ Dataset         │ Status   │ Rows       │ Features  │
├─────────────────┼──────────┼────────────┼───────────┤
│ sales_2024      │ ✅ Ready │ 1,234,567  │ 47        │
│ customers_full  │ ⚠️ Large │ 8,901,234  │ 123       │
│ products_catalog│ ❌ Error │ -          │ -         │
└─────────────────┴──────────┴────────────┴───────────┘

Przykład rozbudowanej tabeli z metadanymi:
╔════════════════════════════════════════════════════════════════╗
║                    Feature Importance Analysis                  ║
╠════════════════════╤══════════╤═════════╤══════════╤══════════╣
║ Feature            │ Import.  │ Stable  │ Correlation │ Type  ║
╟────────────────────┼──────────┼─────────┼──────────┼──────────╢
║ price_elasticity   │ 0.2341 ▰▰▰▰▰▰▰▰▱▱ │ 98.2%   │ Low      │ Num   ║
║ customer_segment   │ 0.1892 ▰▰▰▰▰▰▰▱▱▱ │ 95.1%   │ Medium   │ Cat   ║
║ seasonal_trend     │ 0.1234 ▰▰▰▰▰▱▱▱▱▱ │ 88.7%   │ Low      │ Num   ║
║ promotion_impact   │ 0.0934 ▰▰▰▰▱▱▱▱▱▱ │ 76.3%   │ High     │ Bool  ║
╚════════════════════╧══════════╧═════════╧══════════╧══════════╝
                     Summary: 4 features (stability: 89.6%)
```

#### Tree visualization z kolorami
```
MCTS Exploration Tree (Best Path Highlighted)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌳 Root (baseline: 0.7234)
│
├─● price_features (0.8934) ████████░░ 89%
│  ├─● price_mean_category (0.8656) ███████░░░ 87%
│  │  ├─○ + region (0.8234)
│  │  └─● + seasonality (0.8756) ████████░░ 88% 
│  │     └─✨ + elasticity (0.8934) █████████░ 89% [BEST PATH]
│  │
│  └─◐ price_volatility (0.8123) ████████░░ 81%
│     └─○ + time_windows (not evaluated)
│
├─◐ customer_features (0.8234) ████████░░ 82%
│  ├─● lifetime_value (0.8123) ████████░░ 81%
│  └─○ segmentation (pending)
│
└─○ product_features (0.7892) ███████░░░ 79%
   └─○ category_encoding (not evaluated)

Legend:
● Fully explored   ◐ Partially explored   ○ Not explored
✨ Current best    ░ Performance bar       [%] Score

Node Statistics:
- Total nodes: 127
- Explored: 89 (70.1%)
- Average branching: 3.2
- Max depth reached: 7
```

#### Contextual color coding
```
Schemat kolorów według kontekstu:

SUCCESS (Green #2ECC71):
- ✅ Successful operations
- ↑ Improvements
- "READY", "COMPLETE" states

WARNING (Yellow #F39C12):
- ⚠️ Warnings
- "PENDING", "RUNNING" states
- Degraded performance

ERROR (Red #E74C3C):
- ❌ Errors and failures
- ↓ Degradations
- "FAILED", "ERROR" states

INFO (Blue #3498DB):
- ℹ️ Information
- Hints and suggestions
- Help text

HIGHLIGHT (Magenta #9B59B6):
- Important values
- Best scores
- User selections

MUTED (Gray #95A5A6):
- Secondary information
- Timestamps
- Debug output
```

### Zaawansowane funkcjonalności CLI

#### Tab completion z kontekstem
```
Przykłady inteligentnego dopełniania:

1. Dopełnianie ścieżek:
> mcts run --config configs/<TAB>
configs/quick_test.yaml
configs/production.yaml
configs/experimental/

2. Dopełnianie dataset names:
> manager datasets show <TAB>
sales_2024_q1
sales_2024_q2
customers_full
products_catalog

3. Dopełnianie na podstawie historii:
> mcts resume mcts_<TAB>
mcts_20240702_152419  (PAUSED, 2 hours ago)
mcts_20240702_091232  (COMPLETED, 8 hours ago)
mcts_20240701_184521  (FAILED, yesterday)

4. Dopełnianie parametrów:
> mcts run --config test.yaml --<TAB>
--name            --iterations      --gpu
--dataset         --time-limit      --no-gpu
--sample-size     --memory-limit    --test-mode
```

#### Command aliasing i shortcuts
```
Wbudowane aliasy:
mcts r    → mcts run
mcts s    → mcts status
mcts e    → mcts export
mgr       → manager
m         → manager

Custom aliasy (w ~/.mcts/aliases):
alias quick="mcts run --config configs/quick_test.yaml --test-mode"
alias prod="mcts run --config configs/production.yaml --gpu"
alias last="mcts resume --last"
alias top10="manager features list --top 10 --format table"

Keyboard shortcuts w interactive mode:
Ctrl+R    - Reverse search historii
Ctrl+P/N  - Previous/Next w historii
Ctrl+A/E  - Beginning/End of line
Ctrl+K    - Kill do końca linii
Ctrl+L    - Clear screen
Ctrl+D    - Exit (z potwierdzeniem)
```

#### Multi-format output support
```
Uniwersalne formatowanie:
--format table   (default, human-readable)
--format json    (dla programmatic access)
--format yaml    (human-readable structured)
--format csv     (dla Excel/pandas)
--format html    (rich formatting, charts)
--format latex   (dla dokumentów)

Przykłady różnych formatów:

TABLE (default):
┌────────────┬────────┬───────┐
│ Feature    │ Score  │ Rank  │
├────────────┼────────┼───────┤
│ price_mean │ 0.234  │ 1     │
└────────────┴────────┴───────┘

JSON:
{
  "features": [
    {
      "name": "price_mean",
      "score": 0.234,
      "rank": 1
    }
  ]
}

YAML:
features:
  - name: price_mean
    score: 0.234
    rank: 1

CSV:
Feature,Score,Rank
price_mean,0.234,1
```

### Workflow automation patterns

#### Batch processing script
```bash
#!/bin/bash
# Przykład automated pipeline

# Konfiguracja
DATASETS=("sales_q1" "sales_q2" "sales_q3" "sales_q4")
CONFIG="configs/quarterly_analysis.yaml"
OUTPUT_DIR="results/$(date +%Y%m%d)"

# Setup
mkdir -p "$OUTPUT_DIR"
echo "Starting batch analysis: $(date)" > "$OUTPUT_DIR/log.txt"

# Przetwarzanie
for dataset in "${DATASETS[@]}"; do
    echo "Processing $dataset..."
    
    # Run MCTS
    SESSION=$(mcts run \
        --config "$CONFIG" \
        --dataset "$dataset" \
        --name "batch_${dataset}" \
        --quiet \
        --format json | jq -r '.session_id')
    
    # Wait for completion with timeout
    timeout 6h bash -c "
        while [[ \$(mcts status $SESSION --format json | jq -r '.status') == 'RUNNING' ]]; do
            sleep 60
        done
    "
    
    # Check status
    STATUS=$(mcts status $SESSION --format json | jq -r '.status')
    
    if [[ "$STATUS" == "COMPLETED" ]]; then
        # Export results
        mcts export \
            --session "$SESSION" \
            --format python \
            --output "$OUTPUT_DIR/${dataset}_features.py"
        
        # Generate report
        manager analytics summary \
            --session "$SESSION" \
            --format html \
            --output "$OUTPUT_DIR/${dataset}_report.html"
    else
        echo "ERROR: $dataset failed with status $STATUS" >> "$OUTPUT_DIR/errors.log"
    fi
done

# Comparative analysis
manager analytics compare ${SESSIONS[@]} \
    --output "$OUTPUT_DIR/comparison.html" \
    --include-recommendations
```

#### CI/CD Integration
```yaml
# .github/workflows/feature_discovery.yml
name: Automated Feature Discovery

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday 2 AM
  workflow_dispatch:
    inputs:
      dataset:
        description: 'Dataset to analyze'
        required: true
        default: 'production'

jobs:
  discover:
    runs-on: self-hosted
    timeout-minutes: 360
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup environment
      run: |
        uv venv
        uv pip install -r requirements.txt
    
    - name: Run MCTS discovery
      env:
        MCTS_GPU_ID: 0
        MCTS_LOG_LEVEL: INFO
      run: |
        SESSION=$(mcts run \
          --config configs/ci_discovery.yaml \
          --dataset ${{ github.event.inputs.dataset }} \
          --name "ci_run_${{ github.run_number }}" \
          --iterations 1000 \
          --format json | jq -r '.session_id')
        
        echo "SESSION_ID=$SESSION" >> $GITHUB_ENV
    
    - name: Generate artifacts
      run: |
        # Export features
        mcts export \
          --session ${{ env.SESSION_ID }} \
          --format python \
          --output artifacts/features.py
        
        # Generate report
        manager analytics summary \
          --session ${{ env.SESSION_ID }} \
          --format html \
          --output artifacts/report.html
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: discovery-results-${{ github.run_number }}
        path: artifacts/
        retention-days: 30
    
    - name: Update feature catalog
      if: success()
      run: |
        # Commit best features to repo
        cp artifacts/features.py src/features/discovered/
        git add src/features/discovered/
        git commit -m "feat: Add discovered features from run ${{ github.run_number }}"
        git push
```

### Accessibility i internationalization

#### Obsługa różnych środowisk
```
Terminal detection i fallbacks:

1. Brak kolorów (NO_COLOR=1):
   - Używa ASCII markers: [OK], [WARN], [ERROR]
   - Tabele z podstawowymi ramkami
   - Progress jako procenty

2. Ograniczona szerokość (<80 cols):
   - Automatyczne zawijanie tekstu
   - Kompaktowe tabele
   - Skrócone nazwy kolumn

3. Non-interactive (pipe/redirect):
   - Brak progress bars
   - Brak interaktywnych promptów
   - Machine-readable output

4. Screen readers:
   - Semantyczne markers
   - Opisowe teksty zamiast emoji
   - Strukturalne formatowanie
```

#### Konfiguracja użytkownika
```
~/.mcts/config.yaml:

# UI Preferences
ui:
  theme: dark                    # dark/light/auto
  color_scheme: accessible       # default/accessible/none
  progress_style: detailed       # simple/detailed/none
  table_style: rounded          # ascii/rounded/double
  
# Interaction
interaction:
  confirm_destructive: true      # Ask before dangerous operations
  auto_complete: true           # Enable tab completion
  history_size: 1000            # Command history entries
  
# Output
output:
  default_format: table         # table/json/yaml
  timestamp_format: iso         # iso/unix/relative
  number_format: comma          # comma/space/none
  
# Performance  
performance:
  update_interval: 1.0          # Progress update seconds
  animation: true               # Enable spinners/animations
  
# Localization
locale:
  language: en                  # Language code
  timezone: UTC                 # For timestamps
  decimal_separator: "."        # For numbers
```

### Zaawansowane debugging i troubleshooting

#### Debug mode output
```
Włączenie debug mode:
MCTS_DEBUG=1 mcts run --config test.yaml --debug

Przykład debug output:
[2024-07-02 15:24:31.234] DEBUG [main] Starting MCTS with PID 12345
[2024-07-02 15:24:31.245] DEBUG [config] Loading config from: test.yaml
[2024-07-02 15:24:31.256] DEBUG [config] Merged config:
  mcts:
    exploration_weight: 1.414
    max_tree_depth: 10
    expansion_threshold: 5
[2024-07-02 15:24:31.267] DEBUG [dataset] Loading dataset: sales_2024
[2024-07-02 15:24:31.278] DEBUG [dataset] Cache hit for dataset
[2024-07-02 15:24:31.289] DEBUG [dataset] Shape: (50000, 47), Memory: 125.3 MB
[2024-07-02 15:24:31.301] DEBUG [feature_space] Initializing with 23 operations
[2024-07-02 15:24:31.312] DEBUG [mcts] Tree initialized with root node
[2024-07-02 15:24:31.323] INFO  [mcts] Starting iteration 1/100
[2024-07-02 15:24:31.334] DEBUG [mcts] Selection phase...
[2024-07-02 15:24:31.345] DEBUG [mcts] Selected node: root (visits=0)
[2024-07-02 15:24:31.356] DEBUG [mcts] Expansion phase...
[2024-07-02 15:24:31.367] DEBUG [mcts] Created 5 child nodes
[2024-07-02 15:24:31.378] DEBUG [mcts] Evaluation phase...
[2024-07-02 15:24:31.389] DEBUG [autogluon] Training with config:
  time_limit: 60
  presets: medium_quality_faster_train
  eval_metric: mean_absolute_error
```

#### Performance profiling output
```
Włączenie profiling:
mcts run --config test.yaml --profile

Przykład profiling summary:
Performance Profile Summary
═══════════════════════════════════════════════════════════════

Execution Time Breakdown:
────────────────────────────────────────────────────────────────
Component               Time (s)    Percentage    Calls
────────────────────────────────────────────────────────────────
AutoGluon evaluation    234.56      45.2%         127
Feature generation      156.23      30.1%         423  
Tree operations         45.67       8.8%          5,234
Database I/O            34.12       6.6%          1,892
Data loading            23.45       4.5%          127
Config parsing          12.34       2.4%          1
Other                   12.63       2.4%          -
────────────────────────────────────────────────────────────────
Total                   519.00      100.0%

Memory Usage:
────────────────────────────────────────────────────────────────
Peak memory:            8,234 MB
Average memory:         5,123 MB
Cache size:             1,234 MB
Feature storage:        2,345 MB

Bottlenecks Identified:
────────────────────────────────────────────────────────────────
1. AutoGluon model training (could parallelize)
2. Feature correlation computation (could cache)
3. Database writes not batched (batch size: 100)

Recommendations:
────────────────────────────────────────────────────────────────
- Enable GPU acceleration for 2-3x speedup
- Increase cache size to reduce recomputation
- Use --parallel-features flag for generation
```

### Command cookbook - przykłady zaawansowane

#### Przykład 1: Porównanie wydajności różnych konfiguracji
```bash
# Uruchom testy A/B dla różnych configs
for config in configs/experiment_*.yaml; do
    name=$(basename "$config" .yaml)
    mcts run --config "$config" \
             --dataset production \
             --name "ablation_$name" \
             --iterations 100 \
             --quiet &
done

# Poczekaj na zakończenie
wait

# Porównaj wyniki
manager analytics compare ablation_* \
    --metrics "best_score,time_to_90_percent,total_features" \
    --format latex \
    --output ablation_study.tex
```

#### Przykład 2: Incremental feature discovery
```bash
# Start z baseline
SESSION=$(mcts run --config minimal.yaml --iterations 50 --format json | jq -r '.session_id')

# Iteracyjnie dodawaj complexity
for i in {1..5}; do
    # Eksportuj obecne best features
    mcts export --session $SESSION --top 10 --output "features_v$i.py"
    
    # Wznów z większą głębokością
    SESSION=$(mcts resume $SESSION \
        --additional 50 \
        --config-override "mcts.max_tree_depth=$((5 + i * 2))" \
        --format json | jq -r '.session_id')
    
    # Sprawdź improvement
    improvement=$(mcts status $SESSION --format json | \
        jq '.score_improvement_last_50_iterations')
    
    # Jeśli brak poprawy, zakończ
    if (( $(echo "$improvement < 0.001" | bc -l) )); then
        echo "Converged at iteration $i"
        break
    fi
done
```

#### Przykład 3: Distributed exploration
```bash
# Master node
mcts run --config distributed.yaml \
         --mode master \
         --workers 4 \
         --checkpoint-server redis://localhost:6379 &

# Worker nodes (na różnych maszynach)
for i in {1..4}; do
    ssh worker$i "mcts run --config distributed.yaml \
                          --mode worker \
                          --master master.local:8765 \
                          --gpu-id $i"
done
```

### Future UI considerations

#### Phase 2 - Web Dashboard
```
Planowane komponenty:
1. Real-time visualization
   - D3.js tree visualization
   - Plotly score progression
   - Feature correlation heatmap
   
2. Collaborative features
   - Session sharing via URL
   - Comments on discoveries
   - Team workspaces
   
3. Advanced analytics
   - What-if analysis
   - Feature playground
   - A/B test framework
```

#### Phase 3 - IDE Plugins
```
Planned integrations:
1. VS Code extension
   - IntelliSense for configs
   - Live session monitoring
   - Inline feature preview
   
2. Jupyter widgets
   - Interactive config builder
   - Progress monitoring cell
   - Results visualization
   
3. RStudio addin
   - R bindings
   - Integrated plots
   - Shiny dashboard
```

### Metryki UX i monitoring

#### Zbierane metryki użytkowania
```
Telemetry (opt-in):
- Command frequency
- Error patterns
- Performance bottlenecks
- Feature adoption
- Session success rates

Przykład metryki:
{
  "event": "command_executed",
  "command": "mcts run",
  "duration_seconds": 234.5,
  "success": true,
  "parameters": {
    "iterations": 100,
    "gpu_enabled": true,
    "dataset_size": 50000
  },
  "environment": {
    "os": "linux",
    "python": "3.11.8",
    "terminal": "xterm-256color"
  }
}
```

### Podsumowanie filozofii UX

System został zaprojektowany aby:
1. **Minimalizować friction** - każda operacja możliwa w 1-2 komendach
2. **Maksymalizować feedback** - użytkownik zawsze wie co się dzieje
3. **Wspierać automatyzację** - wszystko skryptowalne i batchowalne
4. **Uczyć podczas używania** - pomocne sugestie i przykłady
5. **Adaptować się do użytkownika** - od nowicjusza do eksperta

Interfejs ewoluuje wraz z potrzebami użytkownika, oferując prostotę dla podstawowych zadań i pełną kontrolę dla zaawansowanych scenariuszy.