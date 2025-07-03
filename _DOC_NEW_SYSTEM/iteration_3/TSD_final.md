# Technical Specification Document - Final Version
## Architektura systemu automatycznego odkrywania cech

### Architektura wysokopoziomowa

System wykorzystuje algorytm Monte Carlo Tree Search (MCTS) do inteligentnej eksploracji przestrzeni możliwych transformacji cech. MCTS, znany z zastosowań w grach strategicznych jak AlphaGo, został zaadaptowany do problemu feature engineering poprzez traktowanie każdej transformacji jako "ruchu" w grze, gdzie nagrodą jest poprawa jakości modelu ML.

Kluczowe innowacje architektury:
- **Lazy evaluation**: Cechy generowane tylko gdy potrzebne
- **Hierarchical caching**: Multi-poziomowy cache dla wydajności  
- **Adaptive exploration**: Dynamiczne dostosowanie parametrów
- **Pluggable operations**: Łatwe dodawanie nowych transformacji
- **Incremental learning**: Wykorzystanie wcześniejszych rezultatów

### Główne komponenty systemu

#### Silnik przeszukiwania przestrzeni cech (MCTS Engine)

##### Struktura danych węzła
```
Węzeł MCTS zawiera:
- Identyfikator: Unikalne ID węzła (UUID)
- Stan: Zbiór operacji prowadzących do węzła
- Statystyki:
  - Liczba wizyt (n)
  - Suma nagród (total_reward)
  - Średnia nagroda (Q = total_reward / n)
  - Najlepsza nagroda (max_reward)
- Relacje:
  - Rodzic: Referencja do węzła nadrzędnego
  - Dzieci: Lista węzłów potomnych
- Metadane:
  - Timestamp utworzenia
  - Głębokość w drzewie
  - Wygenerowane cechy (lazy loaded)
```

##### Algorytm selekcji (UCB1)
```
Formuła UCB1 dla węzła:
UCB(node) = Q(node) + C * sqrt(ln(N) / n(node))

Gdzie:
- Q(node): Średnia nagroda węzła
- C: Parametr eksploracji (domyślnie 1.4)
- N: Całkowita liczba wizyt rodzica
- n(node): Liczba wizyt węzła

Modyfikacje dla feature engineering:
- Progressive widening: Ograniczenie dzieci na start
- Domain-specific priors: Boost dla obiecujących operacji
- Decay factor: Zmniejszanie C w czasie
```

##### Proces ekspansji
```
Ekspansja węzła:
1. Sprawdzenie kryteriów ekspansji:
   - Min wizyt rodzica (threshold: 3)
   - Max dzieci nie osiągnięte
   - Dostępne operacje do zastosowania
   
2. Selekcja operacji:
   - Filtrowanie applicable operations
   - Priorytetyzacja wg category weights
   - Losowanie z rozkładu ważonego
   
3. Tworzenie dziecka:
   - Generowanie nowego stanu
   - Inicjalizacja statystyk
   - Dodanie do drzewa
   
4. Virtual loss:
   - Tymczasowa kara dla parallel search
   - Zapobiega duplikatom
```

##### Symulacja (Ewaluacja)
```
Ewaluacja węzła:
1. Przygotowanie danych:
   - Pobranie cech z cache lub generacja
   - Walidacja kompletności
   - Handling missing values
   
2. Podział danych:
   - Stratified split (jeśli klasyfikacja)
   - Time-based split (jeśli szereg czasowy)
   - Standard random split (pozostałe)
   
3. Trenowanie modelu:
   - AutoGluon z time budget
   - Early stopping na plateau
   - Resource monitoring
   
4. Obliczenie nagrody:
   - Metryka zgodna z problemem
   - Normalizacja do [0,1]
   - Penalty za złożoność
```

##### Propagacja wsteczna
```
Backpropagation:
1. Update bieżącego węzła:
   - Inkrementacja liczby wizyt
   - Dodanie nagrody do sumy
   - Update max_reward jeśli lepsza
   
2. Propagacja do rodzica:
   - Rekurencyjne wywołanie
   - Aż do korzenia drzewa
   
3. Update global best:
   - Jeśli nowy najlepszy wynik
   - Trigger export hooki
   
4. Statistics update:
   - Running averages
   - Confidence intervals
   - Convergence metrics
```

##### Zarządzanie pamięcią
```
Memory management:
1. Node pruning:
   - Usuwanie słabych gałęzi (score < threshold * best)
   - LRU eviction dla rzadko odwiedzanych
   - Zachowanie N najlepszych ścieżek
   
2. Feature cache:
   - Size-based eviction
   - TTL dla stale features
   - Compression dla dużych features
   
3. Garbage collection:
   - Forced GC co N iteracji
   - Memory pressure triggers
   - Incremental cleanup
```

#### Moduł ewaluacji ML (AutoML Evaluator)

##### Konfiguracja modeli
```
Model ensemble configuration:
1. LightGBM (primary):
   - num_leaves: auto (31-255)
   - learning_rate: auto (0.01-0.3)
   - feature_fraction: 0.9
   - bagging_fraction: 0.8
   - lambda_l1: auto
   - lambda_l2: auto
   
2. XGBoost (GPU gdy dostępne):
   - max_depth: auto (3-10)
   - eta: auto (0.01-0.3)
   - subsample: 0.8
   - colsample_bytree: 0.8
   - gpu_id: 0 (jeśli włączone)
   
3. CatBoost (dla categorical heavy):
   - depth: auto (4-10)
   - learning_rate: auto
   - l2_leaf_reg: auto
   - cat_features: auto-detected
   
4. TabNet (optional, deep learning):
   - n_d: 8
   - n_a: 8
   - n_steps: 3
   - gamma: 1.3
   - momentum: 0.02
```

##### Strategie ewaluacji
```
Evaluation strategies:
1. Fast mode (development):
   - Single holdout (80/20)
   - Reduced iterations
   - Smaller ensemble
   - Time limit: 30s
   
2. Standard mode (default):
   - 5-fold CV
   - Full ensemble
   - Hyperparameter tuning
   - Time limit: 60s
   
3. Thorough mode (final):
   - 10-fold CV
   - Extended tuning
   - Stacking ensemble
   - Time limit: 300s
   
4. Custom mode:
   - User-defined splits
   - Custom metrics
   - Specific models
   - No time limit
```

##### Cache mechanizm
```
Result caching:
1. Cache key generation:
   - Hash of feature names (sorted)
   - Hash of model config
   - Hash of data sample
   - Combined 256-bit key
   
2. Storage format:
   - Scores: dict[metric_name, float]
   - Metadata: timing, memory, iterations
   - Model artifacts (optional)
   - Compressed with zstd
   
3. Invalidation:
   - Config change
   - Data drift detection
   - Manual clear
   - TTL expiration (24h default)
```

##### Handling różnych problemów
```
Problem-specific handling:
1. Binary classification:
   - Stratified sampling
   - Class weight balancing
   - Metrics: AUC, F1, Precision, Recall
   - Threshold optimization
   
2. Multi-class classification:
   - Stratified sampling per class
   - One-vs-rest option
   - Metrics: Accuracy, Macro-F1, MAP@K
   - Confusion matrix analysis
   
3. Regression:
   - Random sampling
   - Outlier detection
   - Metrics: RMSE, MAE, R², MAPE
   - Residual analysis
   
4. Ranking:
   - Query-based splitting
   - Position bias correction
   - Metrics: NDCG, MAP, MRR
   - Pairwise/listwise options
```

#### System zarządzania cechami (Feature Space Manager)

##### Architektura operacji
```
Operation hierarchy:
BaseOperation (abstract)
├── GenericOperation (abstract)
│   ├── StatisticalOperation
│   ├── PolynomialOperation
│   ├── BinningOperation
│   ├── RankingOperation
│   ├── TemporalOperation
│   ├── TextOperation
│   └── CategoricalOperation
└── CustomOperation (abstract)
    ├── DomainOperation
    └── UserDefinedOperation

Każda operacja implementuje:
- applicable(data) -> bool
- generate(data, columns) -> features
- get_metadata() -> dict
- estimate_memory() -> int
- validate_output(features) -> bool
```

##### Katalog operacji
```
Przykładowe operacje w katalogu:

Statistical Operations:
- BasicStats: mean, median, std, var, min, max
- AdvancedStats: skew, kurtosis, entropy, gini
- RobustStats: mad, trimmed_mean, winsorized
- Percentiles: q10, q25, q50, q75, q90, iqr
- Correlations: pearson, spearman, kendall

Polynomial Operations:
- Powers: square, cube, sqrt, inverse
- Interactions: multiply_pairs, divide_pairs
- Compositions: log1p, exp, logit

Binning Operations:
- EqualWidth: fixed interval bins
- EqualFreq: same count per bin
- Quantile: percentile-based
- KMeans: cluster-based
- Tree: decision tree splits

Ranking Operations:
- DenseRank: no gaps in ranks
- OrdinalRank: standard ranking
- PercentRank: percentile ranking
- NormalizedRank: [0,1] scaling

Temporal Operations:
- Lags: shift by N periods
- Rolling: window statistics
- Expanding: cumulative stats
- Seasonal: decomposition
- TimeSince: event-based

Text Operations:
- Length: char/word counts
- Statistics: avg word length
- Vectorization: TF-IDF, BoW
- Embeddings: pre-trained models

Categorical Operations:
- OneHot: dummy variables
- Target: mean encoding
- Frequency: count encoding
- Ordinal: ordered mapping
- Hashing: hash trick
```

##### Signal detection
```
Signal detection pipeline:
1. Variance check:
   - Zero variance -> reject
   - Near-zero variance -> warning
   
2. Uniqueness check:
   - All unique -> potential leak
   - <2 unique -> no signal
   
3. Correlation check:
   - Perfect correlation with existing -> reject
   - High correlation (>0.95) -> warning
   
4. Information gain:
   - Mutual information with target
   - Threshold for acceptance
   
5. Stability check:
   - Cross-validation variance
   - Temporal stability (if applicable)
```

##### Feature generation pipeline
```
Generation workflow:
1. Column selection:
   - Type compatibility check
   - Cardinality limits
   - Missing value threshold
   
2. Parameter optimization:
   - Grid search for discrete params
   - Bayesian opt for continuous
   - Early stopping on plateau
   
3. Computation:
   - Vectorized operations (NumPy)
   - Parallel when possible
   - Memory monitoring
   
4. Validation:
   - Output shape check
   - Data type verification
   - Range validation
   - NaN/Inf detection
   
5. Registration:
   - Add to feature catalog
   - Compute metadata
   - Update indices
```

#### Warstwa persystencji (Database Layer)

##### Schema bazy danych
```
Core tables:

1. sessions:
   - id: UUID PRIMARY KEY
   - name: VARCHAR(255)
   - config: JSON
   - status: ENUM('running','completed','failed','paused')
   - created_at: TIMESTAMP
   - updated_at: TIMESTAMP
   - completed_at: TIMESTAMP NULL
   - metadata: JSON
   
2. exploration_tree:
   - id: UUID PRIMARY KEY
   - session_id: UUID REFERENCES sessions
   - parent_id: UUID REFERENCES exploration_tree NULL
   - node_path: VARCHAR(1000)
   - visits: INTEGER
   - total_reward: DOUBLE
   - max_reward: DOUBLE
   - features: JSON
   - created_at: TIMESTAMP
   - metadata: JSON
   INDEX idx_session_parent (session_id, parent_id)
   INDEX idx_best_reward (session_id, max_reward DESC)
   
3. features:
   - id: UUID PRIMARY KEY
   - name: VARCHAR(500) UNIQUE
   - operation: VARCHAR(100)
   - parameters: JSON
   - importance: DOUBLE
   - stability: DOUBLE
   - created_at: TIMESTAMP
   - metadata: JSON
   INDEX idx_importance (importance DESC)
   INDEX idx_operation (operation)
   
4. evaluations:
   - id: UUID PRIMARY KEY
   - feature_set_hash: CHAR(64) UNIQUE
   - scores: JSON
   - model_config: JSON
   - duration_ms: INTEGER
   - created_at: TIMESTAMP
   INDEX idx_hash (feature_set_hash)
   
5. datasets:
   - id: UUID PRIMARY KEY
   - name: VARCHAR(255) UNIQUE
   - path: VARCHAR(1000)
   - hash: CHAR(64)
   - row_count: BIGINT
   - column_count: INTEGER
   - size_bytes: BIGINT
   - metadata: JSON
   - registered_at: TIMESTAMP
   INDEX idx_name (name)
```

##### Optymalizacje wydajności
```
Database optimizations:

1. Partycjonowanie:
   - exploration_tree: BY session_id
   - evaluations: BY created_at (monthly)
   - features: BY importance (buckets)
   
2. Indeksy:
   - Covering indexes dla częstych queries
   - Partial indexes dla filtrowania
   - Expression indexes dla JSON fields
   
3. Materialized views:
   - session_summary: agregaty per session
   - feature_rankings: top features
   - daily_metrics: statystyki dzienne
   
4. Connection pooling:
   - Min connections: 2
   - Max connections: 10
   - Idle timeout: 300s
   - Validation query: SELECT 1
   
5. Query optimization:
   - Prepared statements
   - Batch inserts
   - Async I/O gdzie możliwe
   - Query plan caching
```

##### Backup i recovery
```
Backup strategy:
1. Continuous backup:
   - WAL archiving
   - Point-in-time recovery
   - Retention: 7 days
   
2. Scheduled backups:
   - Full backup: daily
   - Incremental: hourly
   - Compression: zstd
   - Encryption: AES-256
   
3. Export snapshots:
   - Session checkpoints
   - Feature catalog dumps
   - Configuration backups
   
Recovery procedures:
1. Automatic recovery:
   - Detect corruption
   - Restore from last good
   - Replay WAL
   
2. Manual recovery:
   - Point-in-time restore
   - Selective table restore
   - Cross-version migration
```

#### Interfejs użytkownika (CLI Interface)

##### Parser komend
```
Command parsing architecture:
1. Main parser:
   - Global options (--verbose, --config)
   - Subcommands (run, resume, list)
   - Version and help
   
2. Subcommand parsers:
   - Command-specific options
   - Validation rules
   - Default values
   - Mutual exclusions
   
3. Argument types:
   - Path validation
   - Enum constraints
   - Range checks
   - Custom validators
   
4. Configuration precedence:
   - CLI arguments (highest)
   - Environment variables
   - Config file
   - Defaults (lowest)
```

##### Output formatting
```
Output system:
1. Structured output:
   - Tables (rich library)
   - Trees (hierarchy display)
   - Progress bars (tqdm)
   - Syntax highlighting
   
2. Format options:
   - Human readable (default)
   - JSON (--format json)
   - CSV (--format csv)
   - Silent (--quiet)
   
3. Logging integration:
   - Structured logs (JSON)
   - Log levels per module
   - Rotation and compression
   - Syslog support
   
4. Error handling:
   - Exit codes (0-255)
   - Error classes
   - Stack traces (--debug)
   - Suggestions
```

##### Tryb interaktywny
```
REPL mode:
1. Command prompt:
   - Context-aware completion
   - History (readline)
   - Multi-line support
   - Syntax validation
   
2. Commands:
   - All CLI commands
   - Additional REPL commands
   - Variable assignment
   - Script execution
   
3. State management:
   - Session persistence
   - Variable workspace
   - Command history
   - Undo/redo support
```

### Algorytmy kluczowe

#### UCB1 Selection Algorithm
```
Implementacja UCB1 z optymalizacjami:

1. Standard UCB1:
   score = avg_reward + C * sqrt(ln(parent_visits) / node_visits)
   
2. Optymalizacje:
   - Cached score invalidation tylko przy update
   - SIMD operations dla batch scoring
   - Approximate sqrt dla szybkości
   
3. Modyfikacje:
   - Progressive bias: C = C0 * (1 - iteration/max_iterations)
   - Domain priors: score += prior_weight * domain_score
   - Diversity bonus: score += diversity_factor * uniqueness
   
Złożoność: O(1) per node scoring
```

#### Feature Importance Calculation
```
Multi-factor importance scoring:

1. Komponenty:
   - Predictive power (z modelu)
   - Stability (cross-validation variance)  
   - Simplicity (inverse complexity)
   - Novelty (różnica od istniejących)
   
2. Formuła:
   importance = w1 * power + w2 * stability + w3 * simplicity + w4 * novelty
   
   Gdzie wagi są uczone online poprzez:
   - Multi-armed bandit
   - Thompson sampling
   - Decay dla eksploatacji
   
3. Normalizacja:
   - Min-max scaling do [0,1]
   - Percentile ranking
   - Z-score dla outliers
```

#### Memory-Efficient Feature Storage
```
Compressed feature storage:

1. Sparse features:
   - CSR format dla sparse matrices
   - Dictionary encoding dla powtórzeń
   - Bit packing dla boolean
   
2. Dense features:
   - Quantization (float32 -> int8)
   - Delta encoding dla sekwencji
   - Compression (zstd level 3)
   
3. Hierarchical storage:
   - Hot tier: RAM (najczęściej używane)
   - Warm tier: SSD (cache overflow)
   - Cold tier: Disk (archival)
   
Oszczędność: 60-90% pamięci
```

### Przepływy systemowe

#### Inicjalizacja systemu
```
System startup sequence:
1. Configuration loading:
   - Parse CLI arguments
   - Load config file
   - Merge with defaults
   - Validate completeness
   
2. Environment setup:
   - Check Python version
   - Verify dependencies
   - GPU detection
   - Memory limits
   
3. Database initialization:
   - Connection pool setup
   - Schema migration check
   - Integrity verification
   - Cleanup old data
   
4. Component initialization:
   - MCTS engine creation
   - Feature space loading
   - Evaluator preparation
   - Logger configuration
   
5. Recovery check:
   - Detect interrupted sessions
   - Offer resume options
   - Load checkpoint if needed
```

#### Główna pętla MCTS
```
Main exploration loop:
while not stopping_criteria_met():
    # Selection phase
    current = root
    path = []
    while current.is_expanded() and current.has_children():
        current = select_best_child(current)
        path.append(current)
    
    # Expansion phase
    if current.visits >= expansion_threshold:
        new_children = expand_node(current)
        if new_children:
            current = random.choice(new_children)
            path.append(current)
    
    # Evaluation phase
    if current.needs_evaluation():
        features = generate_features(current)
        score = evaluate_features(features)
        current.update_score(score)
    
    # Backpropagation phase
    for node in reversed(path):
        node.update_statistics(score)
    
    # Maintenance
    if iteration % checkpoint_interval == 0:
        save_checkpoint()
    if memory_usage() > threshold:
        prune_tree()
    
    iteration += 1
```

#### Feature generation workflow
```
Feature generation pipeline:
1. Parse node state:
   - Extract operation sequence
   - Identify target columns
   - Load parameters
   
2. Validate applicability:
   - Check column types
   - Verify preconditions
   - Estimate memory needs
   
3. Execute operations:
   for operation in sequence:
       - Load from cache if exists
       - Apply transformation
       - Validate output
       - Update cache
   
4. Post-processing:
   - Handle missing values
   - Clip outliers
   - Normalize if needed
   - Add metadata
   
5. Quality checks:
   - Signal detection
   - Leakage prevention
   - Correlation analysis
   
Return: DataFrame with new features
```

### Obsługa błędów i edge cases

#### Error hierarchy
```
Exception hierarchy:
BaseMinotaurError
├── ConfigurationError
│   ├── InvalidConfigError
│   ├── MissingConfigError
│   └── IncompatibleVersionError
├── DataError
│   ├── DataValidationError
│   ├── InsufficientDataError
│   └── DataTypeError
├── FeatureError
│   ├── FeatureGenerationError
│   ├── InvalidFeatureError
│   └── FeatureLeakageError
├── EvaluationError
│   ├── ModelTrainingError
│   ├── MetricCalculationError
│   └── TimeoutError
└── SystemError
    ├── ResourceError
    ├── DatabaseError
    └── CorruptionError
```

#### Recovery strategies
```
Error recovery patterns:
1. Transient errors (retry):
   - Network timeouts
   - Resource contention
   - Temporary locks
   Strategy: Exponential backoff
   
2. Data errors (skip):
   - Malformed features
   - Numeric overflow
   - Type mismatches
   Strategy: Log and continue
   
3. Critical errors (stop):
   - Out of memory
   - Disk full
   - Database corruption
   Strategy: Graceful shutdown
   
4. Logic errors (rollback):
   - Invalid state
   - Constraint violations
   - Invariant breaks
   Strategy: Restore checkpoint
```

#### Edge case handling
```
Common edge cases:

1. Empty datasets:
   - Validation error
   - Clear message
   - No processing
   
2. Single row data:
   - Disable CV
   - Warning message
   - Limited operations
   
3. All missing column:
   - Skip column
   - Log warning
   - Continue others
   
4. Constant features:
   - Auto-remove
   - Log info
   - No evaluation
   
5. Perfect correlation:
   - Keep first
   - Remove others
   - Document decision
   
6. Memory pressure:
   - Flush cache
   - Reduce batch size
   - Enable swapping
   
7. Infinite/NaN values:
   - Replace with bounds
   - Log occurrence
   - Track frequency
```

### Performance optimizations

#### Computational optimizations
```
Speed improvements:
1. Vectorization:
   - NumPy operations
   - Pandas native methods
   - Numba JIT for loops
   
2. Parallelization:
   - Feature generation (multiprocessing)
   - Model training (native parallel)
   - Tree operations (thread pool)
   
3. Caching:
   - LRU for features
   - Memoization for scores
   - Persistent for sessions
   
4. Lazy evaluation:
   - Delayed computation
   - On-demand loading
   - Streaming processing
   
5. Algorithm tricks:
   - Early stopping
   - Approximate methods
   - Sampling strategies
```

#### Memory optimizations
```
Memory management:
1. Data types:
   - Downcast numerics
   - Categorical for strings
   - Sparse for binary
   
2. Chunking:
   - Process in batches
   - Streaming reads
   - Incremental writes
   
3. Garbage collection:
   - Explicit calls
   - Reference breaking
   - Weak references
   
4. Memory mapping:
   - Large files
   - Shared data
   - Zero-copy views
```

#### I/O optimizations
```
I/O performance:
1. Batching:
   - Group operations
   - Bulk inserts
   - Vectorized queries
   
2. Async I/O:
   - Non-blocking reads
   - Concurrent writes
   - Event-driven
   
3. Compression:
   - Wire protocol
   - Storage format
   - Streaming codec
   
4. Caching:
   - Query results
   - File contents
   - Network responses
```

### Limity i ograniczenia

#### Limity systemowe
```
Hard limits:
- Max tree depth: 20 levels
- Max nodes in memory: 1M nodes
- Max features per dataset: 10k features
- Max dataset size: 10GB in RAM
- Max session duration: 7 days
- Max concurrent operations: 100

Soft limits (configurable):
- Default tree depth: 8
- Default max nodes: 10k
- Default features: 1k
- Default time limit: 8 hours
- Default memory: 16GB
```

#### Ograniczenia algorytmiczne
```
Algorithm limitations:
1. MCTS assumptions:
   - Stationary reward distribution
   - Independent evaluations
   - Discrete action space
   
2. Feature assumptions:
   - Tabular data only
   - Fixed schema
   - No missing target
   
3. Evaluation assumptions:
   - IID data splits
   - Single metric optimization
   - Supervised learning only
```

#### Known issues
```
Znanye problemy:
1. Curse of dimensionality:
   - Exponential growth with depth
   - Mitigation: Pruning, sampling
   
2. Local optima:
   - MCTS can get stuck
   - Mitigation: Random restarts
   
3. Concept drift:
   - Features may become stale
   - Mitigation: Periodic re-evaluation
   
4. Class imbalance:
   - Biased toward majority
   - Mitigation: Stratified sampling, weights
```