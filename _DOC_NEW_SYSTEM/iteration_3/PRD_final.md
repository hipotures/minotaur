# Product Requirements Document - Final Version
## System automatycznego odkrywania cech dla konkurencji ML

### Cel projektu
Stworzenie autonomicznego systemu do automatycznego odkrywania i generowania cech (feature engineering) dla zadań uczenia maszynowego, ze szczególnym fokusem na konkurencje typu Kaggle. System ma wykorzystywać algorytmy przeszukiwania inspirowane sztuczną inteligencją do eksploracji przestrzeni możliwych transformacji danych, automatycznie oceniając ich wartość predykcyjną.

### Główne cele biznesowe

#### 1. Automatyzacja procesu feature engineering
- **Metryka sukcesu**: Redukcja czasu potrzebnego na ręczne tworzenie cech z 2-4 tygodni do 4-8 godzin
- **Baseline**: Ekspert data scientist tworzy 50-100 cech w ciągu tygodnia
- **Target**: System generuje i ocenia 1000+ cech w ciągu 8 godzin
- **ROI**: 10-20x przyspieszenie procesu przy zachowaniu jakości top 20%

#### 2. Poprawa wyników modeli
- **Metryka sukcesu**: Wzrost performance modeli o 5-15% względem baseline
- **Metodologia pomiaru**: A/B testing na 10 różnych datasetach
- **Acceptance criteria**: 
  - 80% przypadków: poprawa ≥5%
  - 50% przypadków: poprawa ≥10%
  - 20% przypadków: poprawa ≥15%
- **Benchmark**: Porównanie z ręcznie stworzonymi cechami przez ekspertów

#### 3. Demokratyzacja ML
- **Metryka sukcesu**: Junior data scientists osiągają 90% performance senior experts
- **Test scenario**: 10 junior DS vs 10 senior DS na tym samym datasecie
- **Time to productivity**: <2 godziny nauki systemu
- **User satisfaction**: NPS score >50

#### 4. Skalowalność rozwiązań
- **Metryka sukcesu**: Jeden system obsługuje 80% typowych problemów ML
- **Wspierane typy**: Classification (binary/multi), regression, ranking
- **Adaptacja do domeny**: <30 minut konfiguracji dla nowej domeny
- **Reużywalność**: 70% cech można przenieść między podobnymi problemami

#### 5. Dokumentacja procesu
- **Metryka sukcesu**: 100% reprodukowalność wyników
- **Generowany kod**: Production-ready, PEP8 compliant, z testami
- **Dokumentacja**: Automatyczne generowanie raportów z uzasadnieniem
- **Compliance**: Audit trail dla wszystkich decyzji algorytmu

### Szczegółowe wymagania funkcjonalne

#### FR1: System eksploracji przestrzeni cech

##### FR1.1: Algorytm przeszukiwania
- **Implementacja**: Monte Carlo Tree Search z modyfikacjami dla feature engineering
- **Parametry kontrolne**:
  - Głębokość eksploracji: 2-20 poziomów
  - Branching factor: 1-50 dzieci per węzeł
  - Exploration vs exploitation: konfigurowalne C w UCB1 (0.1-5.0)
- **Adaptacyjność**: Automatyczne dostosowanie parametrów na podstawie postępu
- **Checkpointing**: Zapis stanu co 5 minut lub 10 iteracji

##### FR1.2: Operacje transformacji
- **Kategorie operacji**:
  1. **Statystyczne** (15+ operacji):
     - Podstawowe: mean, median, std, var, min, max
     - Zaawansowane: skewness, kurtosis, entropy
     - Percentyle: q25, q50, q75, IQR
     - Robust: MAD, trimmed mean, winsorized stats
  2. **Wielomianowe** (6 operacji):
     - Potęgi: x^2, x^3, sqrt(x), 1/x
     - Interakcje: x*y dla wybranych par
     - Kompozycje: log(x+1), exp(x)
  3. **Binning** (5 strategii):
     - Equal width, equal frequency
     - Quantile-based, K-means based
     - Decision tree based (supervised)
  4. **Rankingowe** (4 typy):
     - Dense rank, ordinal rank
     - Percent rank, normalized rank
  5. **Czasowe** (jeśli wykryte):
     - Lag features: t-1, t-7, t-30
     - Rolling statistics: window 3,7,14,30
     - Seasonal decomposition
     - Time since event
  6. **Tekstowe** (jeśli wykryte):
     - Length statistics
     - Word/char counts
     - TF-IDF features
     - Sentiment scores
  7. **Kategoryczne**:
     - One-hot encoding (do 50 kategorii)
     - Target encoding z regularization
     - Frequency encoding
     - Embedding-based (pre-trained)

##### FR1.3: Kompozycja operacji
- **Głębokość**: Do 3 operacji w sekwencji
- **Przykłady kompozycji**:
  - `rank(log(x+1))` - ranking of log-transformed values
  - `bin(normalize(x))` - binning normalized values
  - `mean(x) by category` - grouped aggregations
- **Walidacja**: Automatyczne wykrywanie nonsensownych kompozycji
- **Pamięć**: Cache pośrednich wyników

#### FR2: Automatyczna ewaluacja jakości

##### FR2.1: Framework ewaluacji
- **Silnik**: AutoGluon TabularPredictor z custom configurations
- **Modele bazowe**:
  - LightGBM: primary (fastest)
  - XGBoost: GPU acceleration gdy dostępne
  - CatBoost: dla danych z wieloma kategoriami
  - Neural Networks: TabNet dla głębokich interakcji
- **Ensemble**: Automatyczne ważenie modeli

##### FR2.2: Metryki oceny
- **Classification**:
  - Binary: AUC, accuracy, F1, precision, recall
  - Multi-class: accuracy, macro-F1, MAP@K
  - Imbalanced: balanced accuracy, Cohen's kappa
- **Regression**:
  - RMSE, MAE, MAPE, R²
  - Quantile loss dla uncertainty
- **Ranking**: NDCG, MAP, MRR
- **Custom metrics**: Plugin system dla własnych metryk

##### FR2.3: Strategia walidacji
- **Default**: 5-fold stratified CV
- **Time series**: Time-based splits
- **Small data**: Repeated k-fold (3x5)
- **Large data**: Single holdout (20%)
- **Leakage prevention**: Automatic detection

##### FR2.4: Optymalizacja wydajności
- **Early stopping**: Po 3 rundach bez poprawy
- **Resource limits**: Max 60s per evaluation
- **Sampling**: Dla datasets >100k rows
- **Caching**: Hash-based result storage
- **Parallel evaluation**: Do 4 równoległych

#### FR3: Zarządzanie sesjami eksploracji

##### FR3.1: Lifecycle sesji
- **Inicjalizacja**:
  - Walidacja datasetu i konfiguracji
  - Alokacja zasobów (RAM, CPU, GPU)
  - Setup logging i monitoring
- **Wykonanie**:
  - Real-time progress tracking
  - Resource usage monitoring
  - Automatic checkpointing
- **Zakończenie**:
  - Graceful shutdown
  - Final report generation
  - Cleanup temporary files
- **Post-processing**:
  - Best features extraction
  - Code generation
  - Documentation creation

##### FR3.2: Persystencja stanu
- **Checkpointing**:
  - Full tree serialization
  - Feature cache state
  - Evaluation results
  - Random seeds
- **Recovery**:
  - Automatic detection of interruption
  - State validation
  - Seamless continuation
  - Merge with existing results
- **Versioning**: Compatibility checks

##### FR3.3: Monitoring i telemetria
- **Real-time metrics**:
  - Iterations per minute
  - Features evaluated
  - Best score progression
  - Resource utilization
- **Alerts**:
  - Memory >90% threshold
  - No improvement for N iterations
  - Errors rate >5%
- **Logging levels**:
  - ERROR: Critical failures
  - WARNING: Performance issues
  - INFO: Major milestones
  - DEBUG: Detailed operations

#### FR4: System zarządzania danymi

##### FR4.1: Obsługa formatów
- **Input formats**:
  - CSV: with auto-detection of separators
  - Parquet: preferred for performance
  - Feather: for R interoperability
  - JSON: for nested structures
- **Compression**: gzip, bzip2, xz, zstd
- **Encoding**: UTF-8, Latin-1, auto-detect
- **Size limits**: Up to 10GB in memory

##### FR4.2: Walidacja danych
- **Automatyczne wykrywanie**:
  - Column types (numeric, categorical, datetime, text)
  - Missing value patterns
  - Outliers and anomalies
  - Data quality issues
- **Raportowanie problemów**:
  - Missing values per column
  - Cardinality analysis
  - Distribution statistics
  - Correlation warnings
- **Auto-fix opcje**:
  - Missing value imputation
  - Outlier capping
  - Type conversions
  - Duplicate removal

##### FR4.3: Bezpieczeństwo danych
- **Access control**:
  - Dataset-level permissions
  - User authentication
  - Action audit trail
- **Data protection**:
  - No data leaves local environment
  - Encrypted storage option
  - Secure deletion
- **Compliance**:
  - GDPR considerations
  - PII detection warnings
  - Data retention policies

#### FR5: Analityka i raportowanie

##### FR5.1: Dashboards
- **Session overview**:
  - Progress timeline
  - Score evolution
  - Resource usage
  - Top discoveries
- **Feature analysis**:
  - Importance rankings
  - Correlation matrix
  - Distribution plots
  - Usage frequency
- **Comparative analysis**:
  - Session vs session
  - Feature stability
  - Performance trends

##### FR5.2: Eksport wyników
- **Formats**:
  - Python: sklearn transformers
  - R: recipes format
  - SQL: feature queries
  - JSON: metadata and specs
- **Documentation**:
  - Feature descriptions
  - Statistical properties
  - Usage examples
  - Performance impact
- **Integration ready**:
  - Git-friendly format
  - CI/CD compatible
  - Version controlled

### Wymagania niefunkcjonalne

#### NFR1: Wydajność
- **Throughput**: 100+ features/hour na 8-core CPU
- **Latency**: <1s dla feature generation
- **Skalowanie**: Linear do 16 cores
- **GPU acceleration**: 5-10x dla obsługiwanych operacji

#### NFR2: Niezawodność
- **Uptime**: 99.9% (bez planned maintenance)
- **MTBF**: >168 hours continuous operation
- **Recovery time**: <2 minutes
- **Data integrity**: ACID compliance

#### NFR3: Użyteczność
- **Learning curve**: <2 hours do produktywności
- **Error messages**: Actionable i helpful
- **Documentation**: Comprehensive z przykładami
- **Community**: Discord/Slack support

#### NFR4: Bezpieczeństwo
- **Authentication**: Optional enterprise SSO
- **Authorization**: Role-based access
- **Encryption**: At-rest i in-transit
- **Audit**: Complete action history

#### NFR5: Utrzymanie
- **Code quality**: >80% test coverage
- **Technical debt**: <20% według SonarQube
- **Update frequency**: Monthly patches
- **Breaking changes**: Major versions only

### Przypadki użycia (Use Cases)

#### UC1: Kaggle Competition Workflow
**Aktor**: Competitive Data Scientist
**Trigger**: New competition dataset available
**Przebieg**:
1. Download i register dataset
2. Quick exploration run (1 hour, test mode)
3. Analyze initial results
4. Full run overnight (8 hours)
5. Cherry-pick best features
6. Export dla ensemble models
**Rezultat**: Top 20% leaderboard position

#### UC2: Enterprise Feature Store
**Aktor**: ML Engineer in Enterprise
**Trigger**: Monthly model refresh needed
**Przebieg**:
1. Connect to data warehouse
2. Configure domain-specific operations
3. Run exploration z business constraints
4. Validate features przeciw production rules
5. Deploy do feature store
6. Monitor feature drift
**Rezultat**: Automated feature pipeline

#### UC3: Research Experimentation
**Aktor**: ML Researcher
**Trigger**: Testing new feature generation method
**Przebieg**:
1. Implement custom operation
2. A/B test vs existing methods
3. Statistical significance testing
4. Publish results z reprodukcją
**Rezultat**: Paper-ready experiments

### Acceptance Criteria

#### AC1: Funkcjonalność Core
- [ ] System generuje 1000+ cech w 8 godzin
- [ ] Top 10 cech poprawia baseline o >5%
- [ ] Session recovery działa w 99% przypadków
- [ ] Export generuje working Python code

#### AC2: Wydajność
- [ ] 100k rows dataset: <1 hour full exploration
- [ ] 1M rows dataset: <8 hours full exploration
- [ ] Memory usage: <16GB dla 1M rows
- [ ] CPU utilization: >80% podczas eksploracji

#### AC3: Jakość
- [ ] Test coverage: >80%
- [ ] Documentation coverage: 100% public API
- [ ] Linting: Zero errors (flake8, mypy)
- [ ] Security: Przechodzi OWASP scan

#### AC4: Użyteczność
- [ ] Nowy user produktywny w <2 godziny
- [ ] CLI autocomplete działa
- [ ] Error messages mają suggested fixes
- [ ] --help dostępne dla każdej komendy

### Ograniczenia i założenia

#### Ograniczenia techniczne
1. **Single machine**: Brak distributed computing w v1.0
2. **Tabular data only**: Brak wsparcia dla obrazów/audio
3. **Supervised learning**: Wymaga target variable
4. **Memory bound**: Dataset musi zmieścić się w RAM
5. **Local storage**: Brak cloud integration w MVP

#### Założenia biznesowe
1. Użytkownicy mają podstawową wiedzę ML
2. Datasets są już oczyszczone (podstawowo)
3. Computational resources są dostępne
4. Python environment jest standardem
5. Offline operation jest akceptowalne

#### Out of Scope (v1.0)
1. Real-time feature serving
2. Streaming data support  
3. AutoML pełny pipeline
4. Model deployment
5. Collaborative features
6. Cloud-native architecture
7. Multi-language support
8. GUI/Web interface

### Metryki sukcesu produktu

#### Adoption Metrics
- 1000+ aktywnych użytkowników w 6 miesięcy
- 50+ enterprise pilots w rok 1
- 10k+ GitHub stars w rok 2

#### Usage Metrics
- Average 5 sessions/user/month
- 70% session completion rate
- 80% użytkowników exportuje kod

#### Quality Metrics
- <5% bug reports/release
- <48h average fix time
- >4.5/5.0 user satisfaction

#### Business Impact
- 20% redukcja time-to-model
- 10% średnia poprawa accuracy
- 5x ROI dla enterprise users