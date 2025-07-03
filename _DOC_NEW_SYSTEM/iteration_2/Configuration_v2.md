# Configuration Parameters v2
## Szczegółowa specyfikacja parametrów konfiguracyjnych

### Struktura konfiguracji
System wykorzystuje hierarchiczną konfigurację YAML z dziedziczeniem. Plik bazowy (`mcts_config.yaml`) zawiera domyślne wartości, które mogą być nadpisane przez pliki specyficzne dla przypadków użycia.

### Parametry główne

#### test_mode
- **Typ**: boolean
- **Domyślnie**: false
- **Opis**: Włącza tryb testowy z mock evaluatorem dla szybkiego debugowania
- **Wpływ**: Zastępuje rzeczywistą ewaluację ML deterministycznymi wynikami

### Sekcja: session
Parametry kontrolujące przebieg sesji eksploracji.

#### session.mode
- **Typ**: string (enum: "new", "resume")
- **Domyślnie**: "new"
- **Opis**: Tryb uruchomienia - nowa sesja lub wznowienie istniejącej
- **Użycie**: Automatycznie ustawiane przez CLI

#### session.max_iterations
- **Typ**: integer
- **Domyślnie**: 20
- **Zakres**: 1-10000
- **Opis**: Maksymalna liczba iteracji algorytmu MCTS
- **Wpływ**: Określa głębokość eksploracji przestrzeni cech

#### session.max_runtime_hours
- **Typ**: float
- **Domyślnie**: 12
- **Zakres**: 0.1-168 (tydzień)
- **Opis**: Maksymalny czas działania sesji w godzinach
- **Wpływ**: Zabezpieczenie przed nieskończonym działaniem

#### session.checkpoint_interval
- **Typ**: integer
- **Domyślnie**: 25
- **Opis**: Częstotliwość automatycznego zapisu stanu (co N iteracji)
- **Wpływ**: Balans między bezpieczeństwem a wydajnością I/O

#### session.auto_save
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Automatyczny zapis postępu
- **Wpływ**: Umożliwia recovery po awarii

#### session.session_name
- **Typ**: string lub null
- **Domyślnie**: null (auto-generowane)
- **Opis**: Nazwa sesji dla łatwej identyfikacji
- **Format**: Jeśli null, używa wzorca "mcts_YYYYMMDD_HHMMSS"

### Sekcja: mcts
Parametry algorytmu Monte Carlo Tree Search.

#### mcts.exploration_weight
- **Typ**: float
- **Domyślnie**: 1.4
- **Zakres**: 0.1-5.0
- **Opis**: Parametr C w formule UCB1, kontroluje balans eksploracja vs eksploatacja
- **Wpływ**: Wyższe wartości = więcej eksploracji nowych obszarów

#### mcts.max_tree_depth
- **Typ**: integer
- **Domyślnie**: 8
- **Zakres**: 2-20
- **Opis**: Maksymalna głębokość drzewa eksploracji
- **Wpływ**: Ogranicza złożoność generowanych cech

#### mcts.expansion_threshold
- **Typ**: integer
- **Domyślnie**: 3
- **Opis**: Minimalna liczba wizyt węzła przed ekspansją
- **Wpływ**: Zapobiega przedwczesnej ekspansji

#### mcts.min_visits_for_best
- **Typ**: integer
- **Domyślnie**: 10
- **Opis**: Minimalna liczba wizyt dla uznania węzła za "najlepszy"
- **Wpływ**: Zwiększa pewność wyboru najlepszej ścieżki

#### mcts.ucb1_confidence
- **Typ**: float
- **Domyślnie**: 0.95
- **Zakres**: 0.5-0.99
- **Opis**: Poziom ufności dla obliczeń UCB1
- **Wpływ**: Wpływa na szerokość przedziału ufności

#### mcts.selection_strategy
- **Typ**: string (enum: "ucb1", "thompson", "epsilon_greedy")
- **Domyślnie**: "ucb1"
- **Opis**: Strategia selekcji węzłów do eksploracji
- **Opcje**:
  - ucb1: Upper Confidence Bound
  - thompson: Thompson Sampling (PLANNED)
  - epsilon_greedy: Epsilon-Greedy (PLANNED)

#### mcts.max_children_per_node
- **Typ**: integer
- **Domyślnie**: 5
- **Zakres**: 1-50
- **Opis**: Maksymalna liczba dzieci per węzeł
- **Wpływ**: Kontroluje branching factor drzewa

#### mcts.expansion_budget
- **Typ**: integer
- **Domyślnie**: 20
- **Opis**: Budżet na ekspansję w pojedynczej iteracji
- **Uwaga**: Powinno być ≤ max_children_per_node

#### mcts.max_nodes_in_memory
- **Typ**: integer
- **Domyślnie**: 10000
- **Opis**: Maksymalna liczba węzłów w pamięci
- **Wpływ**: Po przekroczeniu następuje pruning

#### mcts.prune_threshold
- **Typ**: float
- **Domyślnie**: 0.1
- **Zakres**: 0.0-1.0
- **Opis**: Próg dla przycinania słabych gałęzi (względem best score)
- **Wpływ**: 0.1 = przycina węzły z score < 90% best

### Sekcja: autogluon
Parametry ewaluacji modeli ML.

#### autogluon.dataset_name
- **Typ**: string lub null
- **Domyślnie**: null
- **Opis**: Nazwa zarejestrowanego datasetu
- **Użycie**: Alternatywa dla bezpośrednich ścieżek

#### autogluon.target_metric
- **Typ**: string lub null
- **Domyślnie**: null
- **Opis**: Metryka do optymalizacji
- **Opcje**: "accuracy", "map@3", "rmse", "auc", custom

#### autogluon.included_model_types
- **Typ**: list[string]
- **Domyślnie**: ["XGB", "GBM", "CAT"]
- **Opis**: Typy modeli do użycia w ensemble
- **Opcje**:
  - XGB: XGBoost
  - GBM: LightGBM
  - CAT: CatBoost
  - NN_TORCH: Neural Network (TabNet)

#### autogluon.enable_gpu
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Użycie GPU dla treningu (jeśli dostępne)
- **Wpływ**: 5-10x przyspieszenie dla XGBoost/NN

#### autogluon.train_size
- **Typ**: float
- **Domyślnie**: 0.8
- **Zakres**: 0.5-0.95
- **Opis**: Proporcja danych treningowych
- **Wpływ**: Reszta używana do walidacji

#### autogluon.time_limit
- **Typ**: integer
- **Domyślnie**: 60
- **Jednostka**: sekundy
- **Opis**: Limit czasu na trenowanie per iteracja
- **Wpływ**: Trade-off między jakością a szybkością

#### autogluon.presets
- **Typ**: string
- **Domyślnie**: "medium_quality"
- **Opcje**:
  - "best_quality": Najlepsza jakość, wolne
  - "high_quality": Wysoka jakość
  - "medium_quality": Balans jakość/czas
  - "optimize_for_deployment": Szybkie predykcje

#### autogluon.num_bag_folds
- **Typ**: integer
- **Domyślnie**: 3
- **Opis**: Liczba foldów dla baggingu
- **Wpływ**: Więcej = lepsze wyniki, wolniej

#### autogluon.num_bag_sets
- **Typ**: integer
- **Domyślnie**: 1
- **Opis**: Liczba zestawów baggingu
- **Wpływ**: Multiplicative z num_bag_folds

#### autogluon.holdout_frac
- **Typ**: float
- **Domyślnie**: 0.2
- **Opis**: Frakcja danych na holdout validation
- **Wpływ**: Używane gdy train/test split niedostępny

#### autogluon.verbosity
- **Typ**: integer
- **Domyślnie**: 1
- **Zakres**: 0-4
- **Opis**: Poziom logowania AutoGluon
- **Poziomy**:
  - 0: Silent
  - 1: Warnings
  - 2: Info
  - 3: Progress
  - 4: Detailed

### Sekcja: feature_space
Parametry generowania i zarządzania cechami.

#### feature_space.max_features_per_node
- **Typ**: integer
- **Domyślnie**: 300
- **Opis**: Maksymalna liczba cech per węzeł MCTS
- **Wpływ**: Ogranicza memory usage

#### feature_space.min_improvement_threshold
- **Typ**: float
- **Domyślnie**: 0.0005
- **Opis**: Minimalna poprawa score dla akceptacji cechy
- **Wpływ**: Filtruje marginalne ulepszenia

#### feature_space.feature_timeout
- **Typ**: integer
- **Domyślnie**: 300
- **Jednostka**: sekundy
- **Opis**: Timeout dla generowania pojedynczej cechy
- **Wpływ**: Zabezpieczenie przed hanging operations

#### feature_space.max_features_to_build
- **Typ**: integer lub null
- **Domyślnie**: null (unlimited)
- **Opis**: Globalny limit cech do zbudowania
- **Użycie**: Przydatne dla szybkich testów

#### feature_space.max_features_per_iteration
- **Typ**: integer lub null
- **Domyślnie**: null (unlimited)
- **Opis**: Limit cech per iteracja MCTS
- **Wpływ**: Kontroluje przyrost pamięci

#### feature_space.feature_build_timeout
- **Typ**: integer
- **Domyślnie**: 300
- **Opis**: Globalny timeout budowania cech
- **Wpływ**: Zabezpieczenie dla całego procesu

#### feature_space.cache_miss_limit
- **Typ**: integer
- **Domyślnie**: 50
- **Opis**: Max cache misses przed rebuild
- **Wpływ**: Optymalizacja cache performance

#### feature_space.generic_operations
- **Typ**: dict[string, boolean]
- **Opis**: Włączenie/wyłączenie typów operacji
- **Klucze**:
  - statistical_aggregations: true
  - polynomial_features: true
  - binning_features: true
  - ranking_features: true
  - categorical_features: true
  - text_features: true
  - train_features: true

#### feature_space.generic_params
- **Typ**: dict
- **Opis**: Parametry dla operacji generycznych
- **Parametry**:
  - polynomial_degree: 2 (stopień wielomianu)
  - binning_bins: 5 (liczba binów)
  - groupby_columns: [] (kolumny do grupowania)
  - aggregate_columns: [] (kolumny do agregacji)

#### feature_space.category_weights
- **Typ**: dict[string, float]
- **Opis**: Wagi priorytetów dla kategorii operacji
- **Domyślne**:
  - binning: 1.0
  - categorical: 1.1
  - polynomial: 0.8
  - ranking: 0.9
  - statistical: 1.0
  - temporal: 0.7
  - text: 0.6

#### feature_space.lazy_loading
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Ładowanie cech na żądanie
- **Wpływ**: Oszczędność pamięci

#### feature_space.cache_features
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Cache'owanie wygenerowanych cech
- **Wpływ**: Szybsze ponowne użycie

#### feature_space.max_cache_size_mb
- **Typ**: integer
- **Domyślnie**: 2048
- **Opis**: Maksymalny rozmiar cache w MB
- **Wpływ**: Po przekroczeniu - eviction

#### feature_space.cache_cleanup_threshold
- **Typ**: float
- **Domyślnie**: 0.8
- **Opis**: Próg zapełnienia dla cleanup
- **Wpływ**: 0.8 = cleanup przy 80% zapełnienia

### Sekcja: database
Parametry bazy danych i persystencji.

#### database.path
- **Typ**: string
- **Domyślnie**: "data/minotaur.duckdb"
- **Opis**: Ścieżka do głównej bazy danych
- **Format**: Relatywna lub absolutna

#### database.type
- **Typ**: string (enum: "duckdb", "sqlite", "postgresql")
- **Domyślnie**: "duckdb"
- **Opis**: Typ używanej bazy danych
- **Uwaga**: DuckDB rekomendowane dla analityki

#### database.schema
- **Typ**: string
- **Domyślnie**: "main"
- **Opis**: Nazwa schematu bazodanowego
- **Użycie**: Dla multi-tenant deployments

#### database.backup_path
- **Typ**: string
- **Domyślnie**: "data/backups/"
- **Opis**: Katalog dla backupów
- **Uwaga**: Tworzony automatycznie

#### database.backup_interval
- **Typ**: integer
- **Domyślnie**: 50
- **Opis**: Częstotliwość backupów (iteracje)
- **Wpływ**: Balans bezpieczeństwo vs I/O

#### database.backup_prefix
- **Typ**: string
- **Domyślnie**: "minotaur_backup_"
- **Opis**: Prefix nazw plików backup
- **Format**: prefix + timestamp + .duckdb

#### database.max_history_size
- **Typ**: integer
- **Domyślnie**: 50000
- **Opis**: Max rekordów w historii eksploracji
- **Wpływ**: Po przekroczeniu - archiwizacja

#### database.max_backup_files
- **Typ**: integer
- **Domyślnie**: 10
- **Opis**: Liczba zachowywanych backupów
- **Wpływ**: Starsze automatycznie usuwane

#### database.batch_size
- **Typ**: integer
- **Domyślnie**: 10
- **Opis**: Rozmiar batcha dla bulk operations
- **Wpływ**: Trade-off memory vs speed

#### database.sync_mode
- **Typ**: string (enum: "OFF", "NORMAL", "FULL")
- **Domyślnie**: "NORMAL"
- **Opis**: Tryb synchronizacji z dyskiem
- **Trade-off**: Bezpieczeństwo vs wydajność

#### database.journal_mode
- **Typ**: string (enum: "DELETE", "WAL", "MEMORY")
- **Domyślnie**: "WAL"
- **Opis**: Tryb journalowania transakcji
- **Zalecenie**: WAL dla najlepszej wydajności

#### database.auto_cleanup
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Automatyczne czyszczenie starych danych
- **Wpływ**: Utrzymuje rozmiar bazy pod kontrolą

#### database.cleanup_interval_hours
- **Typ**: integer
- **Domyślnie**: 24
- **Opis**: Częstotliwość auto-cleanup
- **Działanie**: Background process

#### database.retention_days
- **Typ**: integer
- **Domyślnie**: 30
- **Opis**: Okres przechowywania danych
- **Wpływ**: Dane starsze są archiwizowane

### Sekcja: logging
Parametry systemu logowania.

#### logging.level
- **Typ**: string (enum: "DEBUG", "INFO", "WARNING", "ERROR")
- **Domyślnie**: "DEBUG"
- **Opis**: Globalny poziom logowania
- **Wpływ**: DEBUG generuje dużo logów

#### logging.log_file
- **Typ**: string
- **Domyślnie**: "logs/minotaur.log"
- **Opis**: Główny plik logów
- **Rotacja**: Automatyczna przy przekroczeniu rozmiaru

#### logging.max_log_size_mb
- **Typ**: integer
- **Domyślnie**: 100
- **Opis**: Max rozmiar pliku logów
- **Działanie**: Po przekroczeniu - rotacja

#### logging.backup_count
- **Typ**: integer
- **Domyślnie**: 5
- **Opis**: Liczba zachowywanych backupów logów
- **Format**: .log, .log.1, .log.2, ...

#### logging.log_feature_code
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Logowanie generowanego kodu cech
- **Użycie**: Debugging transformacji

#### logging.log_timing
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Logowanie czasów wykonania
- **Wpływ**: Performance profiling

#### logging.log_memory_usage
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Logowanie użycia pamięci
- **Częstotliwość**: Co N iteracji

#### logging.log_autogluon_details
- **Typ**: boolean
- **Domyślnie**: false
- **Opis**: Szczegółowe logi z AutoGluon
- **Uwaga**: Generuje bardzo dużo output

#### logging.progress_interval
- **Typ**: integer
- **Domyślnie**: 10
- **Opis**: Częstotliwość raportowania postępu
- **Jednostka**: Iteracje

#### logging.save_intermediate_results
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Zapis wyników pośrednich
- **Użycie**: Analiza post-hoc

#### logging.timing_output_dir
- **Typ**: string
- **Domyślnie**: "logs/timing"
- **Opis**: Katalog dla logów timing
- **Format**: CSV files per session

### Sekcja: resources
Zarządzanie zasobami systemowymi.

#### resources.max_memory_gb
- **Typ**: integer
- **Domyślnie**: 16
- **Opis**: Limit pamięci RAM
- **Działanie**: Soft limit z warnings

#### resources.memory_check_interval
- **Typ**: integer
- **Domyślnie**: 5
- **Jednostka**: Iteracje
- **Opis**: Częstotliwość sprawdzania pamięci

#### resources.force_gc_interval
- **Typ**: integer
- **Domyślnie**: 50
- **Jednostka**: Iteracje
- **Opis**: Wymuszenie garbage collection

#### resources.use_gpu
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Użycie GPU jeśli dostępne
- **Wpływ**: Przyspieszenie XGBoost/NN

#### resources.max_cpu_cores
- **Typ**: integer
- **Domyślnie**: -1 (all available)
- **Opis**: Limit użycia CPU
- **Użycie**: Dla shared environments

#### resources.autogluon_num_cpus
- **Typ**: integer lub null
- **Domyślnie**: null (auto)
- **Opis**: CPU dla AutoGluon
- **Uwaga**: null = AutoGluon decyduje

#### resources.max_disk_usage_gb
- **Typ**: integer
- **Domyślnie**: 50
- **Opis**: Limit użycia dysku
- **Monitoring**: Warnings przy 80%

#### resources.temp_dir
- **Typ**: string
- **Domyślnie**: "/tmp/mcts_features"
- **Opis**: Katalog dla plików tymczasowych
- **Czyszczenie**: Na zakończenie sesji

#### resources.cleanup_temp_on_exit
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Czyszczenie temp na wyjściu
- **Wyjątek**: Zachowuje przy crash

### Sekcja: export
Parametry eksportu wyników.

#### export.formats
- **Typ**: list[string]
- **Domyślnie**: ["python", "json", "html"]
- **Opis**: Formaty eksportu
- **Opcje**: python, json, html, csv

#### export.python_output
- **Typ**: string
- **Domyślnie**: "outputs/best_features_discovered.py"
- **Opis**: Ścieżka dla eksportu Python
- **Format**: sklearn-compatible transformers

#### export.include_dependencies
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Dołącz imports w Python export
- **Wpływ**: Self-contained code

#### export.include_documentation
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Generuj docstrings
- **Wpływ**: Czytelność kodu

#### export.code_style
- **Typ**: string (enum: "pep8", "black", "google")
- **Domyślnie**: "pep8"
- **Opis**: Styl formatowania kodu
- **Działanie**: Auto-formatting

#### export.html_report
- **Typ**: string
- **Domyślnie**: "outputs/discovery_report.html"
- **Opis**: Ścieżka raportu HTML
- **Zawartość**: Interaktywne wykresy

#### export.include_plots
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Generuj wizualizacje
- **Typy**: Feature importance, tree viz

#### export.plot_format
- **Typ**: string (enum: "png", "svg", "pdf")
- **Domyślnie**: "png"
- **Opis**: Format wykresów
- **Trade-off**: Quality vs size

#### export.include_analytics
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Dołącz szczegółową analitykę
- **Zawartość**: Timing, memory, scores

#### export.output_dir
- **Typ**: string
- **Domyślnie**: "outputs/reports"
- **Opis**: Główny katalog output
- **Struktura**: Per-session subdirs

#### export.export_on_completion
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Auto-export po zakończeniu
- **Wpływ**: Convenience feature

#### export.export_on_improvement
- **Typ**: boolean
- **Domyślnie**: true
- **Opis**: Export przy nowym best score
- **Użycie**: Śledzenie postępu

#### export.export_interval
- **Typ**: integer
- **Domyślnie**: 100
- **Jednostka**: Iteracje
- **Opis**: Okresowy export

### Parametry specjalne i zaawansowane

#### validation.*
Sekcja kontroli jakości generowanych cech:
- validate_generated_features: true
- max_validation_time: 60
- cv_folds: 3
- cv_repeats: 1
- significance_level: 0.05
- min_samples_for_test: 10

#### advanced.*
Eksperymentalne funkcje i debugging:
- enable_neural_mcts: false (EXPERIMENTAL)
- enable_parallel_evaluation: false (PLANNED)
- enable_multi_objective: false (PLANNED)
- debug_mode: false
- debug_save_all_features: false
- debug_detailed_timing: false
- auto_recovery: true
- max_recovery_attempts: 3
- recovery_checkpoint_interval: 10