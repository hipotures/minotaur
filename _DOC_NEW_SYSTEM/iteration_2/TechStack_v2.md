# Tech Stack Selection v2
## Szczegółowy wybór technologii dla systemu

### Język programowania i środowisko

#### Python 3.11+
Wybrany jako główny język implementacji ze względu na:
- **Ekosystem ML**: Najbogatszy zestaw bibliotek do machine learning i data science
- **Wydajność**: Wystarczająca dla prototypowania z możliwością optymalizacji (Cython, Numba)
- **Produktywność**: Szybki rozwój i łatwe utrzymanie kodu
- **Społeczność**: Ogromna baza użytkowników i dostępność ekspertów
- **Integracje**: Natywne wsparcie dla wszystkich głównych bibliotek ML

#### Zarządzanie środowiskiem
- **UV**: Nowoczesny, szybki package manager (10-100x szybszy od pip)
- **pyproject.toml**: Standardowa konfiguracja projektu (PEP 517/518)
- **Virtual environments**: Izolacja zależności per projekt
- **Requirements pinning**: Dokładne wersje dla reprodukowalności

### Framework ML i ewaluacja

#### AutoGluon 1.0+
Główny framework do automatycznej ewaluacji modeli:
- **Ensemble learning**: Automatyczne łączenie wielu modeli
- **Hyperparameter optimization**: Wbudowane dostrajanie parametrów
- **Feature preprocessing**: Automatyczna obsługa różnych typów danych
- **Time budgeting**: Kontrola czasu treningu
- **Custom metrics**: Wsparcie dla własnych metryk (MAP@K)

#### Modele bazowe
- **LightGBM 3.3+**: Najszybszy gradient boosting, efektywna pamięć
- **XGBoost 2.0+**: GPU support, najlepsza wydajność
- **CatBoost 1.2+**: Natywna obsługa kategorii, mniej overfittingu
- **TabNet (PyTorch 2.0+)**: Neural networks dla danych tabelarycznych

#### Dodatkowe biblioteki ML
- **Scikit-learn 1.1+**: Preprocessing, metryki, utilities
- **Optuna 3.0+**: Zaawansowana optymalizacja hiperparametrów
- **SHAP**: Interpretability dla feature importance

### Baza danych i persystencja

#### DuckDB 0.9+
Analityczna baza danych jako główny storage:
- **Embedded**: Nie wymaga osobnego serwera
- **Columnar storage**: Optymalna dla zapytań analitycznych
- **Parquet support**: Natywne wsparcie dla efektywnego formatu
- **SQL compliance**: Pełne wsparcie standardu SQL
- **Performance**: 10-100x szybsza od SQLite dla analityki

#### Warstwa abstrakcji
- **SQLAlchemy 2.0+**: ORM i Core dla abstrakcji DB
- **Connection pooling**: Zarządzanie połączeniami
- **Migration support**: Alembic dla wersjonowania schematu
- **Multi-dialect**: Możliwość zmiany backendu (PostgreSQL, MySQL)

### Biblioteki do przetwarzania danych

#### Pandas 2.0+
Główna biblioteka do manipulacji danymi:
- **DataFrame API**: Intuicyjne operacje na danych tabelarycznych
- **PyArrow backend**: 10x szybsze operacje string
- **Copy-on-write**: Efektywniejsze wykorzystanie pamięci
- **Time series**: Bogate wsparcie dla danych czasowych

#### NumPy 1.24+
Obliczenia numeryczne i operacje macierzowe:
- **Vectorization**: Szybkie operacje na tablicach
- **Broadcasting**: Efektywne operacje element-wise
- **Random sampling**: Wysokiej jakości generatory
- **Linear algebra**: Pełne wsparcie BLAS/LAPACK

#### SciPy 1.10+
Zaawansowane funkcje naukowe:
- **Statistics**: Rozkłady, testy statystyczne
- **Optimization**: Algorytmy optymalizacji
- **Signal processing**: Filtry, transformaty
- **Sparse matrices**: Efektywne struktury rzadkie

### Interfejs użytkownika i wizualizacja

#### CLI Framework
- **Click 8.1+**: Deklaratywne definiowanie komend
- **Rich 13.0+**: Piękne formatowanie terminal output
  - Tabele z ramkami i kolorami
  - Progress bars z ETA
  - Syntax highlighting dla kodu
  - Tree views dla hierarchii
- **Typer**: Type hints dla CLI (opcjonalnie)

#### Wizualizacja danych
- **Matplotlib 3.7+**: Podstawowe wykresy, pełna kontrola
- **Seaborn 0.12+**: Statystyczne wizualizacje
- **Plotly 5.0+**: Interaktywne wykresy HTML
- **Altair**: Deklaratywne wizualizacje (opcjonalnie)

### Narzędzia developerskie

#### Testing Framework
- **pytest 7.0+**: Najpopularniejszy framework testowy
  - Fixtures dla reużywalnego setup
  - Parametrized tests
  - Plugin ecosystem
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel test execution
- **pytest-mock**: Mocking utilities
- **Hypothesis**: Property-based testing

#### Code Quality
- **Black**: Automatyczne formatowanie kodu
- **isort**: Sortowanie importów
- **flake8**: Linting i style checking
- **mypy**: Static type checking
- **pre-commit**: Git hooks dla quality checks

#### Profiling i debugging
- **memory-profiler**: Analiza użycia pamięci
- **line-profiler**: Profiling line-by-line
- **py-spy**: Sampling profiler (low overhead)
- **ipdb**: Enhanced debugger

### Infrastruktura i deployment

#### Containerization
- **Docker**: Konteneryzacja aplikacji
  - Multi-stage builds dla małych obrazów
  - Layer caching dla szybkich buildów
  - Health checks
- **Docker Compose**: Orchestracja lokalnego środowiska
- **Kubernetes ready**: Przygotowane dla K8s deployment

#### CI/CD
- **GitHub Actions**: Automatyzacja workflow
  - Matrix testing (multiple Python versions)
  - Automated releases
  - Security scanning
- **tox**: Test automation across environments
- **Poetry/Hatch**: Modern Python packaging

#### Monitoring i observability
- **structlog**: Structured logging
- **OpenTelemetry**: Distributed tracing ready
- **Prometheus client**: Metrics export
- **Sentry SDK**: Error tracking (opcjonalnie)

### Uniwersalne abstrakcje i dobre praktyki

#### Database Abstraction
```
Aplikacja
    ↓
Repository Pattern (abstract interface)
    ↓
SQLAlchemy Core (not ORM for performance)
    ↓
Database-specific optimizations
    ↓
[DuckDB | PostgreSQL | MySQL]
```

#### Feature Engineering Abstraction
```
Feature Request
    ↓
Operation Registry (plugin system)
    ↓
Generic Operations | Domain Operations
    ↓
Signal Detection & Caching Layer
    ↓
Generated Features
```

#### Connection Management
- **Connection pooling**: Min 2, Max 10 connections
- **Retry logic**: Exponential backoff (1s, 2s, 4s)
- **Circuit breaker**: Fail fast po 3 próbach
- **Health checks**: Ping co 30s

#### Security Best Practices
- **Input sanitization**: Wszystkie user inputs
- **Parameterized queries**: Żadnego string concatenation
- **Secrets management**: Environment variables, nie hardcode
- **Least privilege**: Minimalne uprawnienia DB
- **Audit logging**: Wszystkie operacje modyfikujące

### Konfiguracja i zarządzanie

#### Configuration Management
- **YAML**: Human-readable config files
- **Pydantic**: Type-safe settings z walidacją
- **python-dotenv**: Environment variables
- **ConfigArgParse**: CLI args + config files

#### Logging Configuration
```yaml
logging:
  level: INFO
  format: "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
  handlers:
    file:
      class: RotatingFileHandler
      max_bytes: 10485760  # 10MB
      backup_count: 5
    console:
      class: StreamHandler
      formatter: colored
```

### Wymagania systemowe

#### Minimalne
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB SSD
- OS: Linux/macOS/Windows (WSL2)

#### Rekomendowane
- CPU: 8+ cores
- RAM: 16-32 GB
- Storage: 200 GB NVMe SSD
- GPU: NVIDIA with 8GB+ VRAM (optional)

### Roadmap technologiczny

#### Phase 1 (MVP)
- Single machine execution
- Basic feature operations
- CLI interface only

#### Phase 2
- Distributed computing (Ray/Dask)
- REST API
- Web dashboard

#### Phase 3
- Cloud native (Kubernetes)
- Multi-tenancy
- Real-time monitoring