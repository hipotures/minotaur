# Tech Stack Selection - Final Version
## Kompletna specyfikacja technologii dla systemu

### Język programowania i środowisko

#### Python 3.11+ (Required: 3.11.0 - 3.12.x)
**Uzasadnienie wyboru**:
- **Ekosystem ML**: Najpełniejszy zestaw bibliotek (10,000+ pakietów ML/DS)
- **Performance**: Significant improvements in 3.11 (10-60% faster than 3.10)
- **Type hints**: Full support dla static typing i better IDE integration
- **Pattern matching**: Structural pattern matching (PEP 634) dla czystszego kodu
- **Error messages**: Enhanced error reporting dla szybszego debugowania

**Konfiguracja środowiska**:
```
Python configuration:
- Interpreter: CPython 3.11.8+ (recommended)
- Memory: -X dev mode w development
- Optimizations: -O w produkcji
- GC tuning: PYTHONGCTHRESHOLD=700,10,10
- Encoding: UTF-8 (PYTHONUTF8=1)
```

#### UV Package Manager (0.1.0+)
**Specyfikacja**:
```
UV benefits over pip/poetry:
- Installation speed: 10-100x faster
- Resolution speed: Instant dla locked deps
- Disk usage: Shared cache między projektami
- Compatibility: Drop-in pip replacement
- Features: Lock files, workspaces, scripts

Configuration:
- Cache location: ~/.cache/uv
- Index URL: https://pypi.org/simple
- Parallel downloads: 10
- Compile bytecode: Yes
- Verify hashes: Always
```

### Framework ML i ewaluacja

#### AutoGluon 1.0.0+ (Core ML Framework)
**Detailed configuration**:
```
AutoGluon setup:
Core components:
- autogluon.tabular: 1.0.0
- autogluon.core: 1.0.0
- autogluon.features: 1.0.0

Presets optimization:
- best_quality: 
  - Models: All available
  - Bagging: 10 folds
  - Stack levels: 2
  - Time: Unlimited
  
- medium_quality (default):
  - Models: GBM, XGB, CAT
  - Bagging: 5 folds
  - Stack levels: 1
  - Time: Soft limit
  
- optimize_for_deployment:
  - Models: Single best
  - Bagging: None
  - Stack levels: 0
  - Time: Minimal

Resource allocation:
- CPU: ag_args_fit={'num_cpus': os.cpu_count()}
- Memory: ag_args_fit={'memory_limit': 0.9}
- GPU: ag_args_ensemble={'use_cuda': torch.cuda.is_available()}
```

#### Gradient Boosting Implementations

**LightGBM 4.1.0+**:
```
Configuration:
- device_type: 'cpu' or 'cuda'
- num_threads: -1 (all available)
- deterministic: True (for reproducibility)
- force_col_wise: True (for wide datasets)
- force_row_wise: False

Optimizations:
- histogram_pool_size: 1024 MB
- max_bin: 255 (default) or 63 (speed)
- feature_pre_filter: True
- sparse_threshold: 0.8
- zero_as_missing: False

Memory settings:
- workspace_size_mb: 1024
- gpu_memory_fraction: 0.9
- min_data_in_leaf: 20
```

**XGBoost 2.0.3+**:
```
Configuration:
- tree_method: 'hist' (CPU) or 'gpu_hist' (GPU)
- predictor: 'auto'
- n_jobs: -1
- random_state: Fixed for reproducibility

GPU settings:
- gpu_id: 0
- max_bins: 256
- grow_policy: 'depthwise'
- sampling_method: 'gradient_based'

Optimizations:
- updater: 'grow_histmaker,prune'
- refresh_leaf: True
- process_type: 'default'
- multi_strategy: 'multi_output_tree'
```

**CatBoost 1.2.2+**:
```
Configuration:
- task_type: 'CPU' or 'GPU'
- thread_count: -1
- used_ram_limit: '15GB'
- gpu_ram_part: 0.95

Categorical handling:
- cat_features: Auto-detected
- one_hot_max_size: 10
- feature_calcers: ['BoW', 'NaiveBayes']
- text_processing: {'tokenizers': 'Space'}

Training params:
- langevin: True (better generalization)
- posterior_sampling: True
- boost_from_average: True
```

**TabNet (torch-tabnet 4.1.0)**:
```
Neural architecture:
- n_d: 64 (decision dim)
- n_a: 64 (attention dim)
- n_steps: 5 (sequential steps)
- gamma: 1.5 (relaxation)
- n_independent: 2
- n_shared: 2

Training config:
- batch_size: 1024
- virtual_batch_size: 128
- momentum: 0.02
- mask_type: 'entmax'
- lambda_sparse: 0.001

Optimization:
- optimizer: Adam
- lr_scheduler: ReduceLROnPlateau
- early_stopping_patience: 10
```

### Baza danych i persystencja

#### DuckDB 0.10.0+ (Primary Database)
**Production configuration**:
```
DuckDB settings:
Memory:
- memory_limit: '12GB'
- max_memory: '14GB'
- temp_directory: '/tmp/duckdb'

Performance:
- threads: 8
- enable_object_cache: true
- force_compression: 'zstd'
- checkpoint_threshold: '1GB'
- wal_autocheckpoint: '1GB'

Optimization:
- enable_optimizer: true
- enable_profiling: false
- explain_output: false
- force_index_join: false

Storage:
- default_block_size: 262144
- use_direct_io: true
- checkpoint_on_shutdown: true
- access_mode: 'read_write'
```

#### SQLAlchemy 2.0.25+ (Database Abstraction)
**Configuration details**:
```
SQLAlchemy setup:
Engine configuration:
- pool_class: QueuePool
- pool_size: 5
- max_overflow: 10
- pool_timeout: 30
- pool_recycle: 3600
- pool_pre_ping: True

Query optimization:
- echo: False (True in debug)
- echo_pool: False
- query_cache_size: 1200
- use_insertmanyvalues: True
- use_batch_mode: True

Connection args:
- connect_args: {
    'read_only': False,
    'config': {
      'memory_limit': '12GB',
      'threads': 8
    }
  }

Types configuration:
- use_native_datetime: True
- use_native_decimal: True
- use_native_uuid: True
```

### Biblioteki do przetwarzania danych

#### Pandas 2.1.4+ with PyArrow backend
**Optimization configuration**:
```
Pandas settings:
Options:
- pd.options.mode.copy_on_write = True
- pd.options.future.infer_string = True
- pd.options.plotting.backend = 'plotly'
- pd.options.mode.string_storage = 'pyarrow'

Performance:
- pd.options.compute.use_numba = True
- pd.options.mode.data_manager = 'pyarrow'
- pd.options.display.memory_usage = 'deep'

PyArrow integration:
- Default string dtype: 'string[pyarrow]'
- Parquet engine: 'pyarrow'
- Compression: 'zstd'
- Use_dictionary: True
- Row_group_size: 100_000
```

#### NumPy 1.26.3+ with MKL
**Performance settings**:
```
NumPy configuration:
Threading:
- OMP_NUM_THREADS: 8
- MKL_NUM_THREADS: 8
- NUMEXPR_NUM_THREADS: 8
- VECLIB_MAXIMUM_THREADS: 8

Memory:
- NPY_DISABLE_CPU_FEATURES: ''
- NPY_RELAXED_STRIDES_CHECKING: 1
- NPY_USE_HUGEPAGE: 1

BLAS/LAPACK:
- Backend: Intel MKL
- Interface: LP64
- Threading: OpenMP
```

#### SciPy 1.11.4+ 
**Optimization config**:
```
SciPy settings:
- Use BLAS: MKL
- Use LAPACK: MKL
- Threading layer: OpenMP
- Parallel algorithms: Enabled
```

### Interfejs użytkownika i wizualizacja

#### Rich 13.7.0+ (Terminal UI)
**Feature configuration**:
```
Rich components used:
- Console: Custom theme
- Table: BorderStyle.ROUNDED
- Progress: Multiple bars
- Tree: Collapsible nodes
- Syntax: Python highlighting
- Panel: Info boxes
- Layout: Split views

Theme customization:
- success: green bold
- warning: yellow
- error: red bold
- info: blue
- highlight: magenta
- muted: dim white

Console settings:
- force_terminal: True
- width: 120 (or auto)
- color_system: 'auto'
- highlight: True
- markup: True
```

#### Visualization Stack
**Matplotlib 3.8.2+**:
```
Configuration:
- Backend: 'Agg' (headless)
- DPI: 100 (screen), 300 (save)
- Style: 'seaborn-v0_8-darkgrid'
- Font: 'DejaVu Sans'
- Colormap: 'viridis'
```

**Seaborn 0.13.1+**:
```
Settings:
- context: 'notebook'
- palette: 'husl'
- font_scale: 1.2
- rc: {'figure.figsize': (10, 6)}
```

**Plotly 5.18.0+**:
```
Configuration:
- renderer: 'browser' or 'svg'
- theme: 'plotly_dark'
- config: {
    'displayModeBar': True,
    'toImageButtonOptions': {
      'format': 'png',
      'height': 800,
      'width': 1200
    }
  }
```

### Development i Testing Tools

#### Testing Framework Configuration

**pytest 7.4.4+**:
```
pytest.ini configuration:
[pytest]
minversion = 7.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --strict-markers
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    --maxfail=3
    --tb=short
    -p no:warnings
markers =
    slow: marks tests as slow
    integration: integration tests
    unit: unit tests
    gpu: requires GPU
```

**Test plugins**:
```
pytest-cov==4.1.0
pytest-xdist==3.5.0
pytest-timeout==2.2.0
pytest-mock==3.12.0
pytest-asyncio==0.23.3
pytest-benchmark==4.0.0
hypothesis==6.92.1
```

#### Code Quality Tools

**Black configuration**:
```
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

**isort configuration**:
```
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
split_on_trailing_comma = true
```

**mypy configuration**:
```
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = false
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
```

### Infrastructure i Deployment

#### Container Configuration

**Docker settings**:
```dockerfile
# Build stage
FROM python:3.11-slim-bullseye AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime stage  
FROM python:3.11-slim-bullseye
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_SYSTEM_PYTHON=1

# Security
RUN useradd -m -u 1000 minotaur
USER minotaur

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"
```

**Docker Compose**:
```yaml
version: '3.9'
services:
  app:
    build: .
    mem_limit: 16g
    cpus: '8.0'
    environment:
      - PYTHONUNBUFFERED=1
      - OMP_NUM_THREADS=8
    volumes:
      - data:/app/data
      - cache:/app/.cache
    deploy:
      resources:
        limits:
          cpus: '8.0'
          memory: 16G
        reservations:
          cpus: '4.0'
          memory: 8G
```

#### Kubernetes Resources

**Deployment spec**:
```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
    ephemeral-storage: "10Gi"
  limits:
    memory: "16Gi"
    cpu: "8"
    ephemeral-storage: "50Gi"
    nvidia.com/gpu: "1"  # Optional

livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Monitoring i Observability

#### Logging Stack
```
Configuration:
- structlog==24.1.0
  - Processors: TimeStamper, add_log_level, format_exc_info
  - Renderer: JSONRenderer
  - Context_class: dict
  
- python-json-logger==2.0.7
  - Format: '%(timestamp)s %(level)s %(name)s %(message)s'
  
Log shipping:
- Fluent Bit → Elasticsearch → Kibana
- Retention: 30 days
- Index pattern: minotaur-*
```

#### Metrics Stack
```
Prometheus configuration:
- prometheus-client==0.19.0
- Scrape interval: 15s
- Metrics port: 9090

Key metrics:
- mcts_iterations_total
- features_generated_total
- evaluation_duration_seconds
- memory_usage_bytes
- cache_hit_ratio

Grafana dashboards:
- System Overview
- MCTS Performance
- Feature Generation
- Model Training
- Error Rates
```

#### Tracing
```
OpenTelemetry setup:
- opentelemetry-api==1.22.0
- opentelemetry-sdk==1.22.0
- opentelemetry-instrumentation==0.43b0

Exporters:
- Jaeger: Port 6831 (UDP)
- Zipkin: Port 9411 (HTTP)

Auto-instrumentation:
- requests
- sqlalchemy
- redis
- celery
```

### Security Dependencies

#### Authentication/Authorization
```
Libraries:
- python-jose[cryptography]==3.3.0
- passlib[bcrypt]==1.7.4
- python-multipart==0.0.6

JWT configuration:
- Algorithm: RS256
- Token lifetime: 3600s
- Refresh token: 86400s
```

#### Security Scanning
```
Tools:
- bandit==1.7.6
- safety==3.0.1
- pip-audit==2.6.3

Pre-commit hooks:
- detect-secrets==1.4.0
- sqlfluff==3.0.0
```

### Performance Optimization Libraries

#### Compilation/JIT
```
Optional accelerators:
- numba==0.59.0
  - Target: CPU/CUDA
  - Cache: True
  - Parallel: True
  
- cython==3.0.8
  - Language_level: 3
  - Boundscheck: False
  - Wraparound: False
```

#### Profiling
```
Development tools:
- memory-profiler==0.61.0
- line-profiler==4.1.2
- py-spy==0.3.14
- scalene==1.5.38
```

### Data Format Libraries

#### File I/O
```
Supported formats:
- pyarrow==14.0.2 (Parquet, Feather)
- fastparquet==2023.10.1 (Alternative)
- openpyxl==3.1.2 (Excel)
- xlrd==2.0.1 (Legacy Excel)
- tables==3.9.2 (HDF5)
```

#### Compression
```
Libraries:
- zstandard==0.22.0 (Primary)
- lz4==4.3.3 (Speed)
- brotli==1.1.0 (Ratio)
- python-snappy==0.6.1 (Compatibility)
```

### Utility Libraries

#### CLI/Config
```
Configuration:
- pydantic==2.5.3
- pydantic-settings==2.1.0
- python-dotenv==1.0.0
- omegaconf==2.3.0
```

#### Data Validation
```
Validation stack:
- pydantic==2.5.3
- marshmallow==3.20.1
- cerberus==1.3.5
- jsonschema==4.20.0
```

#### Async Support
```
Async libraries:
- asyncio (stdlib)
- aiofiles==23.2.1
- aiohttp==3.9.1
- asyncpg==0.29.0
```

### Version Management

#### Version constraints
```
[tool.uv]
python = ">=3.11,<3.13"
dependencies = [
    "autogluon>=1.0.0,<2.0.0",
    "pandas>=2.1.0,<3.0.0",
    "numpy>=1.26.0,<2.0.0",
    "duckdb>=0.10.0,<1.0.0",
    "sqlalchemy>=2.0.0,<3.0.0",
]
```

#### Lock file management
```
Strategy:
- Lock files: uv.lock
- Update frequency: Monthly
- Security updates: Immediate
- Breaking changes: Major version only
```

### Hardware Requirements

#### Minimum Configuration
```
CPU: 4 cores (x86_64)
RAM: 8 GB
Storage: 50 GB SSD
GPU: None (CPU only)
OS: Linux/macOS/Windows
Python: 3.11+
```

#### Recommended Configuration
```
CPU: 8+ cores (x86_64)
RAM: 16-32 GB
Storage: 200 GB NVMe SSD
GPU: NVIDIA 8GB+ VRAM
OS: Ubuntu 22.04 LTS
Python: 3.11.8
```

#### Production Configuration
```
CPU: 16+ cores (x86_64)
RAM: 64 GB ECC
Storage: 1 TB NVMe RAID
GPU: NVIDIA A100 40GB
OS: Ubuntu 22.04 LTS
Python: 3.11.8
Network: 10 Gbps
```

### Cloud Provider Specifics

#### AWS
```
Services used:
- EC2: g4dn.xlarge minimum
- EBS: gp3 volumes
- S3: Feature storage
- RDS: PostgreSQL option
- EKS: Kubernetes hosting
- CloudWatch: Monitoring
```

#### GCP
```
Services used:
- Compute Engine: n1-standard-4
- Persistent Disk: SSD
- Cloud Storage: Feature bucket
- Cloud SQL: PostgreSQL
- GKE: Kubernetes hosting
- Cloud Monitoring: Metrics
```

#### Azure
```
Services used:
- Virtual Machines: Standard_D4s_v3
- Managed Disks: Premium SSD
- Blob Storage: Feature store
- Database for PostgreSQL
- AKS: Kubernetes hosting
- Monitor: Observability
```