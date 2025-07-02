"""
Configuration Schema Validation using Pydantic

Comprehensive schema definitions for all MCTS configuration sections.
Validates types, ranges, logical constraints, and business rules.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import multiprocessing
import os


class LogLevel(str, Enum):
    """Valid logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SessionMode(str, Enum):
    """Valid session modes."""
    NEW = "new"
    CONTINUE = "continue"


class SelectionStrategy(str, Enum):
    """Valid MCTS selection strategies."""
    UCB1 = "ucb1"
    UCT = "uct"
    EPSILON_GREEDY = "epsilon_greedy"


class AutoGluonPreset(str, Enum):
    """Valid AutoGluon presets."""
    BEST_QUALITY = "best_quality"
    HIGH_QUALITY = "high_quality" 
    GOOD_QUALITY = "good_quality"
    MEDIUM_QUALITY = "medium_quality"
    OPTIMIZE_FOR_DEPLOYMENT = "optimize_for_deployment"
    INTERPRETABLE = "interpretable"


class DataBackend(str, Enum):
    """Valid data backends."""
    AUTO = "auto"
    PANDAS = "pandas"
    DUCKDB = "duckdb"


class ExportFormat(str, Enum):
    """Valid export formats."""
    PYTHON = "python"
    JSON = "json"
    HTML = "html"
    CSV = "csv"


# Configuration section schemas

class SessionConfig(BaseModel):
    """Session configuration validation."""
    mode: SessionMode = SessionMode.NEW
    max_iterations: int = Field(ge=1, le=100000, description="Maximum MCTS iterations")
    max_runtime_hours: float = Field(ge=0.1, le=168.0, description="Max runtime in hours (max 1 week)")
    checkpoint_interval: int = Field(ge=1, le=1000, description="Checkpoint every N iterations")
    auto_save: bool = True
    session_name: Optional[str] = Field(None, max_length=100, description="Human-readable session name")
    resume_session_id: Optional[str] = Field(None, description="Session ID to resume from")

    @field_validator('max_runtime_hours')
    @classmethod
    def validate_runtime(cls, v):
        if v <= 0:
            raise ValueError("Runtime must be positive")
        return v


class MCTSConfig(BaseModel):
    """MCTS algorithm configuration validation."""
    exploration_weight: float = Field(ge=0.1, le=10.0, description="UCB1 exploration weight (sqrt(2) recommended)")
    max_tree_depth: int = Field(ge=1, le=100, description="Maximum tree depth (memory constraint)")
    expansion_threshold: int = Field(ge=1, le=1000, description="Min visits before expansion")
    min_visits_for_best: int = Field(ge=1, le=1000, description="Min visits to consider for best path")
    ucb1_confidence: float = Field(ge=0.5, le=0.99, description="UCB1 confidence level")
    selection_strategy: SelectionStrategy = SelectionStrategy.UCB1
    max_children_per_node: int = Field(ge=1, le=50, description="Max children per node (branching factor)")
    expansion_budget: int = Field(ge=1, le=1000, description="Max expansions per iteration")
    max_nodes_in_memory: int = Field(ge=100, le=1000000, description="Memory limit for tree nodes")
    prune_threshold: float = Field(ge=0.0, le=1.0, description="Pruning threshold for low-performing nodes")

    @field_validator('exploration_weight')
    @classmethod
    def validate_exploration_weight(cls, v):
        if v == 0:
            raise ValueError("Exploration weight cannot be zero (would disable exploration)")
        return v

    @field_validator('prune_threshold')
    @classmethod
    def validate_prune_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Prune threshold must be between 0 and 1")
        return v


class AutoGluonConfig(BaseModel):
    """AutoGluon evaluation configuration validation."""
    dataset_name: Optional[str] = Field(None, description="Registered dataset name")
    target_metric: Optional[str] = Field(None, description="Target evaluation metric")
    included_model_types: List[str] = Field(default_factory=list, description="Model types to include")
    enable_gpu: bool = True
    train_size: float = Field(gt=0.1, lt=1.0, description="Training data fraction (10%-99%)")
    time_limit: int = Field(ge=10, le=7200, description="Time limit per evaluation in seconds")
    presets: AutoGluonPreset = AutoGluonPreset.MEDIUM_QUALITY
    num_bag_folds: int = Field(ge=0, le=20, description="Number of bagging folds (0=disable)")
    num_bag_sets: int = Field(ge=1, le=10, description="Number of bagging sets")
    holdout_frac: float = Field(ge=0.05, le=0.5, description="Holdout fraction (5%-50%)")
    verbosity: int = Field(ge=0, le=4, description="Verbosity level")
    sample_size: Optional[int] = Field(None, ge=100, le=1000000, description="Sample size for evaluation")

    @model_validator(mode='after')
    def validate_data_splits(self):
        train_size = self.train_size
        holdout_frac = self.holdout_frac
        
        # Logical constraint: train_size + holdout_frac should not exceed 1.0
        if train_size + holdout_frac > 0.95:  # Allow 5% margin for validation
            raise ValueError(f"train_size ({train_size}) + holdout_frac ({holdout_frac}) = {train_size + holdout_frac} > 0.95. "
                           "This leaves insufficient data for validation.")
        
        return self

    @field_validator('time_limit')
    @classmethod
    def validate_time_limit(cls, v):
        if v < 10:
            raise ValueError("Time limit too low - AutoGluon needs at least 10 seconds")
        return v


class FeatureSpaceConfig(BaseModel):
    """Feature space configuration validation."""
    max_features_per_node: int = Field(ge=10, le=100000, description="Max features per tree node")
    min_improvement_threshold: float = Field(ge=0.0, le=1.0, description="Min score improvement to accept")
    feature_timeout: int = Field(ge=10, le=3600, description="Feature generation timeout (seconds)")
    max_features_to_build: Optional[int] = Field(None, ge=1, le=1000000)
    max_features_per_iteration: Optional[int] = Field(None, ge=1, le=10000)
    feature_build_timeout: int = Field(ge=30, le=7200, description="Total build timeout (seconds)")
    cache_miss_limit: int = Field(ge=1, le=1000, description="Max cache misses before rebuild")
    
    # Generic operations flags
    generic_operations: Dict[str, bool] = Field(default_factory=dict)
    
    # Generic parameters with validation
    generic_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Category weights
    category_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Cache settings
    lazy_loading: bool = True
    cache_features: bool = True
    max_cache_size_mb: int = Field(ge=100, le=100000, description="Cache size in MB")
    cache_cleanup_threshold: float = Field(ge=0.1, le=0.9, description="Cache cleanup threshold")
    
    # Dynamic categories
    use_dynamic_categories: bool = True
    category_filter: Dict[str, List[str]] = Field(default_factory=dict)

    @field_validator('generic_params')
    @classmethod
    def validate_generic_params(cls, v):
        """Validate generic parameters with specific rules."""
        if 'polynomial_degree' in v:
            degree = v['polynomial_degree']
            if not isinstance(degree, int) or degree < 1 or degree > 6:
                raise ValueError("polynomial_degree must be integer between 1 and 6")
        
        if 'binning_bins' in v:
            bins = v['binning_bins']
            if not isinstance(bins, int) or bins < 2 or bins > 100:
                raise ValueError("binning_bins must be integer between 2 and 100")
        
        if 'groupby_columns' in v:
            if not isinstance(v['groupby_columns'], list):
                raise ValueError("groupby_columns must be a list")
        
        if 'aggregate_columns' in v:
            if not isinstance(v['aggregate_columns'], list):
                raise ValueError("aggregate_columns must be a list")
        
        return v

    @field_validator('category_weights')
    @classmethod
    def validate_category_weights(cls, v):
        """Validate category weights are reasonable."""
        for category, weight in v.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for {category} must be numeric")
            if weight < 0 or weight > 10:
                raise ValueError(f"Weight for {category} must be between 0 and 10")
        return v


class DatabaseConfig(BaseModel):
    """Database configuration validation."""
    path: str = Field(min_length=1, description="Database file path")
    type: str = Field(pattern=r'^(duckdb|sqlite)$', description="Database type")
    schema: str = Field(default="main", description="Database schema")
    backup_path: str = Field(default="data/backups/", description="Backup directory")
    backup_interval: int = Field(ge=1, le=10000, description="Backup every N operations")
    backup_prefix: str = Field(default="minotaur_backup_", max_length=50)
    max_history_size: int = Field(ge=1000, le=10000000, description="Max history records")
    max_backup_files: int = Field(ge=1, le=1000, description="Max backup files to keep")
    batch_size: int = Field(ge=1, le=10000, description="Batch size for operations")
    sync_mode: str = Field(pattern=r'^(NORMAL|FULL|OFF)$', default="NORMAL")
    journal_mode: str = Field(pattern=r'^(DELETE|TRUNCATE|PERSIST|MEMORY|WAL|OFF)$', default="WAL")
    auto_cleanup: bool = True
    cleanup_interval_hours: int = Field(ge=1, le=168, description="Cleanup interval (max 1 week)")
    retention_days: int = Field(ge=1, le=3650, description="Data retention (max 10 years)")


class LoggingConfig(BaseModel):
    """Logging configuration validation."""
    level: LogLevel = LogLevel.INFO
    log_file: str = Field(min_length=1, description="Main log file path")
    max_log_size_mb: int = Field(ge=1, le=1000, description="Max log file size in MB")
    backup_count: int = Field(ge=1, le=100, description="Number of log backup files")
    log_feature_code: bool = True
    log_timing: bool = True
    log_memory_usage: bool = True
    log_autogluon_details: bool = False
    progress_interval: int = Field(ge=1, le=1000, description="Progress logging interval")
    save_intermediate_results: bool = True
    timing_output_dir: str = Field(default="logs/timing", description="Timing data output directory")


class ResourcesConfig(BaseModel):
    """System resources configuration validation."""
    max_memory_gb: int = Field(ge=1, le=2048, description="Max memory usage in GB")
    memory_check_interval: int = Field(ge=1, le=300, description="Memory check interval (seconds)")
    force_gc_interval: int = Field(ge=1, le=1000, description="Forced garbage collection interval")
    use_gpu: bool = True
    max_cpu_cores: int = Field(ge=-1, le=256, description="Max CPU cores (-1 = all available)")
    autogluon_num_cpus: Optional[int] = Field(None, ge=1, le=256)
    max_disk_usage_gb: int = Field(ge=1, le=100000, description="Max disk usage in GB")
    temp_dir: str = Field(default="/tmp/mcts_features", description="Temporary directory")
    cleanup_temp_on_exit: bool = True

    @field_validator('max_cpu_cores')
    @classmethod
    def validate_cpu_cores(cls, v):
        if v == 0:
            raise ValueError("CPU cores cannot be 0 (use -1 for all available)")
        if v > multiprocessing.cpu_count():
            raise ValueError(f"Cannot use more CPU cores ({v}) than available ({multiprocessing.cpu_count()})")
        return v

    @model_validator(mode='after')
    def validate_autogluon_cpus(self):
        if self.autogluon_num_cpus is not None:
            max_cores = self.max_cpu_cores
            if max_cores > 0 and self.autogluon_num_cpus > max_cores:
                raise ValueError(f"AutoGluon CPU count ({self.autogluon_num_cpus}) cannot exceed max_cpu_cores ({max_cores})")
        return self


class DataConfig(BaseModel):
    """Data handling configuration validation."""
    backend: DataBackend = DataBackend.AUTO
    prefer_parquet: bool = True
    auto_convert_csv: bool = True
    dtype_optimization: bool = True
    memory_limit_mb: int = Field(ge=100, le=100000, description="Memory limit for data operations")
    use_small_dataset: bool = False
    small_dataset_size: int = Field(ge=100, le=1000000, description="Small dataset sample size")
    
    # DuckDB specific settings
    duckdb: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('duckdb')
    @classmethod
    def validate_duckdb_config(cls, v):
        """Validate DuckDB specific configuration."""
        if 'max_memory_gb' in v:
            memory = v['max_memory_gb']
            if not isinstance(memory, (int, float)) or memory < 0.1 or memory > 1024:
                raise ValueError("DuckDB max_memory_gb must be between 0.1 and 1024")
        
        if 'max_cached_features' in v:
            cached = v['max_cached_features']
            if not isinstance(cached, int) or cached < 10 or cached > 100000:
                raise ValueError("max_cached_features must be between 10 and 100000")
        
        return v


class LLMConfig(BaseModel):
    """LLM integration configuration validation."""
    enabled: bool = False
    provider: str = Field(pattern=r'^(openai|anthropic|local)$', default="openai")
    model: str = Field(min_length=1, description="Model name")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")
    trigger_interval: int = Field(ge=1, le=1000, description="Trigger every N iterations")
    trigger_on_plateau: bool = True
    plateau_threshold: int = Field(ge=1, le=100, description="Plateau detection threshold")
    max_features_per_request: int = Field(ge=1, le=20, description="Max features per LLM request")
    temperature: float = Field(ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(ge=100, le=8000, description="Max tokens per request")

    @model_validator(mode='after')
    def validate_api_key_env(self):
        """Check if API key environment variable exists when LLM is enabled."""
        if self.enabled and self.api_key_env not in os.environ:
            raise ValueError(f"LLM enabled but API key environment variable '{self.api_key_env}' not found")
        return self


class ExportConfig(BaseModel):
    """Export configuration validation."""
    formats: List[ExportFormat] = Field(default_factory=list, description="Export formats")
    python_output: str = Field(default="outputs/best_features_discovered.py")
    include_dependencies: bool = True
    include_documentation: bool = True
    code_style: str = Field(pattern=r'^(pep8|black|none)$', default="pep8")
    html_report: str = Field(default="outputs/discovery_report.html")
    include_plots: bool = True
    plot_format: str = Field(pattern=r'^(png|jpg|svg|pdf)$', default="png")
    include_analytics: bool = True
    output_dir: str = Field(default="outputs/reports")
    export_on_completion: bool = True
    export_on_improvement: bool = True
    export_interval: int = Field(ge=1, le=10000, description="Export every N iterations")

    @field_validator('formats')
    @classmethod
    def validate_formats(cls, v):
        if not v:
            raise ValueError("At least one export format must be specified")
        return v


class AnalyticsConfig(BaseModel):
    """Analytics configuration validation."""
    figure_size: List[int] = Field(default=[12, 8], min_items=2, max_items=2)
    dpi: int = Field(ge=50, le=300, default=100, description="Plot DPI")
    format: str = Field(pattern=r'^(png|jpg|svg|pdf)$', default="png")
    generate_charts: bool = True
    include_timing_analysis: bool = True

    @field_validator('figure_size')
    @classmethod
    def validate_figure_size(cls, v):
        if len(v) != 2:
            raise ValueError("figure_size must have exactly 2 elements [width, height]")
        if any(x <= 0 or x > 50 for x in v):
            raise ValueError("figure_size elements must be between 1 and 50")
        return v


class ValidationConfig(BaseModel):
    """Validation configuration."""
    validate_generated_features: bool = True
    max_validation_time: int = Field(ge=10, le=3600, description="Max validation time (seconds)")
    cv_folds: int = Field(ge=2, le=20, description="Cross-validation folds")
    cv_repeats: int = Field(ge=1, le=10, description="Cross-validation repeats")
    significance_level: float = Field(ge=0.01, le=0.1, description="Statistical significance level")
    min_samples_for_test: int = Field(ge=5, le=10000, description="Min samples for statistical tests")


class AdvancedConfig(BaseModel):
    """Advanced/experimental configuration."""
    enable_neural_mcts: bool = False
    enable_parallel_evaluation: bool = False
    enable_multi_objective: bool = False
    debug_mode: bool = False
    debug_save_all_features: bool = False
    debug_detailed_timing: bool = False
    auto_recovery: bool = True
    max_recovery_attempts: int = Field(ge=1, le=10, description="Max recovery attempts")
    recovery_checkpoint_interval: int = Field(ge=1, le=1000, description="Recovery checkpoint interval")


class MCTSConfigurationSchema(BaseModel):
    """Complete MCTS configuration schema."""
    test_mode: bool = False
    session: SessionConfig = Field(default_factory=SessionConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    autogluon: AutoGluonConfig = Field(default_factory=AutoGluonConfig)
    feature_space: FeatureSpaceConfig = Field(default_factory=FeatureSpaceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    resources: ResourcesConfig = Field(default_factory=ResourcesConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Forbid extra fields not defined in schema
        validate_assignment = True  # Validate on assignment
        arbitrary_types_allowed = False
        
    @model_validator(mode='after')
    def validate_global_constraints(self):
        """Cross-section validation rules."""
        
        # Memory consistency check
        if isinstance(self.resources, ResourcesConfig) and isinstance(self.data, DataConfig) and isinstance(self.feature_space, FeatureSpaceConfig):
            max_memory_gb = self.resources.max_memory_gb
            data_memory_mb = self.data.memory_limit_mb
            cache_memory_mb = self.feature_space.max_cache_size_mb
            
            total_memory_mb = data_memory_mb + cache_memory_mb
            if total_memory_mb > max_memory_gb * 1024 * 0.8:  # 80% of max memory
                raise ValueError(f"Data memory ({data_memory_mb}MB) + cache memory ({cache_memory_mb}MB) "
                               f"exceeds 80% of max memory ({max_memory_gb}GB)")
        
        # Session iteration vs runtime consistency
        if isinstance(self.session, SessionConfig) and isinstance(self.autogluon, AutoGluonConfig):
            max_iterations = self.session.max_iterations
            max_runtime_hours = self.session.max_runtime_hours
            time_limit_seconds = self.autogluon.time_limit
            
            # Rough estimate: each iteration takes at least time_limit seconds
            min_runtime_hours = (max_iterations * time_limit_seconds) / 3600
            if min_runtime_hours > max_runtime_hours:
                raise ValueError(f"Unrealistic time constraint: {max_iterations} iterations "
                               f"× {time_limit_seconds}s = {min_runtime_hours:.1f}h > {max_runtime_hours}h max runtime")
        
        return self


# Helper functions for validation

def validate_config_dict(config_dict: Dict[str, Any]) -> MCTSConfigurationSchema:
    """
    Validate a configuration dictionary against the schema.
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        Validated configuration model
        
    Raises:
        ValidationError: If validation fails
    """
    return MCTSConfigurationSchema(**config_dict)


def get_validation_errors(config_dict: Dict[str, Any]) -> List[str]:
    """
    Get list of validation errors without raising exception.
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        List of validation error messages
    """
    try:
        validate_config_dict(config_dict)
        return []
    except Exception as e:
        # Extract individual error messages from Pydantic ValidationError
        if hasattr(e, 'errors'):
            errors = []
            for error in e.errors():
                loc = ' -> '.join(str(x) for x in error['loc'])
                msg = error['msg']
                errors.append(f"{loc}: {msg}")
            return errors
        else:
            return [str(e)]


# Export commonly used types and functions
__all__ = [
    'MCTSConfigurationSchema',
    'validate_config_dict', 
    'get_validation_errors',
    'LogLevel',
    'SessionMode',
    'SelectionStrategy',
    'AutoGluonPreset',
    'DataBackend',
    'ExportFormat'
]