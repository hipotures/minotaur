"""
Configuration Validation and Compatibility System

Provides comprehensive validation for MCTS configuration including:
1. Value validation using Pydantic schemas
2. Configuration compatibility checking for session resumption  
3. Critical/Warning/Safe parameter categorization
4. Configuration hash calculation for change detection
"""

import hashlib
import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from pydantic import ValidationError

from .config_schema import (
    MCTSConfigurationSchema, 
    validate_config_dict, 
    get_validation_errors
)

logger = logging.getLogger(__name__)


class CompatibilityLevel(str, Enum):
    """Configuration compatibility levels."""
    COMPATIBLE = "compatible"     # Safe to resume session
    WARNING = "warning"          # Requires --force-resume flag
    INCOMPATIBLE = "incompatible" # Cannot resume session


@dataclass
class CompatibilityResult:
    """Result of configuration compatibility check."""
    level: CompatibilityLevel
    changes: List[str]           # List of changed parameters
    critical_changes: List[str]  # Critical incompatible changes
    warning_changes: List[str]   # Warning-level changes
    messages: List[str]          # Human-readable messages
    
    @property
    def is_compatible(self) -> bool:
        """Check if configuration is compatible."""
        return self.level == CompatibilityLevel.COMPATIBLE
    
    @property
    def requires_force(self) -> bool:
        """Check if --force-resume is required."""
        return self.level == CompatibilityLevel.WARNING
    
    @property
    def is_blocked(self) -> bool:
        """Check if resumption is blocked."""
        return self.level == CompatibilityLevel.INCOMPATIBLE


@dataclass 
class ValidationResult:
    """Result of configuration value validation."""
    is_valid: bool
    errors: List[str]            # Validation error messages
    warnings: List[str]          # Non-blocking warnings
    config_model: Optional[MCTSConfigurationSchema] = None
    
    @property
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return bool(self.errors)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return bool(self.warnings)


class ConfigValidator:
    """
    Configuration validator for MCTS system.
    
    Handles both value validation (using Pydantic) and compatibility checking
    for session resumption scenarios.
    """
    
    # Critical parameters that block session resumption when changed
    CRITICAL_PARAMETERS = {
        # MCTS Algorithm - Core behavior
        'mcts.exploration_weight',
        'mcts.max_tree_depth', 
        'mcts.expansion_threshold',
        'mcts.selection_strategy',
        'mcts.max_children_per_node',
        'mcts.ucb1_confidence',
        
        # Dataset and Evaluation - Data compatibility
        'autogluon.dataset_name',
        'autogluon.target_metric',
        'autogluon.train_size',
        'autogluon.holdout_frac',
        
        # Feature Operations - Available feature space
        'feature_space.generic_operations',
        'feature_space.generic_params.polynomial_degree',
        'feature_space.generic_params.binning_bins',
        'feature_space.generic_params.groupby_columns',
        'feature_space.generic_params.aggregate_columns',
    }
    
    # Warning parameters that require --force-resume but don't block
    WARNING_PARAMETERS = {
        # Performance and Resource Limits
        'mcts.expansion_budget',
        'mcts.max_nodes_in_memory',
        'session.max_iterations',  # Can be safely increased
        'session.max_runtime_hours',  # Can be safely increased
        
        # AutoGluon Performance Settings
        'autogluon.time_limit',
        'autogluon.sample_size',
        'autogluon.presets',
        'autogluon.num_bag_folds',
        'autogluon.verbosity',
        
        # Feature Space Performance
        'feature_space.max_features_per_node',
        'feature_space.feature_timeout',
        'feature_space.feature_build_timeout',
        'feature_space.min_improvement_threshold',
        
        # Resource Constraints
        'resources.max_memory_gb',
        'resources.max_cpu_cores',
        'resources.max_disk_usage_gb',
        
        # Data Processing
        'data.memory_limit_mb',
        'data.use_small_dataset',
        'data.small_dataset_size',
    }
    
    # Safe parameters that can be changed without issues
    SAFE_PARAMETERS = {
        # Caching and Optimization (no algorithmic impact)
        'feature_space.max_cache_size_mb',
        'feature_space.cache_cleanup_threshold',
        'feature_space.lazy_loading',
        'feature_space.cache_features',
        
        # Logging and Output
        'logging.level',
        'logging.log_file',
        'logging.max_log_size_mb',
        'logging.backup_count',
        'logging.log_feature_code',
        'logging.log_timing',
        'logging.log_memory_usage',
        'logging.progress_interval',
        
        # Database Operational Settings
        'database.backup_interval',
        'database.backup_prefix',
        'database.max_backup_files',
        'database.batch_size',
        'database.auto_cleanup',
        'database.cleanup_interval_hours',
        
        # Export and Analytics
        'export.formats',
        'export.python_output',
        'export.include_dependencies',
        'export.include_documentation',
        'export.include_plots',
        'export.export_on_completion',
        'export.export_interval',
        
        # Analytics
        'analytics.figure_size',
        'analytics.dpi',
        'analytics.format',
        'analytics.generate_charts',
        'analytics.include_timing_analysis',
        
        # Advanced/Debug Settings  
        'advanced.debug_mode',
        'advanced.debug_save_all_features',
        'advanced.debug_detailed_timing',
        'advanced.auto_recovery',
        
        # LLM Integration (experimental)
        'llm.enabled',
        'llm.provider',
        'llm.model',
        'llm.temperature',
        'llm.trigger_interval',
        'llm.max_features_per_request',
        
        # Validation Settings
        'validation.validate_generated_features',
        'validation.max_validation_time',
        'validation.cv_folds',
        'validation.significance_level',
    }
    
    def __init__(self):
        """Initialize configuration validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_values(self, config_dict: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration values using Pydantic schema.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        try:
            # Attempt validation with Pydantic
            config_model = validate_config_dict(config_dict)
            
            # Check for warnings (non-blocking issues)
            warnings = self._check_configuration_warnings(config_dict)
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=warnings,
                config_model=config_model
            )
            
        except ValidationError as e:
            # Extract detailed error messages
            errors = []
            for error in e.errors():
                loc = ' -> '.join(str(x) for x in error['loc'])
                msg = error['msg']
                errors.append(f"{loc}: {msg}")
            
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=[],
                config_model=None
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unexpected validation error: {str(e)}"],
                warnings=[],
                config_model=None
            )
    
    def calculate_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """
        Calculate hash of critical configuration parameters.
        
        Only includes parameters that affect algorithm correctness and
        data compatibility. Changes to these parameters make sessions
        incompatible.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SHA256 hash of critical parameters
        """
        critical_config = {}
        
        # Extract critical parameters only
        for param_path in self.CRITICAL_PARAMETERS:
            value = self._get_nested_value(config_dict, param_path)
            if value is not None:
                critical_config[param_path] = value
        
        # Create deterministic hash
        config_json = json.dumps(critical_config, sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode('utf-8')).hexdigest()
    
    def check_compatibility(self, 
                          stored_config: Dict[str, Any],
                          current_config: Dict[str, Any]) -> CompatibilityResult:
        """
        Check compatibility between stored and current configuration.
        
        Args:
            stored_config: Configuration stored in session
            current_config: Current configuration to validate
            
        Returns:
            CompatibilityResult with compatibility level and details
        """
        changes = []
        critical_changes = []
        warning_changes = []
        messages = []
        
        # Find all changed parameters
        all_changes = self._find_config_changes(stored_config, current_config)
        
        # Categorize changes by severity
        for param_path, (old_val, new_val) in all_changes.items():
            change_desc = f"{param_path}: {old_val} → {new_val}"
            changes.append(change_desc)
            
            if param_path in self.CRITICAL_PARAMETERS:
                critical_changes.append(change_desc)
                messages.append(f"🚫 Critical change: {change_desc}")
                
            elif param_path in self.WARNING_PARAMETERS:
                warning_changes.append(change_desc)
                messages.append(f"⚠️  Warning change: {change_desc}")
                
            elif param_path in self.SAFE_PARAMETERS:
                messages.append(f"✅ Safe change: {change_desc}")
                
            else:
                # Unknown parameter - treat as warning
                warning_changes.append(change_desc)
                messages.append(f"❓ Unknown parameter change: {change_desc}")
        
        # Determine compatibility level
        if critical_changes:
            level = CompatibilityLevel.INCOMPATIBLE
            messages.insert(0, "❌ Session resumption BLOCKED due to critical configuration changes")
            messages.append("💡 Start a new session with --new-session")
            
        elif warning_changes:
            level = CompatibilityLevel.WARNING
            messages.insert(0, "⚠️  Configuration changes detected that may affect results")
            messages.append("💡 Use --force-resume to continue anyway")
            
        else:
            level = CompatibilityLevel.COMPATIBLE
            if changes:
                messages.insert(0, "✅ Configuration changes are safe for session resumption")
            else:
                messages.insert(0, "✅ No configuration changes detected")
        
        return CompatibilityResult(
            level=level,
            changes=changes,
            critical_changes=critical_changes,
            warning_changes=warning_changes,
            messages=messages
        )
    
    def _find_config_changes(self, 
                           old_config: Dict[str, Any],
                           new_config: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Find all changes between two configuration dictionaries.
        
        Returns:
            Dictionary mapping parameter paths to (old_value, new_value) tuples
        """
        changes = {}
        
        # Check all known parameters
        all_params = self.CRITICAL_PARAMETERS | self.WARNING_PARAMETERS | self.SAFE_PARAMETERS
        
        for param_path in all_params:
            old_val = self._get_nested_value(old_config, param_path)
            new_val = self._get_nested_value(new_config, param_path)
            
            if old_val != new_val:
                changes[param_path] = (old_val, new_val)
        
        return changes
    
    def _get_nested_value(self, config_dict: Dict[str, Any], param_path: str) -> Any:
        """
        Get nested value from configuration dictionary using dot notation.
        
        Args:
            config_dict: Configuration dictionary
            param_path: Dot-separated path (e.g., 'mcts.exploration_weight')
            
        Returns:
            Value at the specified path, or None if not found
        """
        try:
            keys = param_path.split('.')
            value = config_dict
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return value
        except (KeyError, TypeError, AttributeError):
            return None
    
    def _check_configuration_warnings(self, config_dict: Dict[str, Any]) -> List[str]:
        """
        Check for configuration warnings (non-blocking issues).
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Performance warnings
        mcts_config = config_dict.get('mcts', {})
        if mcts_config.get('max_tree_depth', 8) > 20:
            warnings.append("Very deep MCTS tree (>20) may cause memory issues")
        
        if mcts_config.get('expansion_budget', 20) > 100:
            warnings.append("High expansion budget (>100) may slow down iterations")
        
        # Memory warnings
        resources = config_dict.get('resources', {})
        feature_space = config_dict.get('feature_space', {})
        
        max_memory = resources.get('max_memory_gb', 16)
        cache_size_mb = feature_space.get('max_cache_size_mb', 2048)
        
        if cache_size_mb > max_memory * 1024 * 0.5:  # Cache > 50% of total memory
            warnings.append(f"Feature cache ({cache_size_mb}MB) is >50% of total memory ({max_memory}GB)")
        
        # Session duration warnings
        session = config_dict.get('session', {})
        autogluon = config_dict.get('autogluon', {})
        
        max_iterations = session.get('max_iterations', 20)
        time_limit = autogluon.get('time_limit', 60)
        
        estimated_hours = (max_iterations * time_limit) / 3600
        if estimated_hours > 24:
            warnings.append(f"Estimated runtime ({estimated_hours:.1f}h) is very long (>24h)")
        
        return warnings
    
    def get_parameter_category(self, param_path: str) -> str:
        """
        Get the category of a configuration parameter.
        
        Args:
            param_path: Dot-separated parameter path
            
        Returns:
            Parameter category: 'critical', 'warning', 'safe', or 'unknown'
        """
        if param_path in self.CRITICAL_PARAMETERS:
            return 'critical'
        elif param_path in self.WARNING_PARAMETERS:
            return 'warning'
        elif param_path in self.SAFE_PARAMETERS:
            return 'safe'
        else:
            return 'unknown'
    
    def get_parameter_info(self) -> Dict[str, List[str]]:
        """
        Get information about parameter categorization.
        
        Returns:
            Dictionary with lists of parameters by category
        """
        return {
            'critical': sorted(list(self.CRITICAL_PARAMETERS)),
            'warning': sorted(list(self.WARNING_PARAMETERS)),
            'safe': sorted(list(self.SAFE_PARAMETERS))
        }


# Convenience functions

def validate_configuration(config_dict: Dict[str, Any]) -> ValidationResult:
    """
    Validate configuration values.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        ValidationResult
    """
    validator = ConfigValidator()
    return validator.validate_values(config_dict)


def check_config_compatibility(stored_config: Dict[str, Any],
                             current_config: Dict[str, Any]) -> CompatibilityResult:
    """
    Check configuration compatibility for session resumption.
    
    Args:
        stored_config: Stored configuration from session
        current_config: Current configuration
        
    Returns:
        CompatibilityResult
    """
    validator = ConfigValidator()
    return validator.check_compatibility(stored_config, current_config)


def calculate_configuration_hash(config_dict: Dict[str, Any]) -> str:
    """
    Calculate configuration hash for change detection.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        SHA256 hash string
    """
    validator = ConfigValidator()
    return validator.calculate_config_hash(config_dict)


# Export public interface
__all__ = [
    'ConfigValidator',
    'CompatibilityLevel',
    'CompatibilityResult', 
    'ValidationResult',
    'validate_configuration',
    'check_config_compatibility',
    'calculate_configuration_hash'
]