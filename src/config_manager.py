"""Secure Configuration Management for Minotaur MCTS System.

This module provides:
- Environment variable management for sensitive data
- Configuration schema validation
- Secure config loading with override support
- Secrets redaction in logs
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from copy import deepcopy
from jsonschema import validate, ValidationError, Draft7Validator

from .security import InputValidator, setup_secure_logging

logger = setup_secure_logging(__name__)


# Configuration schema for validation
CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["database", "mcts", "session"],
    "properties": {
        "test_mode": {"type": "boolean"},
        "database": {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "pattern": "^[a-zA-Z0-9_/.-]+\\.db$"},
                "backup_path": {"type": "string"},
                "type": {"type": "string", "enum": ["duckdb", "sqlite"]},
                "schema": {"type": "string"}
            }
        },
        "mcts": {
            "type": "object",
            "required": ["exploration_weight", "max_tree_depth"],
            "properties": {
                "exploration_weight": {"type": "number", "minimum": 0},
                "max_tree_depth": {"type": "integer", "minimum": 1},
                "selection_strategy": {"type": "string", "enum": ["ucb1", "thompson"]},
                "enable_pruning": {"type": "boolean"},
                "pruning_threshold": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "session": {
            "type": "object",
            "required": ["mode", "max_iterations"],
            "properties": {
                "mode": {"type": "string", "enum": ["new", "continue", "resume_best"]},
                "max_iterations": {"type": "integer", "minimum": 1},
                "max_runtime_hours": {"type": "number", "minimum": 0}
            }
        },
        "autogluon": {
            "type": "object",
            "properties": {
                "train_path": {"type": "string"},
                "test_path": {"type": ["string", "null"]},
                "target_column": {"type": "string"},
                "target_metric": {"type": "string"}
            }
        },
        "data": {
            "type": "object",
            "properties": {
                "backend": {"type": "string", "enum": ["pandas", "duckdb"]},
                "cache_dir": {"type": "string"},
                "duckdb": {
                    "type": "object",
                    "properties": {
                        "db_dir": {"type": "string"},
                        "connection_pool_size": {"type": "integer", "minimum": 1},
                        "max_memory_gb": {"type": "string"}
                    }
                }
            }
        }
    }
}


class SecureConfigManager:
    """Manages configuration with security best practices."""
    
    # Environment variable mappings for sensitive data
    ENV_MAPPINGS = {
        'autogluon.api_key': 'MINOTAUR_API_KEY',
        'database.connection_string': 'MINOTAUR_DB_CONNECTION',
        'logging.remote_endpoint': 'MINOTAUR_LOG_ENDPOINT',
        'cloud.aws_access_key': 'AWS_ACCESS_KEY_ID',
        'cloud.aws_secret_key': 'AWS_SECRET_ACCESS_KEY'
    }
    
    def __init__(self, base_config_path: Union[str, Path]):
        """Initialize config manager with base configuration file.
        
        Args:
            base_config_path: Path to base configuration YAML file
        """
        self.base_config_path = Path(base_config_path)
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_config_path}")
        
        self.base_config = self._load_yaml_file(self.base_config_path)
        self.merged_config = deepcopy(self.base_config)
        self._apply_env_overrides()
    
    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        """Safely load YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            with open(path, 'r') as f:
                # Use safe_load to prevent arbitrary code execution
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ValueError(f"Config file must contain a dictionary, got {type(config)}")
            
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {path}: {e}")
            raise
    
    def merge_override(self, override_path: Union[str, Path]) -> None:
        """Merge override configuration file.
        
        Args:
            override_path: Path to override configuration file
        """
        override_path = Path(override_path)
        if not override_path.exists():
            raise FileNotFoundError(f"Override config not found: {override_path}")
        
        override_config = self._load_yaml_file(override_path)
        self.merged_config = self._deep_merge(self.merged_config, override_config)
        self._apply_env_overrides()
        
        logger.info(f"Merged override config from: {override_path}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        for config_path, env_var in self.ENV_MAPPINGS.items():
            env_value = os.environ.get(env_var)
            if env_value:
                self._set_nested_value(self.merged_config, config_path, env_value)
                logger.info(f"Applied environment override for {config_path}")
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., 'autogluon.api_key')
            value: Value to set
        """
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_nested_value(self, config: Dict[str, Any], path: str, 
                         default: Any = None) -> Any:
        """Get a nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """Validate configuration against schema.
        
        Args:
            schema: JSON schema to validate against (uses default if None)
            
        Raises:
            ValidationError: If configuration is invalid
        """
        schema = schema or CONFIG_SCHEMA
        
        try:
            validate(self.merged_config, schema)
            logger.info("Configuration validation successful")
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e.message}")
            logger.error(f"Failed at path: {'.'.join(str(p) for p in e.path)}")
            raise
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            path: Dot-separated path (e.g., 'database.path')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        return self._get_nested_value(self.merged_config, path, default)
    
    def get_config(self, redact_secrets: bool = True) -> Dict[str, Any]:
        """Get the merged configuration.
        
        Args:
            redact_secrets: Whether to redact sensitive values
            
        Returns:
            Configuration dictionary
        """
        if redact_secrets:
            return self._redact_secrets(deepcopy(self.merged_config))
        return deepcopy(self.merged_config)
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """Get configuration for a specific component.
        
        Args:
            component: Component name (e.g., 'mcts', 'database')
            
        Returns:
            Component configuration
        """
        if component not in self.merged_config:
            raise KeyError(f"Component '{component}' not found in configuration")
        
        return deepcopy(self.merged_config[component])
    
    def _redact_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive values in configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Redacted configuration
        """
        sensitive_keys = [
            'api_key', 'secret', 'password', 'token', 
            'credential', 'private_key', 'access_key'
        ]
        
        def redact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in d.items():
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in sensitive_keys):
                    d[key] = '***REDACTED***'
                elif isinstance(value, dict):
                    d[key] = redact_dict(value)
                elif isinstance(value, list):
                    d[key] = [
                        redact_dict(item) if isinstance(item, dict) else item
                        for item in value
                    ]
            return d
        
        return redact_dict(config)
    
    def save_merged_config(self, output_path: Union[str, Path], 
                          redact_secrets: bool = True) -> None:
        """Save merged configuration to file.
        
        Args:
            output_path: Path to save configuration
            redact_secrets: Whether to redact sensitive values
        """
        output_path = Path(output_path)
        config_to_save = self.get_config(redact_secrets=redact_secrets)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=True)
        
        logger.info(f"Saved merged configuration to: {output_path}")
    
    def get_required_env_vars(self) -> List[str]:
        """Get list of required environment variables.
        
        Returns:
            List of environment variable names
        """
        required = []
        for config_path, env_var in self.ENV_MAPPINGS.items():
            # Check if the config path exists in base config
            if self._get_nested_value(self.base_config, config_path) is None:
                # If not in base config, it might be required from env
                if os.environ.get(env_var) is None:
                    required.append(env_var)
        
        return required
    
    def check_required_env_vars(self) -> None:
        """Check that all required environment variables are set.
        
        Raises:
            EnvironmentError: If required variables are missing
        """
        missing = self.get_required_env_vars()
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


def load_secure_config(base_path: Union[str, Path], 
                      override_path: Optional[Union[str, Path]] = None,
                      validate_schema: bool = True) -> Dict[str, Any]:
    """Convenience function to load configuration securely.
    
    Args:
        base_path: Path to base configuration file
        override_path: Optional path to override configuration
        validate_schema: Whether to validate against schema
        
    Returns:
        Merged and validated configuration
    """
    manager = SecureConfigManager(base_path)
    
    if override_path:
        manager.merge_override(override_path)
    
    if validate_schema:
        manager.validate()
    
    return manager.get_config(redact_secrets=False)


# Example environment variable template
ENV_TEMPLATE = """
# Minotaur MCTS Feature Discovery System - Environment Variables

# API Keys and Credentials
export MINOTAUR_API_KEY=""

# Database Configuration
export MINOTAUR_DB_CONNECTION=""

# Logging Configuration
export MINOTAUR_LOG_ENDPOINT=""

# Cloud Storage (Optional)
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
"""


def generate_env_template(output_path: Union[str, Path] = ".env.template") -> None:
    """Generate environment variable template file.
    
    Args:
        output_path: Path to save template file
    """
    with open(output_path, 'w') as f:
        f.write(ENV_TEMPLATE)
    
    # Set restrictive permissions
    os.chmod(output_path, 0o600)
    
    logger.info(f"Generated environment template at: {output_path}")