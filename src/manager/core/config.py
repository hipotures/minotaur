"""
Configuration management for the manager system
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Central configuration management for the manager system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Try to find project root by looking for config directory
            current = Path.cwd()
            while current != current.parent:
                if (current / 'config' / 'mcts_config.yaml').exists():
                    config_path = current / 'config' / 'mcts_config.yaml'
                    break
                current = current.parent
            
            if config_path is None:
                config_path = Path('config/mcts_config.yaml')
        
        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent.parent
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'database': {
                'path': 'data/minotaur.duckdb',
                'backup_path': 'data/backups/',
                'backup_prefix': 'minotaur_backup_',
                'max_memory': '4GB',
                'threads': 4
            },
            'export': {
                'default_format': 'csv',
                'output_dir': 'outputs/duckdb_exports'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    @property
    def database_path(self) -> Path:
        """Get absolute database path."""
        db_path = self._config.get('database', {}).get('path', 'data/minotaur.duckdb')
        return self.project_root / db_path
    
    @property
    def backup_path(self) -> Path:
        """Get absolute backup directory path."""
        backup_path = self._config.get('database', {}).get('backup_path', 'data/backups/')
        return self.project_root / backup_path
    
    @property
    def backup_prefix(self) -> str:
        """Get backup file prefix."""
        return self._config.get('database', {}).get('backup_prefix', 'minotaur_backup_')
    
    @property
    def export_dir(self) -> Path:
        """Get export directory path."""
        export_dir = self._config.get('export', {}).get('output_dir', 'outputs/duckdb_exports')
        return self.project_root / export_dir
    
    @property
    def duckdb_settings(self) -> Dict[str, Any]:
        """Get DuckDB-specific settings."""
        return {
            'max_memory': self._config.get('database', {}).get('max_memory', '4GB'),
            'threads': self._config.get('database', {}).get('threads', 4)
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key (e.g., 'database.path')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value