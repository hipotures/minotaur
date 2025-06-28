"""Unit tests for secure configuration manager."""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
from jsonschema import ValidationError

from src.config_manager import SecureConfigManager, load_secure_config, generate_env_template


class TestSecureConfigManager:
    """Test secure configuration management functionality."""
    
    @pytest.fixture
    def base_config(self):
        """Create a base configuration for testing."""
        return {
            'database': {
                'path': 'test.db',
                'type': 'duckdb'
            },
            'mcts': {
                'exploration_weight': 1.4,
                'max_tree_depth': 5
            },
            'session': {
                'mode': 'new',
                'max_iterations': 100
            }
        }
    
    @pytest.fixture
    def config_file(self, base_config):
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(base_config, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_init_with_valid_config(self, config_file):
        """Test initialization with valid configuration file."""
        manager = SecureConfigManager(config_file)
        
        assert manager.base_config_path == Path(config_file)
        assert 'database' in manager.base_config
        assert manager.base_config['mcts']['exploration_weight'] == 1.4
    
    def test_init_with_missing_file(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            SecureConfigManager('non_existent_file.yaml')
    
    def test_merge_override(self, config_file):
        """Test merging override configuration."""
        manager = SecureConfigManager(config_file)
        
        # Create override config
        override_config = {
            'mcts': {
                'exploration_weight': 2.0,
                'new_param': 'test'
            },
            'new_section': {
                'param': 'value'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(override_config, f)
            override_path = f.name
        
        try:
            manager.merge_override(override_path)
            
            # Check merged values
            config = manager.get_config(redact_secrets=False)
            assert config['mcts']['exploration_weight'] == 2.0
            assert config['mcts']['max_tree_depth'] == 5  # Original value
            assert config['mcts']['new_param'] == 'test'
            assert config['new_section']['param'] == 'value'
            
        finally:
            if os.path.exists(override_path):
                os.unlink(override_path)
    
    @patch.dict(os.environ, {'MINOTAUR_API_KEY': 'test_key_123'})
    def test_env_override(self, config_file):
        """Test environment variable overrides."""
        # Add autogluon section to config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        config['autogluon'] = {'api_key': 'original_key'}
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        manager = SecureConfigManager(config_file)
        
        # Check that env variable overrides config
        final_config = manager.get_config(redact_secrets=False)
        assert final_config['autogluon']['api_key'] == 'test_key_123'
    
    def test_config_validation(self, config_file):
        """Test configuration validation."""
        manager = SecureConfigManager(config_file)
        
        # Should validate successfully
        manager.validate()
        
        # Modify config to be invalid
        manager.merged_config['mcts']['exploration_weight'] = -1  # Invalid: negative
        
        with pytest.raises(ValidationError):
            manager.validate()
    
    def test_secret_redaction(self, config_file):
        """Test that secrets are properly redacted."""
        manager = SecureConfigManager(config_file)
        
        # Add some secrets
        manager.merged_config['autogluon'] = {
            'api_key': 'secret_key_123',
            'password': 'my_password',
            'normal_param': 'visible'
        }
        
        redacted = manager.get_config(redact_secrets=True)
        
        assert redacted['autogluon']['api_key'] == '***REDACTED***'
        assert redacted['autogluon']['password'] == '***REDACTED***'
        assert redacted['autogluon']['normal_param'] == 'visible'
    
    def test_component_config(self, config_file):
        """Test getting component-specific configuration."""
        manager = SecureConfigManager(config_file)
        
        mcts_config = manager.get_component_config('mcts')
        
        assert 'exploration_weight' in mcts_config
        assert 'database' not in mcts_config
        
        # Test non-existent component
        with pytest.raises(KeyError):
            manager.get_component_config('non_existent')
    
    def test_save_merged_config(self, config_file):
        """Test saving merged configuration."""
        manager = SecureConfigManager(config_file)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            manager.save_merged_config(output_path)
            
            # Load saved config
            with open(output_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config['mcts']['exploration_weight'] == 1.4
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_required_env_vars(self, config_file):
        """Test checking required environment variables."""
        manager = SecureConfigManager(config_file)
        
        # Since we don't have required env vars in base config, should be empty
        required = manager.get_required_env_vars()
        assert isinstance(required, list)


def test_load_secure_config():
    """Test the convenience function for loading config."""
    # Create temporary configs
    base_config = {
        'database': {'path': 'test.db'},
        'mcts': {'exploration_weight': 1.4, 'max_tree_depth': 5},
        'session': {'mode': 'new', 'max_iterations': 100}
    }
    
    override_config = {
        'mcts': {'exploration_weight': 2.0}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(base_config, f)
        base_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(override_config, f)
        override_path = f.name
    
    try:
        # Load with override
        config = load_secure_config(base_path, override_path)
        
        assert config['mcts']['exploration_weight'] == 2.0
        assert config['mcts']['max_tree_depth'] == 5
        
    finally:
        if os.path.exists(base_path):
            os.unlink(base_path)
        if os.path.exists(override_path):
            os.unlink(override_path)


def test_generate_env_template():
    """Test environment template generation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        template_path = f.name
    
    try:
        generate_env_template(template_path)
        
        assert os.path.exists(template_path)
        
        # Check file permissions (should be 0o600)
        stat_info = os.stat(template_path)
        assert stat_info.st_mode & 0o777 == 0o600
        
        # Check content
        with open(template_path, 'r') as f:
            content = f.read()
        
        assert 'MINOTAUR_API_KEY' in content
        assert 'AWS_ACCESS_KEY_ID' in content
        
    finally:
        if os.path.exists(template_path):
            os.unlink(template_path)