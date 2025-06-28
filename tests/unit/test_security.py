"""Unit tests for security module."""

import pytest
import os
import tempfile
from pathlib import Path
import logging

from src.security import (
    SecurePathManager, SecureQueryBuilder, InputValidator,
    PathTraversalError, QueryInjectionError, SecurityError,
    SensitiveDataFilter, setup_secure_logging
)


class TestSecurePathManager:
    """Test secure path management functionality."""
    
    def test_path_validation_allowed(self):
        """Test that allowed paths are validated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecurePathManager([tmpdir])
            
            # Test valid path
            valid_path = os.path.join(tmpdir, "test.txt")
            result = manager.validate_path(valid_path)
            assert isinstance(result, Path)
            assert str(result) == os.path.abspath(valid_path)
    
    def test_path_validation_denied(self):
        """Test that paths outside allowed directories are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecurePathManager([tmpdir])
            
            # Test path outside allowed directory
            with pytest.raises(PathTraversalError):
                manager.validate_path("/etc/passwd")
    
    def test_path_traversal_prevention(self):
        """Test prevention of directory traversal attacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecurePathManager([tmpdir])
            
            # Test various traversal attempts
            traversal_attempts = [
                os.path.join(tmpdir, "../../../etc/passwd"),
                os.path.join(tmpdir, "test/../../../etc/passwd"),
                os.path.join(tmpdir, "test/../../..", "etc/passwd"),
            ]
            
            for attempt in traversal_attempts:
                with pytest.raises(PathTraversalError):
                    manager.validate_path(attempt)
    
    def test_join_path_secure(self):
        """Test secure path joining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecurePathManager([tmpdir])
            
            # Test normal join
            result = manager.join_path(tmpdir, "subdir", "file.txt")
            expected = Path(tmpdir) / "subdir" / "file.txt"
            assert result == expected
            
            # Test traversal attempt in join
            result = manager.join_path(tmpdir, "test", "file.txt")
            # Should result in tmpdir/test/file.txt
            assert "test" in str(result)
            assert str(result).startswith(tmpdir)


class TestSecureQueryBuilder:
    """Test secure query building functionality."""
    
    def test_table_name_validation(self):
        """Test table name validation against whitelist."""
        # Valid tables
        for table in ['train_data', 'test_data', 'features_cache']:
            result = SecureQueryBuilder.validate_table_name(table)
            assert result == table
        
        # Invalid table
        with pytest.raises(QueryInjectionError):
            SecureQueryBuilder.validate_table_name("users; DROP TABLE train_data;")
    
    def test_column_name_validation(self):
        """Test column name validation."""
        # Valid column names
        valid_names = ['column1', 'test_col', 'feature_123', 'col_name']
        for name in valid_names:
            result = SecureQueryBuilder.validate_column_name(name)
            assert result == name
        
        # Invalid column names
        invalid_names = [
            "col; DROP TABLE",
            "col'name",
            'col"name',
            "col/*comment*/name",
            "col--comment",
            "col-name",  # Dashes not allowed as they can be operators
        ]
        for name in invalid_names:
            with pytest.raises(QueryInjectionError):
                SecureQueryBuilder.validate_column_name(name)
    
    def test_identifier_escaping(self):
        """Test proper identifier escaping."""
        result = SecureQueryBuilder.escape_identifier("column_name")
        assert result == '"column_name"'
    
    def test_csv_describe_query(self):
        """Test building secure DESCRIBE query."""
        test_path = Path("/tmp/test.csv")
        query, params = SecureQueryBuilder.build_csv_describe_query(test_path)
        
        assert query == "DESCRIBE SELECT * FROM read_csv_auto(?)"
        assert params == ("/tmp/test.csv",)
    
    def test_csv_load_query(self):
        """Test building secure INSERT query for CSV loading."""
        test_path = Path("/tmp/test.csv")
        column_mapping = {
            'id': 'ID',
            'name': 'Name',
            'value': 'Value'
        }
        
        query, params = SecureQueryBuilder.build_csv_load_query(
            test_path, 'train_data', column_mapping
        )
        
        # Check query structure
        assert "INSERT INTO train_data" in query
        assert "read_csv_auto(?)" in query
        assert params == ("/tmp/test.csv",)
        
        # Check that columns are properly escaped
        assert '"Name"' in query
        assert '"Value"' in query
    
    def test_count_query(self):
        """Test building secure COUNT query."""
        query, params = SecureQueryBuilder.build_count_query('train_data')
        
        assert query == "SELECT COUNT(*) FROM train_data"
        assert params == ()


class TestInputValidator:
    """Test input validation functionality."""
    
    def test_config_value_validation_type(self):
        """Test configuration value type validation."""
        # Valid type
        result = InputValidator.validate_config_value(
            'max_iterations', 100, expected_type=int
        )
        assert result == 100
        
        # Invalid type
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_config_value(
                'max_iterations', "100", expected_type=int
            )
        assert "expected type int" in str(exc_info.value)
    
    def test_config_value_validation_allowed_values(self):
        """Test configuration value allowed values validation."""
        # Valid value
        result = InputValidator.validate_config_value(
            'mode', 'train', allowed_values=['train', 'test', 'eval']
        )
        assert result == 'train'
        
        # Invalid value
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_config_value(
                'mode', 'invalid', allowed_values=['train', 'test', 'eval']
            )
        assert "not in allowed values" in str(exc_info.value)
    
    def test_log_message_sanitization(self):
        """Test sensitive data removal from log messages."""
        # API key sanitization
        msg = "Connecting with api_key=sk-1234567890abcdef"
        sanitized = InputValidator.sanitize_log_message(msg)
        assert sanitized == "Connecting with api_key=***REDACTED***"
        
        # Password sanitization
        msg = "Login with password: supersecret123"
        sanitized = InputValidator.sanitize_log_message(msg)
        assert "***REDACTED***" in sanitized
        assert "supersecret123" not in sanitized
        
        # File path sanitization
        msg = "Loading file from /home/username/project/data.csv"
        sanitized = InputValidator.sanitize_log_message(msg)
        assert sanitized == "Loading file from /home/***/project/data.csv"
        
        # Email sanitization
        msg = "User email: user@example.com logged in"
        sanitized = InputValidator.sanitize_log_message(msg)
        assert sanitized == "User email: ***@***.*** logged in"
    
    def test_multiple_sensitive_data(self):
        """Test sanitization of multiple sensitive items."""
        msg = "User /home/john/app with token=abc123 and email john@example.com"
        sanitized = InputValidator.sanitize_log_message(msg)
        
        assert "/home/john/" not in sanitized
        assert "token=abc123" not in sanitized
        assert "john@example.com" not in sanitized
        assert "***" in sanitized


class TestSensitiveDataFilter:
    """Test logging filter functionality."""
    
    def test_filter_log_record(self):
        """Test that log records are properly filtered."""
        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="API call with key=secret123",
            args=(),
            exc_info=None
        )
        
        # Apply filter
        filter = SensitiveDataFilter()
        result = filter.filter(record)
        
        assert result is True  # Record should pass through
        assert "secret123" not in record.msg
        assert "***REDACTED***" in record.msg
    
    def test_filter_with_args(self):
        """Test filtering with message arguments."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User %s with password %s",
            args=("user@example.com", "password123"),
            exc_info=None
        )
        
        filter = SensitiveDataFilter()
        filter.filter(record)
        
        # Check args are sanitized
        assert "***@***.***" in record.args[0]
        assert "password123" not in str(record.args)


class TestSecureLogging:
    """Test secure logging setup."""
    
    def test_setup_secure_logging(self):
        """Test that secure logging is properly configured."""
        # Create test logger
        test_logger = setup_secure_logging("test_secure_logger")
        
        # Add a test handler to capture output
        import io
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)
        
        # Log sensitive information
        test_logger.info("Processing with api_key=test123")
        
        # Check output
        output = stream.getvalue()
        assert "test123" not in output
        assert "***REDACTED***" in output


@pytest.fixture
def temp_allowed_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        allowed = [
            os.path.join(tmpdir, "allowed1"),
            os.path.join(tmpdir, "allowed2")
        ]
        for dir in allowed:
            os.makedirs(dir, exist_ok=True)
        yield allowed


def test_integration_path_and_query(temp_allowed_dirs):
    """Integration test combining path validation and query building."""
    # Set up path manager
    path_manager = SecurePathManager(temp_allowed_dirs)
    
    # Create a test CSV file
    test_file = os.path.join(temp_allowed_dirs[0], "test_data.csv")
    Path(test_file).touch()
    
    # Validate path
    validated_path = path_manager.validate_path(test_file)
    
    # Build query with validated path
    query, params = SecureQueryBuilder.build_csv_describe_query(validated_path)
    
    assert query == "DESCRIBE SELECT * FROM read_csv_auto(?)"
    assert params[0] == str(validated_path)