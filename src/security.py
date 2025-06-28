"""Security utilities for the Minotaur MCTS system.

This module provides security-focused utilities including:
- Path validation and sanitization
- Query parameterization helpers
- Input validation
- Log sanitization
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class PathTraversalError(SecurityError):
    """Raised when path traversal is detected."""
    pass


class QueryInjectionError(SecurityError):
    """Raised when potential query injection is detected."""
    pass


class SecurePathManager:
    """Manages secure file path operations preventing directory traversal."""
    
    def __init__(self, allowed_base_dirs: List[Union[str, Path]]):
        """Initialize with list of allowed base directories.
        
        Args:
            allowed_base_dirs: List of directories that are allowed for file operations
        """
        self.allowed_dirs = [os.path.abspath(str(d)) for d in allowed_base_dirs]
        logger.info(f"SecurePathManager initialized with allowed dirs: {self.allowed_dirs}")
    
    def validate_path(self, path: Union[str, Path]) -> Path:
        """Validate that a path is within allowed directories.
        
        Args:
            path: Path to validate
            
        Returns:
            Validated absolute path
            
        Raises:
            PathTraversalError: If path is outside allowed directories
        """
        abs_path = os.path.abspath(str(path))
        
        for allowed in self.allowed_dirs:
            if abs_path.startswith(allowed):
                return Path(abs_path)
        
        raise PathTraversalError(
            f"Path '{path}' resolves to '{abs_path}' which is outside allowed directories"
        )
    
    def join_path(self, base: Union[str, Path], *parts: str) -> Path:
        """Safely join path components preventing traversal.
        
        Args:
            base: Base directory
            *parts: Path components to join
            
        Returns:
            Validated joined path
            
        Raises:
            PathTraversalError: If resulting path is outside allowed directories
        """
        # First validate the base
        validated_base = self.validate_path(base)
        
        # Join parts
        result = validated_base
        for part in parts:
            # Remove any directory traversal attempts
            clean_part = part.replace('..', '').replace('~', '')
            result = result / clean_part
        
        # Validate final result
        return self.validate_path(result)


class SecureQueryBuilder:
    """Builds secure parameterized queries for DuckDB."""
    
    # Whitelist of allowed table names
    ALLOWED_TABLES = {
        'train_data', 'test_data', 'features_cache',
        'exploration_history', 'best_scores', 'sessions'
    }
    
    # Whitelist of allowed functions in queries
    ALLOWED_FUNCTIONS = {
        'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'ROW_NUMBER',
        'COALESCE', 'CAST', 'to_json', 'read_csv_auto',
        'read_parquet', 'DESCRIBE'
    }
    
    @classmethod
    def validate_table_name(cls, table_name: str) -> str:
        """Validate table name against whitelist.
        
        Args:
            table_name: Table name to validate
            
        Returns:
            Validated table name
            
        Raises:
            QueryInjectionError: If table name is not allowed
        """
        if table_name not in cls.ALLOWED_TABLES:
            raise QueryInjectionError(f"Table '{table_name}' is not in allowed list")
        return table_name
    
    @classmethod
    def validate_column_name(cls, column_name: str) -> str:
        """Validate column name for safety.
        
        Args:
            column_name: Column name to validate
            
        Returns:
            Validated column name
            
        Raises:
            QueryInjectionError: If column name contains dangerous characters
        """
        # Allow alphanumeric and underscore only (dash can be SQL operator)
        if not re.match(r'^[a-zA-Z0-9_]+$', column_name):
            raise QueryInjectionError(
                f"Column name '{column_name}' contains invalid characters"
            )
        return column_name
    
    @classmethod
    def escape_identifier(cls, identifier: str) -> str:
        """Escape identifier for safe use in queries.
        
        Args:
            identifier: Identifier to escape
            
        Returns:
            Escaped identifier
        """
        # DuckDB uses double quotes for identifiers
        return f'"{identifier}"'
    
    @classmethod
    def build_csv_describe_query(cls, file_path: Path) -> tuple[str, tuple]:
        """Build a safe DESCRIBE query for CSV file.
        
        Args:
            file_path: Path to CSV file (should be pre-validated)
            
        Returns:
            Tuple of (query_template, parameters)
        """
        # Use parameterized query with ? placeholder
        query = "DESCRIBE SELECT * FROM read_csv_auto(?)"
        return query, (str(file_path),)
    
    @classmethod
    def build_csv_load_query(cls, file_path: Path, table_name: str, 
                           column_mapping: Dict[str, str]) -> tuple[str, tuple]:
        """Build a safe INSERT query for loading CSV data.
        
        Args:
            file_path: Path to CSV file (should be pre-validated)
            table_name: Target table name
            column_mapping: Mapping of internal to CSV column names
            
        Returns:
            Tuple of (query_template, parameters)
        """
        # Validate table name
        safe_table = cls.validate_table_name(table_name)
        
        # Build column expressions safely
        json_fields = []
        for internal_name, csv_column in column_mapping.items():
            if csv_column and internal_name != 'id':
                safe_internal = cls.validate_column_name(internal_name)
                safe_csv = cls.escape_identifier(csv_column)
                json_fields.append(
                    f"'{safe_internal}': COALESCE(CAST({safe_csv} AS VARCHAR), 'Unknown')"
                )
        
        # Add id field
        if 'id' in column_mapping and column_mapping['id']:
            safe_id = cls.escape_identifier(column_mapping['id'])
            json_fields.insert(0, f"'id': COALESCE({safe_id}, ROW_NUMBER() OVER ())")
        else:
            json_fields.insert(0, "'id': ROW_NUMBER() OVER ()")
        
        json_construction = "{ " + ", ".join(json_fields) + " }"
        
        # Build query with parameter placeholder
        query = f"""
        INSERT INTO {safe_table} (id, data)
        SELECT ROW_NUMBER() OVER () as id, 
               to_json({json_construction}) as data
        FROM read_csv_auto(?)
        """
        
        return query, (str(file_path),)
    
    @classmethod
    def build_count_query(cls, table_name: str) -> tuple[str, tuple]:
        """Build a safe COUNT query.
        
        Args:
            table_name: Table to count rows from
            
        Returns:
            Tuple of (query_template, parameters)
        """
        safe_table = cls.validate_table_name(table_name)
        return f"SELECT COUNT(*) FROM {safe_table}", ()


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    @staticmethod
    def validate_config_value(key: str, value: Any, 
                            expected_type: type = None,
                            allowed_values: List[Any] = None) -> Any:
        """Validate a configuration value.
        
        Args:
            key: Configuration key name
            value: Value to validate
            expected_type: Expected type (optional)
            allowed_values: List of allowed values (optional)
            
        Returns:
            Validated value
            
        Raises:
            ValueError: If validation fails
        """
        if expected_type and not isinstance(value, expected_type):
            raise ValueError(
                f"Config key '{key}' expected type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        
        if allowed_values and value not in allowed_values:
            raise ValueError(
                f"Config key '{key}' value '{value}' not in allowed values: {allowed_values}"
            )
        
        return value
    
    @staticmethod
    def sanitize_log_message(message: str) -> str:
        """Remove sensitive data from log messages.
        
        Args:
            message: Log message to sanitize
            
        Returns:
            Sanitized message
        """
        # Remove potential API keys
        message = re.sub(
            r'(api_key|password|token|secret|key)\s*[=:]\s*[^\s]+',
            r'\1=***REDACTED***',
            message,
            flags=re.IGNORECASE
        )
        
        # Remove file paths that might expose system structure
        message = re.sub(r'/home/[^/\s]+/', '/home/***/', message)
        message = re.sub(r'C:\\\\Users\\\\[^\\\\]+\\\\', r'C:\\Users\\***\\', message)
        
        # Remove potential email addresses
        message = re.sub(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            '***@***.***',
            message
        )
        
        return message


class SensitiveDataFilter(logging.Filter):
    """Logging filter that removes sensitive data."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to remove sensitive data.
        
        Args:
            record: Log record to filter
            
        Returns:
            True (always allow record through after sanitization)
        """
        if hasattr(record, 'msg'):
            record.msg = InputValidator.sanitize_log_message(str(record.msg))
        if hasattr(record, 'args') and record.args:
            sanitized_args = []
            for arg in record.args:
                sanitized_args.append(InputValidator.sanitize_log_message(str(arg)))
            record.args = tuple(sanitized_args)
        return True


def setup_secure_logging(logger_name: Optional[str] = None) -> logging.Logger:
    """Set up logger with security filters.
    
    Args:
        logger_name: Name of logger (None for root logger)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(logger_name)
    
    # Add sensitive data filter
    sensitive_filter = SensitiveDataFilter()
    
    # Add to all handlers
    for handler in logger.handlers:
        handler.addFilter(sensitive_filter)
    
    # Also add to logger itself
    logger.addFilter(sensitive_filter)
    
    return logger