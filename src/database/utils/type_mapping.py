"""
Type mapping utilities for different database engines
"""

from typing import Dict, Any, Type
from sqlalchemy import types
import logging

logger = logging.getLogger(__name__)


class TypeMapper:
    """Utility for mapping types between different database engines"""
    
    # Common type mappings from string representations to SQLAlchemy types
    COMMON_TYPE_MAPPING = {
        # Integer types
        'integer': types.Integer,
        'int': types.Integer,
        'bigint': types.BigInteger,
        'smallint': types.SmallInteger,
        
        # Float types
        'float': types.Float,
        'double': types.Float,
        'real': types.Float,
        'decimal': types.Numeric,
        'numeric': types.Numeric,
        
        # String types
        'varchar': types.String,
        'text': types.Text,
        'char': types.CHAR,
        'string': types.String,
        
        # Date/Time types
        'date': types.Date,
        'time': types.Time,
        'timestamp': types.DateTime,
        'datetime': types.DateTime,
        
        # Boolean type
        'boolean': types.Boolean,
        'bool': types.Boolean,
        
        # Binary types
        'blob': types.LargeBinary,
        'binary': types.LargeBinary,
        
        # JSON type
        'json': types.JSON,
    }
    
    # Database-specific type preferences
    DB_TYPE_PREFERENCES = {
        'duckdb': {
            'string_type': types.String,
            'integer_type': types.Integer,
            'float_type': types.Double,
            'datetime_type': types.DateTime,
            'json_type': types.JSON,
        },
        'sqlite': {
            'string_type': types.Text,
            'integer_type': types.Integer,
            'float_type': types.Float,
            'datetime_type': types.DateTime,
            'json_type': types.Text,  # SQLite doesn't have native JSON
        },
        'postgresql': {
            'string_type': types.String,
            'integer_type': types.Integer,
            'float_type': types.Float,
            'datetime_type': types.DateTime,
            'json_type': types.JSON,
        }
    }
    
    @classmethod
    def map_type(cls, type_str: str, target_db: str = None) -> Type[types.TypeEngine]:
        """
        Map a type string to appropriate SQLAlchemy type
        
        Args:
            type_str: String representation of type
            target_db: Target database type for optimization
            
        Returns:
            SQLAlchemy type class
        """
        type_str_lower = type_str.lower().strip()
        
        # Remove size specifications like VARCHAR(255)
        if '(' in type_str_lower:
            type_str_lower = type_str_lower.split('(')[0]
        
        # Check common mappings first
        if type_str_lower in cls.COMMON_TYPE_MAPPING:
            base_type = cls.COMMON_TYPE_MAPPING[type_str_lower]
            
            # Apply database-specific preferences if available
            if target_db and target_db in cls.DB_TYPE_PREFERENCES:
                preferences = cls.DB_TYPE_PREFERENCES[target_db]
                
                # Map to preferred types for target database
                if base_type == types.String:
                    return preferences['string_type']
                elif base_type == types.Integer:
                    return preferences['integer_type']
                elif base_type == types.Float:
                    return preferences['float_type']
                elif base_type == types.DateTime:
                    return preferences['datetime_type']
                elif base_type == types.JSON:
                    return preferences['json_type']
            
            return base_type
        
        # Fallback to Text for unknown types
        logger.warning(f"Unknown type '{type_str}', defaulting to Text")
        return types.Text
    
    @classmethod
    def get_optimal_type_for_data(cls, sample_data: Any, target_db: str = None) -> Type[types.TypeEngine]:
        """
        Determine optimal SQLAlchemy type based on sample data
        
        Args:
            sample_data: Sample data to analyze
            target_db: Target database type
            
        Returns:
            Optimal SQLAlchemy type
        """
        if sample_data is None:
            return types.Text
        
        # Try to infer type from data
        if isinstance(sample_data, bool):
            return types.Boolean
        elif isinstance(sample_data, int):
            return cls._get_integer_type(sample_data, target_db)
        elif isinstance(sample_data, float):
            return cls._get_float_type(target_db)
        elif isinstance(sample_data, str):
            return cls._get_string_type(sample_data, target_db)
        elif isinstance(sample_data, (dict, list)):
            return cls._get_json_type(target_db)
        else:
            # Fallback to text
            return types.Text
    
    @classmethod
    def _get_integer_type(cls, value: int, target_db: str = None) -> Type[types.TypeEngine]:
        """Get appropriate integer type based on value range"""
        if -32768 <= value <= 32767:
            return types.SmallInteger
        elif -2147483648 <= value <= 2147483647:
            return types.Integer
        else:
            return types.BigInteger
    
    @classmethod
    def _get_float_type(cls, target_db: str = None) -> Type[types.TypeEngine]:
        """Get appropriate float type for target database"""
        if target_db == 'duckdb':
            return types.Double
        else:
            return types.Float
    
    @classmethod
    def _get_string_type(cls, value: str, target_db: str = None) -> Type[types.TypeEngine]:
        """Get appropriate string type based on length and target database"""
        if target_db == 'sqlite':
            return types.Text  # SQLite prefers TEXT
        elif len(value) > 255:
            return types.Text
        else:
            return types.String(length=max(255, len(value) * 2))  # Some buffer
    
    @classmethod
    def _get_json_type(cls, target_db: str = None) -> Type[types.TypeEngine]:
        """Get appropriate JSON type for target database"""
        if target_db == 'sqlite':
            return types.Text  # SQLite doesn't have native JSON
        else:
            return types.JSON
    
    @classmethod
    def convert_pandas_dtype(cls, pandas_dtype: str, target_db: str = None) -> Type[types.TypeEngine]:
        """
        Convert pandas dtype to SQLAlchemy type
        
        Args:
            pandas_dtype: Pandas dtype string
            target_db: Target database type
            
        Returns:
            SQLAlchemy type class
        """
        dtype_str = str(pandas_dtype).lower()
        
        # Integer types
        if 'int' in dtype_str:
            if 'int8' in dtype_str or 'int16' in dtype_str:
                return types.SmallInteger
            elif 'int32' in dtype_str:
                return types.Integer
            elif 'int64' in dtype_str:
                return types.BigInteger
            else:
                return types.Integer
        
        # Float types
        elif 'float' in dtype_str:
            return cls._get_float_type(target_db)
        
        # Boolean type
        elif 'bool' in dtype_str:
            return types.Boolean
        
        # Object/string types
        elif 'object' in dtype_str or 'string' in dtype_str:
            return cls._get_string_type('', target_db)
        
        # Datetime types
        elif 'datetime' in dtype_str:
            return types.DateTime
        
        # Fallback
        else:
            logger.warning(f"Unknown pandas dtype '{pandas_dtype}', defaulting to Text")
            return types.Text


def create_table_from_dataframe(df, table_name: str, target_db: str = None) -> Dict[str, Type[types.TypeEngine]]:
    """
    Create column type mapping from DataFrame for table creation
    
    Args:
        df: Pandas DataFrame
        table_name: Name of target table
        target_db: Target database type
        
    Returns:
        Dictionary mapping column names to SQLAlchemy types
    """
    type_mapping = {}
    
    for column in df.columns:
        # Get sample data (first non-null value)
        sample_data = None
        for value in df[column].dropna():
            sample_data = value
            break
        
        # Determine type based on pandas dtype and sample data
        pandas_dtype = str(df[column].dtype)
        
        if sample_data is not None:
            # Use sample data for better type inference
            inferred_type = TypeMapper.get_optimal_type_for_data(sample_data, target_db)
        else:
            # Fall back to pandas dtype conversion
            inferred_type = TypeMapper.convert_pandas_dtype(pandas_dtype, target_db)
        
        type_mapping[column] = inferred_type
        
        logger.debug(f"Column '{column}': {pandas_dtype} -> {inferred_type}")
    
    return type_mapping