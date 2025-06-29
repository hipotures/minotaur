"""
Dataset data models.

This module defines Pydantic models for dataset registry operations,
including dataset metadata, file information, and usage tracking.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class DatasetFormat(str):
    """Dataset file format enumeration."""
    CSV = "csv"
    PARQUET = "parquet"
    CSV_GZ = "csv.gz"
    JSON = "json"
    XLSX = "xlsx"


class Dataset(BaseModel):
    """
    Dataset registry model.
    
    Represents a registered dataset with all file information,
    metadata, and usage statistics.
    """
    
    dataset_id: str = Field(..., description="Unique dataset identifier (MD5 hash)")
    dataset_name: str = Field(..., description="Human-readable dataset name")
    train_path: str = Field(..., description="Path to training data file")
    test_path: Optional[str] = Field(None, description="Path to test data file")
    submission_path: Optional[str] = Field(None, description="Path to submission template")
    validation_path: Optional[str] = Field(None, description="Path to validation data file")
    target_column: str = Field(..., description="Target column name")
    id_column: Optional[str] = Field(None, description="ID column name")
    competition_name: Optional[str] = Field(None, description="Kaggle competition name")
    description: Optional[str] = Field(None, description="Dataset description")
    
    # File statistics
    train_records: Optional[int] = Field(None, ge=0, description="Number of records in train file")
    train_columns: Optional[int] = Field(None, ge=0, description="Number of columns in train file")
    test_records: Optional[int] = Field(None, ge=0, description="Number of records in test file")
    test_columns: Optional[int] = Field(None, ge=0, description="Number of columns in test file")
    submission_records: Optional[int] = Field(None, ge=0, description="Number of records in submission file")
    submission_columns: Optional[int] = Field(None, ge=0, description="Number of columns in submission file")
    validation_records: Optional[int] = Field(None, ge=0, description="Number of records in validation file")
    validation_columns: Optional[int] = Field(None, ge=0, description="Number of columns in validation file")
    
    # File formats
    train_format: Optional[str] = Field(None, description="Format of train file")
    test_format: Optional[str] = Field(None, description="Format of test file")
    submission_format: Optional[str] = Field(None, description="Format of submission file")
    validation_format: Optional[str] = Field(None, description="Format of validation file")
    
    # Legacy columns for backward compatibility
    column_count: Optional[int] = Field(None, ge=0, description="DEPRECATED: use train_columns")
    train_row_count: Optional[int] = Field(None, ge=0, description="DEPRECATED: use train_records")
    test_row_count: Optional[int] = Field(None, ge=0, description="DEPRECATED: use test_records")
    data_size_mb: Optional[float] = Field(None, ge=0.0, description="DEPRECATED: calculated from files")
    feature_types: Optional[Dict[str, str]] = Field(None, description="DEPRECATED: extracted from data")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="When dataset was registered")
    last_used: Optional[datetime] = Field(None, description="When dataset was last used")
    is_active: bool = Field(default=True, description="Whether dataset is actively used")
    
    @field_validator('dataset_id')
    @classmethod
    def validate_dataset_id(cls, v):
        """Validate dataset ID format (MD5 hash or shortened version)."""
        if not v:
            raise ValueError('Dataset ID cannot be empty')
        # Accept both full MD5 hash (32 chars) and shortened versions (8+ chars)
        if len(v) < 8 or len(v) > 32:
            raise ValueError('Dataset ID must be 8-32 characters long')
        if not all(c in '0123456789abcdef' for c in v.lower()):
            raise ValueError('Dataset ID must be a valid hexadecimal hash')
        return v.lower()
    
    @field_validator('dataset_name')
    @classmethod
    def validate_dataset_name(cls, v):
        """Validate dataset name."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Dataset name cannot be empty')
        return v.strip()
    
    @field_validator('train_path')
    @classmethod
    def validate_train_path(cls, v):
        """Validate train path."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Train path cannot be empty')
        return v.strip()
    
    @field_validator('target_column')
    @classmethod
    def validate_target_column(cls, v):
        """Validate target column name."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Target column cannot be empty')
        return v.strip()
    
    @field_validator('train_format', 'test_format', 'submission_format', 'validation_format')
    @classmethod
    def validate_file_format(cls, v):
        """Validate file format."""
        if v is not None:
            valid_formats = ['csv', 'parquet', 'csv.gz', 'json', 'xlsx']
            if v.lower() not in valid_formats:
                raise ValueError(f'Invalid file format. Must be one of: {valid_formats}')
            return v.lower()
        return v
    
    @property
    def total_train_size(self) -> Optional[int]:
        """Total size of training data."""
        if self.train_records and self.train_columns:
            return self.train_records * self.train_columns
        return None
    
    @property
    def total_test_size(self) -> Optional[int]:
        """Total size of test data."""
        if self.test_records and self.test_columns:
            return self.test_records * self.test_columns
        return None
    
    @property
    def has_test_data(self) -> bool:
        """Check if dataset has test data."""
        return self.test_path is not None and len(self.test_path.strip()) > 0
    
    @property
    def has_validation_data(self) -> bool:
        """Check if dataset has validation data."""
        return self.validation_path is not None and len(self.validation_path.strip()) > 0
    
    @property
    def file_paths(self) -> Dict[str, Optional[str]]:
        """Get all file paths as dictionary."""
        return {
            'train': self.train_path,
            'test': self.test_path,
            'submission': self.submission_path,
            'validation': self.validation_path
        }
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "dataset_id": "abc123def456789012345678901234",
                "dataset_name": "Titanic Survival Prediction",
                "train_path": "/path/to/train.csv",
                "test_path": "/path/to/test.csv",
                "target_column": "Survived",
                "id_column": "PassengerId",
                "competition_name": "titanic",
                "description": "Predict survival on the Titanic",
                "train_records": 891,
                "train_columns": 12,
                "test_records": 418,
                "test_columns": 11,
                "train_format": "csv",
                "test_format": "csv"
            }
        }


class DatasetUsage(BaseModel):
    """
    Dataset usage tracking model.
    
    Tracks how datasets are used across different sessions
    and provides usage analytics.
    """
    
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    dataset_id: str = Field(..., description="Dataset identifier")
    session_id: str = Field(..., description="Session that used the dataset")
    usage_timestamp: datetime = Field(default_factory=datetime.now, description="When dataset was used")
    data_subset_used: Optional[str] = Field(None, description="Which subset was used (train/test/validation)")
    sample_size: Optional[int] = Field(None, ge=0, description="Size of data sample used")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Time taken to process data")
    memory_usage_mb: Optional[float] = Field(None, ge=0.0, description="Memory usage in MB")
    notes: Optional[str] = Field(None, description="Additional usage notes")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetCreate(BaseModel):
    """
    Model for creating new datasets.
    
    Used for dataset registration operations with validation.
    """
    
    dataset_name: str = Field(..., description="Human-readable dataset name")
    train_path: str = Field(..., description="Path to training data file")
    test_path: Optional[str] = Field(None, description="Path to test data file")
    submission_path: Optional[str] = Field(None, description="Path to submission template")
    validation_path: Optional[str] = Field(None, description="Path to validation data file")
    target_column: str = Field(..., description="Target column name")
    id_column: Optional[str] = Field(None, description="ID column name")
    competition_name: Optional[str] = Field(None, description="Kaggle competition name")
    description: Optional[str] = Field(None, description="Dataset description")
    
    # Auto-detection will fill in file statistics and formats
    auto_detect: bool = Field(default=True, description="Whether to auto-detect file properties")
    
    @field_validator('train_path', 'test_path', 'submission_path', 'validation_path')
    @classmethod
    def validate_file_paths(cls, v):
        """Validate file paths exist if provided."""
        if v is not None and len(v.strip()) > 0:
            path = Path(v.strip())
            if not path.exists():
                raise ValueError(f'File does not exist: {v}')
            return str(path.resolve())
        return v


class DatasetUpdate(BaseModel):
    """
    Model for updating existing datasets.
    
    All fields are optional to allow partial updates.
    """
    
    dataset_name: Optional[str] = None
    test_path: Optional[str] = None
    submission_path: Optional[str] = None
    validation_path: Optional[str] = None
    target_column: Optional[str] = None
    id_column: Optional[str] = None
    competition_name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    
    # Allow updating file statistics manually
    train_records: Optional[int] = Field(None, ge=0)
    train_columns: Optional[int] = Field(None, ge=0)
    test_records: Optional[int] = Field(None, ge=0)
    test_columns: Optional[int] = Field(None, ge=0)


class DatasetAnalysis(BaseModel):
    """
    Dataset analysis results model.
    
    Provides statistical analysis and insights about datasets
    in the registry.
    """
    
    total_datasets: int = Field(..., ge=0, description="Total number of datasets")
    active_datasets: int = Field(..., ge=0, description="Number of active datasets")
    datasets_with_test: int = Field(..., ge=0, description="Datasets with test data")
    datasets_with_validation: int = Field(..., ge=0, description="Datasets with validation data")
    avg_train_size: Optional[float] = Field(None, ge=0.0, description="Average training set size")
    total_data_size_gb: Optional[float] = Field(None, ge=0.0, description="Total data size in GB")
    most_used_datasets: List[str] = Field(..., description="Most frequently used datasets")
    file_format_distribution: Dict[str, int] = Field(..., description="Distribution of file formats")
    competition_distribution: Dict[str, int] = Field(..., description="Distribution by competition")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "total_datasets": 5,
                "active_datasets": 4,
                "datasets_with_test": 3,
                "datasets_with_validation": 1,
                "avg_train_size": 15000.0,
                "total_data_size_gb": 2.5,
                "most_used_datasets": ["Titanic", "Fertilizer S5E6"],
                "file_format_distribution": {"csv": 4, "parquet": 1},
                "competition_distribution": {"titanic": 1, "playground-series-s5e6": 1}
            }
        }


class DatasetFileInfo(BaseModel):
    """
    Dataset file information model.
    
    Contains detailed information about dataset files
    extracted during auto-detection.
    """
    
    file_path: str = Field(..., description="Path to the file")
    file_format: str = Field(..., description="Detected file format")
    file_size_mb: float = Field(..., ge=0.0, description="File size in MB")
    record_count: int = Field(..., ge=0, description="Number of records (rows)")
    column_count: int = Field(..., ge=0, description="Number of columns")
    column_names: List[str] = Field(..., description="List of column names")
    column_types: Dict[str, str] = Field(..., description="Column data types")
    missing_values: Dict[str, int] = Field(..., description="Missing value counts per column")
    sample_data: Optional[Dict[str, Any]] = Field(None, description="Sample of first few rows")
    
    @property
    def total_cells(self) -> int:
        """Total number of cells in the dataset."""
        return self.record_count * self.column_count
    
    @property
    def missing_percentage(self) -> float:
        """Percentage of missing values overall."""
        if self.total_cells > 0:
            total_missing = sum(self.missing_values.values())
            return (total_missing / self.total_cells) * 100.0
        return 0.0
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/train.csv",
                "file_format": "csv",
                "file_size_mb": 0.8,
                "record_count": 891,
                "column_count": 12,
                "column_names": ["PassengerId", "Survived", "Pclass", "Name"],
                "column_types": {"PassengerId": "int64", "Survived": "int64", "Name": "object"},
                "missing_values": {"Age": 177, "Cabin": 687, "Embarked": 2},
                "sample_data": {"PassengerId": [1, 2, 3], "Survived": [0, 1, 1]}
            }
        }