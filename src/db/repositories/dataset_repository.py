"""
Dataset repository implementation.

This module provides database operations for dataset registry management,
including dataset registration, file analysis, and usage tracking.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from ..core.base_repository import BaseRepository
from ..core.connection import DuckDBConnectionManager
from ..models.dataset import (
    Dataset, DatasetUsage, DatasetCreate, DatasetUpdate,
    DatasetAnalysis, DatasetFileInfo
)


class DatasetRepository(BaseRepository[Dataset]):
    """
    Repository for dataset registry operations.
    
    Handles all database operations related to dataset management,
    including registration, file analysis, and usage tracking.
    """
    
    @property
    def table_name(self) -> str:
        """Return the datasets table name."""
        return "datasets"
    
    @property
    def model_class(self) -> type:
        """Return the Dataset model class."""
        return Dataset
    
    def _get_conflict_target(self) -> Optional[str]:
        """
        Get the primary conflict target column for ON CONFLICT operations.
        
        For datasets table, the primary key is 'dataset_id', not 'id'.
        
        Returns:
            'dataset_id' as the conflict resolution column
        """
        return 'dataset_id'
    
    def _row_to_model(self, row: Any) -> Dataset:
        """Convert database row to Dataset model."""
        # Handle both tuple and dict-like row objects
        if hasattr(row, 'keys'):
            # Dict-like object (SQLite Row)
            data = dict(row)
        else:
            # Tuple - map to known column order
            columns = [
                'dataset_id', 'dataset_name', 'train_path', 'test_path',
                'submission_path', 'validation_path', 'target_column', 'id_column',
                'competition_name', 'description', 'train_records', 'train_columns',
                'test_records', 'test_columns', 'submission_records', 'submission_columns',
                'validation_records', 'validation_columns', 'train_format', 'test_format',
                'submission_format', 'validation_format', 'column_count', 'train_row_count',
                'test_row_count', 'data_size_mb', 'feature_types', 'created_at',
                'last_used', 'is_active'
            ]
            data = dict(zip(columns, row))
        
        # Parse JSON fields
        if isinstance(data.get('feature_types'), str):
            try:
                data['feature_types'] = json.loads(data['feature_types'])
            except (json.JSONDecodeError, TypeError):
                data['feature_types'] = None
        
        # Convert timestamp fields
        for field in ['created_at', 'last_used']:
            if isinstance(data.get(field), str):
                try:
                    data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    if field == 'created_at':
                        data[field] = datetime.now()
                    else:
                        data[field] = None
        
        # Set defaults for missing fields
        data.setdefault('is_active', True)
        
        # Handle None values for optional paths
        for path_field in ['test_path', 'submission_path', 'validation_path']:
            if data.get(path_field) == '':
                data[path_field] = None
        
        return Dataset(**data)
    
    def _model_to_dict(self, model: Dataset) -> Dict[str, Any]:
        """Convert Dataset model to dictionary for database operations."""
        data = {
            'dataset_id': model.dataset_id,
            'dataset_name': model.dataset_name,
            'train_path': model.train_path,
            'test_path': model.test_path,
            'submission_path': model.submission_path,
            'validation_path': model.validation_path,
            'target_column': model.target_column,
            'id_column': model.id_column,
            'competition_name': model.competition_name,
            'description': model.description,
            'train_records': model.train_records,
            'train_columns': model.train_columns,
            'test_records': model.test_records,
            'test_columns': model.test_columns,
            'submission_records': model.submission_records,
            'submission_columns': model.submission_columns,
            'validation_records': model.validation_records,
            'validation_columns': model.validation_columns,
            'train_format': model.train_format,
            'test_format': model.test_format,
            'submission_format': model.submission_format,
            'validation_format': model.validation_format,
            'column_count': model.column_count,
            'train_row_count': model.train_row_count,
            'test_row_count': model.test_row_count,
            'data_size_mb': model.data_size_mb,
            'feature_types': json.dumps(model.feature_types) if model.feature_types else None,
            'created_at': model.created_at.isoformat(),
            'last_used': model.last_used.isoformat() if model.last_used else None,
            'is_active': model.is_active
        }
        
        return data
    
    def register_dataset(self, dataset_data: DatasetCreate) -> Dataset:
        """
        Register a new dataset in the registry.
        
        Args:
            dataset_data: Dataset creation data
            
        Returns:
            Registered dataset with generated ID and metadata
        """
        # Generate dataset ID from file paths
        dataset_id = self._generate_dataset_id(dataset_data.train_path, dataset_data.test_path)
        
        # Auto-detect file properties if requested
        file_info = {}
        if dataset_data.auto_detect:
            file_info = self._analyze_dataset_files(dataset_data)
        
        # Create dataset with detected/provided information
        dataset = Dataset(
            dataset_id=dataset_id,
            dataset_name=dataset_data.dataset_name,
            train_path=dataset_data.train_path,
            test_path=dataset_data.test_path,
            submission_path=dataset_data.submission_path,
            validation_path=dataset_data.validation_path,
            target_column=dataset_data.target_column,
            id_column=dataset_data.id_column,
            competition_name=dataset_data.competition_name,
            description=dataset_data.description,
            **file_info  # Include auto-detected information
        )
        
        saved_dataset = self.save(dataset)
        self.logger.info(f"Registered dataset: {dataset_data.dataset_name} (ID: {dataset_id[:8]}...)")
        return saved_dataset
    
    def _generate_dataset_id(self, train_path: str, test_path: Optional[str]) -> str:
        """Generate dataset ID from file paths using MD5 hash."""
        path_string = f"{train_path}|{test_path or ''}"
        return hashlib.md5(path_string.encode()).hexdigest()
    
    def _analyze_dataset_files(self, dataset_data: DatasetCreate) -> Dict[str, Any]:
        """
        Analyze dataset files to extract metadata.
        
        Args:
            dataset_data: Dataset creation data
            
        Returns:
            Dictionary with file analysis results
        """
        analysis_results = {}
        
        # Analyze train file
        if dataset_data.train_path:
            train_info = self._analyze_single_file(dataset_data.train_path)
            if train_info:
                analysis_results.update({
                    'train_records': train_info.record_count,
                    'train_columns': train_info.column_count,
                    'train_format': train_info.file_format
                })
        
        # Analyze test file
        if dataset_data.test_path:
            test_info = self._analyze_single_file(dataset_data.test_path)
            if test_info:
                analysis_results.update({
                    'test_records': test_info.record_count,
                    'test_columns': test_info.column_count,
                    'test_format': test_info.file_format
                })
        
        # Analyze other files
        for file_type, file_path in [
            ('submission', dataset_data.submission_path),
            ('validation', dataset_data.validation_path)
        ]:
            if file_path:
                file_info = self._analyze_single_file(file_path)
                if file_info:
                    analysis_results.update({
                        f'{file_type}_records': file_info.record_count,
                        f'{file_type}_columns': file_info.column_count,
                        f'{file_type}_format': file_info.file_format
                    })
        
        return analysis_results
    
    def _analyze_single_file(self, file_path: str) -> Optional[DatasetFileInfo]:
        """
        Analyze a single dataset file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            File information or None if analysis failed
        """
        try:
            import pandas as pd
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.logger.warning(f"File does not exist: {file_path}")
                return None
            
            # Detect file format
            file_format = file_path_obj.suffix.lower().lstrip('.')
            if file_format == 'gz' and file_path_obj.suffixes:
                # Handle .csv.gz format
                file_format = ''.join(file_path_obj.suffixes[-2:]).lstrip('.')
            
            # Load file based on format
            if file_format in ['csv', 'csv.gz']:
                df = pd.read_csv(file_path)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_format == 'json':
                df = pd.read_json(file_path)
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                self.logger.warning(f"Unsupported file format: {file_format}")
                return None
            
            # Get file size
            file_size_mb = file_path_obj.stat().st_size / 1024 / 1024
            
            # Analyze columns
            column_names = df.columns.tolist()
            column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
            missing_values = df.isnull().sum().to_dict()
            
            # Get sample data (first 3 rows)
            sample_data = {}
            for col in column_names[:5]:  # Limit to first 5 columns
                sample_data[col] = df[col].head(3).tolist()
            
            return DatasetFileInfo(
                file_path=file_path,
                file_format=file_format,
                file_size_mb=file_size_mb,
                record_count=len(df),
                column_count=len(df.columns),
                column_names=column_names,
                column_types=column_types,
                missing_values=missing_values,
                sample_data=sample_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze file {file_path}: {e}")
            return None
    
    def update_dataset(self, dataset_id: str, update_data: DatasetUpdate) -> Optional[Dataset]:
        """
        Update an existing dataset.
        
        Args:
            dataset_id: Dataset ID to update
            update_data: Fields to update
            
        Returns:
            Updated dataset or None if not found
        """
        # Get current dataset
        dataset = self.find_by_id(dataset_id, 'dataset_id')
        if not dataset:
            return None
        
        # Apply updates
        update_dict = update_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(dataset, field, value)
        
        return self.save(dataset)
    
    def find_by_name(self, dataset_name: str) -> Optional[Dataset]:
        """
        Find dataset by name.
        
        Args:
            dataset_name: Dataset name to search for
            
        Returns:
            Dataset or None if not found
        """
        return super().find_by_id(dataset_name, 'dataset_name')
    
    def get_active_datasets(self, limit: Optional[int] = None) -> List[Dataset]:
        """
        Get all active datasets.
        
        Args:
            limit: Maximum number of datasets to return
            
        Returns:
            List of active datasets
        """
        return self.find_all(
            where_clause="is_active = TRUE",
            order_by="last_used DESC NULLS LAST, created_at DESC",
            limit=limit
        )
    
    def search_datasets(self, search_term: str, active_only: bool = True) -> List[Dataset]:
        """
        Search datasets by name, competition, or description.
        
        Args:
            search_term: Term to search for
            active_only: Whether to include only active datasets
            
        Returns:
            List of matching datasets
        """
        where_clause = """
        (dataset_name LIKE ? OR competition_name LIKE ? OR description LIKE ?)
        """
        params = [f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"]
        
        if active_only:
            where_clause += " AND is_active = TRUE"
        
        return self.find_all(
            where_clause=where_clause,
            params=tuple(params),
            order_by="dataset_name ASC"
        )
    
    def get_datasets_by_competition(self, competition_name: str,
                                  active_only: bool = True) -> List[Dataset]:
        """
        Get datasets for a specific competition.
        
        Args:
            competition_name: Competition name to filter by
            active_only: Whether to include only active datasets
            
        Returns:
            List of datasets for the competition
        """
        where_clause = "competition_name = ?"
        params = [competition_name]
        
        if active_only:
            where_clause += " AND is_active = TRUE"
        
        return self.find_all(
            where_clause=where_clause,
            params=tuple(params),
            order_by="created_at DESC"
        )
    
    def mark_dataset_used(self, dataset_id: str) -> bool:
        """
        Mark dataset as recently used.
        
        Args:
            dataset_id: Dataset ID to mark as used
            
        Returns:
            True if dataset was updated, False if not found
        """
        update_data = DatasetUpdate(last_used=datetime.now())
        updated_dataset = self.update_dataset(dataset_id, update_data)
        
        if updated_dataset:
            self.logger.debug(f"Marked dataset {dataset_id[:8]}... as used")
            return True
        
        return False
    
    def deactivate_dataset(self, dataset_id: str) -> bool:
        """
        Deactivate a dataset instead of deleting it.
        
        Args:
            dataset_id: Dataset ID to deactivate
            
        Returns:
            True if dataset was deactivated, False if not found
        """
        update_data = DatasetUpdate(is_active=False)
        updated_dataset = self.update_dataset(dataset_id, update_data)
        
        if updated_dataset:
            self.logger.info(f"Deactivated dataset: {dataset_id[:8]}...")
            return True
        
        return False
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get overall dataset registry statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats_query = """
        SELECT 
            COUNT(*) as total_datasets,
            COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_datasets,
            COUNT(CASE WHEN test_path IS NOT NULL THEN 1 END) as datasets_with_test,
            COUNT(CASE WHEN validation_path IS NOT NULL THEN 1 END) as datasets_with_validation,
            COUNT(DISTINCT competition_name) as unique_competitions,
            AVG(train_records) as avg_train_size,
            SUM(CASE WHEN data_size_mb IS NOT NULL THEN data_size_mb ELSE 0 END) as total_size_mb
        FROM datasets
        """
        
        result = self.execute_custom_query(stats_query, fetch='one')
        
        if result:
            return {
                'total_datasets': result[0] or 0,
                'active_datasets': result[1] or 0,
                'datasets_with_test': result[2] or 0,
                'datasets_with_validation': result[3] or 0,
                'unique_competitions': result[4] or 0,
                'avg_train_size': float(result[5]) if result[5] else 0.0,
                'total_size_gb': (result[6] or 0) / 1024.0
            }
        
        return {
            'total_datasets': 0,
            'active_datasets': 0,
            'datasets_with_test': 0,
            'datasets_with_validation': 0,
            'unique_competitions': 0,
            'avg_train_size': 0.0,
            'total_size_gb': 0.0
        }
    
    def get_dataset_analysis(self) -> DatasetAnalysis:
        """
        Get comprehensive analysis of the dataset registry.
        
        Returns:
            Dataset analysis results
        """
        # Get basic statistics
        stats = self.get_dataset_statistics()
        
        # Get most used datasets (by last_used and session count)
        usage_query = """
        SELECT d.dataset_name, COUNT(s.session_id) as usage_count
        FROM datasets d
        LEFT JOIN sessions s ON d.dataset_id = s.dataset_hash
        WHERE d.is_active = TRUE
        GROUP BY d.dataset_id, d.dataset_name
        ORDER BY usage_count DESC, d.last_used DESC NULLS LAST
        LIMIT 5
        """
        
        usage_results = self.execute_custom_query(usage_query, fetch='all')
        most_used_datasets = [row[0] for row in usage_results] if usage_results else []
        
        # Get file format distribution
        format_query = """
        SELECT train_format, COUNT(*) as count
        FROM datasets
        WHERE is_active = TRUE AND train_format IS NOT NULL
        GROUP BY train_format
        ORDER BY count DESC
        """
        
        format_results = self.execute_custom_query(format_query, fetch='all')
        file_format_distribution = {row[0]: row[1] for row in format_results}
        
        # Get competition distribution
        competition_query = """
        SELECT 
            COALESCE(competition_name, 'Unknown') as competition,
            COUNT(*) as count
        FROM datasets
        WHERE is_active = TRUE
        GROUP BY competition_name
        ORDER BY count DESC
        """
        
        competition_results = self.execute_custom_query(competition_query, fetch='all')
        competition_distribution = {row[0]: row[1] for row in competition_results}
        
        return DatasetAnalysis(
            total_datasets=stats['total_datasets'],
            active_datasets=stats['active_datasets'],
            datasets_with_test=stats['datasets_with_test'],
            datasets_with_validation=stats['datasets_with_validation'],
            avg_train_size=stats['avg_train_size'],
            total_data_size_gb=stats['total_size_gb'],
            most_used_datasets=most_used_datasets,
            file_format_distribution=file_format_distribution,
            competition_distribution=competition_distribution
        )
    
    def cleanup_unused_datasets(self, days: int = 90) -> int:
        """
        Mark datasets as inactive if they haven't been used for a long time.
        
        Args:
            days: Number of days after which to mark datasets as inactive
            
        Returns:
            Number of datasets marked as inactive
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        update_query = """
        UPDATE datasets 
        SET is_active = FALSE
        WHERE is_active = TRUE 
        AND (last_used IS NULL OR last_used < ?)
        AND created_at < ?
        """
        
        self.execute_custom_query(
            update_query,
            (cutoff_date.isoformat(), cutoff_date.isoformat()),
            fetch='none'
        )
        
        # Get count of updated datasets
        count_query = "SELECT changes()"
        result = self.execute_custom_query(count_query, fetch='one')
        
        cleanup_count = result[0] if result else 0
        if cleanup_count > 0:
            self.logger.info(f"Marked {cleanup_count} unused datasets as inactive")
        
        return cleanup_count
    
    def get_by_name(self, dataset_name: str) -> Optional[Dataset]:
        """Get dataset by name."""
        return self.find_by_name(dataset_name)
    
    def list_all(self, active_only: bool = True) -> List[Dataset]:
        """List all datasets."""
        if active_only:
            return self.get_active_datasets()
        else:
            return self.find_all(order_by="dataset_name ASC")


class DatasetUsageRepository(BaseRepository[DatasetUsage]):
    """
    Repository for dataset usage tracking.
    
    Handles tracking how datasets are used across sessions.
    """
    
    @property
    def table_name(self) -> str:
        """Return the dataset_usage table name (if it exists)."""
        # Note: This table doesn't exist in current schema, but could be added
        return "dataset_usage"
    
    @property
    def model_class(self) -> type:
        """Return the DatasetUsage model class."""
        return DatasetUsage
    
    def _row_to_model(self, row: Any) -> DatasetUsage:
        """Convert database row to DatasetUsage model."""
        # This would be implemented if dataset_usage table is created
        pass
    
    def _model_to_dict(self, model: DatasetUsage) -> Dict[str, Any]:
        """Convert DatasetUsage model to dictionary for database operations."""
        # This would be implemented if dataset_usage table is created
        pass