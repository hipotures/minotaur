"""
Service layer for dataset-related business logic using SQLAlchemy abstraction layer.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime
from ...database.engine_factory import DatabaseFactory
from ..core.utils import format_number, format_bytes, format_datetime


class DatasetService:
    """Handles dataset-related business logic using new database abstraction."""
    
    def __init__(self, db_manager):
        """Initialize service with database manager.
        
        Args:
            db_manager: Database manager instance from factory
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Ensure datasets table exists
        self._ensure_datasets_table()
        
        # Legacy compatibility - modules expect a repository attribute
        self.repository = self
    
    def _ensure_datasets_table(self):
        """Ensure datasets table exists."""
        self.logger.info("Creating datasets table if not exists...")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS datasets (
            name VARCHAR UNIQUE NOT NULL,
            dataset_path VARCHAR NOT NULL,
            train_path VARCHAR,
            test_path VARCHAR,
            validation_path VARCHAR,
            submission_path VARCHAR,
            target_column VARCHAR,
            id_column VARCHAR,
            dataset_type VARCHAR DEFAULT 'csv',
            description VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT true
        )
        """
        self.db_manager.execute_ddl(create_table_query)
        self.logger.info("Datasets table creation completed")
    
    def list_datasets(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """List all datasets with formatted information.
        
        Args:
            include_inactive: Whether to include inactive datasets
            
        Returns:
            List of dataset dictionaries with formatted information
        """
        where_clause = "" if include_inactive else "WHERE is_active = true"
        query = f"""
        SELECT * FROM datasets 
        {where_clause}
        ORDER BY created_at DESC
        """
        
        datasets = self.db_manager.execute_query(query)
        
        # Format datasets for display
        formatted_datasets = []
        for dataset in datasets:
            formatted = dict(dataset)
            
            # Format file size
            if formatted.get('file_size_bytes'):
                formatted['file_size_display'] = format_bytes(formatted['file_size_bytes'])
            
            # Format dates
            if formatted.get('created_at'):
                formatted['created_at_display'] = format_datetime(formatted['created_at'])
            
            # Add row/column count display
            if formatted.get('rows_count') and formatted.get('columns_count'):
                formatted['shape_display'] = f"{format_number(formatted['rows_count'])} × {formatted['columns_count']}"
            
            formatted_datasets.append(formatted)
        
        return formatted_datasets
    
    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dataset by name.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset dictionary or None if not found
        """
        query = """
        SELECT name, dataset_path, train_path, test_path, validation_path, 
               submission_path, target_column, id_column, dataset_type, 
               description, created_at, updated_at, is_active
        FROM datasets 
        WHERE name = :name
        """
        results = self.db_manager.execute_query(query, {'name': name})
        return results[0] if results else None
    
    def register_dataset(self, name: str, dataset_path: str, target_column: str = None,
                        description: str = None, auto_analyze: bool = True) -> Dict[str, Any]:
        """Register a new dataset.
        
        Args:
            name: Dataset name (must be unique)
            dataset_path: Path to dataset file
            target_column: Target column for ML tasks
            description: Optional description
            auto_analyze: Whether to automatically analyze dataset
            
        Returns:
            Dictionary with registration result
        """
        # Ensure table exists first  
        self._ensure_datasets_table()
        
        path = Path(dataset_path)
        
        # Validate path exists
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Auto-analyze if requested
        rows_count = None
        columns_count = None
        file_size_bytes = None
        
        if auto_analyze:
            try:
                file_size_bytes = path.stat().st_size
                
                # Basic dataset analysis
                if path.suffix.lower() == '.csv':
                    import pandas as pd
                    # Read just a sample to get column count
                    df_sample = pd.read_csv(path, nrows=0)
                    columns_count = len(df_sample.columns)
                    
                    # Estimate row count efficiently
                    import subprocess
                    result = subprocess.run(['wc', '-l', str(path)], capture_output=True, text=True)
                    if result.returncode == 0:
                        rows_count = int(result.stdout.split()[0]) - 1  # Subtract header
                        
            except Exception as e:
                self.logger.warning(f"Could not analyze dataset {name}: {e}")
        
        # Insert dataset record (id will be auto-generated)
        insert_query = """
        INSERT INTO datasets 
        (name, dataset_path, target_column, description, rows_count, columns_count, file_size_bytes)
        VALUES (:name, :dataset_path, :target_column, :description, :rows_count, :columns_count, :file_size_bytes)
        """
        
        params = {
            'name': name,
            'dataset_path': str(path.resolve()),
            'target_column': target_column,
            'description': description,
            'rows_count': rows_count,
            'columns_count': columns_count,
            'file_size_bytes': file_size_bytes
        }
        
        try:
            self.db_manager.execute_dml(insert_query, params)
            self.logger.info(f"Registered dataset: {name}")
            
            return {
                'success': True,
                'message': f"Dataset '{name}' registered successfully",
                'dataset': self.get_dataset(name)
            }
            
        except Exception as e:
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Dataset with name '{name}' already exists")
            else:
                raise
    
    def update_dataset(self, name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update dataset information.
        
        Args:
            name: Dataset name
            updates: Dictionary of fields to update
            
        Returns:
            Updated dataset information
        """
        if not updates:
            return self.get_dataset(name)
        
        # Build update query
        set_clauses = []
        params = {'name': name}
        
        for key, value in updates.items():
            if key != 'name':  # Don't allow name changes
                set_clauses.append(f"{key} = :{key}")
                params[key] = value
        
        if set_clauses:
            # Add updated_at timestamp
            set_clauses.append("updated_at = :updated_at")
            params['updated_at'] = datetime.now()
            
            query = f"UPDATE datasets SET {', '.join(set_clauses)} WHERE name = :name"
            self.db_manager.execute_query(query, params)
            
            self.logger.info(f"Updated dataset: {name}")
        
        return self.get_dataset(name)
    
    def deactivate_dataset(self, name: str) -> Dict[str, Any]:
        """Deactivate a dataset (soft delete).
        
        Args:
            name: Dataset name
            
        Returns:
            Result of deactivation
        """
        query = "UPDATE datasets SET is_active = false, updated_at = :updated_at WHERE name = :name"
        params = {'name': name, 'updated_at': datetime.now()}
        
        self.db_manager.execute_query(query, params)
        self.logger.info(f"Deactivated dataset: {name}")
        
        return {'success': True, 'message': f"Dataset '{name}' deactivated"}
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get overall dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats_query = """
        SELECT 
            COUNT(*) as total_datasets,
            COUNT(CASE WHEN is_active = true THEN 1 END) as active_datasets,
            SUM(file_size_bytes) as total_size_bytes,
            SUM(rows_count) as total_rows,
            AVG(columns_count) as avg_columns
        FROM datasets
        """
        
        results = self.db_manager.execute_query(stats_query)
        stats = results[0] if results else {}
        
        # Format results
        formatted_stats = {
            'total_datasets': stats.get('total_datasets', 0),
            'active_datasets': stats.get('active_datasets', 0),
            'total_size_display': format_bytes(stats.get('total_size_bytes', 0)),
            'total_rows_display': format_number(stats.get('total_rows', 0)),
            'avg_columns': round(stats.get('avg_columns', 0), 1) if stats.get('avg_columns') else 0
        }
        
        return formatted_stats
    
    def search_datasets(self, query: str) -> List[Dict[str, Any]]:
        """Search datasets by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching datasets
        """
        search_query = """
        SELECT * FROM datasets 
        WHERE (name LIKE :query OR description LIKE :query)
        AND is_active = true
        ORDER BY created_at DESC
        """
        
        params = {'query': f'%{query}%'}
        return self.db_manager.execute_query(search_query, params)
    
    def get_all_datasets(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """Legacy compatibility method - same as list_datasets."""
        return self.list_datasets(include_inactive)
    
    def register_dataset_auto(self, name: str, path: str, target_column: str = None,
                             id_column: str = None, competition_name: str = None,
                             description: str = None, force_update: bool = False,
                             mcts_feature: bool = False) -> Dict[str, Any]:
        """Auto-register dataset by scanning directory for standard files.
        
        Args:
            name: Dataset name
            path: Directory path containing dataset files
            target_column: Target column for ML
            id_column: ID column (optional)
            competition_name: Competition name (optional)
            description: Dataset description
            force_update: Whether to update if exists
            mcts_feature: Whether to enable MCTS features
            
        Returns:
            Registration result dictionary
        """
        try:
            # Ensure table exists first
            self._ensure_datasets_table()
            
            # Check if dataset already exists
            existing = self.get_dataset(name)
            if existing and not force_update:
                return {
                    'success': False,
                    'message': f"Dataset '{name}' already exists. Use --force-update to overwrite.",
                    'dataset': existing
                }
            
            # Auto-detect files in directory
            dataset_dir = Path(path)
            if not dataset_dir.exists():
                return {
                    'success': False,
                    'message': f"Directory not found: {path}",
                    'error': 'Directory not found'
                }
            
            if not dataset_dir.is_dir():
                return {
                    'success': False,
                    'message': f"Path is not a directory: {path}",
                    'error': 'Not a directory'
                }
            
            # Scan for standard file patterns
            file_paths = self._scan_dataset_directory(dataset_dir)
            
            if not file_paths.get('train_path'):
                return {
                    'success': False,
                    'message': f"No train file found in directory: {path}",
                    'error': 'Missing train file'
                }
            
            # Build description
            if not description:
                description = f"Auto-registered dataset from {path}"
                if competition_name:
                    description += f" (Competition: {competition_name})"
            
            # Prepare dataset info
            dataset_info = {
                'name': name,
                'dataset_path': str(dataset_dir.resolve()),
                'train_path': file_paths.get('train_path'),
                'test_path': file_paths.get('test_path'),
                'validation_path': file_paths.get('validation_path'),
                'submission_path': file_paths.get('submission_path'),
                'target_column': target_column,
                'id_column': id_column,
                'description': description
            }
            
            # Register or update
            if existing and force_update:
                return self._update_dataset_registration(name, dataset_info)
            else:
                return self._create_dataset_registration(dataset_info)
                
        except Exception as e:
            self.logger.error(f"Auto-registration failed for {name}: {e}")
            return {
                'success': False,
                'message': f"Auto-registration failed: {str(e)}",
                'error': str(e)
            }
    
    def _scan_dataset_directory(self, directory: Path) -> Dict[str, str]:
        """Scan directory for standard dataset files.
        
        Args:
            directory: Path to dataset directory
            
        Returns:
            Dictionary with paths to found files
        """
        files = {}
        
        # Common patterns for dataset files
        patterns = {
            'train_path': ['train.csv', 'training.csv', 'train_data.csv', 'train_set.csv'],
            'test_path': ['test.csv', 'testing.csv', 'test_data.csv', 'test_set.csv'],
            'validation_path': ['val.csv', 'valid.csv', 'validation.csv', 'dev.csv'],
            'submission_path': ['sample_submission.csv', 'submission.csv', 'sample_sub.csv', 
                               'gender_submission.csv', 'submission_format.csv']
        }
        
        # Scan for each file type
        for file_type, file_patterns in patterns.items():
            for pattern in file_patterns:
                file_path = directory / pattern
                if file_path.exists() and file_path.is_file():
                    files[file_type] = str(file_path.resolve())
                    break
        
        return files
    
    def _create_dataset_registration(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create new dataset registration.
        
        Args:
            dataset_info: Dataset information dictionary
            
        Returns:
            Registration result
        """
        try:
            # Insert dataset record
            insert_query = """
            INSERT INTO datasets 
            (name, dataset_path, train_path, test_path, validation_path, submission_path,
             target_column, id_column, description)
            VALUES (:name, :dataset_path, :train_path, :test_path, :validation_path, 
                    :submission_path, :target_column, :id_column, :description)
            """
            
            self.db_manager.execute_dml(insert_query, dataset_info)
            self.logger.info(f"Registered dataset: {dataset_info['name']}")
            
            return {
                'success': True,
                'message': f"Dataset '{dataset_info['name']}' registered successfully",
                'dataset': self.get_dataset(dataset_info['name'])
            }
            
        except Exception as e:
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Dataset with name '{dataset_info['name']}' already exists")
            else:
                raise
    
    def _update_dataset_registration(self, name: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing dataset registration.
        
        Args:
            name: Dataset name
            dataset_info: New dataset information
            
        Returns:
            Update result
        """
        # Build update query
        update_query = """
        UPDATE datasets 
        SET dataset_path = :dataset_path,
            train_path = :train_path,
            test_path = :test_path,
            validation_path = :validation_path,
            submission_path = :submission_path,
            target_column = :target_column,
            id_column = :id_column,
            description = :description,
            updated_at = :updated_at
        WHERE name = :name
        """
        
        dataset_info['updated_at'] = datetime.now()
        dataset_info['name'] = name
        
        self.db_manager.execute_dml(update_query, dataset_info)
        self.logger.info(f"Updated dataset: {name}")
        
        return {
            'success': True,
            'message': f"Dataset '{name}' updated successfully",
            'dataset': self.get_dataset(name)
        }
    
    def register_dataset_manual(self, name: str, train_path: str, test_path: str = None,
                              submission_path: str = None, validation_path: str = None,
                              target_column: str = None, id_column: str = None,
                              competition_name: str = None, description: str = None,
                              mcts_feature: bool = False) -> Dict[str, Any]:
        """Manually register dataset with explicit file paths.
        
        Args:
            name: Dataset name
            train_path: Path to training file
            test_path: Path to test file (optional)
            submission_path: Path to submission template (optional)
            validation_path: Path to validation file (optional)
            target_column: Target column name
            id_column: ID column name
            competition_name: Competition name
            description: Dataset description
            mcts_feature: Whether to enable MCTS features
            
        Returns:
            Registration result
        """
        try:
            # Ensure table exists
            self._ensure_datasets_table()
            
            # Validate train path exists
            train_file = Path(train_path)
            if not train_file.exists():
                return {
                    'success': False,
                    'message': f"Training file not found: {train_path}",
                    'error': 'File not found'
                }
            
            # Determine dataset directory from train file
            dataset_dir = train_file.parent
            
            # Build description
            if not description:
                description = f"Manually registered dataset"
                if competition_name:
                    description += f" (Competition: {competition_name})"
            
            # Prepare dataset info
            dataset_info = {
                'name': name,
                'dataset_path': str(dataset_dir.resolve()),
                'train_path': str(train_file.resolve()),
                'test_path': str(Path(test_path).resolve()) if test_path else None,
                'validation_path': str(Path(validation_path).resolve()) if validation_path else None,
                'submission_path': str(Path(submission_path).resolve()) if submission_path else None,
                'target_column': target_column,
                'id_column': id_column,
                'description': description
            }
            
            # Check if exists
            existing = self.get_dataset(name)
            if existing:
                return self._update_dataset_registration(name, dataset_info)
            else:
                return self._create_dataset_registration(dataset_info)
                
        except Exception as e:
            self.logger.error(f"Manual registration failed for {name}: {e}")
            return {
                'success': False,
                'message': f"Manual registration failed: {str(e)}",
                'error': str(e)
            }