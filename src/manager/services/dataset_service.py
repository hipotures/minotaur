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
            target_column VARCHAR,
            dataset_type VARCHAR DEFAULT 'csv',
            description VARCHAR,
            rows_count INTEGER,
            columns_count INTEGER,
            file_size_bytes INTEGER,
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
        query = "SELECT * FROM datasets WHERE name = :name"
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
        """Auto-register dataset with full analysis and feature extraction.
        
        Args:
            name: Dataset name
            path: Dataset path
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
            
            # Build description
            if not description:
                description = f"Auto-registered dataset from {path}"
                if competition_name:
                    description += f" (Competition: {competition_name})"
            
            # Register using standard method
            if existing and force_update:
                # Update existing dataset
                updates = {
                    'dataset_path': str(Path(path).resolve()),
                    'target_column': target_column,
                    'description': description
                }
                
                # Re-analyze if requested
                if force_update:
                    dataset_path = Path(path)
                    if dataset_path.exists() and dataset_path.suffix.lower() == '.csv':
                        try:
                            import pandas as pd
                            df_sample = pd.read_csv(dataset_path, nrows=0)
                            updates['columns_count'] = len(df_sample.columns)
                            
                            # Estimate row count
                            import subprocess
                            result = subprocess.run(['wc', '-l', str(dataset_path)], capture_output=True, text=True)
                            if result.returncode == 0:
                                updates['rows_count'] = int(result.stdout.split()[0]) - 1
                                
                            updates['file_size_bytes'] = dataset_path.stat().st_size
                        except Exception as e:
                            self.logger.warning(f"Could not re-analyze dataset {name}: {e}")
                
                updated_dataset = self.update_dataset(name, updates)
                return {
                    'success': True,
                    'message': f"Dataset '{name}' updated successfully",
                    'dataset': updated_dataset
                }
            else:
                # Register new dataset
                return self.register_dataset(
                    name=name,
                    dataset_path=path,
                    target_column=target_column,
                    description=description,
                    auto_analyze=True
                )
                
        except Exception as e:
            self.logger.error(f"Auto-registration failed for {name}: {e}")
            return {
                'success': False,
                'message': f"Auto-registration failed: {str(e)}",
                'error': str(e)
            }