"""
Service layer for dataset-related business logic
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import json
from ...db.repositories.dataset_repository import DatasetRepository
from ..core.utils import format_number, format_bytes, format_datetime


class DatasetService:
    """Handles dataset-related business logic."""
    
    def __init__(self, dataset_repository: DatasetRepository):
        """Initialize service with repository.
        
        Args:
            dataset_repository: Dataset repository instance
        """
        self.repository = dataset_repository
        self.logger = logging.getLogger(__name__)
    
    def list_datasets(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """List all datasets with formatted information.
        
        Args:
            include_inactive: Whether to include inactive datasets
            
        Returns:
            List of formatted dataset summaries
        """
        datasets = self.repository.get_all_datasets(include_inactive)
        
        formatted_datasets = []
        for dataset in datasets:
            # Parse metadata
            metadata = json.loads(dataset['metadata']) if dataset['metadata'] else {}
            
            formatted_datasets.append({
                'name': dataset['name'],
                'description': dataset.get('description', 'No description'),
                'created': format_datetime(dataset['created_at'], 'short'),
                'last_used': format_datetime(dataset['last_used'], 'relative') if dataset['last_used'] else 'Never',
                'sessions': format_number(dataset['session_count']),
                'status': 'Active' if dataset['is_active'] else 'Inactive',
                'size': format_bytes(metadata.get('file_size_bytes', 0)),
                'records': format_number(metadata.get('record_count', 0))
            })
        
        return formatted_datasets
    
    def get_dataset_details(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Detailed dataset information or None
        """
        dataset = self.repository.get_dataset_by_name(dataset_name)
        
        if not dataset:
            return None
        
        # Get statistics
        stats = self.repository.get_dataset_statistics(dataset_name)
        
        # Parse metadata
        metadata = json.loads(dataset['metadata']) if dataset['metadata'] else {}
        
        return {
            'basic_info': {
                'name': dataset['name'],
                'id': dataset['dataset_id'],
                'description': dataset.get('description', 'No description'),
                'file_path': dataset['file_path'],
                'created': format_datetime(dataset['created_at'], 'long'),
                'updated': format_datetime(dataset['updated_at'], 'long'),
                'status': 'Active' if dataset['is_active'] else 'Inactive'
            },
            'metadata': {
                'file_size': format_bytes(metadata.get('file_size_bytes', 0)),
                'record_count': format_number(metadata.get('record_count', 0)),
                'column_count': metadata.get('column_count', 0),
                'file_format': metadata.get('file_format', 'Unknown'),
                'target_column': metadata.get('target_column'),
                'id_column': metadata.get('id_column')
            },
            'usage_statistics': {
                'total_sessions': stats['session_statistics']['total_sessions'],
                'completed_sessions': stats['session_statistics']['completed_sessions'],
                'failed_sessions': stats['session_statistics']['failed_sessions'],
                'success_rate': (stats['session_statistics']['completed_sessions'] / 
                               max(1, stats['session_statistics']['total_sessions'])),
                'avg_score': stats['session_statistics']['avg_score'],
                'max_score': stats['session_statistics']['max_score']
            },
            'feature_statistics': stats['feature_statistics'],
            'top_features': stats['top_features'][:5]  # Top 5 features
        }
    
    def register_dataset(self, name: str, file_path: str, 
                        description: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a new dataset.
        
        Args:
            name: Dataset name
            file_path: Path to dataset file
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            Registration result
        """
        try:
            # Check if dataset already exists
            existing = self.repository.get_dataset_by_name(name)
            if existing:
                return {
                    'success': False,
                    'error': f'Dataset {name} already exists'
                }
            
            # Validate file exists
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {
                    'success': False,
                    'error': f'File not found: {file_path}'
                }
            
            # Add file info to metadata
            if metadata is None:
                metadata = {}
            
            metadata['file_size_bytes'] = file_path_obj.stat().st_size
            metadata['file_name'] = file_path_obj.name
            
            # Create dataset
            dataset_data = {
                'dataset_name': name,
                'train_path': str(file_path),
                'description': description,
                'target_column': 'unknown',  # Will be overridden by auto/manual registration
                'metadata': json.dumps(metadata, default=str) if metadata else None
            }
            dataset_id = self.repository.create_dataset(dataset_data)
            
            return {
                'success': True,
                'dataset_id': dataset_id,
                'message': f'Successfully registered dataset: {name}'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to register dataset: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_dataset(self, dataset_name: str, **updates) -> Dict[str, Any]:
        """Update dataset information.
        
        Args:
            dataset_name: Name of dataset to update
            **updates: Fields to update
            
        Returns:
            Update result
        """
        dataset = self.repository.get_dataset_by_name(dataset_name)
        
        if not dataset:
            return {
                'success': False,
                'error': f'Dataset {dataset_name} not found'
            }
        
        try:
            success = self.repository.update_dataset(dataset['dataset_id'], **updates)
            
            return {
                'success': success,
                'message': f'Successfully updated dataset: {dataset_name}' if success 
                          else 'No changes made'
            }
        except Exception as e:
            self.logger.error(f"Failed to update dataset: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_datasets(self, query: str) -> List[Dict[str, Any]]:
        """Search for datasets by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching datasets
        """
        results = self.repository.search_datasets(query)
        
        return [
            {
                'name': r['name'],
                'description': r.get('description', 'No description'),
                'created': format_datetime(r['created_at'], 'short'),
                'file_path': r['file_path']
            }
            for r in results
        ]
    
    def get_dataset_comparisons(self) -> Dict[str, Any]:
        """Compare performance across all datasets.
        
        Returns:
            Dataset comparison data
        """
        all_datasets = self.repository.get_all_datasets()
        
        comparisons = []
        for dataset in all_datasets:
            stats = self.repository.get_dataset_statistics(dataset['name'])
            
            comparisons.append({
                'dataset': dataset['name'],
                'sessions': stats['session_statistics']['total_sessions'],
                'avg_score': stats['session_statistics']['avg_score'],
                'max_score': stats['session_statistics']['max_score'],
                'unique_features': stats['feature_statistics']['unique_features'],
                'avg_feature_impact': stats['feature_statistics']['avg_feature_impact']
            })
        
        # Sort by average score
        comparisons.sort(key=lambda x: x['avg_score'], reverse=True)
        
        return {
            'datasets': comparisons,
            'summary': {
                'total_datasets': len(comparisons),
                'best_performing': comparisons[0]['dataset'] if comparisons else None,
                'most_active': max(comparisons, key=lambda x: x['sessions'])['dataset'] 
                             if comparisons else None
            }
        }
    
    def register_dataset_auto(self, name: str, path: str, 
                             target_column: Optional[str] = None,
                             id_column: Optional[str] = None,
                             competition_name: Optional[str] = None,
                             description: Optional[str] = None,
                             force_update: bool = False) -> Dict[str, Any]:
        """Auto-register dataset by detecting files in directory.
        
        Args:
            name: Dataset name
            path: Path to dataset directory
            target_column: Target column name
            id_column: ID column name
            competition_name: Competition name
            description: Dataset description
            
        Returns:
            Registration result
        """
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return {
                    'success': False,
                    'error': f'Directory not found: {path}'
                }
            
            # Import dataset importer dynamically
            import sys
            import os
            # Get the absolute path to src directory
            current_dir = Path(__file__).resolve()
            src_dir = current_dir.parent.parent.parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            
            from dataset_importer import DatasetImporter
            
            # Initialize importer and auto-detect files
            importer = DatasetImporter(name)
            file_mappings = importer.auto_detect_files(str(path_obj))
            
            if 'train' not in file_mappings:
                return {
                    'success': False,
                    'error': 'No training file found in directory'
                }
            
            # Auto-detect target column if not provided
            if not target_column:
                target_column = importer.detect_target_column(file_mappings['train'])
                if not target_column:
                    return {
                        'success': False,
                        'error': 'Could not auto-detect target column. Please specify --target-column'
                    }
            
            # Auto-detect ID column if not provided
            if not id_column:
                id_column = importer.detect_id_column(file_mappings['train'])
            
            # Analyze files for metadata
            metadata = {}
            for table_name, file_path in file_mappings.items():
                try:
                    file_metadata = importer.analyze_file(file_path)
                    metadata[table_name] = file_metadata
                except Exception as e:
                    metadata[table_name] = {'error': str(e)}
            
            # Generate dataset ID BEFORE processing to check for duplicates
            dataset_id = importer.generate_dataset_id(file_mappings)
            
            # Check if dataset already exists
            existing_dataset = self.repository.get_dataset_by_name(name)
            if existing_dataset:
                if not force_update:
                    return {
                        'success': False,
                        'error': f'Dataset with name "{name}" already exists'
                    }
                else:
                    # Force update: remove existing dataset
                    self.repository.delete(existing_dataset.dataset_id, 'dataset_id')
                    logger = logging.getLogger(__name__)
                    logger.info(f"Force update: removed existing dataset '{name}' (ID: {existing_dataset.dataset_id[:8]})")
            
            # Check if dataset ID already exists (same data, different name)
            existing_by_id = self.repository.get_dataset_by_id(dataset_id)
            if existing_by_id:
                return {
                    'success': False,
                    'error': f'Dataset with identical data already exists as "{existing_by_id["name"]}"'
                }
            
            # Import data to DuckDB (only if no conflicts)
            duckdb_path = importer.create_duckdb_dataset(file_mappings, target_column, id_column)
            
            # Build metadata
            full_metadata = {
                'auto_detected': True,
                'source_directory': str(path_obj),
                'duckdb_path': str(duckdb_path),
                'file_mappings': file_mappings,
                'file_analysis': metadata,
                'target_column': target_column,
                'id_column': id_column,
                'competition_name': competition_name
            }
            
            # Register dataset
            dataset_data = {
                'dataset_id': dataset_id,
                'dataset_name': name,
                'train_path': file_mappings.get('train', ''),
                'test_path': file_mappings.get('test'),
                'submission_path': file_mappings.get('submission'),
                'validation_path': file_mappings.get('validation'),
                'target_column': target_column,
                'id_column': id_column,
                'competition_name': competition_name,
                'description': description or f'Auto-detected dataset from {path}',
                'metadata': json.dumps(full_metadata, default=str) if full_metadata else None
            }
            dataset_id = self.repository.create_dataset(dataset_data)
            
            return {
                'success': True,
                'dataset_id': dataset_id,
                'message': f'Successfully registered dataset: {name}'
            }
            
        except Exception as e:
            self.logger.error(f"Auto-registration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def register_dataset_manual(self, name: str, train_path: str,
                               test_path: Optional[str] = None,
                               submission_path: Optional[str] = None,
                               validation_path: Optional[str] = None,
                               target_column: Optional[str] = None,
                               id_column: Optional[str] = None,
                               competition_name: Optional[str] = None,
                               description: Optional[str] = None) -> Dict[str, Any]:
        """Manually register dataset with specified file paths.
        
        Args:
            name: Dataset name
            train_path: Training data file path
            test_path: Test data file path
            submission_path: Submission template file path
            validation_path: Validation data file path
            target_column: Target column name
            id_column: ID column name
            competition_name: Competition name
            description: Dataset description
            
        Returns:
            Registration result
        """
        try:
            # Validate train file exists
            if not Path(train_path).exists():
                return {
                    'success': False,
                    'error': f'Training file not found: {train_path}'
                }
            
            # Build metadata
            metadata = {
                'auto_detected': False,
                'train_file': train_path,
                'test_file': test_path,
                'submission_file': submission_path,
                'validation_file': validation_path,
                'target_column': target_column,
                'id_column': id_column,
                'competition_name': competition_name
            }
            
            # Register using the train file
            dataset_data = {
                'dataset_name': name,
                'train_path': train_path,
                'test_path': test_path,
                'submission_path': submission_path,
                'validation_path': validation_path,
                'target_column': target_column,
                'id_column': id_column,
                'competition_name': competition_name,
                'description': description or f'Manually registered dataset: {name}',
                'metadata': json.dumps(metadata, default=str) if metadata else None
            }
            dataset_id = self.repository.create_dataset(dataset_data)
            
            return {
                'success': True,
                'dataset_id': dataset_id,
                'message': f'Successfully registered dataset: {name}'
            }
            
        except Exception as e:
            self.logger.error(f"Manual registration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def cleanup_dataset(self, dataset_name: str, remove_sessions: bool = False) -> Dict[str, Any]:
        """Clean up dataset and optionally remove associated sessions.
        
        Args:
            dataset_name: Name of dataset to clean up
            remove_sessions: Whether to remove associated sessions
            
        Returns:
            Cleanup result
        """
        dataset = self.repository.get_dataset_by_name(dataset_name)
        
        if not dataset:
            return {
                'success': False,
                'error': f'Dataset {dataset_name} not found'
            }
        
        try:
            # Mark dataset as inactive
            self.repository.update_dataset(dataset['dataset_id'], is_active=False)
            
            result = {
                'success': True,
                'dataset_deactivated': True,
                'sessions_removed': 0
            }
            
            if remove_sessions:
                # This would require session repository access
                # For now, just return the intent
                result['message'] = 'Dataset deactivated. Session removal requires manual intervention.'
            else:
                result['message'] = f'Dataset {dataset_name} has been deactivated.'
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup dataset: {e}")
            return {
                'success': False,
                'error': str(e)
            }