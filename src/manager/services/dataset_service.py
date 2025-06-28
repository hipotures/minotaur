"""
Service layer for dataset-related business logic
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import json
from ..repositories.dataset_repository import DatasetRepository
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
            dataset_id = self.repository.create_dataset(
                name=name,
                file_path=str(file_path),
                description=description,
                metadata=metadata
            )
            
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