"""
Repository for dataset-related database operations
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from .base import BaseRepository


class DatasetRepository(BaseRepository):
    """Handles all dataset-related database operations."""
    
    def get_all_datasets(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """Get all registered datasets.
        
        Args:
            include_inactive: Whether to include inactive datasets
            
        Returns:
            List of dataset dictionaries
        """
        query = """
        SELECT 
            dataset_id,
            dataset_name,
            train_path,
            test_path,
            submission_path,
            validation_path,
            target_column,
            id_column,
            competition_name,
            description,
            train_records,
            train_columns,
            test_records,
            test_columns,
            created_at,
            last_used,
            is_active,
            data_size_mb
        FROM datasets
        """
        
        if not include_inactive:
            query += " WHERE is_active = true"
        
        query += " ORDER BY created_at DESC"
        
        return self.fetch_all(query)
    
    def get_dataset_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dataset by name.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset dictionary or None
        """
        query = """
        SELECT 
            d.dataset_id,
            d.dataset_name,
            d.train_path,
            d.test_path,
            d.submission_path,
            d.validation_path,
            d.target_column,
            d.id_column,
            d.competition_name,
            d.description,
            d.train_records,
            d.train_columns,
            d.test_records,
            d.test_columns,
            d.created_at,
            d.last_used,
            d.is_active,
            d.data_size_mb,
            d.feature_types,
            COUNT(DISTINCT s.session_id) as session_count,
            AVG(s.best_score) as avg_score,
            MAX(s.best_score) as max_score
        FROM datasets d
        LEFT JOIN sessions s ON d.dataset_id = s.dataset_hash
        WHERE d.dataset_name = ?
        GROUP BY d.dataset_id, d.dataset_name, d.train_path, d.test_path, 
                 d.submission_path, d.validation_path, d.target_column, d.id_column,
                 d.competition_name, d.description, d.train_records, d.train_columns,
                 d.test_records, d.test_columns, d.created_at, d.last_used,
                 d.is_active, d.data_size_mb, d.feature_types
        """
        
        return self.fetch_one(query, (name,))
    
    def get_dataset_by_id(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset by ID.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset dictionary or None
        """
        query = """
        SELECT 
            dataset_id,
            dataset_name,
            train_path,
            test_path,
            submission_path,
            validation_path,
            target_column,
            id_column,
            competition_name,
            description,
            train_records,
            train_columns,
            test_records,
            test_columns,
            created_at,
            last_used,
            is_active,
            data_size_mb,
            feature_types
        FROM datasets
        WHERE dataset_id = ?
        """
        
        return self.fetch_one(query, (dataset_id,))
    
    def create_dataset(self, dataset_data: Dict[str, Any]) -> str:
        """Create a new dataset entry.
        
        Args:
            dataset_data: Dictionary containing dataset information
            
        Returns:
            Created dataset ID
        """
        import uuid
        import json
        
        dataset_id = dataset_data.get('dataset_id', str(uuid.uuid4()))
        
        # Required fields
        required_fields = ['dataset_name', 'train_path', 'target_column']
        for field in required_fields:
            if field not in dataset_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Prepare values with defaults
        values = {
            'dataset_id': dataset_id,
            'dataset_name': dataset_data['dataset_name'],
            'train_path': dataset_data['train_path'],
            'test_path': dataset_data.get('test_path'),
            'submission_path': dataset_data.get('submission_path'),
            'validation_path': dataset_data.get('validation_path'),
            'target_column': dataset_data['target_column'],
            'id_column': dataset_data.get('id_column'),
            'competition_name': dataset_data.get('competition_name'),
            'description': dataset_data.get('description'),
            'train_records': dataset_data.get('train_records'),
            'train_columns': dataset_data.get('train_columns'),
            'test_records': dataset_data.get('test_records'),
            'test_columns': dataset_data.get('test_columns'),
            'submission_records': dataset_data.get('submission_records'),
            'submission_columns': dataset_data.get('submission_columns'),
            'validation_records': dataset_data.get('validation_records'),
            'validation_columns': dataset_data.get('validation_columns'),
            'train_format': dataset_data.get('train_format'),
            'test_format': dataset_data.get('test_format'),
            'submission_format': dataset_data.get('submission_format'),
            'validation_format': dataset_data.get('validation_format'),
            'column_count': dataset_data.get('column_count'),
            'train_row_count': dataset_data.get('train_row_count'),
            'test_row_count': dataset_data.get('test_row_count'),
            'data_size_mb': dataset_data.get('data_size_mb'),
            'feature_types': json.dumps(dataset_data.get('feature_types', {})),
            'created_at': datetime.now().isoformat(),
            'is_active': True
        }
        
        # Build INSERT query
        columns = list(values.keys())
        placeholders = ['?' for _ in columns]
        
        query = f"""
        INSERT INTO datasets ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        self.execute(query, tuple(values.values()))
        
        return dataset_id
    
    def update_dataset(self, dataset_id: str, **kwargs) -> bool:
        """Update dataset fields.
        
        Args:
            dataset_id: Dataset ID to update
            **kwargs: Fields to update
            
        Returns:
            True if updated
        """
        # Allowed update fields
        allowed_fields = {
            'dataset_name', 'description', 'test_path', 'submission_path',
            'validation_path', 'is_active', 'last_used', 'feature_types'
        }
        
        update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not update_fields:
            return False
        
        # Build UPDATE query
        set_clauses = []
        values = []
        
        for field, value in update_fields.items():
            set_clauses.append(f"{field} = ?")
            if field == 'feature_types' and isinstance(value, dict):
                values.append(json.dumps(value))
            else:
                values.append(value)
        
        values.append(dataset_id)
        
        query = f"""
        UPDATE datasets 
        SET {', '.join(set_clauses)}
        WHERE dataset_id = ?
        """
        
        self.execute(query, tuple(values))
        return True
    
    def update_last_used(self, dataset_id: str) -> None:
        """Update the last_used timestamp for a dataset.
        
        Args:
            dataset_id: Dataset ID to update
        """
        query = """
        UPDATE datasets 
        SET last_used = ?
        WHERE dataset_id = ?
        """
        
        self.execute(query, (datetime.now().isoformat(), dataset_id))
    
    def get_dataset_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dictionary of statistics
        """
        # Session statistics
        session_stats_query = """
        SELECT 
            COUNT(*) as total_sessions,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions,
            AVG(CASE WHEN status = 'completed' THEN best_score END) as avg_score,
            MAX(CASE WHEN status = 'completed' THEN best_score END) as max_score,
            AVG(CASE WHEN status = 'completed' THEN total_iterations END) as avg_iterations,
            SUM(CASE WHEN status = 'completed' THEN total_iterations END) as total_iterations
        FROM sessions
        WHERE dataset_hash = ?
        """
        
        session_stats = self.fetch_one(session_stats_query, (dataset_id,))
        
        # Feature statistics
        feature_stats_query = """
        SELECT 
            COUNT(DISTINCT fi.feature_name) as unique_features,
            COUNT(*) as total_feature_evaluations,
            AVG(fi.impact_delta) as avg_feature_impact
        FROM feature_impact fi
        JOIN sessions s ON fi.session_id = s.session_id
        WHERE s.dataset_hash = ?
        """
        
        feature_stats = self.fetch_one(feature_stats_query, (dataset_id,))
        
        # Top features
        top_features_query = """
        SELECT 
            fi.feature_name,
            fc.feature_category,
            AVG(fi.impact_delta) as avg_impact,
            COUNT(*) as use_count
        FROM feature_impact fi
        JOIN sessions s ON fi.session_id = s.session_id
        JOIN feature_catalog fc ON fi.feature_name = fc.feature_name
        WHERE s.dataset_hash = ?
        GROUP BY fi.feature_name, fc.feature_category
        ORDER BY avg_impact DESC
        LIMIT 10
        """
        
        top_features = self.fetch_all(top_features_query, (dataset_id,))
        
        return {
            'session_statistics': session_stats,
            'feature_statistics': feature_stats,
            'top_features': top_features
        }
    
    def search_datasets(self, query: str) -> List[Dict[str, Any]]:
        """Search datasets by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching datasets
        """
        search_pattern = f"%{query}%"
        
        query_sql = """
        SELECT 
            dataset_id,
            dataset_name,
            description,
            train_path,
            target_column,
            created_at,
            is_active,
            train_records,
            test_records
        FROM datasets
        WHERE (dataset_name ILIKE ? OR description ILIKE ? OR competition_name ILIKE ?)
        AND is_active = true
        ORDER BY 
            CASE 
                WHEN dataset_name = ? THEN 1
                WHEN dataset_name ILIKE ? THEN 2
                ELSE 3
            END,
            created_at DESC
        """
        
        # Priority: exact match, starts with, contains
        exact_pattern = query
        starts_pattern = f"{query}%"
        
        return self.fetch_all(query_sql, (
            search_pattern, search_pattern, search_pattern,
            exact_pattern, starts_pattern
        ))
    
    def get_datasets_by_competition(self, competition_name: str) -> List[Dict[str, Any]]:
        """Get all datasets for a specific competition.
        
        Args:
            competition_name: Name of the competition
            
        Returns:
            List of dataset dictionaries
        """
        query = """
        SELECT 
            dataset_id,
            dataset_name,
            description,
            train_path,
            test_path,
            target_column,
            created_at,
            train_records,
            test_records,
            data_size_mb
        FROM datasets
        WHERE competition_name = ? AND is_active = true
        ORDER BY created_at DESC
        """
        
        return self.fetch_all(query, (competition_name,))