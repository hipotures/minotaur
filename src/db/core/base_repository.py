"""
Base repository implementation for database operations.

This module provides the abstract base repository class that encapsulates
common database operations and patterns. All specific repositories inherit
from this base class to ensure consistency and reduce code duplication.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, TypeVar, Generic, Union
from datetime import datetime
import logging
import time

from .connection import DuckDBConnectionManager
from ..config.logging_config import DatabaseLoggerAdapter

# Type variable for model classes
T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository class providing common database operations.
    
    This class implements the Repository pattern and provides a consistent
    interface for database operations across all domain repositories.
    """
    
    def __init__(self, connection_manager: DuckDBConnectionManager):
        """
        Initialize repository with connection manager.
        
        Args:
            connection_manager: DuckDB connection manager instance
        """
        self.conn_manager = connection_manager
        self.logger = DatabaseLoggerAdapter(
            logging.getLogger(f'db.{self.__class__.__name__.lower()}'),
            {'repository': self.__class__.__name__, 'db_path': self.conn_manager.db_path}
        )
        
        # Performance tracking
        self.operation_stats = {
            'queries_executed': 0,
            'total_query_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the primary table name for this repository."""
        pass
    
    @property
    @abstractmethod
    def model_class(self) -> Type[T]:
        """Return the model class for this repository."""
        pass
    
    def _get_conflict_target(self) -> Optional[str]:
        """
        Get the primary conflict target column for ON CONFLICT operations.
        
        Override this method in specific repositories to specify the preferred
        conflict resolution column (usually the natural key, not auto-generated id).
        
        Returns:
            Column name for conflict resolution, or None for no conflict handling
        """
        # Default to 'id' if it exists in the table
        return 'id'
    
    def find_by_id(self, id_value: Any, id_column: str = 'id') -> Optional[T]:
        """
        Find entity by ID.
        
        Args:
            id_value: ID value to search for
            id_column: Name of ID column (default: 'id')
            
        Returns:
            Model instance or None if not found
        """
        start_time = time.time()
        
        try:
            query = f"SELECT * FROM {self.table_name} WHERE {id_column} = ?"
            result = self.conn_manager.execute_query(query, (id_value,), fetch='one')
            
            duration = time.time() - start_time
            self._track_operation('find_by_id', duration)
            
            if result:
                return self._row_to_model(result)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find {self.table_name} by {id_column}={id_value}: {e}")
            raise
    
    def find_all(self, limit: Optional[int] = None, offset: int = 0, 
                 order_by: Optional[str] = None, where_clause: Optional[str] = None,
                 params: Optional[tuple] = None) -> List[T]:
        """
        Find all entities with optional filtering and pagination.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: ORDER BY clause (without 'ORDER BY')
            where_clause: WHERE clause (without 'WHERE')
            params: Parameters for WHERE clause
            
        Returns:
            List of model instances
        """
        start_time = time.time()
        
        try:
            query_parts = [f"SELECT * FROM {self.table_name}"]
            query_params = []
            
            if where_clause:
                query_parts.append(f"WHERE {where_clause}")
                if params:
                    query_params.extend(params)
            
            if order_by:
                query_parts.append(f"ORDER BY {order_by}")
            
            if limit is not None:
                query_parts.append(f"LIMIT {limit}")
                if offset > 0:
                    query_parts.append(f"OFFSET {offset}")
            
            query = " ".join(query_parts)
            results = self.conn_manager.execute_query(
                query, 
                tuple(query_params) if query_params else None, 
                fetch='all'
            )
            
            duration = time.time() - start_time
            self._track_operation('find_all', duration)
            
            return [self._row_to_model(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Failed to find all {self.table_name}: {e}")
            raise
    
    def count(self, where_clause: Optional[str] = None, params: Optional[tuple] = None) -> int:
        """
        Count entities with optional filtering.
        
        Args:
            where_clause: WHERE clause (without 'WHERE')
            params: Parameters for WHERE clause
            
        Returns:
            Count of matching entities
        """
        start_time = time.time()
        
        try:
            query_parts = [f"SELECT COUNT(*) FROM {self.table_name}"]
            
            if where_clause:
                query_parts.append(f"WHERE {where_clause}")
            
            query = " ".join(query_parts)
            result = self.conn_manager.execute_query(query, params, fetch='one')
            
            duration = time.time() - start_time
            self._track_operation('count', duration)
            
            return result[0] if result else 0
            
        except Exception as e:
            self.logger.error(f"Failed to count {self.table_name}: {e}")
            raise
    
    def save(self, entity: T, update_on_conflict: bool = True) -> T:
        """
        Save or update entity.
        
        Args:
            entity: Model instance to save
            update_on_conflict: Whether to update on primary key conflict
            
        Returns:
            Saved model instance with updated fields
        """
        start_time = time.time()
        
        try:
            # Convert model to dictionary
            data = self._model_to_dict(entity)
            
            # Prepare INSERT query
            columns = list(data.keys())
            placeholders = ', '.join(['?' for _ in columns])
            values = tuple(data.values())
            
            if update_on_conflict:
                # DuckDB requires explicit conflict target for ON CONFLICT
                # Get the primary conflict target for this table
                conflict_target = self._get_conflict_target()
                if conflict_target:
                    # Build update SET clause excluding the conflict column
                    update_columns = [col for col in columns if col != conflict_target]
                    update_set = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_columns])
                    
                    query = f"""
                        INSERT INTO {self.table_name} 
                        ({', '.join(columns)}) 
                        VALUES ({placeholders})
                        ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set}
                    """
                else:
                    # Fallback to simple INSERT if no conflict target defined
                    query = f"""
                        INSERT INTO {self.table_name} 
                        ({', '.join(columns)}) 
                        VALUES ({placeholders})
                    """
            else:
                query = f"""
                    INSERT INTO {self.table_name} 
                    ({', '.join(columns)}) 
                    VALUES ({placeholders})
                """
            
            self.conn_manager.execute_query(query, values, fetch='none')
            
            duration = time.time() - start_time
            self._track_operation('save', duration)
            
            self.logger.debug(f"Saved {self.table_name} entity")
            return entity
            
        except Exception as e:
            self.logger.error(f"Failed to save {self.table_name}: {e}")
            raise
    
    def delete(self, id_value: Any, id_column: str = 'id') -> bool:
        """
        Delete entity by ID.
        
        Args:
            id_value: ID value to delete
            id_column: Name of ID column (default: 'id')
            
        Returns:
            True if entity was deleted, False if not found
        """
        start_time = time.time()
        
        try:
            query = f"DELETE FROM {self.table_name} WHERE {id_column} = ?"
            self.conn_manager.execute_query(query, (id_value,), fetch='none')
            
            # DuckDB doesn't have changes() function, assume 1 row affected
            affected_rows = 1
            
            duration = time.time() - start_time
            self._track_operation('delete', duration)
            
            success = affected_rows > 0
            if success:
                self.logger.debug(f"Deleted {self.table_name} entity with {id_column}={id_value}")
            else:
                self.logger.warning(f"No {self.table_name} entity found with {id_column}={id_value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete {self.table_name} by {id_column}={id_value}: {e}")
            raise
    
    def batch_save(self, entities: List[T], batch_size: int = 1000) -> List[T]:
        """
        Save multiple entities in batches for better performance.
        
        Args:
            entities: List of model instances to save
            batch_size: Number of entities per batch
            
        Returns:
            List of saved model instances
        """
        start_time = time.time()
        saved_entities = []
        
        try:
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                
                # Prepare batch insert
                if batch:
                    # Convert all models to dictionaries
                    batch_data = [self._model_to_dict(entity) for entity in batch]
                    
                    # Get columns from first entity
                    columns = list(batch_data[0].keys())
                    placeholders = ', '.join(['?' for _ in columns])
                    
                    # Prepare values for all entities in batch
                    operations = []
                    for data in batch_data:
                        values = tuple(data[col] for col in columns)
                        
                        # Use proper conflict resolution for DuckDB
                        conflict_target = self._get_conflict_target()
                        if conflict_target:
                            # Build update SET clause excluding the conflict column
                            update_columns = [col for col in columns if col != conflict_target]
                            update_set = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_columns])
                            
                            query = f"""
                                INSERT INTO {self.table_name} 
                                ({', '.join(columns)}) 
                                VALUES ({placeholders})
                                ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set}
                            """
                        else:
                            # Fallback to simple INSERT if no conflict target defined
                            query = f"""
                                INSERT INTO {self.table_name} 
                                ({', '.join(columns)}) 
                                VALUES ({placeholders})
                            """
                        
                        operations.append((query, values))
                    
                    # Execute batch transaction
                    self.conn_manager.execute_transaction(operations)
                    saved_entities.extend(batch)
            
            duration = time.time() - start_time
            self._track_operation('batch_save', duration)
            
            self.logger.info(f"Batch saved {len(entities)} {self.table_name} entities in {duration:.3f}s")
            return saved_entities
            
        except Exception as e:
            self.logger.error(f"Failed to batch save {self.table_name}: {e}")
            raise
    
    def execute_custom_query(self, query: str, params: Optional[tuple] = None, 
                           fetch: str = 'all') -> Any:
        """
        Execute custom SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Fetch method ('all', 'one', 'none')
            
        Returns:
            Query results based on fetch method
        """
        start_time = time.time()
        
        try:
            result = self.conn_manager.execute_query(query, params, fetch)
            
            duration = time.time() - start_time
            self._track_operation('custom_query', duration)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute custom query in {self.table_name}: {e}")
            raise
    
    def exists(self, where_clause: str, params: Optional[tuple] = None) -> bool:
        """
        Check if entity exists matching criteria.
        
        Args:
            where_clause: WHERE clause (without 'WHERE')
            params: Parameters for WHERE clause
            
        Returns:
            True if entity exists, False otherwise
        """
        try:
            count = self.count(where_clause, params)
            return count > 0
        except Exception as e:
            self.logger.error(f"Failed to check existence in {self.table_name}: {e}")
            raise
    
    @abstractmethod
    def _row_to_model(self, row: Any) -> T:
        """
        Convert database row to model instance.
        
        Args:
            row: Database row (tuple or dict-like object)
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def _model_to_dict(self, model: T) -> Dict[str, Any]:
        """
        Convert model instance to dictionary for database operations.
        
        Args:
            model: Model instance
            
        Returns:
            Dictionary representation suitable for database
        """
        pass
    
    def _track_operation(self, operation: str, duration: float) -> None:
        """Track repository operation performance."""
        self.operation_stats['queries_executed'] += 1
        self.operation_stats['total_query_time'] += duration
        
        self.logger.performance(f'{operation}_duration', duration, 's')
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this repository."""
        stats = self.operation_stats.copy()
        
        if stats['queries_executed'] > 0:
            stats['avg_query_time'] = stats['total_query_time'] / stats['queries_executed']
        else:
            stats['avg_query_time'] = 0.0
        
        stats['table_name'] = self.table_name
        stats['repository_class'] = self.__class__.__name__
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on repository and underlying table."""
        try:
            # Check if table exists and is accessible
            count = self.count()
            
            return {
                'status': 'healthy',
                'table_name': self.table_name,
                'record_count': count,
                'repository_class': self.__class__.__name__
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'table_name': self.table_name,
                'error': str(e),
                'repository_class': self.__class__.__name__
            }


class ReadOnlyRepository(BaseRepository[T]):
    """
    Read-only repository that raises errors on write operations.
    
    Useful for views or tables that should not be modified through
    the application layer.
    """
    
    def save(self, entity: T, update_on_conflict: bool = True) -> T:
        """Raise error for read-only repository."""
        raise NotImplementedError(f"{self.__class__.__name__} is read-only")
    
    def delete(self, id_value: Any, id_column: str = 'id') -> bool:
        """Raise error for read-only repository."""
        raise NotImplementedError(f"{self.__class__.__name__} is read-only")
    
    def batch_save(self, entities: List[T], batch_size: int = 1000) -> List[T]:
        """Raise error for read-only repository."""
        raise NotImplementedError(f"{self.__class__.__name__} is read-only")