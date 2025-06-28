"""
Feature repository implementation.

This module provides database operations for feature catalog management,
including feature registration, impact analysis, and performance tracking.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import json

from ..core.base_repository import BaseRepository
from ..core.connection import DuckDBConnectionManager
from ..models.feature import (
    Feature, FeatureImpact, OperationPerformance,
    FeatureCreate, FeatureUpdate, FeatureAnalysis,
    FeatureCategory, FeatureCreator
)


class FeatureRepository(BaseRepository[Feature]):
    """
    Repository for feature catalog operations.
    
    Handles all database operations related to feature management,
    including registration, impact tracking, and analysis.
    """
    
    @property
    def table_name(self) -> str:
        """Return the feature_catalog table name."""
        return "feature_catalog"
    
    @property
    def model_class(self) -> type:
        """Return the Feature model class."""
        return Feature
    
    def _get_conflict_target(self) -> Optional[str]:
        """Use feature_name as conflict target since it's the natural key."""
        return 'feature_name'
    
    def _row_to_model(self, row: Any) -> Feature:
        """Convert database row to Feature model."""
        # Handle both tuple and dict-like row objects
        if hasattr(row, 'keys'):
            # Dict-like object (SQLite Row)
            data = dict(row)
        else:
            # Tuple - map to known column order
            columns = [
                'id', 'feature_name', 'feature_category', 'python_code',
                'dependencies', 'description', 'created_by', 'creation_timestamp',
                'is_active', 'computational_cost', 'data_type'
            ]
            data = dict(zip(columns, row))
        
        # Parse JSON dependencies field
        if isinstance(data.get('dependencies'), str):
            try:
                data['dependencies'] = json.loads(data['dependencies'])
            except (json.JSONDecodeError, TypeError):
                data['dependencies'] = []
        
        # Convert string timestamp to datetime object
        if isinstance(data.get('creation_timestamp'), str):
            try:
                data['creation_timestamp'] = datetime.fromisoformat(data['creation_timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                data['creation_timestamp'] = datetime.now()
        
        # Handle enum fields
        if isinstance(data.get('feature_category'), str):
            try:
                data['feature_category'] = FeatureCategory(data['feature_category'])
            except ValueError:
                data['feature_category'] = FeatureCategory.FEATURE_TRANSFORMATIONS
        
        if isinstance(data.get('created_by'), str):
            try:
                data['created_by'] = FeatureCreator(data['created_by'])
            except ValueError:
                data['created_by'] = FeatureCreator.MCTS
        
        # Set defaults for missing fields
        data.setdefault('dependencies', [])
        data.setdefault('is_active', True)
        data.setdefault('computational_cost', 1.0)
        data.setdefault('data_type', 'float64')
        
        return Feature(**data)
    
    def _model_to_dict(self, model: Feature) -> Dict[str, Any]:
        """Convert Feature model to dictionary for database operations."""
        data = {
            'feature_name': model.feature_name,
            'feature_category': model.feature_category.value,
            'python_code': model.python_code,
            'dependencies': json.dumps(model.dependencies),
            'description': model.description,
            'created_by': model.created_by.value,
            'creation_timestamp': model.creation_timestamp.isoformat(),
            'is_active': model.is_active,
            'computational_cost': model.computational_cost,
            'data_type': model.data_type
        }
        
        # Include ID if it exists (for updates)
        if model.id is not None:
            data['id'] = model.id
        
        return data
    
    def register_feature(self, feature_data: FeatureCreate) -> Feature:
        """
        Register a new feature in the catalog.
        
        Args:
            feature_data: Feature creation data
            
        Returns:
            Registered feature with ID
        """
        feature = Feature(
            feature_name=feature_data.feature_name,
            feature_category=feature_data.feature_category,
            python_code=feature_data.python_code,
            dependencies=feature_data.dependencies,
            description=feature_data.description,
            created_by=feature_data.created_by,
            computational_cost=feature_data.computational_cost,
            data_type=feature_data.data_type
        )
        
        saved_feature = self.save(feature)
        self.logger.info(f"Registered feature: {feature_data.feature_name}")
        return saved_feature
    
    def update_feature(self, feature_name: str, update_data: FeatureUpdate) -> Optional[Feature]:
        """
        Update an existing feature.
        
        Args:
            feature_name: Feature name to update
            update_data: Fields to update
            
        Returns:
            Updated feature or None if not found
        """
        # Get current feature
        feature = self.find_by_name(feature_name)
        if not feature:
            return None
        
        # Apply updates
        update_dict = update_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(feature, field, value)
        
        return self.save(feature)
    
    def find_by_name(self, feature_name: str) -> Optional[Feature]:
        """
        Find feature by name.
        
        Args:
            feature_name: Feature name to search for
            
        Returns:
            Feature or None if not found
        """
        return super().find_by_id(feature_name, 'feature_name')
    
    def get_features_by_category(self, category: FeatureCategory, 
                               active_only: bool = True) -> List[Feature]:
        """
        Get features by category.
        
        Args:
            category: Feature category to filter by
            active_only: Whether to include only active features
            
        Returns:
            List of features in the category
        """
        where_clause = "feature_category = ?"
        params = [category.value]
        
        if active_only:
            where_clause += " AND is_active = TRUE"
        
        return self.find_all(
            where_clause=where_clause,
            params=tuple(params),
            order_by="feature_name ASC"
        )
    
    def get_features_by_creator(self, creator: FeatureCreator,
                              active_only: bool = True) -> List[Feature]:
        """
        Get features by creator.
        
        Args:
            creator: Feature creator to filter by
            active_only: Whether to include only active features
            
        Returns:
            List of features by the creator
        """
        where_clause = "created_by = ?"
        params = [creator.value]
        
        if active_only:
            where_clause += " AND is_active = TRUE"
        
        return self.find_all(
            where_clause=where_clause,
            params=tuple(params),
            order_by="creation_timestamp DESC"
        )
    
    def search_features(self, search_term: str, active_only: bool = True) -> List[Feature]:
        """
        Search features by name or description.
        
        Args:
            search_term: Term to search for
            active_only: Whether to include only active features
            
        Returns:
            List of matching features
        """
        where_clause = "(feature_name LIKE ? OR description LIKE ?)"
        params = [f"%{search_term}%", f"%{search_term}%"]
        
        if active_only:
            where_clause += " AND is_active = TRUE"
        
        return self.find_all(
            where_clause=where_clause,
            params=tuple(params),
            order_by="feature_name ASC"
        )
    
    def get_feature_code(self, feature_name: str) -> Optional[str]:
        """
        Get Python code for a specific feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Python code or None if feature not found or inactive
        """
        query = """
        SELECT python_code FROM feature_catalog 
        WHERE feature_name = ? AND is_active = TRUE
        """
        
        result = self.execute_custom_query(query, (feature_name,), fetch='one')
        return result[0] if result else None
    
    def get_feature_dependencies(self, feature_name: str) -> List[str]:
        """
        Get dependencies for a feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            List of dependency names
        """
        feature = self.find_by_name(feature_name)
        return feature.dependencies if feature else []
    
    def get_dependent_features(self, dependency_name: str) -> List[Feature]:
        """
        Get features that depend on a specific feature/column.
        
        Args:
            dependency_name: Name of the dependency
            
        Returns:
            List of features that depend on the given name
        """
        # Use JSON search to find features with this dependency
        query = """
        SELECT * FROM feature_catalog
        WHERE is_active = TRUE 
        AND json_extract(dependencies, '$') LIKE ?
        ORDER BY feature_name
        """
        
        # Search pattern for JSON array containing the dependency
        search_pattern = f'%"{dependency_name}"%'
        
        results = self.execute_custom_query(query, (search_pattern,), fetch='all')
        return [self._row_to_model(row) for row in results]
    
    def deactivate_feature(self, feature_name: str) -> bool:
        """
        Deactivate a feature instead of deleting it.
        
        Args:
            feature_name: Feature name to deactivate
            
        Returns:
            True if feature was deactivated, False if not found
        """
        update_data = FeatureUpdate(is_active=False)
        updated_feature = self.update_feature(feature_name, update_data)
        
        if updated_feature:
            self.logger.info(f"Deactivated feature: {feature_name}")
            return True
        
        return False
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Get overall feature catalog statistics.
        
        Returns:
            Dictionary with feature statistics
        """
        stats_query = """
        SELECT 
            COUNT(*) as total_features,
            COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_features,
            COUNT(DISTINCT feature_category) as unique_categories,
            COUNT(DISTINCT created_by) as unique_creators,
            AVG(computational_cost) as avg_computational_cost,
            MIN(creation_timestamp) as first_created,
            MAX(creation_timestamp) as last_created
        FROM feature_catalog
        """
        
        result = self.execute_custom_query(stats_query, fetch='one')
        
        if result:
            return {
                'total_features': result[0] or 0,
                'active_features': result[1] or 0,
                'unique_categories': result[2] or 0,
                'unique_creators': result[3] or 0,
                'avg_computational_cost': float(result[4]) if result[4] else 0.0,
                'first_created': result[5],
                'last_created': result[6]
            }
        
        return {
            'total_features': 0,
            'active_features': 0,
            'unique_categories': 0,
            'unique_creators': 0,
            'avg_computational_cost': 0.0,
            'first_created': None,
            'last_created': None
        }
    
    def get_features_analysis(self) -> FeatureAnalysis:
        """
        Get comprehensive analysis of the feature catalog.
        
        Returns:
            Feature analysis results
        """
        # Get basic statistics
        stats = self.get_feature_statistics()
        
        # Get features by category
        category_query = """
        SELECT feature_category, COUNT(*) as count
        FROM feature_catalog
        WHERE is_active = TRUE
        GROUP BY feature_category
        ORDER BY count DESC
        """
        
        category_results = self.execute_custom_query(category_query, fetch='all')
        features_by_category = {row[0]: row[1] for row in category_results}
        
        # Get features by creator
        creator_query = """
        SELECT created_by, COUNT(*) as count
        FROM feature_catalog
        WHERE is_active = TRUE
        GROUP BY created_by
        ORDER BY count DESC
        """
        
        creator_results = self.execute_custom_query(creator_query, fetch='all')
        features_by_creator = {row[0]: row[1] for row in creator_results}
        
        # Get most used dependencies
        dependencies_query = """
        SELECT dependency, COUNT(*) as usage_count
        FROM (
            SELECT json_extract(dependencies, '$[' || (value-1) || ']') as dependency
            FROM feature_catalog, json_each('[' || replace(replace(json_extract(dependencies, '$'), '[', ''), ']', '') || ']')
            WHERE is_active = TRUE AND json_array_length(dependencies) > 0
        )
        WHERE dependency IS NOT NULL AND dependency != ''
        GROUP BY dependency
        ORDER BY usage_count DESC
        LIMIT 10
        """
        
        dependencies_results = self.execute_custom_query(dependencies_query, fetch='all')
        most_used_dependencies = [row[0].strip('"') for row in dependencies_results]
        
        # Get top performing features (placeholder - would need feature_impact data)
        top_performing_features = []  # TODO: Implement when feature_impact is available
        
        return FeatureAnalysis(
            total_features=stats['total_features'],
            active_features=stats['active_features'],
            features_by_category=features_by_category,
            features_by_creator=features_by_creator,
            avg_computational_cost=stats['avg_computational_cost'],
            most_used_dependencies=most_used_dependencies,
            top_performing_features=top_performing_features
        )


class FeatureImpactRepository(BaseRepository[FeatureImpact]):
    """
    Repository for feature impact analysis operations.
    
    Handles tracking and analysis of feature performance impact.
    """
    
    @property
    def table_name(self) -> str:
        """Return the feature_impact table name."""
        return "feature_impact"
    
    @property
    def model_class(self) -> type:
        """Return the FeatureImpact model class."""
        return FeatureImpact
    
    def _row_to_model(self, row: Any) -> FeatureImpact:
        """Convert database row to FeatureImpact model."""
        # Handle both tuple and dict-like row objects
        if hasattr(row, 'keys'):
            data = dict(row)
        else:
            columns = [
                'id', 'feature_name', 'baseline_score', 'with_feature_score',
                'impact_delta', 'impact_percentage', 'evaluation_context',
                'sample_size', 'confidence_interval', 'statistical_significance',
                'first_discovered', 'last_evaluated', 'session_id'
            ]
            data = dict(zip(columns, row))
        
        # Parse JSON fields
        for json_field in ['evaluation_context', 'confidence_interval']:
            if isinstance(data.get(json_field), str):
                try:
                    data[json_field] = json.loads(data[json_field])
                except (json.JSONDecodeError, TypeError):
                    data[json_field] = [] if json_field == 'evaluation_context' else None
        
        # Convert timestamp fields
        for field in ['first_discovered', 'last_evaluated']:
            if isinstance(data.get(field), str):
                try:
                    data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    data[field] = datetime.now()
        
        # Set defaults
        data.setdefault('evaluation_context', [])
        data.setdefault('sample_size', 1)
        
        return FeatureImpact(**data)
    
    def _model_to_dict(self, model: FeatureImpact) -> Dict[str, Any]:
        """Convert FeatureImpact model to dictionary for database operations."""
        data = {
            'feature_name': model.feature_name,
            'baseline_score': model.baseline_score,
            'with_feature_score': model.with_feature_score,
            'impact_delta': model.impact_delta,
            'impact_percentage': model.impact_percentage,
            'evaluation_context': json.dumps(model.evaluation_context),
            'sample_size': model.sample_size,
            'confidence_interval': json.dumps(model.confidence_interval) if model.confidence_interval else None,
            'statistical_significance': model.statistical_significance,
            'first_discovered': model.first_discovered.isoformat(),
            'last_evaluated': model.last_evaluated.isoformat(),
            'session_id': model.session_id
        }
        
        if model.id is not None:
            data['id'] = model.id
        
        return data
    
    def update_feature_impact(self, feature_name: str, baseline_score: float,
                            with_feature_score: float, session_id: str,
                            context_features: List[str] = None) -> FeatureImpact:
        """
        Update impact analysis for a feature.
        
        Args:
            feature_name: Feature name
            baseline_score: Baseline score without feature
            with_feature_score: Score with feature included
            session_id: Session where impact was measured
            context_features: Other features in evaluation set
            
        Returns:
            Updated or created feature impact record
        """
        impact_delta = with_feature_score - baseline_score
        impact_percentage = (impact_delta / baseline_score) * 100 if baseline_score > 0 else 0
        
        # Check if record exists for this feature in current session
        existing = self.find_all(
            where_clause="feature_name = ? AND session_id = ?",
            params=(feature_name, session_id),
            limit=1
        )
        
        if existing:
            # Update existing record with running average
            impact = existing[0]
            old_sample_size = impact.sample_size
            new_sample_size = old_sample_size + 1
            
            # Running average of impact
            new_avg_impact = ((impact.impact_delta * old_sample_size) + impact_delta) / new_sample_size
            new_avg_percentage = (new_avg_impact / baseline_score) * 100 if baseline_score > 0 else 0
            
            impact.baseline_score = baseline_score
            impact.with_feature_score = with_feature_score
            impact.impact_delta = new_avg_impact
            impact.impact_percentage = new_avg_percentage
            impact.evaluation_context = context_features or []
            impact.sample_size = new_sample_size
            impact.last_evaluated = datetime.now()
            
            return self.save(impact)
        else:
            # Create new record
            impact = FeatureImpact(
                feature_name=feature_name,
                baseline_score=baseline_score,
                with_feature_score=with_feature_score,
                impact_delta=impact_delta,
                impact_percentage=impact_percentage,
                evaluation_context=context_features or [],
                sample_size=1,
                session_id=session_id
            )
            
            return self.save(impact)
    
    def get_top_performing_features(self, limit: int = 10,
                                  session_id: Optional[str] = None) -> List[FeatureImpact]:
        """
        Get top performing features by impact delta.
        
        Args:
            limit: Maximum number of features to return
            session_id: Optional session filter
            
        Returns:
            List of top performing features
        """
        where_clause = "impact_delta > 0"
        params = []
        
        if session_id:
            where_clause += " AND session_id = ?"
            params.append(session_id)
        
        return self.find_all(
            where_clause=where_clause,
            params=tuple(params) if params else None,
            order_by="impact_delta DESC",
            limit=limit
        )


class OperationPerformanceRepository(BaseRepository[OperationPerformance]):
    """
    Repository for operation performance tracking.
    
    Handles tracking and analysis of feature operation performance.
    """
    
    @property
    def table_name(self) -> str:
        """Return the operation_performance table name."""
        return "operation_performance"
    
    @property
    def model_class(self) -> type:
        """Return the OperationPerformance model class."""
        return OperationPerformance
    
    def _row_to_model(self, row: Any) -> OperationPerformance:
        """Convert database row to OperationPerformance model."""
        # Handle both tuple and dict-like row objects
        if hasattr(row, 'keys'):
            data = dict(row)
        else:
            columns = [
                'id', 'operation_name', 'operation_category', 'total_applications',
                'success_count', 'avg_improvement', 'best_improvement', 'worst_result',
                'avg_execution_time', 'last_used', 'effectiveness_score', 'session_id'
            ]
            data = dict(zip(columns, row))
        
        # Convert timestamp field
        if isinstance(data.get('last_used'), str):
            try:
                data['last_used'] = datetime.fromisoformat(data['last_used'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                data['last_used'] = datetime.now()
        
        # Set defaults for missing fields
        for field in ['total_applications', 'success_count']:
            data.setdefault(field, 0)
        
        for field in ['avg_improvement', 'best_improvement', 'worst_result', 'avg_execution_time', 'effectiveness_score']:
            data.setdefault(field, 0.0)
        
        return OperationPerformance(**data)
    
    def _model_to_dict(self, model: OperationPerformance) -> Dict[str, Any]:
        """Convert OperationPerformance model to dictionary for database operations."""
        data = {
            'operation_name': model.operation_name,
            'operation_category': model.operation_category,
            'total_applications': model.total_applications,
            'success_count': model.success_count,
            'avg_improvement': model.avg_improvement,
            'best_improvement': model.best_improvement,
            'worst_result': model.worst_result,
            'avg_execution_time': model.avg_execution_time,
            'last_used': model.last_used.isoformat(),
            'effectiveness_score': model.effectiveness_score,
            'session_id': model.session_id
        }
        
        if model.id is not None:
            data['id'] = model.id
        
        return data
    
    def update_operation_performance(self, operation_name: str, category: str,
                                   improvement: float, execution_time: float,
                                   session_id: str, success: bool = True) -> OperationPerformance:
        """
        Update performance statistics for an operation.
        
        Args:
            operation_name: Name of the operation
            category: Operation category
            improvement: Score improvement achieved
            execution_time: Time taken to execute
            session_id: Session where operation was used
            success: Whether operation was successful
            
        Returns:
            Updated or created operation performance record
        """
        # Get existing stats
        existing = self.find_all(
            where_clause="operation_name = ? AND session_id = ?",
            params=(operation_name, session_id),
            limit=1
        )
        
        if existing:
            perf = existing[0]
            
            # Update statistics
            new_total = perf.total_applications + 1
            new_success = perf.success_count + (1 if success else 0)
            new_avg_imp = ((perf.avg_improvement * perf.total_applications) + improvement) / new_total
            new_best_imp = max(perf.best_improvement, improvement)
            new_worst = min(perf.worst_result, improvement)
            new_avg_time = ((perf.avg_execution_time * perf.total_applications) + execution_time) / new_total
            
            # Calculate effectiveness score (success rate * avg improvement)
            effectiveness = (new_success / new_total) * max(0, new_avg_imp)
            
            perf.total_applications = new_total
            perf.success_count = new_success
            perf.avg_improvement = new_avg_imp
            perf.best_improvement = new_best_imp
            perf.worst_result = new_worst
            perf.avg_execution_time = new_avg_time
            perf.effectiveness_score = effectiveness
            perf.last_used = datetime.now()
            
            return self.save(perf)
        else:
            # Create new record
            effectiveness = (1 if success else 0) * max(0, improvement)
            
            perf = OperationPerformance(
                operation_name=operation_name,
                operation_category=category,
                total_applications=1,
                success_count=1 if success else 0,
                avg_improvement=improvement,
                best_improvement=improvement,
                worst_result=improvement,
                avg_execution_time=execution_time,
                effectiveness_score=effectiveness,
                session_id=session_id
            )
            
            return self.save(perf)
    
    def get_operation_rankings(self, session_id: Optional[str] = None,
                             limit: int = 20) -> List[OperationPerformance]:
        """
        Get operation effectiveness rankings.
        
        Args:
            session_id: Optional session filter
            limit: Maximum number of operations to return
            
        Returns:
            List of operations ranked by effectiveness
        """
        where_clause = "total_applications > 0"
        params = []
        
        if session_id:
            where_clause += " AND session_id = ?"
            params.append(session_id)
        
        return self.find_all(
            where_clause=where_clause,
            params=tuple(params) if params else None,
            order_by="effectiveness_score DESC",
            limit=limit
        )