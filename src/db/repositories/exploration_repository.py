"""
Exploration repository implementation.

This module provides database operations for MCTS exploration tracking,
including exploration steps, node management, and path analysis.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import json

from ..core.base_repository import BaseRepository
from ..core.connection import DuckDBConnectionManager
from ..models.exploration import (
    ExplorationStep, ExplorationNode, ExplorationPath,
    ExplorationCreate, ExplorationAnalysis
)


class ExplorationRepository(BaseRepository[ExplorationStep]):
    """
    Repository for MCTS exploration operations.
    
    Handles all database operations related to exploration tracking,
    including step logging, node management, and path analysis.
    """
    
    @property
    def table_name(self) -> str:
        """Return the exploration_history table name."""
        return "exploration_history"
    
    @property
    def model_class(self) -> type:
        """Return the ExplorationStep model class."""
        return ExplorationStep
    
    def _row_to_model(self, row: Any) -> ExplorationStep:
        """Convert database row to ExplorationStep model."""
        # Handle both tuple and dict-like row objects
        if hasattr(row, 'keys'):
            # Dict-like object (SQLite Row)
            data = dict(row)
        else:
            # Tuple - map to known column order
            columns = [
                'id', 'session_id', 'iteration', 'timestamp', 'parent_node_id',
                'operation_applied', 'features_before', 'features_after',
                'evaluation_score', 'target_metric', 'evaluation_time',
                'autogluon_config', 'mcts_ucb1_score', 'node_visits',
                'is_best_so_far', 'memory_usage_mb', 'notes', 'mcts_node_id'
            ]
            data = dict(zip(columns, row))
        
        # Parse JSON fields
        for json_field in ['features_before', 'features_after', 'autogluon_config']:
            if isinstance(data.get(json_field), str):
                try:
                    data[json_field] = json.loads(data[json_field])
                except (json.JSONDecodeError, TypeError):
                    if json_field in ['features_before', 'features_after']:
                        data[json_field] = []
                    else:
                        data[json_field] = {}
        
        # Convert string timestamps to datetime objects
        if isinstance(data.get('timestamp'), str):
            try:
                data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                data['timestamp'] = datetime.now()
        
        # Handle None values and set defaults
        data.setdefault('node_visits', 1)
        data.setdefault('is_best_so_far', False)
        
        # Ensure required fields have valid values
        if not data.get('features_before'):
            data['features_before'] = []
        if not data.get('features_after'):
            data['features_after'] = []
        if not data.get('autogluon_config'):
            data['autogluon_config'] = {}
        
        return ExplorationStep(**data)
    
    def _model_to_dict(self, model: ExplorationStep) -> Dict[str, Any]:
        """Convert ExplorationStep model to dictionary for database operations."""
        data = {
            'session_id': model.session_id,
            'iteration': model.iteration,
            'timestamp': model.timestamp.isoformat(),
            'parent_node_id': model.parent_node_id,
            'operation_applied': model.operation_applied,
            'features_before': json.dumps(model.features_before),
            'features_after': json.dumps(model.features_after),
            'evaluation_score': model.evaluation_score,
            'target_metric': model.target_metric,
            'evaluation_time': model.evaluation_time,
            'autogluon_config': json.dumps(model.autogluon_config),
            'mcts_ucb1_score': model.mcts_ucb1_score,
            'node_visits': model.node_visits,
            'is_best_so_far': model.is_best_so_far,
            'memory_usage_mb': model.memory_usage_mb,
            'notes': model.notes,
            'mcts_node_id': model.mcts_node_id
        }
        
        # Include ID if it exists (for updates)
        if model.id is not None:
            data['id'] = model.id
        
        return data
    
    def log_exploration_step(self, step_data: ExplorationCreate) -> ExplorationStep:
        """
        Log a new MCTS exploration step.
        
        Args:
            step_data: Exploration step creation data
            
        Returns:
            Created exploration step with ID
        """
        # Check if this is the best score so far in the session
        is_best = self._is_best_score_so_far(step_data.session_id, step_data.evaluation_score)
        
        # Create exploration step
        step = ExplorationStep(
            session_id=step_data.session_id,
            iteration=step_data.iteration,
            parent_node_id=step_data.parent_node_id,
            operation_applied=step_data.operation_applied,
            features_before=step_data.features_before,
            features_after=step_data.features_after,
            evaluation_score=step_data.evaluation_score,
            target_metric=step_data.target_metric,
            evaluation_time=step_data.evaluation_time,
            autogluon_config=step_data.autogluon_config,
            mcts_ucb1_score=step_data.mcts_ucb1_score,
            is_best_so_far=is_best,
            memory_usage_mb=step_data.memory_usage_mb,
            notes=step_data.notes,
            mcts_node_id=step_data.mcts_node_id,
            node_visits=step_data.node_visits
        )
        
        saved_step = self.save(step)
        
        # Update session statistics
        self._update_session_stats(step_data.session_id, step_data.iteration, step_data.evaluation_score)
        
        self.logger.info(f"Logged exploration step {step_data.iteration} for session {step_data.session_id}")
        return saved_step
    
    def _is_best_score_so_far(self, session_id: str, score: float) -> bool:
        """Check if the given score is the best so far in the session."""
        query = "SELECT MAX(evaluation_score) FROM exploration_history WHERE session_id = ?"
        result = self.execute_custom_query(query, (session_id,), fetch='one')
        
        current_best = result[0] if result and result[0] is not None else 0.0
        return score > current_best
    
    def _update_session_stats(self, session_id: str, iteration: int, score: float) -> None:
        """Update session statistics after logging a step."""
        update_query = """
        UPDATE sessions SET 
            total_iterations = ?,
            best_score = CASE WHEN best_score > ? THEN best_score ELSE ? END
        WHERE session_id = ?
        """
        
        self.execute_custom_query(
            update_query,
            (iteration, score, score, session_id),
            fetch='none'
        )
    
    def get_session_exploration_history(self, session_id: str, 
                                      limit: Optional[int] = None) -> List[ExplorationStep]:
        """
        Get exploration history for a session.
        
        Args:
            session_id: Session ID to get history for
            limit: Maximum number of steps to return
            
        Returns:
            List of exploration steps ordered by iteration
        """
        return self.find_all(
            where_clause="session_id = ?",
            params=(session_id,),
            order_by="iteration ASC",
            limit=limit
        )
    
    def get_best_exploration_steps(self, session_id: str, 
                                 limit: int = 10) -> List[ExplorationStep]:
        """
        Get the best exploration steps for a session.
        
        Args:
            session_id: Session ID to get steps for
            limit: Maximum number of steps to return
            
        Returns:
            List of best exploration steps ordered by score
        """
        return self.find_all(
            where_clause="session_id = ?",
            params=(session_id,),
            order_by="evaluation_score DESC",
            limit=limit
        )
    
    def get_exploration_path(self, session_id: str, target_node_id: int) -> Optional[ExplorationPath]:
        """
        Get the complete exploration path to a specific node.
        
        Args:
            session_id: Session ID
            target_node_id: Target node ID to trace path to
            
        Returns:
            Complete exploration path or None if not found
        """
        # Build recursive query to trace path from target to root
        path_query = """
        WITH RECURSIVE path_trace(id, session_id, parent_node_id, iteration, operation_applied, 
                                 features_after, evaluation_score, level) AS (
            -- Base case: target node
            SELECT id, session_id, parent_node_id, iteration, operation_applied, 
                   features_after, evaluation_score, 0 as level
            FROM exploration_history
            WHERE session_id = ? AND id = ?
            
            UNION ALL
            
            -- Recursive case: parent nodes
            SELECT eh.id, eh.session_id, eh.parent_node_id, eh.iteration, eh.operation_applied,
                   eh.features_after, eh.evaluation_score, pt.level + 1
            FROM exploration_history eh
            INNER JOIN path_trace pt ON eh.id = pt.parent_node_id
            WHERE pt.parent_node_id IS NOT NULL
        )
        SELECT id, session_id, parent_node_id, iteration, operation_applied, 
               features_after, evaluation_score, level
        FROM path_trace
        ORDER BY level DESC
        """
        
        results = self.execute_custom_query(
            path_query,
            (session_id, target_node_id),
            fetch='all'
        )
        
        if not results:
            return None
        
        # Convert results to ExplorationNode objects
        path_nodes = []
        operations_sequence = []
        
        for row in results:
            # Parse features_after JSON
            features = []
            if row[5]:  # features_after
                try:
                    features = json.loads(row[5])
                except (json.JSONDecodeError, TypeError):
                    features = []
            
            node = ExplorationNode(
                node_id=row[0],  # id
                session_id=row[1],  # session_id
                parent_node_id=row[2],  # parent_node_id
                operation_applied=row[4],  # operation_applied
                features=features,
                best_score=float(row[6]),  # evaluation_score
                depth=row[7]  # level (reversed)
            )
            
            path_nodes.append(node)
            if row[4]:  # operation_applied
                operations_sequence.append(row[4])
        
        # Reverse to get root-to-target order
        path_nodes.reverse()
        operations_sequence.reverse()
        
        # Adjust depth values
        for i, node in enumerate(path_nodes):
            node.depth = i
        
        return ExplorationPath(
            session_id=session_id,
            target_node_id=target_node_id,
            path_nodes=path_nodes,
            total_depth=len(path_nodes) - 1,
            final_score=path_nodes[-1].best_score if path_nodes else 0.0,
            operations_sequence=operations_sequence
        )
    
    def get_exploration_analysis(self, session_id: str) -> ExplorationAnalysis:
        """
        Get comprehensive analysis of exploration for a session.
        
        Args:
            session_id: Session ID to analyze
            
        Returns:
            Exploration analysis results
        """
        # Get basic statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_steps,
            COUNT(DISTINCT operation_applied) as unique_operations,
            MAX(evaluation_score) as best_score,
            MAX(evaluation_score) - MIN(evaluation_score) as score_improvement,
            AVG(evaluation_time) as avg_evaluation_time
        FROM exploration_history
        WHERE session_id = ?
        """
        
        stats_result = self.execute_custom_query(stats_query, (session_id,), fetch='one')
        
        # Get most effective operations
        operations_query = """
        SELECT operation_applied, AVG(evaluation_score) as avg_score
        FROM exploration_history
        WHERE session_id = ?
        GROUP BY operation_applied
        ORDER BY avg_score DESC
        LIMIT 5
        """
        
        operations_results = self.execute_custom_query(
            operations_query, 
            (session_id,), 
            fetch='all'
        )
        
        # Get feature count progression
        progression_query = """
        SELECT iteration, json_array_length(features_after) as feature_count
        FROM exploration_history
        WHERE session_id = ?
        ORDER BY iteration
        """
        
        progression_results = self.execute_custom_query(
            progression_query,
            (session_id,),
            fetch='all'
        )
        
        # Find convergence point (where improvement plateaus)
        convergence_iteration = None
        if stats_result and stats_result[2]:  # best_score exists
            best_score = float(stats_result[2])
            convergence_threshold = best_score * 0.95  # 95% of best score
            
            convergence_query = """
            SELECT MIN(iteration) as convergence_iteration
            FROM exploration_history
            WHERE session_id = ? AND evaluation_score >= ?
            """
            
            conv_result = self.execute_custom_query(
                convergence_query,
                (session_id, convergence_threshold),
                fetch='one'
            )
            
            if conv_result and conv_result[0]:
                convergence_iteration = conv_result[0]
        
        # Build analysis result
        if stats_result:
            most_effective_ops = [row[0] for row in operations_results] if operations_results else []
            feature_counts = [row[1] for row in progression_results] if progression_results else []
            
            analysis = ExplorationAnalysis(
                session_id=session_id,
                total_steps=stats_result[0] or 0,
                unique_operations=stats_result[1] or 0,
                best_score=float(stats_result[2]) if stats_result[2] else 0.0,
                score_improvement=float(stats_result[3]) if stats_result[3] else 0.0,
                avg_evaluation_time=float(stats_result[4]) if stats_result[4] else 0.0,
                most_effective_operations=most_effective_ops,
                feature_count_progression=feature_counts,
                convergence_iteration=convergence_iteration
            )
        else:
            # Empty analysis for sessions with no data
            analysis = ExplorationAnalysis(
                session_id=session_id,
                total_steps=0,
                unique_operations=0,
                best_score=0.0,
                score_improvement=0.0,
                avg_evaluation_time=0.0,
                most_effective_operations=[],
                feature_count_progression=[],
                convergence_iteration=None
            )
        
        return analysis
    
    def get_operation_performance(self, session_id: Optional[str] = None, 
                                limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get performance statistics for different operations.
        
        Args:
            session_id: Optional session ID to filter by
            limit: Maximum number of operations to return
            
        Returns:
            List of operation performance statistics
        """
        base_query = """
        SELECT 
            operation_applied,
            COUNT(*) as total_applications,
            AVG(evaluation_score) as avg_score,
            MAX(evaluation_score) as best_score,
            MIN(evaluation_score) as worst_score,
            AVG(evaluation_time) as avg_time,
            COUNT(CASE WHEN is_best_so_far = TRUE THEN 1 END) as best_discoveries
        FROM exploration_history
        """
        
        params = []
        if session_id:
            base_query += " WHERE session_id = ?"
            params.append(session_id)
        
        base_query += """
        GROUP BY operation_applied
        ORDER BY avg_score DESC
        LIMIT ?
        """
        params.append(limit)
        
        results = self.execute_custom_query(base_query, tuple(params), fetch='all')
        
        performance_data = []
        for row in results:
            performance_data.append({
                'operation_name': row[0],
                'total_applications': row[1] or 0,
                'avg_score': float(row[2]) if row[2] else 0.0,
                'best_score': float(row[3]) if row[3] else 0.0,
                'worst_score': float(row[4]) if row[4] else 0.0,
                'avg_evaluation_time': float(row[5]) if row[5] else 0.0,
                'best_discoveries': row[6] or 0
            })
        
        return performance_data
    
    def cleanup_old_exploration_data(self, session_id: str, keep_best: int = 100) -> int:
        """
        Clean up old exploration data, keeping only the best N steps.
        
        Args:
            session_id: Session ID to clean up
            keep_best: Number of best steps to keep
            
        Returns:
            Number of records deleted
        """
        delete_query = """
        DELETE FROM exploration_history
        WHERE session_id = ? AND id NOT IN (
            SELECT id FROM exploration_history
            WHERE session_id = ?
            ORDER BY evaluation_score DESC, iteration ASC
            LIMIT ?
        )
        """
        
        self.execute_custom_query(
            delete_query,
            (session_id, session_id, keep_best),
            fetch='none'
        )
        
        # Get count of deleted records
        count_query = "SELECT changes()"
        result = self.execute_custom_query(count_query, fetch='one')
        
        deleted_count = result[0] if result else 0
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} exploration records for session {session_id[:8]}...")
        
        return deleted_count