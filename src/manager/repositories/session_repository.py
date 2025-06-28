"""
Repository for session-related database operations
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from .base import BaseRepository


class SessionRepository(BaseRepository):
    """Handles all session-related database operations."""
    
    def get_all_sessions(self, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """Get all sessions with basic information.
        
        Args:
            include_deleted: Whether to include sessions (no deleted_at column exists)
            
        Returns:
            List of session dictionaries
        """
        query = """
        SELECT 
            session_id,
            session_name,
            start_time,
            end_time,
            total_iterations,
            best_score,
            config_snapshot,
            status,
            strategy,
            is_test_mode,
            notes,
            dataset_hash
        FROM sessions
        ORDER BY start_time DESC
        """
        
        return self.fetch_all(query)
    
    def get_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific session.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            Session dictionary or None if not found
        """
        query = """
        SELECT 
            session_id,
            session_name,
            start_time,
            end_time,
            total_iterations,
            best_score,
            config_snapshot,
            status,
            strategy,
            is_test_mode,
            notes,
            dataset_hash
        FROM sessions
        WHERE session_id = ?
        """
        
        return self.fetch_one(query, (session_id,))
    
    def get_sessions_by_dataset(self, dataset_hash: str) -> List[Dict[str, Any]]:
        """Get all sessions for a specific dataset.
        
        Args:
            dataset_hash: Hash of the dataset
            
        Returns:
            List of session dictionaries
        """
        query = """
        SELECT 
            session_id,
            session_name,
            start_time,
            end_time,
            total_iterations,
            best_score,
            status,
            strategy,
            is_test_mode
        FROM sessions
        WHERE dataset_hash = ?
        ORDER BY start_time DESC
        """
        
        return self.fetch_all(query, (dataset_hash,))
    
    def get_session_statistics(self, days: Optional[int] = None) -> Dict[str, Any]:
        """Get aggregate statistics for sessions.
        
        Args:
            days: Limit to sessions from last N days
            
        Returns:
            Dictionary of statistics
        """
        where_clause = "WHERE 1=1"
        params = []
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            where_clause += " AND start_time >= ?"
            params.append(cutoff_date.isoformat())
        
        # Total sessions by status
        status_query = f"""
        SELECT status, COUNT(*) as count
        FROM sessions
        {where_clause}
        GROUP BY status
        """
        
        status_rows = self.fetch_all(status_query, tuple(params))
        status_counts = {row['status']: row['count'] for row in status_rows}
        
        # Performance statistics
        perf_query = f"""
        SELECT 
            COUNT(*) as total,
            AVG(best_score) as avg_score,
            MAX(best_score) as max_score,
            AVG(total_iterations) as avg_iterations,
            SUM(total_iterations) as total_iterations,
            AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration,
            SUM(EXTRACT(EPOCH FROM (end_time - start_time))) as total_duration
        FROM sessions
        {where_clause}
        AND status = 'completed'
        AND end_time IS NOT NULL
        """
        
        perf_row = self.fetch_one(perf_query, tuple(params))
        
        # Get nodes and features from exploration_history
        exploration_query = f"""
        SELECT 
            COUNT(DISTINCT h.id) as total_nodes,
            COUNT(DISTINCT h.operation_applied) as total_operations
        FROM exploration_history h
        JOIN sessions s ON h.session_id = s.session_id
        {where_clause}
        """
        
        exploration_row = self.fetch_one(exploration_query, tuple(params))
        
        return {
            'status_counts': status_counts,
            'total_sessions': sum(status_counts.values()),
            'performance': {
                'completed_sessions': perf_row['total'] if perf_row else 0,
                'avg_score': perf_row['avg_score'] if perf_row else 0,
                'max_score': perf_row['max_score'] if perf_row else 0,
                'avg_iterations': perf_row['avg_iterations'] if perf_row else 0,
                'total_iterations': perf_row['total_iterations'] if perf_row else 0,
                'avg_duration': perf_row['avg_duration'] if perf_row else 0,
                'total_duration': perf_row['total_duration'] if perf_row else 0,
                'total_nodes': exploration_row['total_nodes'] if exploration_row else 0,
                'total_features': exploration_row['total_operations'] if exploration_row else 0
            }
        }
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dictionaries
        """
        query = """
        SELECT 
            s.session_id,
            s.session_name,
            s.start_time,
            s.end_time,
            s.status,
            s.best_score,
            s.total_iterations,
            s.strategy,
            s.is_test_mode,
            d.dataset_name
        FROM sessions s
        LEFT JOIN datasets d ON s.dataset_hash = d.dataset_id
        ORDER BY s.start_time DESC
        LIMIT ?
        """
        
        return self.fetch_all(query, (limit,))
    
    def get_session_performance_trends(self, session_id: str) -> List[Dict[str, Any]]:
        """Get performance trend data for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of performance metrics over time
        """
        query = """
        SELECT 
            iteration,
            timestamp,
            best_score,
            nodes_explored,
            features_generated
        FROM system_performance
        WHERE session_id = ?
        ORDER BY iteration
        """
        
        return self.fetch_all(query, (session_id,))
    
    def get_session_exploration_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get exploration history for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of records to return
            
        Returns:
            List of exploration records
        """
        query = """
        SELECT 
            id,
            parent_node_id,
            iteration,
            operation_applied,
            features_before,
            features_after,
            evaluation_score,
            target_metric,
            evaluation_time,
            mcts_ucb1_score,
            node_visits,
            is_best_so_far,
            timestamp
        FROM exploration_history
        WHERE session_id = ?
        ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.fetch_all(query, (session_id,))
    
    def update_session_status(self, session_id: str, status: str, 
                            end_time: Optional[datetime] = None,
                            error_message: Optional[str] = None) -> bool:
        """Update session status.
        
        Args:
            session_id: Session ID to update
            status: New status
            end_time: End time (for completed/failed sessions)
            error_message: Error message (for failed sessions)
            
        Returns:
            True if updated, False if not found
        """
        updates = ["status = ?"]
        params = [status]
        
        if end_time:
            updates.append("end_time = ?")
            params.append(end_time.isoformat())
        
        if error_message and status == 'failed':
            updates.append("notes = ?")
            params.append(f"Error: {error_message}")
        
        params.append(session_id)
        
        query = f"""
        UPDATE sessions 
        SET {', '.join(updates)}
        WHERE session_id = ?
        """
        
        result = self.execute(query, tuple(params))
        return True  # DuckDB doesn't provide rowcount easily
    
    def delete_session(self, session_id: str, hard_delete: bool = True) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID to delete
            hard_delete: Always true (no soft delete in current schema)
            
        Returns:
            True if deleted
        """
        # Delete related data first
        queries = [
            "DELETE FROM exploration_history WHERE session_id = ?",
            "DELETE FROM feature_impact WHERE session_id = ?",
            "DELETE FROM system_performance WHERE session_id = ?",
            "DELETE FROM sessions WHERE session_id = ?"
        ]
        
        for query in queries:
            self.execute(query, (session_id,))
        
        return True