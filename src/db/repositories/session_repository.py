"""
Session repository implementation.

This module provides database operations for MCTS session management,
including session creation, updates, and analytics queries.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import json

from ..core.base_repository import BaseRepository
from ..core.connection import DuckDBConnectionManager
from ..models.session import Session, SessionSummary, SessionCreate, SessionUpdate, SessionStatus


class SessionRepository(BaseRepository[Session]):
    """
    Repository for MCTS session operations.
    
    Handles all database operations related to session management,
    including CRUD operations and analytical queries.
    """
    
    @property
    def table_name(self) -> str:
        """Return the sessions table name."""
        return "sessions"
    
    @property
    def model_class(self) -> type:
        """Return the Session model class."""
        return Session
    
    def _get_conflict_target(self) -> Optional[str]:
        """Disable ON CONFLICT for sessions to avoid foreign key issues."""
        return None  # Force use of INSERT without ON CONFLICT
    
    def save(self, entity: Session, update_on_conflict: bool = True) -> Session:
        """
        Override save method to handle foreign key constraints properly.
        
        Args:
            entity: Session entity to save
            update_on_conflict: Ignored - we handle INSERT/UPDATE logic manually
            
        Returns:
            Saved session entity
        """
        try:
            # Check if session already exists
            existing = self.find_by_id(entity.session_id, 'session_id')
            
            if existing:
                # Session exists, do UPDATE
                return self._update_session_direct(entity)
            else:
                # Session doesn't exist, do INSERT
                return super().save(entity, update_on_conflict=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save session {entity.session_id}: {e}")
            # If INSERT fails due to race condition, try UPDATE
            if "UNIQUE constraint failed" in str(e) or "already exists" in str(e):
                return self._update_session_direct(entity)
            raise
    
    def _row_to_model(self, row: Any) -> Session:
        """Convert database row to Session model."""
        # Handle both tuple and dict-like row objects
        if hasattr(row, 'keys'):
            # Dict-like object (SQLite Row)
            data = dict(row)
        else:
            # Tuple - map to known column order
            columns = [
                'session_id', 'session_name', 'start_time', 'end_time',
                'total_iterations', 'best_score', 'config_snapshot', 'status',
                'strategy', 'is_test_mode', 'notes', 'dataset_hash'
            ]
            data = dict(zip(columns, row))
        
        # Parse JSON fields
        if isinstance(data.get('config_snapshot'), str):
            try:
                data['config_snapshot'] = json.loads(data['config_snapshot'])
            except (json.JSONDecodeError, TypeError):
                data['config_snapshot'] = {}
        
        # Convert string timestamps to datetime objects
        for field in ['start_time', 'end_time']:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    data[field] = None
        
        # Handle None values for optional fields
        for field in ['session_name', 'end_time', 'notes', 'dataset_hash']:
            if field not in data:
                data[field] = None
        
        # Set defaults for missing fields
        data.setdefault('total_iterations', 0)
        data.setdefault('best_score', 0.0)
        data.setdefault('config_snapshot', {})
        data.setdefault('status', SessionStatus.ACTIVE)
        data.setdefault('strategy', 'default')
        data.setdefault('is_test_mode', False)
        
        return Session(**data)
    
    def _model_to_dict(self, model: Session) -> Dict[str, Any]:
        """Convert Session model to dictionary for database operations."""
        data = {
            'session_id': model.session_id,
            'session_name': model.session_name,
            'start_time': model.start_time.isoformat(),
            'end_time': model.end_time.isoformat() if model.end_time else None,
            'total_iterations': model.total_iterations,
            'best_score': model.best_score,
            'config_snapshot': json.dumps(model.config_snapshot),
            'status': model.status.value,
            'strategy': model.strategy.value,
            'is_test_mode': model.is_test_mode,
            'notes': model.notes,
            'dataset_hash': model.dataset_hash
        }
        return data
    
    def create_session(self, session_create: SessionCreate) -> Session:
        """
        Create a new session.
        
        Args:
            session_create: Session creation data
            
        Returns:
            Created session model
        """
        session = Session(
            session_id=session_create.session_id,
            session_name=session_create.session_name,
            config_snapshot=session_create.config_snapshot,
            strategy=session_create.strategy,
            is_test_mode=session_create.is_test_mode,
            dataset_hash=session_create.dataset_hash,
            notes=session_create.notes
        )
        
        return self.save(session)
    
    def update_session(self, session_id: str, update_data: SessionUpdate) -> Optional[Session]:
        """
        Update an existing session.
        
        Args:
            session_id: Session ID to update
            update_data: Fields to update
            
        Returns:
            Updated session model or None if not found
        """
        # Get current session
        session = self.find_by_id(session_id, 'session_id')
        if not session:
            return None
        
        # Apply updates
        update_dict = update_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(session, field, value)
        
        return self.save(session)
    
    def _update_session_direct(self, session: Session) -> Session:
        """
        Direct UPDATE operation for sessions to avoid foreign key conflicts.
        
        Args:
            session: Session model to update
            
        Returns:
            Updated session model
        """
        try:
            # Convert model to dict and exclude session_id from updates
            data = self._model_to_dict(session)
            data.pop('session_id', None)  # Remove session_id from updates
            
            if not data:
                return session  # Nothing to update
            
            # Build UPDATE query
            set_clauses = [f"{col} = ?" for col in data.keys()]
            values = list(data.values()) + [session.session_id]
            
            query = f"""
                UPDATE {self.table_name} 
                SET {', '.join(set_clauses)}
                WHERE session_id = ?
            """
            
            with self.connection_manager.get_connection() as conn:
                conn.execute(query, values)
            
            self.logger.debug(f"Successfully updated session {session.session_id} using direct UPDATE")
            return session
            
        except Exception as e:
            self.logger.error(f"Direct session update failed for {session.session_id}: {e}")
            raise

    def get_active_sessions(self, limit: Optional[int] = None) -> List[Session]:
        """
        Get all active sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of active sessions
        """
        return self.find_all(
            where_clause="status = ?",
            params=(SessionStatus.ACTIVE.value,),
            order_by="start_time DESC",
            limit=limit
        )
    
    def get_recent_sessions(self, days: int = 30, limit: Optional[int] = None) -> List[Session]:
        """
        Get sessions from the last N days.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of sessions to return
            
        Returns:
            List of recent sessions
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return self.find_all(
            where_clause="start_time >= ?",
            params=(cutoff_date.isoformat(),),
            order_by="start_time DESC",
            limit=limit
        )
    
    def get_sessions_by_dataset(self, dataset_hash: str, limit: Optional[int] = None) -> List[Session]:
        """
        Get sessions that used a specific dataset.
        
        Args:
            dataset_hash: Dataset hash to filter by
            limit: Maximum number of sessions to return
            
        Returns:
            List of sessions using the dataset
        """
        return self.find_all(
            where_clause="dataset_hash = ?",
            params=(dataset_hash,),
            order_by="start_time DESC",
            limit=limit
        )
    
    def get_session_summary(self, session_id: str) -> Optional[SessionSummary]:
        """
        Get session summary with aggregated statistics.
        
        Args:
            session_id: Session ID to summarize
            
        Returns:
            Session summary or None if not found
        """
        query = """
        SELECT 
            s.session_id,
            s.session_name,
            s.start_time,
            s.end_time,
            s.total_iterations,
            COALESCE(MIN(eh.evaluation_score), 0) as min_score,
            COALESCE(MAX(eh.evaluation_score), 0) as max_score,
            COALESCE(MAX(eh.evaluation_score) - MIN(eh.evaluation_score), 0) as improvement,
            COALESCE(AVG(eh.evaluation_time), 0) as avg_eval_time,
            COALESCE(SUM(eh.evaluation_time), 0) as total_eval_time,
            s.status,
            COALESCE(ANY_VALUE(eh.target_metric), 'unknown') as target_metric
        FROM sessions s
        LEFT JOIN exploration_history eh ON s.session_id = eh.session_id
        WHERE s.session_id = ?
        GROUP BY s.session_id, s.session_name, s.start_time, s.end_time, 
                 s.total_iterations, s.status
        """
        
        result = self.execute_custom_query(query, (session_id,), fetch='one')
        if not result:
            return None
        
        # Convert result to SessionSummary
        def safe_datetime_parse(value):
            if not value:
                return None
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    return None
            return None
        
        data = {
            'session_id': result[0],
            'session_name': result[1],
            'start_time': safe_datetime_parse(result[2]) or datetime.now(),
            'end_time': safe_datetime_parse(result[3]),
            'total_iterations': result[4] or 0,
            'min_score': float(result[5]) if result[5] is not None else 0.0,
            'max_score': float(result[6]) if result[6] is not None else 0.0,
            'improvement': float(result[7]) if result[7] is not None else 0.0,
            'avg_eval_time': float(result[8]) if result[8] is not None else 0.0,
            'total_eval_time': float(result[9]) if result[9] is not None else 0.0,
            'status': SessionStatus(result[10]) if result[10] else SessionStatus.ACTIVE,
            'target_metric': result[11] if result[11] else 'unknown'
        }
        
        return SessionSummary(**data)
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get overall session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        stats_query = """
        SELECT 
            COUNT(*) as total_sessions,
            COUNT(CASE WHEN status = ? THEN 1 END) as active_sessions,
            COUNT(CASE WHEN status = ? THEN 1 END) as completed_sessions,
            COUNT(CASE WHEN is_test_mode = TRUE THEN 1 END) as test_sessions,
            AVG(total_iterations) as avg_iterations,
            AVG(best_score) as avg_best_score,
            MAX(best_score) as best_score_overall,
            COUNT(DISTINCT dataset_hash) as unique_datasets_used
        FROM sessions
        WHERE start_time >= datetime('now', '-30 days')
        """
        
        result = self.execute_custom_query(
            stats_query, 
            (SessionStatus.ACTIVE.value, SessionStatus.COMPLETED.value), 
            fetch='one'
        )
        
        if result:
            return {
                'total_sessions': result[0] or 0,
                'active_sessions': result[1] or 0,
                'completed_sessions': result[2] or 0,
                'test_sessions': result[3] or 0,
                'avg_iterations': float(result[4]) if result[4] else 0.0,
                'avg_best_score': float(result[5]) if result[5] else 0.0,
                'best_score_overall': float(result[6]) if result[6] else 0.0,
                'unique_datasets_used': result[7] or 0
            }
        
        return {
            'total_sessions': 0,
            'active_sessions': 0,
            'completed_sessions': 0,
            'test_sessions': 0,
            'avg_iterations': 0.0,
            'avg_best_score': 0.0,
            'best_score_overall': 0.0,
            'unique_datasets_used': 0
        }
    
    def close_session(self, session_id: str, status: SessionStatus = SessionStatus.COMPLETED) -> bool:
        """
        Close a session with the specified status.
        
        Args:
            session_id: Session ID to close
            status: Final status for the session
            
        Returns:
            True if session was closed, False if not found
        """
        update_data = SessionUpdate(
            end_time=datetime.now(),
            status=status
        )
        
        updated_session = self.update_session(session_id, update_data)
        return updated_session is not None
    
    def cleanup_old_sessions(self, days: int = 90) -> int:
        """
        Mark old sessions as completed if they're still active.
        
        Args:
            days: Number of days after which to mark sessions as completed
            
        Returns:
            Number of sessions updated
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        update_query = """
        UPDATE sessions 
        SET status = ?, end_time = CURRENT_TIMESTAMP
        WHERE status = ? AND start_time < ?
        """
        
        self.execute_custom_query(
            update_query,
            (SessionStatus.COMPLETED.value, SessionStatus.ACTIVE.value, cutoff_date.isoformat()),
            fetch='none'
        )
        
        # Get count of updated sessions
        count_query = "SELECT changes()"
        result = self.execute_custom_query(count_query, fetch='one')
        
        return result[0] if result else 0
    
    def get_session_performance_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get session performance trend over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of daily performance statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        trend_query = """
        SELECT 
            DATE(start_time) as date,
            COUNT(*) as sessions_started,
            COUNT(CASE WHEN status = ? THEN 1 END) as sessions_completed,
            AVG(best_score) as avg_score,
            MAX(best_score) as best_score
        FROM sessions
        WHERE start_time >= ?
        GROUP BY DATE(start_time)
        ORDER BY date
        """
        
        results = self.execute_custom_query(
            trend_query,
            (SessionStatus.COMPLETED.value, cutoff_date.isoformat()),
            fetch='all'
        )
        
        trend_data = []
        for row in results:
            trend_data.append({
                'date': row[0],
                'sessions_started': row[1] or 0,
                'sessions_completed': row[2] or 0,
                'avg_score': float(row[3]) if row[3] else 0.0,
                'best_score': float(row[4]) if row[4] else 0.0
            })
        
        return trend_data