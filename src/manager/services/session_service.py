"""
Service layer for session-related business logic using SQLAlchemy abstraction layer.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from ..core.utils import format_duration, format_datetime, format_number


class SessionService:
    """Handles session-related business logic using new database abstraction."""
    
    def __init__(self, db_manager):
        """Initialize service with database manager.
        
        Args:
            db_manager: Database manager instance from factory
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Ensure sessions table exists
        self._ensure_sessions_table()
        
        # Legacy compatibility - modules expect a repository attribute
        self.repository = self
    
    def _ensure_sessions_table(self):
        """Ensure sessions table exists."""
        self.logger.info("Creating sessions table if not exists...")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR PRIMARY KEY,
            session_name VARCHAR DEFAULT '',
            dataset_name VARCHAR,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            status VARCHAR DEFAULT 'running',
            total_iterations INTEGER DEFAULT 0,
            best_score DOUBLE DEFAULT 0.0,
            config_hash VARCHAR,
            error_message VARCHAR,
            strategy VARCHAR DEFAULT 'mcts',
            is_test_mode BOOLEAN DEFAULT false,
            config_snapshot TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.db_manager.execute_ddl(create_table_query)
        self.logger.info("Sessions table creation completed")
    
    def list_sessions(self, dataset: Optional[str] = None,
                     status: Optional[str] = None,
                     days: Optional[int] = None) -> List[Dict[str, Any]]:
        """List sessions with filtering and formatting.
        
        Args:
            dataset: Filter by dataset name
            status: Filter by session status
            days: Limit to sessions from last N days
            
        Returns:
            List of formatted session summaries
        """
        # Build query with filters
        where_clauses = []
        params = {}
        
        if dataset:
            where_clauses.append("dataset_name = :dataset")
            params['dataset'] = dataset
        
        if status:
            where_clauses.append("status = :status")
            params['status'] = status
        
        if days:
            where_clauses.append("start_time >= :cutoff_date")
            cutoff_date = datetime.now() - timedelta(days=days)
            params['cutoff_date'] = cutoff_date
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
        SELECT * FROM sessions 
        WHERE {where_clause}
        ORDER BY start_time DESC
        """
        
        sessions = self.db_manager.execute_query(query, params)
        
        # Format sessions for display
        formatted_sessions = []
        for session in sessions:
            formatted = dict(session)
            
            # Format duration
            if formatted.get('start_time') and formatted.get('end_time'):
                start = datetime.fromisoformat(str(formatted['start_time']))
                end = datetime.fromisoformat(str(formatted['end_time']))
                duration = end - start
                formatted['duration_display'] = format_duration(duration.total_seconds())
            elif formatted.get('start_time'):
                start = datetime.fromisoformat(str(formatted['start_time']))
                duration = datetime.now() - start
                formatted['duration_display'] = format_duration(duration.total_seconds()) + " (running)"
            else:
                formatted['duration_display'] = "Unknown"
            
            # Format dates
            if formatted.get('start_time'):
                formatted['start_time_display'] = format_datetime(formatted['start_time'])
            
            # Format iterations and score
            formatted['iterations_display'] = format_number(formatted.get('total_iterations', 0))
            formatted['score_display'] = f"{formatted.get('best_score', 0.0):.4f}"
            
            formatted_sessions.append(formatted)
        
        return formatted_sessions
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session dictionary or None if not found
        """
        query = "SELECT * FROM sessions WHERE session_id = :session_id"
        results = self.db_manager.execute_query(query, {'session_id': session_id})
        return results[0] if results else None
    
    def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new session.
        
        Args:
            session_data: Session data dictionary
            
        Returns:
            Session ID
        """
        session_id = session_data.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        insert_query = """
        INSERT INTO sessions 
        (session_id, dataset_name, config_hash, status)
        VALUES (:session_id, :dataset_name, :config_hash, :status)
        """
        
        params = {
            'session_id': session_id,
            'dataset_name': session_data.get('dataset_name', ''),
            'config_hash': session_data.get('config_hash', ''),
            'status': 'running'
        }
        
        self.db_manager.execute_dml(insert_query, params)
        self.logger.info(f"Created session: {session_id}")
        return session_id
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update session information.
        
        Args:
            session_id: Session ID
            updates: Dictionary of fields to update
        """
        if not updates:
            return
        
        # Build update query
        set_clauses = []
        params = {'session_id': session_id}
        
        for key, value in updates.items():
            if key != 'session_id':
                set_clauses.append(f"{key} = :{key}")
                params[key] = value
        
        if set_clauses:
            # Add updated_at timestamp
            set_clauses.append("updated_at = :updated_at")
            params['updated_at'] = datetime.now()
            
            query = f"UPDATE sessions SET {', '.join(set_clauses)} WHERE session_id = :session_id"
            self.db_manager.execute_dml(query, params)
            
            self.logger.info(f"Updated session: {session_id}")
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if session was deleted
        """
        query = "DELETE FROM sessions WHERE session_id = :session_id"
        rows_affected = self.db_manager.execute_dml(query, {'session_id': session_id})
        
        if rows_affected > 0:
            self.logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        stats_query = """
        SELECT 
            COUNT(*) as total_sessions,
            COUNT(CASE WHEN status = 'running' THEN 1 END) as running_sessions,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions,
            AVG(total_iterations) as avg_iterations,
            MAX(best_score) as best_score
        FROM sessions
        """
        
        results = self.db_manager.execute_query(stats_query)
        stats = results[0] if results else {}
        
        # Format results
        formatted_stats = {
            'total_sessions': stats.get('total_sessions', 0),
            'running_sessions': stats.get('running_sessions', 0),
            'completed_sessions': stats.get('completed_sessions', 0),
            'failed_sessions': stats.get('failed_sessions', 0),
            'avg_iterations': round(stats.get('avg_iterations', 0), 1) if stats.get('avg_iterations') else 0,
            'best_score': stats.get('best_score', 0.0)
        }
        
        return formatted_stats
    
    # Legacy compatibility methods
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Legacy compatibility method."""
        return self.list_sessions()
    
    def get_sessions_by_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Legacy compatibility method."""
        return self.list_sessions(dataset=dataset_name)
    
    def get_latest_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of latest sessions
        """
        query = """
        SELECT * FROM sessions 
        ORDER BY start_time DESC 
        LIMIT :limit
        """
        
        return self.db_manager.execute_query(query, {'limit': limit})
    
    # Legacy repository compatibility methods
    def fetch_all(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Legacy compatibility method for raw SQL queries.
        
        Args:
            query: SQL query string
            params: List of query parameters
            
        Returns:
            List of result dictionaries
        """
        # Convert list params to dict for SQLAlchemy
        if params:
            # Create numbered parameter placeholders
            sql_params = {}
            for i, param in enumerate(params):
                sql_params[f'param_{i}'] = param
            
            # Replace ? placeholders with :param_n
            formatted_query = query
            for i in range(len(params)):
                formatted_query = formatted_query.replace('?', f':param_{i}', 1)
            
            return self.db_manager.execute_query(formatted_query, sql_params)
        else:
            return self.db_manager.execute_query(query)
    
    def fetch_one(self, query: str, params: Optional[List] = None) -> Optional[Dict[str, Any]]:
        """Legacy compatibility method for single row queries."""
        results = self.fetch_all(query, params)
        return results[0] if results else None