"""
Database Service Manager for Minotaur MCTS System.

This module provides a high-level service layer using the new SQLAlchemy-based
database abstraction layer. It replaces the old repository-based architecture
with a simplified interface.
"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from .database.engine_factory import DatabaseFactory
from .database.base_manager import DatabaseManager
from .session_output_manager import SessionOutputManager


class DatabaseService:
    """
    High-level database service using SQLAlchemy abstraction layer.
    
    This service provides a simplified interface for database operations
    using the new multi-engine database system.
    """
    
    def __init__(self, config: Dict[str, Any], read_only: bool = False):
        """
        Initialize database service with configuration.
        
        Args:
            config: Main MCTS configuration dictionary
            read_only: If True, opens database in read-only mode
        """
        self.config = config
        self.read_only = read_only
        self.logger = logging.getLogger(__name__)
        
        # Get database configuration
        db_config = config.get('database', {})
        db_type = db_config.get('type', 'duckdb')
        
        # Extract connection parameters
        connection_params = {
            'database': db_config.get('path', 'data/minotaur.duckdb'),
            'read_only': read_only
        }
        
        # Create database manager using factory
        try:
            self.db_manager = DatabaseFactory.create_manager(db_type, connection_params)
            self.logger.info(f"Initialized DatabaseService with {db_type} backend")
        except Exception as e:
            self.logger.error(f"Failed to initialize database service: {e}")
            raise
        
        # Session output manager will be initialized after session creation
        self.session_output_manager = None
        self.output_manager = None  # Alias for compatibility
        self.current_session_id = None
        self.session_name = None
        
        # Connection manager compatibility
        self.connection_manager = self  # Self-reference for compatibility
        self.db_path = connection_params['database']
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results."""
        return self.db_manager.execute_query(query, params)
    
    def execute_query_df(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Execute a SQL query and return as DataFrame."""
        return self.db_manager.execute_query_df(query, params)
    
    def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new MCTS session."""
        session_id = session_data.get('session_id', 'default')
        
        # Create sessions table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR PRIMARY KEY,
            start_time TIMESTAMP,
            config_hash VARCHAR,
            dataset_name VARCHAR,
            total_iterations INTEGER DEFAULT 0,
            best_score DOUBLE DEFAULT 0.0,
            status VARCHAR DEFAULT 'running',
            end_time TIMESTAMP,
            error_message VARCHAR
        )
        """
        
        self.db_manager.execute_query(create_table_query)
        
        # Insert new session
        insert_query = """
        INSERT OR REPLACE INTO sessions 
        (session_id, start_time, config_hash, dataset_name, status)
        VALUES (?, ?, ?, ?, ?)
        """
        
        params = {
            'session_id': session_id,
            'start_time': datetime.now(),
            'config_hash': session_data.get('config_hash', ''),
            'dataset_name': session_data.get('dataset_name', ''),
            'status': 'running'
        }
        
        self.db_manager.execute_query(insert_query, params)
        self.logger.info(f"Created session: {session_id}")
        return session_id
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update session information."""
        if not updates:
            return
        
        # Build dynamic UPDATE query
        set_clauses = []
        params = {'session_id': session_id}
        
        for key, value in updates.items():
            if key != 'session_id':  # Don't update the primary key
                set_clauses.append(f"{key} = :{key}")
                params[key] = value
        
        if set_clauses:
            query = f"UPDATE sessions SET {', '.join(set_clauses)} WHERE session_id = :session_id"
            self.db_manager.execute_query(query, params)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        query = "SELECT * FROM sessions WHERE session_id = :session_id"
        results = self.db_manager.execute_query(query, {'session_id': session_id})
        return results[0] if results else None
    
    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent sessions."""
        query = """
        SELECT * FROM sessions 
        ORDER BY start_time DESC 
        LIMIT :limit
        """
        return self.db_manager.execute_query(query, {'limit': limit})
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        return {
            'backend': 'sqlalchemy',
            'database_type': self.db_manager.__class__.__name__,
            'engine_info': str(self.db_manager.engine.url) if hasattr(self.db_manager, 'engine') else 'unknown'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            # Simple connectivity test
            result = self.db_manager.execute_query("SELECT 1 as test")
            if result and result[0].get('test') == 1:
                return {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': 'Query test failed',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def initialize_session(self, session_mode: str, resume_session_id: Optional[str] = None, 
                          force_resume: bool = False) -> str:
        """Initialize or resume a session."""
        import uuid
        from datetime import datetime
        
        if session_mode == 'continue' and resume_session_id:
            # Resume existing session
            session = self.get_session(resume_session_id)
            if session:
                self.current_session_id = resume_session_id
                self.session_name = session.get('session_name', f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                self.logger.info(f"Resumed session: {self.current_session_id}")
            else:
                raise ValueError(f"Session {resume_session_id} not found")
        else:
            # Create new session
            self.current_session_id = str(uuid.uuid4())
            self.session_name = f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
            # Create session record
            session_data = {
                'session_id': self.current_session_id,
                'session_name': self.session_name,
                'dataset_name': self.config.get('autogluon', {}).get('dataset_name', ''),
                'config_hash': hashlib.md5(str(self.config).encode()).hexdigest()[:16]
            }
            
            # Ensure sessions table exists
            self._ensure_sessions_table()
            
            # Insert new session
            insert_query = """
            INSERT INTO sessions (session_id, session_name, dataset_name, start_time, status, config_hash)
            VALUES (:session_id, :session_name, :dataset_name, :start_time, :status, :config_hash)
            """
            
            params = {
                'session_id': self.current_session_id,
                'session_name': self.session_name,
                'dataset_name': session_data['dataset_name'],
                'start_time': datetime.now(),
                'status': 'running',
                'config_hash': session_data['config_hash']
            }
            
            self.db_manager.execute_dml(insert_query, params)
            self.logger.info(f"Created new session: {self.current_session_id}")
        
        # Initialize session output manager
        base_output_dir = self.config.get('output', {}).get('base_dir', 'outputs')
        self.session_output_manager = SessionOutputManager(self.session_name, self.current_session_id, base_output_dir)
        self.output_manager = self.session_output_manager  # Alias
        
        return self.current_session_id
    
    def _ensure_sessions_table(self) -> None:
        """Ensure sessions table exists with all required columns."""
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
            config_snapshot VARCHAR DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.db_manager.execute_ddl(create_table_query)
    
    def log_exploration_step(self, **kwargs) -> int:
        """Log an exploration step - placeholder for compatibility."""
        # This would need to be properly implemented based on the database schema
        self.logger.debug(f"Logging exploration step: {kwargs.get('operation', 'unknown')}")
        return 0
    
    def ensure_mcts_node_exists(self, **kwargs) -> None:
        """Ensure MCTS node exists in the database."""
        # Placeholder implementation - would need proper MCTS nodes table
        node_id = kwargs.get('node_id')
        parent_node_id = kwargs.get('parent_node_id')
        operation = kwargs.get('operation_applied')
        self.logger.debug(f"Ensuring MCTS node exists: {node_id} (parent={parent_node_id}, op={operation})")
    
    def update_session_progress(self, iteration: int, best_score: float) -> None:
        """Update session progress with current iteration and best score."""
        if not self.current_session_id:
            return
        
        update_query = """
        UPDATE sessions 
        SET total_iterations = :iterations, best_score = :best_score, updated_at = :updated_at
        WHERE session_id = :session_id
        """
        
        params = {
            'iterations': iteration,
            'best_score': best_score,
            'updated_at': datetime.now(),
            'session_id': self.current_session_id
        }
        
        try:
            rows_affected = self.db_manager.execute_dml(update_query, params)
            self.logger.info(f"Updated session progress: iteration={iteration}, best_score={best_score} (rows affected: {rows_affected})")
        except Exception as e:
            self.logger.error(f"Failed to update session progress: {e}")
    
    def close_session(self, status: str = 'completed') -> None:
        """Close the current session and update its status."""
        if not self.current_session_id:
            self.logger.warning("No active session to close")
            return
        
        update_query = """
        UPDATE sessions 
        SET status = :status, end_time = :end_time, updated_at = :updated_at
        WHERE session_id = :session_id
        """
        
        params = {
            'session_id': self.current_session_id,
            'status': status,
            'end_time': datetime.now(),
            'updated_at': datetime.now()
        }
        
        self.db_manager.execute_dml(update_query, params)
        self.logger.info(f"Closed session {self.current_session_id} with status: {status}")
    
    def get_resume_parameters(self) -> dict:
        """Get parameters for resuming MCTS from current session."""
        if not self.current_session_id:
            return {
                'next_iteration': 0,
                'loaded_node_count': 0,
                'best_score': 0.0,
                'resume_operation': None,
                'has_history': False,
                'root_score': None,
                'total_evaluations': 0
            }
        
        # Get session info
        session_query = """
        SELECT total_iterations, best_score 
        FROM sessions 
        WHERE session_id = :session_id
        """
        results = self.db_manager.execute_query(session_query, {'session_id': self.current_session_id})
        
        if results:
            session = results[0]
            has_history = session.get('total_iterations', 0) > 0
            return {
                'next_iteration': session.get('total_iterations', 0) + 1,
                'loaded_node_count': 0,  # Would need exploration history to get this
                'best_score': session.get('best_score', 0.0),
                'resume_operation': None,  # Would need last operation from exploration history
                'has_history': has_history,
                'root_score': session.get('best_score', 0.0) if has_history else None,
                'total_evaluations': session.get('total_iterations', 0) * 2  # Rough estimate
            }
        
        return {
            'next_iteration': 0,
            'loaded_node_count': 0,
            'best_score': 0.0,
            'resume_operation': None,
            'has_history': False,
            'root_score': None,
            'total_evaluations': 0
        }
    
    def get_session_progress(self) -> Dict[str, Any]:
        """Get current session progress information."""
        if not self.current_session_id:
            return {
                'total_iterations': 0,
                'best_score': 0.0,
                'exploration_count': 0,
                'unique_features': 0
            }
        
        # Get session info
        session_query = """
        SELECT total_iterations, best_score 
        FROM sessions 
        WHERE session_id = :session_id
        """
        results = self.db_manager.execute_query(session_query, {'session_id': self.current_session_id})
        
        if results:
            session = results[0]
            return {
                'total_iterations': session.get('total_iterations', 0),
                'best_score': session.get('best_score', 0.0),
                'exploration_count': 0,  # Would need exploration history count
                'unique_features': 0     # Would need feature count
            }
        
        return {
            'total_iterations': 0,
            'best_score': 0.0,
            'exploration_count': 0,
            'unique_features': 0
        }
    
    def export_best_features_to_session(self, limit: int = 20) -> Optional[str]:
        """Export Python code for best features to session directory."""
        # Placeholder implementation
        if self.output_manager:
            output_file = self.output_manager.get_file_path('best_features.py', 'exports')
            # Would need to implement actual feature export logic
            with open(output_file, 'w') as f:
                f.write("# Best features export not yet implemented\n")
            return output_file
        return None
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self.db_manager, 'close'):
            self.db_manager.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()