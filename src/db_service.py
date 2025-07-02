"""
Database Service Manager for Minotaur MCTS System.

This module provides a high-level service layer using the new SQLAlchemy-based
database abstraction layer. It replaces the old repository-based architecture
with a simplified interface.
"""

import logging
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
        
        # Initialize session output manager
        self.session_output_manager = SessionOutputManager(config)
    
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
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self.db_manager, 'close'):
            self.db_manager.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()