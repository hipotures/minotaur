"""
Universal Session Resolver for Minotaur MCTS System

Provides unified session identification and resolution across all tools.
Supports multiple input formats: full UUID, partial UUID, session name, keywords.
"""

import logging
import re
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Standardized session information."""
    session_id: str
    session_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    total_iterations: int
    best_score: Optional[float]
    is_test_mode: bool
    config_hash: Optional[str]
    

class SessionResolutionError(Exception):
    """Raised when session cannot be resolved."""
    pass


class SessionResolver:
    """
    Universal session resolver supporting multiple identification formats.
    
    Supported formats:
    - Full UUID: "99cde13b-9f49-443e-a59b-1950d497617f"
    - Partial UUID (≥6 chars): "99cde13b", "99cde13b-9f49"
    - Session name: "session_20250702_030849"
    - Keywords: "latest", "last", "recent"
    """
    
    def __init__(self, db_connection_manager):
        """Initialize resolver with database connection manager."""
        self.connection_manager = db_connection_manager
        
    def resolve_session(self, identifier: str) -> SessionInfo:
        """
        Resolve session from any supported identifier format.
        
        Args:
            identifier: Session identifier (UUID, name, or keyword)
            
        Returns:
            SessionInfo object with resolved session data
            
        Raises:
            SessionResolutionError: If session cannot be resolved
        """
        if not identifier or not identifier.strip():
            raise SessionResolutionError("Session identifier cannot be empty")
            
        identifier = identifier.strip()
        
        # Handle keywords
        if identifier.lower() in ['latest', 'last', 'recent']:
            return self._get_latest_session()
            
        # Check if it looks like a UUID (contains hyphens or is hex-like)
        if self._is_uuid_like(identifier):
            return self._resolve_by_uuid(identifier)
            
        # Try as session name
        session = self._resolve_by_name(identifier)
        if session:
            return session
            
        # If nothing worked, try partial UUID match
        session = self._resolve_by_partial_uuid(identifier)
        if session:
            return session
            
        # Generate helpful error message
        suggestions = self._get_session_suggestions(identifier)
        error_msg = f"Session not found: '{identifier}'"
        if suggestions:
            error_msg += f"\n\nDid you mean:\n" + "\n".join(f"  - {s}" for s in suggestions[:3])
        error_msg += f"\n\nUse one of these formats:"
        error_msg += f"\n  - Full UUID: 99cde13b-9f49-443e-a59b-1950d497617f"
        error_msg += f"\n  - Partial UUID: 99cde13b (≥6 chars)"
        error_msg += f"\n  - Session name: session_20250702_030849"
        error_msg += f"\n  - Keyword: latest"
        
        raise SessionResolutionError(error_msg)
    
    def resolve_multiple_sessions(self, identifiers: List[str]) -> List[SessionInfo]:
        """
        Resolve multiple sessions, ensuring they are all the same type.
        
        Args:
            identifiers: List of session identifiers
            
        Returns:
            List of SessionInfo objects
            
        Raises:
            SessionResolutionError: If any session cannot be resolved
        """
        if not identifiers:
            raise SessionResolutionError("No session identifiers provided")
            
        sessions = []
        for identifier in identifiers:
            session = self.resolve_session(identifier)
            sessions.append(session)
            
        return sessions
    
    def _is_uuid_like(self, identifier: str) -> bool:
        """Check if identifier looks like a UUID."""
        # Has hyphens = definitely UUID format
        if '-' in identifier:
            return True
            
        # All hex characters and reasonable length = probably UUID
        if re.match(r'^[0-9a-fA-F]{6,}$', identifier):
            return True
            
        return False
    
    def _resolve_by_uuid(self, uuid_identifier: str) -> SessionInfo:
        """Resolve session by full or partial UUID."""
        # If it looks like a full UUID, try exact match first
        if len(uuid_identifier) >= 32 and '-' in uuid_identifier:
            session = self._fetch_session_by_exact_id(uuid_identifier)
            if session:
                return session
                
        # Try partial match
        return self._resolve_by_partial_uuid(uuid_identifier)
    
    def _resolve_by_partial_uuid(self, partial_uuid: str) -> Optional[SessionInfo]:
        """Resolve session by partial UUID (≥6 characters)."""
        if len(partial_uuid) < 6:
            return None
            
        # Clean up the partial UUID - remove hyphens for partial matching
        clean_partial = partial_uuid.replace('-', '')
        
        query = """
        SELECT session_id, session_name, start_time, end_time, status, 
               total_iterations, best_score, is_test_mode, config_hash
        FROM sessions 
        WHERE REPLACE(session_id, '-', '') LIKE ?
        ORDER BY start_time DESC 
        LIMIT 2
        """
        
        with self.connection_manager.get_connection() as conn:
            results = conn.execute(query, [f"{clean_partial}%"]).fetchall()
            
        if not results:
            return None
            
        if len(results) > 1:
            # Multiple matches - need more specific identifier
            matches = [r[0][:8] + "..." for r in results[:3]]
            raise SessionResolutionError(
                f"Ambiguous session identifier '{partial_uuid}'. "
                f"Multiple sessions match:\n" + 
                "\n".join(f"  - {m}" for m in matches) +
                f"\nPlease provide a more specific identifier."
            )
            
        return self._result_to_session_info(results[0])
    
    def _resolve_by_name(self, session_name: str) -> Optional[SessionInfo]:
        """Resolve session by exact session name."""
        query = """
        SELECT session_id, session_name, start_time, end_time, status, 
               total_iterations, best_score, is_test_mode, config_hash
        FROM sessions 
        WHERE session_name = ?
        ORDER BY start_time DESC 
        LIMIT 1
        """
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute(query, [session_name]).fetchone()
            
        if result:
            return self._result_to_session_info(result)
        return None
    
    def _fetch_session_by_exact_id(self, session_id: str) -> Optional[SessionInfo]:
        """Fetch session by exact session ID."""
        query = """
        SELECT session_id, session_name, start_time, end_time, status, 
               total_iterations, best_score, is_test_mode, config_hash
        FROM sessions 
        WHERE session_id = ?
        """
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute(query, [session_id]).fetchone()
            
        if result:
            return self._result_to_session_info(result)
        return None
    
    def _get_latest_session(self) -> SessionInfo:
        """Get the most recent session."""
        query = """
        SELECT session_id, session_name, start_time, end_time, status, 
               total_iterations, best_score, is_test_mode, config_hash
        FROM sessions 
        ORDER BY start_time DESC 
        LIMIT 1
        """
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute(query).fetchone()
            
        if not result:
            raise SessionResolutionError(
                "No sessions found in database. "
                "Run MCTS to create a session first."
            )
            
        return self._result_to_session_info(result)
    
    def _get_session_suggestions(self, failed_identifier: str) -> List[str]:
        """Get session suggestions for failed resolution."""
        # Get recent sessions for suggestions
        query = """
        SELECT session_id, session_name, start_time
        FROM sessions 
        ORDER BY start_time DESC 
        LIMIT 5
        """
        
        try:
            with self.connection_manager.get_connection() as conn:
                results = conn.execute(query).fetchall()
                
            suggestions = []
            for result in results:
                session_id, session_name, start_time = result
                short_id = session_id[:8] + "..."
                if session_name:
                    suggestions.append(f"{short_id} ({session_name})")
                else:
                    suggestions.append(short_id)
                    
            return suggestions
            
        except Exception as e:
            logger.warning(f"Could not fetch session suggestions: {e}")
            return []
    
    def _result_to_session_info(self, result) -> SessionInfo:
        """Convert database result to SessionInfo object."""
        (session_id, session_name, start_time, end_time, status, 
         total_iterations, best_score, is_test_mode, config_hash) = result
         
        # Handle datetime conversion
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
        return SessionInfo(
            session_id=session_id,
            session_name=session_name or "",
            start_time=start_time,
            end_time=end_time,
            status=status or "unknown",
            total_iterations=total_iterations or 0,
            best_score=best_score,
            is_test_mode=is_test_mode or False,
            config_hash=config_hash
        )


def create_session_resolver(config: Dict[str, Any]) -> SessionResolver:
    """
    Factory function to create SessionResolver with appropriate database connection.
    
    Args:
        config: MCTS configuration dictionary
        
    Returns:
        Configured SessionResolver instance
    """
    from src.database.engine_factory import DatabaseFactory
    
    # Get database configuration
    db_config = config.get('database', {})
    db_type = db_config.get('type', 'duckdb')
    
    # Extract connection parameters
    connection_params = {
        'database': db_config.get('path', 'data/minotaur.duckdb'),
        'read_only': True
    }
    
    # Create database manager using factory
    db_manager = DatabaseFactory.create_manager(db_type, connection_params)
    
    # Create a wrapper that provides get_connection method for compatibility
    class ConnectionManagerWrapper:
        def __init__(self, db_manager):
            self.db_manager = db_manager
            self.db_path = connection_params['database']
            
        def close(self):
            """Close database connection."""
            if hasattr(self.db_manager, 'close'):
                self.db_manager.close()
            
        def get_connection(self):
            # Return a context manager that provides execute method
            class ConnectionWrapper:
                def __init__(self, db_manager):
                    self.db_manager = db_manager
                    
                def __enter__(self):
                    return self
                    
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
                    
                def execute(self, query, params=None):
                    # Convert list params to dict for SQLAlchemy
                    if isinstance(params, list):
                        # For LIKE queries, we need to handle positional parameters
                        import re
                        # Replace ? with :param0, :param1, etc.
                        param_dict = {}
                        for i, param in enumerate(params):
                            param_dict[f'param{i}'] = param
                        # Replace ? with named parameters
                        query_parts = query.split('?')
                        new_query = query_parts[0]
                        for i in range(1, len(query_parts)):
                            new_query += f':param{i-1}' + query_parts[i]
                        query = new_query
                        params = param_dict
                    
                    results = self.db_manager.execute_query(query, params)
                    
                    # Convert results to match expected format
                    class ResultWrapper:
                        def __init__(self, results):
                            self.results = results
                            
                        def fetchall(self):
                            return [tuple(row.values()) for row in self.results]
                            
                        def fetchone(self):
                            if self.results:
                                return tuple(self.results[0].values())
                            return None
                    
                    return ResultWrapper(results)
            
            return ConnectionWrapper(self.db_manager)
    
    connection_manager = ConnectionManagerWrapper(db_manager)
    return SessionResolver(connection_manager)