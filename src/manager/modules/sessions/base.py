"""
Base Sessions Command - Common functionality for sessions commands.

Provides shared utilities, formatting, and database access patterns
for all sessions module commands.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import json
from manager.core.command_base import BaseCommand


class BaseSessionsCommand(BaseCommand, ABC):
    """Base class for all sessions commands."""
    
    def __init__(self):
        super().__init__()
        self.session_service = None
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """Inject required services."""
        super().inject_services(services)
        self.session_service = services.get('session_service')
        if not self.session_service:
            raise ValueError("SessionService is required for sessions commands")
    
    @abstractmethod
    def execute(self, args) -> None:
        """Execute the command with given arguments."""
        pass
    
    def format_duration(self, start_time: str, end_time: str = None) -> str:
        """Format duration between start and end time."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else datetime.now()
            
            duration = end - start
            total_seconds = int(duration.total_seconds())
            
            if total_seconds < 60:
                return f"{total_seconds}s"
            elif total_seconds < 3600:
                return f"{total_seconds // 60}m {total_seconds % 60}s"
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                return f"{hours}h {minutes}m"
                
        except:
            return "Unknown"
    
    def extract_target_metric(self, config_json: str) -> str:
        """Extract target metric from configuration JSON."""
        if not config_json:
            return "unknown"
        
        try:
            config = json.loads(config_json) if isinstance(config_json, str) else config_json
            return config.get('autogluon', {}).get('target_metric', 'unknown')
        except:
            return "unknown"
    
    def format_score(self, score: float, metric: str = None) -> str:
        """Format score with optional metric suffix."""
        if not score or score == 0:
            return "No score"
        
        score_str = f"{score:.5f}"
        if metric and metric != "unknown":
            metric_short = metric[:3] if len(metric) > 3 else metric
            score_str += f" ({metric_short})"
        
        return score_str
    
    def format_session_identifier(self, session_id: str, name: str = None) -> str:
        """Format session display with ID and optional name."""
        session_short = session_id[:8] if session_id else "Unknown"
        if name:
            return f"{session_short}... | {name}"
        return f"{session_short}..."
    
    def get_session_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get session by ID or name."""
        try:
            # First try by UUID
            query = """
                SELECT 
                    session_id, session_name, start_time, end_time, 
                    total_iterations, best_score, config_snapshot, 
                    status, strategy, is_test_mode, notes, dataset_hash
                FROM sessions 
                WHERE session_id = ? OR session_name = ?
            """
            result = self.session_service.repository.fetch_one(query, [identifier, identifier])
            
            if result:
                return {
                    'session_id': result['session_id'],
                    'session_name': result['session_name'],
                    'start_time': result['start_time'],
                    'end_time': result['end_time'],
                    'total_iterations': result['total_iterations'],
                    'best_score': result['best_score'],
                    'config_snapshot': result['config_snapshot'],
                    'status': result['status'],
                    'strategy': result['strategy'],
                    'is_test_mode': result['is_test_mode'],
                    'notes': result['notes'],
                    'dataset_hash': result['dataset_hash']
                }
            
            return None
            
        except Exception as e:
            self.print_error(f"Failed to get session: {e}")
            return None
    
    def get_exploration_statistics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get exploration statistics for a session."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_explorations,
                    MIN(evaluation_score) as min_score,
                    MAX(evaluation_score) as max_score,
                    AVG(evaluation_score) as avg_score,
                    SUM(evaluation_time) as total_eval_time,
                    AVG(evaluation_time) as avg_eval_time,
                    COUNT(DISTINCT operation_applied) as unique_operations
                FROM exploration_history 
                WHERE session_id = ?
            """
            result = self.session_service.repository.fetch_one(query, [session_id])
            
            if result and result.get('total_explorations', 0) > 0:
                return {
                    'total_explorations': result['total_explorations'],
                    'min_score': result['min_score'],
                    'max_score': result['max_score'],
                    'avg_score': result['avg_score'],
                    'total_eval_time': result['total_eval_time'],
                    'avg_eval_time': result['avg_eval_time'],
                    'unique_operations': result['unique_operations']
                }
            
            return None
            
        except Exception as e:
            self.print_error(f"Failed to get exploration statistics: {e}")
            return None
    
    def get_top_operations(self, session_id: str, limit: int = 5) -> list:
        """Get top performing operations for a session."""
        try:
            query = """
                SELECT 
                    operation_applied,
                    COUNT(*) as usage_count,
                    AVG(evaluation_score) as avg_score,
                    MAX(evaluation_score) as best_score
                FROM exploration_history 
                WHERE session_id = ?
                GROUP BY operation_applied
                ORDER BY avg_score DESC
                LIMIT ?
            """
            return self.session_service.repository.fetch_all(query, [session_id, limit])
            
        except Exception as e:
            self.print_error(f"Failed to get top operations: {e}")
            return []