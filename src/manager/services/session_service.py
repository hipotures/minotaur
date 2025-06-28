"""
Service layer for session-related business logic
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from ..repositories.session_repository import SessionRepository
from ..core.utils import format_duration, format_datetime, format_number


class SessionService:
    """Handles session-related business logic."""
    
    def __init__(self, session_repository: SessionRepository):
        """Initialize service with repository.
        
        Args:
            session_repository: Session repository instance
        """
        self.repository = session_repository
        self.logger = logging.getLogger(__name__)
    
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
        # Get all sessions
        if dataset:
            sessions = self.repository.get_sessions_by_dataset(dataset)
        else:
            sessions = self.repository.get_all_sessions()
        
        # Apply filters
        if status:
            sessions = [s for s in sessions if s['status'] == status]
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            sessions = [s for s in sessions 
                       if datetime.fromisoformat(s['created_at']) >= cutoff_date]
        
        # Format sessions
        formatted_sessions = []
        for session in sessions:
            formatted_sessions.append({
                'session_id': session['session_id'],
                'created_at': format_datetime(session['created_at'], 'short'),
                'status': session['status'],
                'dataset': session.get('dataset_name', 'Unknown'),
                'best_score': f"{session.get('best_score', 0):.4f}",
                'nodes': format_number(session.get('nodes_explored', 0)),
                'features': format_number(session.get('features_generated', 0)),
                'duration': format_duration(session.get('elapsed_time', 0))
            })
        
        return formatted_sessions
    
    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session information with formatted output.
        
        Args:
            session_id: Session ID
            
        Returns:
            Detailed session information or None
        """
        session = self.repository.get_session_by_id(session_id)
        
        if not session:
            return None
        
        # Format timestamps
        created_at = datetime.fromisoformat(session['created_at'])
        updated_at = datetime.fromisoformat(session['updated_at']) if session['updated_at'] else created_at
        
        # Calculate derived metrics
        duration = session.get('elapsed_time', 0)
        nodes_per_minute = (session.get('nodes_explored', 0) / (duration / 60)) if duration > 0 else 0
        features_per_node = (session.get('features_generated', 0) / 
                           max(1, session.get('nodes_explored', 0)))
        
        return {
            'basic_info': {
                'session_id': session['session_id'],
                'status': session['status'],
                'dataset': session.get('dataset_name', 'Unknown'),
                'created': format_datetime(created_at, 'long'),
                'updated': format_datetime(updated_at, 'long'),
                'duration': format_duration(duration),
                'error': session.get('error_message')
            },
            'performance': {
                'best_score': session.get('best_score', 0),
                'nodes_explored': session.get('nodes_explored', 0),
                'features_generated': session.get('features_generated', 0),
                'nodes_per_minute': round(nodes_per_minute, 2),
                'features_per_node': round(features_per_node, 2)
            },
            'config': session.get('config_data', {})
        }
    
    def compare_sessions(self, session_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple sessions side by side.
        
        Args:
            session_ids: List of session IDs to compare
            
        Returns:
            Comparison data
        """
        sessions = []
        for session_id in session_ids:
            session = self.repository.get_session_by_id(session_id)
            if session:
                sessions.append(session)
        
        if not sessions:
            return {'error': 'No valid sessions found'}
        
        # Find best values for highlighting
        best_score = max(s.get('best_score', 0) for s in sessions)
        min_time = min(s.get('elapsed_time', float('inf')) for s in sessions)
        max_nodes = max(s.get('nodes_explored', 0) for s in sessions)
        
        # Build comparison
        comparison = {
            'sessions': [],
            'summary': {
                'best_score': best_score,
                'fastest_time': min_time,
                'most_nodes': max_nodes
            }
        }
        
        for session in sessions:
            score = session.get('best_score', 0)
            time = session.get('elapsed_time', 0)
            nodes = session.get('nodes_explored', 0)
            
            comparison['sessions'].append({
                'session_id': session['session_id'],
                'dataset': session.get('dataset_name', 'Unknown'),
                'status': session['status'],
                'score': score,
                'is_best_score': score == best_score,
                'time': time,
                'is_fastest': time == min_time,
                'nodes': nodes,
                'is_most_nodes': nodes == max_nodes,
                'features': session.get('features_generated', 0),
                'efficiency': round(score / max(1, time / 60), 4)  # Score per minute
            })
        
        return comparison
    
    def get_session_trends(self, dataset: Optional[str] = None,
                          days: int = 30) -> Dict[str, Any]:
        """Analyze session trends over time.
        
        Args:
            dataset: Optional dataset filter
            days: Number of days to analyze
            
        Returns:
            Trend analysis data
        """
        stats = self.repository.get_session_statistics(days)
        
        # Calculate trend indicators
        if dataset:
            recent_sessions = self.repository.get_sessions_by_dataset(dataset)
        else:
            recent_sessions = self.repository.get_recent_sessions(100)
        
        # Group by day for trend analysis
        daily_stats = {}
        for session in recent_sessions:
            date = datetime.fromisoformat(session['created_at']).date()
            if date not in daily_stats:
                daily_stats[date] = {
                    'count': 0,
                    'total_score': 0,
                    'total_time': 0
                }
            
            daily_stats[date]['count'] += 1
            daily_stats[date]['total_score'] += session.get('best_score', 0)
            daily_stats[date]['total_time'] += session.get('elapsed_time', 0)
        
        # Calculate moving averages
        dates = sorted(daily_stats.keys())
        trends = []
        
        for i, date in enumerate(dates):
            # 7-day moving average
            start_idx = max(0, i - 6)
            window_dates = dates[start_idx:i + 1]
            
            window_count = sum(daily_stats[d]['count'] for d in window_dates)
            window_score = sum(daily_stats[d]['total_score'] for d in window_dates)
            window_time = sum(daily_stats[d]['total_time'] for d in window_dates)
            
            avg_score = window_score / max(1, window_count)
            avg_time = window_time / max(1, window_count)
            
            trends.append({
                'date': date.isoformat(),
                'sessions': daily_stats[date]['count'],
                'avg_score': round(avg_score, 4),
                'avg_time': round(avg_time, 2),
                'ma7_score': round(avg_score, 4),
                'ma7_sessions': round(window_count / len(window_dates), 1)
            })
        
        return {
            'overall_stats': stats,
            'daily_trends': trends[-30:],  # Last 30 days
            'summary': {
                'total_sessions': stats['total_sessions'],
                'success_rate': stats['status_counts'].get('completed', 0) / 
                              max(1, stats['total_sessions']),
                'avg_score_trend': 'improving' if len(trends) >= 2 and 
                                 trends[-1]['avg_score'] > trends[-7]['avg_score'] 
                                 else 'stable'
            }
        }
    
    def cleanup_sessions(self, days_old: int = 30,
                        dry_run: bool = True) -> Dict[str, Any]:
        """Clean up old sessions.
        
        Args:
            days_old: Delete sessions older than this many days
            dry_run: If True, only report what would be deleted
            
        Returns:
            Cleanup results
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        all_sessions = self.repository.get_all_sessions(include_deleted=True)
        
        to_delete = []
        for session in all_sessions:
            created = datetime.fromisoformat(session['created_at'])
            if created < cutoff_date and not session.get('deleted_at'):
                to_delete.append(session)
        
        if not dry_run:
            deleted_count = 0
            for session in to_delete:
                if self.repository.delete_session(session['session_id']):
                    deleted_count += 1
            
            return {
                'deleted': deleted_count,
                'message': f"Deleted {deleted_count} sessions older than {days_old} days"
            }
        else:
            return {
                'would_delete': len(to_delete),
                'sessions': [
                    {
                        'session_id': s['session_id'],
                        'created': format_datetime(s['created_at'], 'short'),
                        'dataset': s.get('dataset_name', 'Unknown'),
                        'score': s.get('best_score', 0)
                    }
                    for s in to_delete
                ],
                'message': f"Would delete {len(to_delete)} sessions (dry run)"
            }