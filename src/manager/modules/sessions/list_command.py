"""
List Command - List recent sessions with summary statistics.

Provides session listing with filtering options including:
- Status filtering (active, completed, interrupted, all)
- Strategy filtering
- Limit control for result size
- Summary statistics and performance metrics
"""

from typing import Dict, Any, List
from .base import BaseSessionsCommand


class ListCommand(BaseSessionsCommand):
    """Handle --list command for sessions."""
    
    def execute(self, args) -> None:
        """Execute the sessions list command."""
        try:
            # Get session list with filters
            sessions = self._get_sessions_list(args)
            
            if not sessions:
                self.print_info("No sessions found.")
                self.print_info("Sessions are created when running MCTS discovery.")
                return
            
            # Get summary statistics
            summary_stats = self._get_summary_statistics()
            
            # Output results
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(sessions, summary_stats)
            else:
                self._output_sessions_table(sessions, summary_stats, args)
                
        except Exception as e:
            self.print_error(f"Failed to list sessions: {e}")
    
    def _get_sessions_list(self, args) -> List[Dict[str, Any]]:
        """Get filtered list of sessions."""
        try:
            # Build WHERE clause for filters
            conditions = []
            params = []
            
            # Status filter
            status = getattr(args, 'status', 'all')
            if status != 'all':
                conditions.append("status = ?")
                params.append(status)
            
            # Strategy filter
            strategy = getattr(args, 'strategy', None)
            if strategy:
                conditions.append("strategy = ?")
                params.append(strategy)
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            # Build final query
            limit = getattr(args, 'limit', 10)
            query = f"""
                SELECT 
                    session_id,
                    session_name,
                    start_time,
                    end_time,
                    total_iterations,
                    best_score,
                    status,
                    strategy,
                    is_test_mode,
                    config_snapshot
                FROM sessions 
                {where_clause}
                ORDER BY start_time DESC 
                LIMIT ?
            """
            params.append(limit)
            
            results = self.session_service.repository.fetch_all(query, params)
            
            # Convert to session objects
            sessions = []
            for row in results:
                sessions.append({
                    'session_id': row['session_id'],
                    'session_name': row['session_name'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'total_iterations': row['total_iterations'],
                    'best_score': row['best_score'],
                    'status': row['status'],
                    'strategy': row['strategy'],
                    'is_test_mode': row['is_test_mode'],
                    'config_snapshot': row['config_snapshot']
                })
            
            return sessions
            
        except Exception as e:
            self.print_error(f"Failed to get sessions list: {e}")
            return []
    
    def _get_summary_statistics(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_sessions,
                    AVG(best_score) as avg_score,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_sessions,
                    COUNT(CASE WHEN status = 'interrupted' THEN 1 END) as interrupted_sessions
                FROM sessions 
                WHERE best_score > 0
            """
            result = self.session_service.repository.fetch_one(query)
            
            if result:
                return {
                    'total_sessions': result['total_sessions'],
                    'avg_score': result['avg_score'],
                    'completed_sessions': result['completed_sessions'],
                    'active_sessions': result['active_sessions'],
                    'interrupted_sessions': result['interrupted_sessions']
                }
            
            return {}
            
        except Exception as e:
            self.print_error(f"Failed to get summary statistics: {e}")
            return {}
    
    def _output_sessions_table(self, sessions: List[Dict[str, Any]], 
                              summary_stats: Dict[str, Any], args) -> None:
        """Output sessions in formatted table view."""
        print("📅 MCTS DISCOVERY SESSIONS")
        print("=" * 60)
        
        # Show summary statistics
        if summary_stats:
            total = summary_stats.get('total_sessions', 0)
            avg_score = summary_stats.get('avg_score', 0)
            completed = summary_stats.get('completed_sessions', 0)
            active = summary_stats.get('active_sessions', 0)
            interrupted = summary_stats.get('interrupted_sessions', 0)
            
            avg_score_display = f"{avg_score:.5f}" if avg_score else "N/A"
            print(f"📊 Total sessions: {total}, Completed: {completed}, Active: {active}, Interrupted: {interrupted}")
            print(f"📈 Average score: {avg_score_display}")
            print()
        
        # Show sessions
        for i, session in enumerate(sessions, 1):
            session_short = session['session_id'][:8]
            name_display = session['session_name'] or "Unnamed"
            duration = "Running" if not session['end_time'] else self.format_duration(
                session['start_time'], session['end_time']
            )
            
            # Extract metric and format score
            metric = self.extract_target_metric(session['config_snapshot'])
            score_display = self.format_score(session['best_score'], metric)
            
            test_flag = " [TEST]" if session['is_test_mode'] else ""
            
            print(f"{i:2}. {session_short}... | {name_display}{test_flag}")
            print(f"    Status: {session['status']} | Strategy: {session['strategy']}")
            print(f"    Started: {session['start_time']} | Duration: {duration}")
            print(f"    Iterations: {session['total_iterations']} | Best Score: {score_display}")
            print()
        
        # Show filter info if applied
        status_filter = getattr(args, 'status', 'all')
        strategy_filter = getattr(args, 'strategy', None)
        limit = getattr(args, 'limit', 10)
        
        filters_applied = []
        if status_filter != 'all':
            filters_applied.append(f"status={status_filter}")
        if strategy_filter:
            filters_applied.append(f"strategy={strategy_filter}")
        
        if filters_applied:
            print(f"🔍 Filters applied: {', '.join(filters_applied)}")
            print(f"📄 Showing {len(sessions)} of {limit} max results")
        else:
            print(f"📄 Showing latest {len(sessions)} sessions (limit: {limit})")
        
        # Quick actions
        print("\n💡 Quick Actions:")
        print("   Show details: python manager.py sessions --show <SESSION_ID>")
        print("   Compare sessions: python manager.py sessions --compare <ID1> <ID2>")
        if sessions:
            example_id = sessions[0]['session_id'][:8]
            print(f"   Example: python manager.py sessions --show {example_id}")
    
    def _output_json(self, sessions: List[Dict[str, Any]], 
                     summary_stats: Dict[str, Any]) -> None:
        """Output sessions in JSON format."""
        output_data = {
            'summary': summary_stats,
            'sessions': sessions
        }
        self.print_json(output_data, "Sessions List")