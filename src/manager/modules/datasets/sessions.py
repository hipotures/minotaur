"""
Sessions Command - Show sessions using a specific dataset.

Provides detailed session listing for a dataset including:
- Session information (ID, status, scores, timing)
- Performance metrics and trends
- Session comparison capabilities
- Filtering and sorting options
"""

from datetime import datetime
from typing import Dict, Any, List
from .base import BaseDatasetsCommand


class SessionsCommand(BaseDatasetsCommand):
    """Handle --sessions command for datasets."""
    
    def execute(self, args) -> None:
        """Execute the show dataset sessions command."""
        try:
            dataset_identifier = args.sessions
            dataset = self.find_dataset_by_identifier(dataset_identifier)
            
            if not dataset:
                self.print_error(f"Dataset '{dataset_identifier}' not found.")
                return
            
            # Get sessions for this dataset
            sessions = self.session_service.get_sessions_by_dataset(dataset['dataset_id'])
            
            if not sessions:
                self.print_info(f"No sessions found for dataset '{dataset['dataset_name']}'.")
                self.print_info(f"Start a session: python mcts.py --config config/mcts_config.yaml")
                return
            
            # Output in requested format
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(dataset, sessions)
            else:
                self._output_formatted_sessions(dataset, sessions)
                
        except Exception as e:
            self.print_error(f"Failed to show dataset sessions: {e}")
    
    def _output_formatted_sessions(self, dataset: Dict[str, Any], sessions: List[Dict[str, Any]]) -> None:
        """Output sessions in formatted table view."""
        dataset_name = dataset['dataset_name']
        
        print(f"\n📊 SESSIONS FOR DATASET: {dataset_name}")
        print("=" * 60)
        print(f"🔑 Dataset ID: {dataset['dataset_id'][:8]}")
        print(f"📈 Total Sessions: {len(sessions)}")
        
        # Sort sessions by start time (most recent first)
        sorted_sessions = sorted(sessions, 
                               key=lambda s: s.get('start_time', ''), 
                               reverse=True)
        
        # Calculate summary statistics
        completed_sessions = [s for s in sessions if s.get('status') == 'completed']
        success_rate = len(completed_sessions) / len(sessions) * 100 if sessions else 0
        
        scores = [s.get('best_score', 0) for s in completed_sessions if s.get('best_score')]
        avg_score = sum(scores) / len(scores) if scores else 0
        best_score = max(scores) if scores else 0
        
        print(f"✅ Success Rate: {success_rate:.1f}%")
        if avg_score > 0:
            print(f"📊 Average Score: {avg_score:.5f}")
            print(f"🏆 Best Score: {best_score:.5f}")
        
        # Show sessions table
        print(f"\n📋 SESSION DETAILS")
        print("-" * 80)
        
        headers = ['Session ID', 'Status', 'Start Time', 'Duration', 'Iterations', 'Best Score']
        rows = []
        
        for session in sorted_sessions:
            # Format duration
            duration = self._calculate_session_duration(session)
            duration_str = self.format_duration(duration) if duration else "N/A"
            
            # Format start time
            start_time = session.get('start_time', '')
            start_time_str = start_time[:16].replace('T', ' ') if start_time else 'Unknown'
            
            # Format score
            best_score = session.get('best_score')
            score_str = f"{best_score:.5f}" if best_score else "N/A"
            
            # Status icon
            status = session.get('status', 'unknown')
            status_icon = {
                'completed': '✅ Completed',
                'failed': '❌ Failed', 
                'running': '⏳ Running',
                'interrupted': '⚠️  Interrupted'
            }.get(status, f"❔ {status.title()}")
            
            rows.append([
                session.get('session_id', '')[:8],
                status_icon,
                start_time_str,
                duration_str,
                str(session.get('total_iterations', 0)),
                score_str
            ])
        
        self.print_table(headers, rows)
        
        # Show recent activity
        recent_sessions = [s for s in sorted_sessions[:5]]
        if recent_sessions:
            print(f"\n🕒 RECENT ACTIVITY")
            print("-" * 40)
            for session in recent_sessions:
                session_id = session.get('session_id', '')[:8]
                status = session.get('status', 'unknown')
                start_time = session.get('start_time', '')[:10]
                
                status_icon = {"completed": "✅", "failed": "❌", "running": "⏳"}.get(status, "❔")
                print(f"{status_icon} {session_id} - {start_time}")
        
        # Show quick actions
        print(f"\n💡 QUICK ACTIONS")
        print("-" * 40)
        if completed_sessions:
            best_session = max(completed_sessions, key=lambda s: s.get('best_score', 0), default=None)
            if best_session:
                best_session_id = best_session.get('session_id', '')[:8]
                print(f"   View best session: python manager.py sessions --show {best_session_id}")
        
        print(f"   Compare sessions: python manager.py sessions --compare SESSION1 SESSION2")
        print(f"   Dataset details: python manager.py datasets --details {dataset_name}")
    
    def _calculate_session_duration(self, session: Dict[str, Any]) -> float:
        """Calculate session duration in seconds."""
        try:
            start_time = session.get('start_time')
            end_time = session.get('end_time')
            
            if start_time and end_time:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                return (end_dt - start_dt).total_seconds()
            elif start_time and session.get('status') == 'running':
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                now = datetime.now()
                return (now - start_dt.replace(tzinfo=None)).total_seconds()
            else:
                return 0.0
                
        except (ValueError, TypeError):
            return 0.0
    
    def _output_json(self, dataset: Dict[str, Any], sessions: List[Dict[str, Any]]) -> None:
        """Output sessions in JSON format."""
        output = {
            'dataset': {
                'name': dataset['dataset_name'],
                'id': dataset['dataset_id'],
            },
            'sessions': sessions,
            'summary': {
                'total_sessions': len(sessions),
                'completed_sessions': len([s for s in sessions if s.get('status') == 'completed']),
                'success_rate': len([s for s in sessions if s.get('status') == 'completed']) / len(sessions) * 100 if sessions else 0,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        self.print_json(output, f"Sessions for Dataset: {dataset['dataset_name']}")