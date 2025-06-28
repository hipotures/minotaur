"""
Verify Last Command - Verify last N sessions.

Provides verification of the last N sessions with:
- Configurable session count
- Batch verification with table display
- Filtered results based on verification status
"""

from typing import Dict, Any, List
from .verify_all_command import VerifyAllCommand


class VerifyLastCommand(VerifyAllCommand):
    """Handle --last command for verification."""
    
    def execute(self, args) -> None:
        """Execute the verify last N sessions command."""
        try:
            count = args.last
            
            print(f"🔍 VERIFYING LAST {count} SESSION{'S' if count != 1 else ''}")
            print("=" * 60)
            
            # Get last N sessions
            sessions = self._get_last_sessions(count)
            
            if not sessions:
                self.print_info("No sessions found in database")
                self.print_info("Sessions are created when running MCTS discovery")
                return
            
            # Reverse to show oldest to newest (but only the last N)
            sessions = list(reversed(sessions))
            
            print(f"Found {len(sessions)} session{'s' if len(sessions) != 1 else ''} to verify")
            print()
            
            # Use the batch verification logic from parent class
            self._verify_sessions_batch(sessions, args)
            
        except Exception as e:
            self.print_error(f"Failed to verify last sessions: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
    
    def _get_last_sessions(self, count: int) -> List[tuple]:
        """Get last N sessions from database."""
        try:
            query = """
                SELECT session_id, session_name, status 
                FROM sessions 
                ORDER BY start_time DESC
                LIMIT ?
            """
            results = self.session_service.repository.fetch_all(query, [count])
            
            # Convert to tuples for compatibility
            sessions = []
            for row in results:
                sessions.append((
                    row['session_id'],
                    row['session_name'],
                    row['status']
                ))
            
            return sessions
            
        except Exception as e:
            self.print_error(f"Failed to get last sessions: {e}")
            return []