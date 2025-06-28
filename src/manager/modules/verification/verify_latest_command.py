"""
Verify Latest Command - Verify most recent session.

Provides verification of the most recently created session with:
- Automatic latest session detection
- Full verification suite
- User-friendly session identification
"""

from typing import Dict, Any, Optional
from .verify_session_command import VerifySessionCommand


class VerifyLatestCommand(VerifySessionCommand):
    """Handle --verify-latest command for verification."""
    
    def execute(self, args) -> None:
        """Execute the verify latest session command."""
        try:
            # Find the latest session
            latest_session = self._get_latest_session()
            
            if not latest_session:
                self.print_error("No sessions found in database")
                self.print_info("Sessions are created when running MCTS discovery")
                self.print_info("Try: python mcts.py --config config/mcts_config_s5e6_fast_test.yaml --test-mode")
                return
            
            session_id, session_name = latest_session
            session_identifier = session_name or session_id
            
            print(f"🔍 Verifying latest session: {session_name} ({session_id[:8]}...)")
            print()
            
            # Use the session verification logic from parent class
            # but override the session identifier
            args.verify_session = session_identifier
            super().execute(args)
            
        except Exception as e:
            self.print_error(f"Failed to verify latest session: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
    
    def _get_latest_session(self) -> Optional[tuple]:
        """Get the most recent session from database."""
        try:
            query = """
                SELECT session_id, session_name 
                FROM sessions 
                ORDER BY start_time DESC 
                LIMIT 1
            """
            result = self.session_service.repository.fetch_one(query)
            
            if result:
                return (result['session_id'], result['session_name'])
            
            return None
            
        except Exception as e:
            self.print_error(f"Failed to get latest session: {e}")
            return None