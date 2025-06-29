"""
Cleanup Command - Remove old or incomplete sessions.

Provides session cleanup capabilities including:
- Identification of cleanup candidates (interrupted, incomplete, test sessions)
- Safe removal with confirmation prompts
- Cascade deletion of related data
- Dry-run mode for testing
"""

from typing import Dict, Any, List
from .base import BaseSessionsCommand


class CleanupCommand(BaseSessionsCommand):
    """Handle --cleanup command for sessions."""
    
    def execute(self, args) -> None:
        """Execute the session cleanup command."""
        try:
            # Find sessions that need cleanup
            cleanup_candidates = self._find_cleanup_candidates()
            
            if not cleanup_candidates:
                self.print_success("✅ No sessions need cleanup")
                return
            
            # Show candidates
            self._show_cleanup_candidates(cleanup_candidates)
            
            # Check for dry-run mode
            if getattr(args, 'dry_run', False):
                self.print_info("🧪 DRY-RUN MODE: Cleanup simulation completed successfully")
                self.print_info(f"Would have deleted {len(cleanup_candidates)} sessions")
                return
            
            # Confirm deletion
            if self._confirm_cleanup():
                deleted_count = self._perform_cleanup(cleanup_candidates)
                self.print_success(f"✅ Deleted {deleted_count} sessions and related data")
            else:
                self.print_info("❌ Cleanup cancelled")
                
        except Exception as e:
            self.print_error(f"Failed to cleanup sessions: {e}")
    
    def _find_cleanup_candidates(self) -> List[Dict[str, Any]]:
        """Find sessions that are candidates for cleanup."""
        try:
            query = """
                SELECT 
                    session_id, 
                    session_name, 
                    status, 
                    total_iterations, 
                    is_test_mode,
                    start_time,
                    best_score
                FROM sessions
                WHERE status = 'interrupted' 
                   OR (total_iterations = 0 AND status = 'active')
                   OR is_test_mode = TRUE
                ORDER BY start_time DESC
            """
            
            results = self.session_service.repository.fetch_all(query)
            
            candidates = []
            for row in results:
                candidates.append({
                    'session_id': row['session_id'],
                    'session_name': row['session_name'],
                    'status': row['status'],
                    'total_iterations': row['total_iterations'],
                    'is_test_mode': row['is_test_mode'],
                    'start_time': row['start_time'],
                    'best_score': row['best_score']
                })
            
            return candidates
            
        except Exception as e:
            self.print_error(f"Failed to find cleanup candidates: {e}")
            return []
    
    def _show_cleanup_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        """Show sessions that will be cleaned up."""
        print("🧹 CLEANING UP SESSIONS")
        print("=" * 60)
        print(f"Found {len(candidates)} sessions for cleanup:")
        print()
        
        for i, session in enumerate(candidates, 1):
            session_id = session['session_id']
            name = session['session_name'] or 'Unnamed'
            status = session['status']
            iterations = session['total_iterations']
            is_test = session['is_test_mode']
            start_time = session['start_time']
            best_score = session['best_score']
            
            # Determine cleanup reasons
            reasons = []
            if status == 'interrupted':
                reasons.append("interrupted")
            if iterations == 0 and status == 'active':
                reasons.append("no progress")
            if is_test:
                reasons.append("test mode")
            
            # Format display
            session_short = session_id[:8]
            score_display = f", score: {best_score:.5f}" if best_score else ""
            
            print(f"{i:2}. {session_short}... ({name})")
            print(f"    Reason: {', '.join(reasons)}")
            print(f"    Status: {status}, Iterations: {iterations}{score_display}")
            print(f"    Started: {start_time}")
            
            # Check for related data
            related_data = self._check_related_data(session_id)
            if any(related_data.values()):
                related_items = []
                for table, count in related_data.items():
                    if count > 0:
                        related_items.append(f"{table}: {count}")
                print(f"    Related data: {', '.join(related_items)}")
            
            print()
    
    def _check_related_data(self, session_id: str) -> Dict[str, int]:
        """Check for related data that will be deleted."""
        try:
            related_tables = {
                'exploration_history': 'exploration_history',
                'feature_impact': 'feature_impact',
                'operation_performance': 'operation_performance',
                'system_performance': 'system_performance'
            }
            
            related_counts = {}
            
            for name, table in related_tables.items():
                try:
                    query = f"SELECT COUNT(*) as count FROM {table} WHERE session_id = ?"
                    result = self.session_service.repository.fetch_one(query, [session_id])
                    related_counts[name] = result.get('count', 0) if result else 0
                except Exception:
                    # Table might not exist or have different schema
                    related_counts[name] = 0
            
            return related_counts
            
        except Exception as e:
            self.print_error(f"Failed to check related data: {e}")
            return {}
    
    def _confirm_cleanup(self) -> bool:
        """Confirm cleanup operation with user."""
        print("⚠️  WARNING: This operation will permanently delete sessions and all related data.")
        print("   This includes exploration history, feature impacts, and performance data.")
        print()
        
        try:
            response = input("Delete these sessions? (yes/no): ").strip().lower()
            return response == 'yes'
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled by user")
            return False
    
    def _perform_cleanup(self, candidates: List[Dict[str, Any]]) -> int:
        """Perform the actual cleanup operation."""
        deleted_count = 0
        
        try:
            # Delete each session and its related data
            for session in candidates:
                session_id = session['session_id']
                
                try:
                    # Delete from all related tables (in dependency order)
                    self._delete_related_data(session_id)
                    
                    # Delete the session itself
                    delete_query = "DELETE FROM sessions WHERE session_id = ?"
                    self.session_service.repository.execute_query(delete_query, [session_id])
                    
                    deleted_count += 1
                    
                    # Log successful deletion
                    session_short = session_id[:8]
                    session_name = session['session_name'] or 'Unnamed'
                    self.print_info(f"   Deleted {session_short}... ({session_name})")
                    
                except Exception as e:
                    session_short = session_id[:8]
                    self.print_warning(f"   Failed to delete {session_short}...: {e}")
                    continue
            
            return deleted_count
            
        except Exception as e:
            self.print_error(f"Failed during cleanup operation: {e}")
            return deleted_count
    
    def _delete_related_data(self, session_id: str) -> None:
        """Delete all data related to a session."""
        # Order matters - delete children before parents
        related_tables = [
            'exploration_history',
            'feature_impact', 
            'operation_performance',
            'system_performance'
        ]
        
        for table in related_tables:
            try:
                delete_query = f"DELETE FROM {table} WHERE session_id = ?"
                self.session_service.repository.execute_query(delete_query, [session_id])
            except Exception as e:
                # Table might not exist or have different schema
                # This is not a critical error for cleanup
                pass