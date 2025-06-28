"""
Verify All Command - Verify all sessions in database.

Provides batch verification of all sessions with:
- Quick verification mode for batch processing
- Status filtering (failed-only, warn-only, pass-only)
- Batch summary statistics
- Performance optimization for large datasets
"""

from typing import Dict, Any, List
from .base import BaseVerificationCommand


class VerifyAllCommand(BaseVerificationCommand):
    """Handle --verify-all command for verification."""
    
    def execute(self, args) -> None:
        """Execute the verify all sessions command."""
        try:
            print("🔍 BATCH SESSION VERIFICATION")
            print("=" * 60)
            
            # Get all sessions
            sessions = self._get_all_sessions()
            
            if not sessions:
                self.print_info("No sessions found in database")
                self.print_info("Sessions are created when running MCTS discovery")
                return
            
            print(f"Found {len(sessions)} sessions to verify (oldest to newest)")
            print()
            
            # Use common batch verification logic
            self._verify_sessions_batch(sessions, args)
            
        except Exception as e:
            self.print_error(f"Batch verification failed: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
    
    def _get_all_sessions(self) -> List[tuple]:
        """Get all sessions from database."""
        try:
            query = """
                SELECT session_id, session_name, status 
                FROM sessions 
                ORDER BY start_time ASC
            """
            results = self.session_service.repository.fetch_all(query)
            
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
            self.print_error(f"Failed to get sessions: {e}")
            return []
    
    def _verify_sessions_batch(self, sessions: List[tuple], args) -> None:
        """Common batch verification logic with filtering support."""
        
        # Print table header
        print(f"{'#':<4} {'Session Name':<25} {'Status':<8} {'Result':<6}")
        print("-" * 50)
        
        batch_results = []
        displayed_count = 0
        
        for i, (session_id, session_name, status) in enumerate(sessions, 1):
            # Run quick verification for batch mode
            original_quick = getattr(args, 'quick', False)
            args.quick = True
            
            try:
                verification_result = self._verify_session_quick(session_name or session_id, args)
                batch_results.append(verification_result)
                
                # Format result with icon
                result_status = verification_result['overall_status']
                if result_status == 'PASS':
                    result_display = "✅ PASS"
                elif result_status == 'WARN':
                    result_display = "⚠️ WARN"
                elif result_status == 'FAIL':
                    result_display = "❌ FAIL"
                else:
                    result_display = "💥 ERROR"
                
            except Exception as e:
                batch_results.append({
                    'session_id': session_id,
                    'overall_status': 'ERROR',
                    'error': str(e)
                })
                result_display = "💥 ERROR"
                result_status = 'ERROR'
            
            # Apply filters
            should_display = self.should_display_result(result_status, args)
            
            if should_display:
                # Print table row
                session_display = (session_name or session_id[:8])[:24]
                status_display = status[:7] if status else "unknown"
                print(f"{i:<4} {session_display:<25} {status_display:<8} {result_display}")
                displayed_count += 1
            
            # Restore original quick setting
            args.quick = original_quick
        
        print()
        
        # Print filtering info if filters were applied
        if displayed_count != len(sessions):
            filter_info = self.get_filter_description(args)
            print(f"📋 Showing {displayed_count}/{len(sessions)} sessions ({filter_info})")
            print()
        
        # Print batch summary
        self._print_batch_summary(batch_results)
    
    def _verify_session_quick(self, session_identifier: str, args) -> Dict[str, Any]:
        """Quick verification for batch processing."""
        session_info = self.get_session_info(session_identifier)
        if not session_info:
            return {'session_id': session_identifier, 'overall_status': 'NOT_FOUND'}
        
        # Run only critical checks
        results = {
            'session_id': session_identifier,
            'session_name': session_info.get('session_name'),
            'checks_run': 'quick',
            'categories': {}
        }
        
        # Quick database check
        db_results = self.verify_database_integrity(session_info)
        results['categories']['database'] = {'status': db_results['status']}
        
        # Quick MCTS check
        mcts_results = self.verify_mcts_correctness(session_info)
        results['categories']['mcts'] = {'status': mcts_results['status']}
        
        # Calculate overall status
        results['overall_status'] = self.calculate_overall_status(results)
        
        return results
    
    def _print_batch_summary(self, batch_results: List[Dict[str, Any]]) -> None:
        """Print summary for batch verification."""
        total_sessions = len(batch_results)
        status_counts = {}
        
        for result in batch_results:
            status = result.get('overall_status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("=" * 60)
        print("📊 BATCH VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total Sessions Verified: {total_sessions}")
        print()
        
        for status in ['PASS', 'WARN', 'FAIL', 'ERROR', 'NOT_FOUND']:
            count = status_counts.get(status, 0)
            if count > 0:
                percentage = (count / total_sessions) * 100
                icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'ERROR': '💥', 'NOT_FOUND': '❓'}.get(status, '?')
                print(f"{icon} {status}: {count} ({percentage:.1f}%)")
        
        print()
        
        # Recommendations based on results
        failed_count = status_counts.get('FAIL', 0) + status_counts.get('ERROR', 0)
        warn_count = status_counts.get('WARN', 0)
        
        if failed_count > 0:
            print("🔧 RECOMMENDATIONS:")
            print(f"   • {failed_count} session(s) have critical issues that need attention")
            print("   • Use --failed-only filter to focus on problematic sessions")
            print("   • Run detailed verification on failed sessions with --verify-session")
        elif warn_count > 0:
            print("💡 RECOMMENDATIONS:")
            print(f"   • {warn_count} session(s) have minor warnings")
            print("   • Use --warn-only filter to review warning details")
            print("   • Most sessions are healthy overall")
        else:
            print("🎉 All sessions verified successfully!")
            print("   • No critical issues found")
            print("   • MCTS system is operating correctly")
        
        print()
        
        # Quick actions
        print("💡 Quick Actions:")
        print("   Review failed sessions: python manager.py verification --verify-all --failed-only")
        print("   Check warnings only: python manager.py verification --verify-all --warn-only")
        if failed_count > 0:
            print("   Detailed verification: python manager.py verification --verify-session <SESSION_ID>")