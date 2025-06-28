#!/usr/bin/env python3
"""
Sessions Module - MCTS session management

Provides commands for listing, analyzing, and managing MCTS discovery sessions.
"""

import argparse
from typing import Dict, List, Any
from datetime import datetime
from . import ModuleInterface

class SessionsModule(ModuleInterface):
    """Module for managing MCTS sessions."""
    
    @property
    def name(self) -> str:
        return "sessions"
    
    @property
    def description(self) -> str:
        return "Manage and analyze MCTS discovery sessions"
    
    @property
    def commands(self) -> Dict[str, str]:
        return {
            "--list": "List recent sessions with summary statistics",
            "--show": "Show detailed information about a specific session",
            "--compare": "Compare performance between multiple sessions",
            "--export": "Export session data to CSV/JSON",
            "--cleanup": "Remove old or incomplete sessions",
            "--help": "Show detailed help for sessions module"
        }
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add session-specific arguments."""
        session_group = parser.add_argument_group('Sessions Module')
        session_group.add_argument('--list', action='store_true',
                                 help='List recent sessions')
        session_group.add_argument('--show', type=str, metavar='SESSION_ID',
                                 help='Show detailed session information')
        session_group.add_argument('--compare', nargs='+', metavar='SESSION_ID',
                                 help='Compare multiple sessions')
        session_group.add_argument('--export', type=str, metavar='FORMAT',
                                 choices=['csv', 'json'], 
                                 help='Export session data')
        session_group.add_argument('--cleanup', action='store_true',
                                 help='Remove incomplete sessions')
        session_group.add_argument('--limit', type=int, default=10,
                                 help='Limit number of sessions to show')
        session_group.add_argument('--status', type=str,
                                 choices=['active', 'completed', 'interrupted', 'all'],
                                 default='all', help='Filter by session status')
    
    def execute(self, args: argparse.Namespace, manager) -> None:
        """Execute sessions module commands."""
        
        if getattr(args, 'list', False):
            self._list_sessions(args, manager)
        elif getattr(args, 'show', None):
            self._show_session(args.show, args, manager)
        elif getattr(args, 'compare', None):
            self._compare_sessions(args.compare, args, manager)
        elif getattr(args, 'export', None):
            self._export_sessions(args.export, args, manager)
        elif getattr(args, 'cleanup', False):
            self._cleanup_sessions(args, manager)
        else:
            print("❌ No sessions command specified. Use --help for options.")
    
    def _list_sessions(self, args: argparse.Namespace, manager) -> None:
        """List recent sessions with summary."""
        print("📅 MCTS DISCOVERY SESSIONS")
        print("=" * 50)
        
        if not manager.duckdb_path.exists():
            print(f"❌ Database not found: {manager.duckdb_path}")
            return
        
        try:
            with manager._connect() as conn:
                # Build WHERE clause for status filter
                status_filter = ""
                limit = getattr(args, 'limit', 10)
                status = getattr(args, 'status', 'all')
                params = [limit]
                if status != 'all':
                    status_filter = "WHERE status = ?"
                    params = [status, limit]
                
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
                    {status_filter}
                    ORDER BY start_time DESC 
                    LIMIT ?
                """
                
                if status != 'all':
                    result = conn.execute(query, [status, limit]).fetchall()
                else:
                    result = conn.execute(query, [limit]).fetchall()
                
                if not result:
                    print("No sessions found.")
                    return
                
                # Show summary statistics
                all_sessions = conn.execute("SELECT COUNT(*), AVG(best_score) FROM sessions WHERE best_score > 0").fetchone()
                total_sessions, avg_score = all_sessions
                avg_score_display = f"{avg_score:.5f}" if avg_score else "N/A"
                print(f"📊 Total sessions: {total_sessions}, Average score: {avg_score_display}")
                print()
                
                for i, row in enumerate(result, 1):
                    session_id, name, start_time, end_time, iterations, score, status, strategy, is_test, config_json = row
                    
                    # Format display
                    session_short = session_id[:8]
                    name_display = name or "Unnamed"
                    duration = "Running" if not end_time else self._format_duration(start_time, end_time)
                    score_display = f"{score:.5f}" if score and score > 0 else "No score"
                    test_flag = " [TEST]" if is_test else ""
                    
                    print(f"{i:2}. {session_short}... | {name_display}{test_flag}")
                    print(f"    Status: {status} | Strategy: {strategy}")
                    print(f"    Started: {start_time} | Duration: {duration}")
                    # Extract target metric from config
                    metric = "unknown"
                    if config_json:
                        try:
                            import json
                            config = json.loads(config_json)
                            metric = config.get('autogluon', {}).get('target_metric', 'unknown')
                        except:
                            pass
                    
                    print(f"    Iterations: {iterations} | Best Score: {score_display} ({metric})")
                    print()
                
        except Exception as e:
            print(f"❌ Error listing sessions: {e}")
    
    def _show_session(self, session_identifier: str, args: argparse.Namespace, manager) -> None:
        """Show detailed information about a specific session."""
        print("🔍 SESSION DETAILS")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Try to find session by ID (UUID) or by name
                session_info = conn.execute("""
                    SELECT * FROM sessions 
                    WHERE session_id = ? OR session_name = ?
                """, [session_identifier, session_identifier]).fetchone()
                
                if not session_info:
                    print(f"❌ Session not found: {session_identifier}")
                    print("💡 Use either full UUID (e.g., 8208bed6-...) or session name (e.g., session_20250628_001011)")
                    return
                
                # Unpack session info (includes dataset_hash column)
                sid, name, start_time, end_time, iterations, best_score, config, status, strategy, is_test, notes, dataset_hash = session_info
                
                print(f"📋 BASIC INFORMATION:")
                print(f"   Session ID: {sid}")
                print(f"   Name: {name or 'Unnamed'}")
                print(f"   Status: {status}")
                print(f"   Strategy: {strategy}")
                print(f"   Test Mode: {'Yes' if is_test else 'No'}")
                print(f"   Started: {start_time}")
                print(f"   Ended: {end_time or 'Still running'}")
                print(f"   Duration: {self._format_duration(start_time, end_time) if end_time else 'Running'}")
                print(f"   Total Iterations: {iterations}")
                best_score_display = f"{best_score:.5f}" if best_score else "No score"
                
                # Extract target metric from config
                metric = "unknown"
                if config:
                    try:
                        import json
                        config_data = json.loads(config) if isinstance(config, str) else config
                        metric = config_data.get('autogluon', {}).get('target_metric', 'unknown')
                    except:
                        pass
                
                print(f"   Best Score: {best_score_display} ({metric})")
                print(f"   Dataset Hash: {dataset_hash[:8]}..." if dataset_hash else "   Dataset Hash: Not set")
                if notes:
                    print(f"   Notes: {notes}")
                print()
                
                # Get exploration statistics
                exploration_stats = conn.execute("""
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
                """, [sid]).fetchone()
                
                if exploration_stats and exploration_stats[0] > 0:
                    total_exp, min_score, max_score, avg_score, total_time, avg_time, unique_ops = exploration_stats
                    
                    print("📊 EXPLORATION STATISTICS:")
                    print(f"   Total Explorations: {total_exp}")
                    print(f"   Score Range: {min_score:.5f} - {max_score:.5f}")
                    print(f"   Average Score: {avg_score:.5f}")
                    print(f"   Total Evaluation Time: {total_time:.1f}s ({total_time/60:.1f}m)")
                    print(f"   Average Evaluation Time: {avg_time:.1f}s")
                    print(f"   Unique Operations: {unique_ops}")
                    print()
                    
                    # Show top operations
                    top_operations = conn.execute("""
                        SELECT 
                            operation_applied,
                            COUNT(*) as usage_count,
                            AVG(evaluation_score) as avg_score,
                            MAX(evaluation_score) as best_score
                        FROM exploration_history 
                        WHERE session_id = ?
                        GROUP BY operation_applied
                        ORDER BY avg_score DESC
                        LIMIT 5
                    """, [sid]).fetchall()
                    
                    if top_operations:
                        print("🏆 TOP OPERATIONS:")
                        for op, count, avg_score, best_score in top_operations:
                            print(f"   • {op}: {count} uses, avg: {avg_score:.5f}, best: {best_score:.5f}")
                        print()
                
                # Show configuration if available
                if config:
                    try:
                        import json
                        config_data = json.loads(config)
                        print("⚙️  CONFIGURATION:")
                        print(f"   Train Path: {config_data.get('autogluon', {}).get('train_path', 'N/A')}")
                        print(f"   Target Metric: {config_data.get('autogluon', {}).get('target_metric', 'N/A')}")
                        print(f"   Model Types: {config_data.get('autogluon', {}).get('included_model_types', 'N/A')}")
                        print(f"   Max Iterations: {config_data.get('session', {}).get('max_iterations', 'N/A')}")
                        print()
                    except json.JSONDecodeError:
                        print("⚙️  Configuration data available but not readable as JSON")
                        print()
                
        except Exception as e:
            print(f"❌ Error showing session details: {e}")
    
    def _compare_sessions(self, session_ids: List[str], args: argparse.Namespace, manager) -> None:
        """Compare performance between multiple sessions."""
        print(f"📊 COMPARING {len(session_ids)} SESSIONS")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                comparison_data = []
                
                for session_id in session_ids:
                    # Get session summary
                    session_data = conn.execute("""
                        SELECT 
                            s.session_id,
                            s.session_name,
                            s.strategy,
                            s.total_iterations,
                            s.best_score,
                            s.start_time,
                            s.end_time,
                            COUNT(eh.id) as exploration_count,
                            AVG(eh.evaluation_score) as avg_score,
                            SUM(eh.evaluation_time) as total_time,
                            s.config_snapshot
                        FROM sessions s
                        LEFT JOIN exploration_history eh ON s.session_id = eh.session_id
                        WHERE s.session_id = ?
                        GROUP BY s.session_id, s.session_name, s.strategy, s.total_iterations, 
                                 s.best_score, s.start_time, s.end_time, s.config_snapshot
                    """, [session_id]).fetchone()
                    
                    if session_data:
                        comparison_data.append(session_data)
                    else:
                        print(f"⚠️  Session not found: {session_id[:8]}...")
                
                if not comparison_data:
                    print("❌ No valid sessions found for comparison")
                    return
                
                # Display comparison table
                print(f"{'Session':<10} {'Name':<15} {'Strategy':<12} {'Iterations':<10} {'Best Score':<14} {'Avg Score':<12} {'Total Time':<12}")
                print("-" * 90)
                
                for data in comparison_data:
                    sid, name, strategy, iterations, best_score, start_time, end_time, exp_count, avg_score, total_time, config_json = data
                    
                    # Extract target metric
                    metric = ""
                    if config_json:
                        try:
                            import json
                            config = json.loads(config_json)
                            metric = config.get('autogluon', {}).get('target_metric', 'unknown')
                            metric = f"({metric[:3]})" if len(metric) > 3 else f"({metric})"
                        except:
                            metric = ""
                    
                    session_short = sid[:8]
                    name_short = (name or "Unnamed")[:14]
                    strategy_short = strategy[:11]
                    best_display = f"{best_score:.5f}{metric}" if best_score else "N/A"
                    avg_display = f"{avg_score:.5f}" if avg_score else "N/A"
                    time_display = f"{total_time:.1f}s" if total_time else "N/A"
                    
                    print(f"{session_short:<10} {name_short:<15} {strategy_short:<12} {iterations:<10} {best_display:<14} {avg_display:<12} {time_display:<12}")
                
                print()
                
                # Performance summary
                valid_scores = [data[4] for data in comparison_data if data[4]]
                if valid_scores:
                    best_overall = max(valid_scores)
                    worst_overall = min(valid_scores)
                    best_session = next(data for data in comparison_data if data[4] == best_overall)
                    
                    print("🏆 COMPARISON SUMMARY:")
                    print(f"   Best Performance: {best_session[0][:8]}... ({best_overall:.5f})")
                    print(f"   Score Range: {worst_overall:.5f} - {best_overall:.5f}")
                    print(f"   Average Score: {sum(valid_scores)/len(valid_scores):.5f}")
                
        except Exception as e:
            print(f"❌ Error comparing sessions: {e}")
    
    def _export_sessions(self, format_type: str, args: argparse.Namespace, manager) -> None:
        """Export session data to file."""
        print(f"📦 EXPORTING SESSIONS TO {format_type.upper()}")
        print("=" * 50)
        
        try:
            from pathlib import Path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format_type == 'csv':
                # Use dedicated DuckDB export directory
                export_config = manager.get_export_config()
                exports_dir = manager.project_root / export_config['export_dir']
                exports_dir.mkdir(parents=True, exist_ok=True)
                output_file = exports_dir / f"sessions_export_{timestamp}.csv"
                
                with manager._connect() as conn:
                    # Use string formatting for COPY TO since parameter binding doesn't work
                    query = f"""
                        COPY (
                            SELECT 
                                session_id,
                                session_name,
                                start_time,
                                end_time,
                                total_iterations,
                                best_score,
                                status,
                                strategy,
                                is_test_mode
                            FROM sessions 
                            ORDER BY start_time DESC
                        ) TO '{str(output_file)}' (HEADER, DELIMITER ',')
                    """
                    conn.execute(query)
                    
                print(f"✅ Exported to: {output_file}")
                
            elif format_type == 'json':
                import json
                # Use dedicated DuckDB export directory
                export_config = manager.get_export_config()
                exports_dir = manager.project_root / export_config['export_dir']
                exports_dir.mkdir(parents=True, exist_ok=True)
                output_file = exports_dir / f"sessions_export_{timestamp}.json"
                
                with manager._connect() as conn:
                    sessions = conn.execute("""
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
                            config_snapshot,
                            notes
                        FROM sessions 
                        ORDER BY start_time DESC
                    """).fetchall()
                    
                    sessions_data = []
                    for session in sessions:
                        session_dict = {
                            'session_id': session[0],
                            'session_name': session[1],
                            'start_time': session[2],
                            'end_time': session[3],
                            'total_iterations': session[4],
                            'best_score': session[5],
                            'status': session[6],
                            'strategy': session[7],
                            'is_test_mode': session[8],
                            'config_snapshot': json.loads(session[9]) if session[9] else None,
                            'notes': session[10]
                        }
                        sessions_data.append(session_dict)
                    
                    with open(output_file, 'w') as f:
                        json.dump(sessions_data, f, indent=2, default=str)
                    
                print(f"✅ Exported {len(sessions_data)} sessions to: {output_file}")
                
        except Exception as e:
            print(f"❌ Error exporting sessions: {e}")
    
    def _cleanup_sessions(self, args: argparse.Namespace, manager) -> None:
        """Remove incomplete or test sessions."""
        print("🧹 CLEANING UP SESSIONS")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Find sessions to cleanup
                cleanup_candidates = conn.execute("""
                    SELECT session_id, session_name, status, total_iterations, is_test_mode
                    FROM sessions
                    WHERE status = 'interrupted' 
                       OR (total_iterations = 0 AND status = 'active')
                       OR is_test_mode = TRUE
                    ORDER BY start_time DESC
                """).fetchall()
                
                if not cleanup_candidates:
                    print("✅ No sessions need cleanup")
                    return
                
                print(f"Found {len(cleanup_candidates)} sessions for cleanup:")
                for session_id, name, status, iterations, is_test in cleanup_candidates:
                    reason = []
                    if status == 'interrupted':
                        reason.append("interrupted")
                    if iterations == 0 and status == 'active':
                        reason.append("no progress")
                    if is_test:
                        reason.append("test mode")
                    
                    print(f"   • {session_id[:8]}... ({name or 'Unnamed'}) - {', '.join(reason)}")
                
                print()
                
                # Check for dry-run mode (used in testing)
                if getattr(args, 'dry_run', False):
                    print("🧪 DRY-RUN MODE: Cleanup simulation completed successfully")
                    print(f"Would have deleted {len(cleanup_candidates)} sessions")
                    return
                
                response = input("Delete these sessions? (yes/no): ")
                
                if response.lower() == 'yes':
                    deleted_count = 0
                    for session_id, _, _, _, _ in cleanup_candidates:
                        # Delete from all related tables
                        conn.execute("DELETE FROM exploration_history WHERE session_id = ?", [session_id])
                        conn.execute("DELETE FROM feature_impact WHERE session_id = ?", [session_id])
                        conn.execute("DELETE FROM operation_performance WHERE session_id = ?", [session_id])
                        conn.execute("DELETE FROM system_performance WHERE session_id = ?", [session_id])
                        conn.execute("DELETE FROM sessions WHERE session_id = ?", [session_id])
                        deleted_count += 1
                    
                    print(f"✅ Deleted {deleted_count} sessions and related data")
                else:
                    print("❌ Cleanup cancelled")
                
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    def _format_duration(self, start_time: str, end_time: str = None) -> str:
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