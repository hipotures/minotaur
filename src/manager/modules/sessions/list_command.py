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

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


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
            elif RICH_AVAILABLE and getattr(args, 'rich', True):
                self._output_sessions_rich(sessions, summary_stats, args)
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
            
            # Build final query - use DISTINCT to avoid duplicates
            limit = getattr(args, 'limit', 10)
            query = f"""
                SELECT DISTINCT
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
                    MAX(best_score) as max_score,
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
                    'max_score': result['max_score'],
                    'completed_sessions': result['completed_sessions'],
                    'active_sessions': result['active_sessions'],
                    'interrupted_sessions': result['interrupted_sessions']
                }
            
            return {}
            
        except Exception as e:
            self.print_error(f"Failed to get summary statistics: {e}")
            return {}
    
    def _get_iteration_details_for_sessions(self, session_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Get iteration details for multiple sessions."""
        if not session_ids:
            return {}
        
        try:
            # Create placeholders for query
            placeholders = ','.join(['?' for _ in session_ids])
            
            # Get all explorations, including multiple per iteration
            query = f"""
                SELECT 
                    session_id,
                    iteration,
                    evaluation_score as score,
                    operation_applied as operation,
                    evaluation_time as eval_time,
                    timestamp
                FROM exploration_history
                WHERE session_id IN ({placeholders})
                ORDER BY session_id, timestamp, iteration
            """
            
            results = self.session_service.repository.fetch_all(query, session_ids)
            
            # Group by session_id
            iteration_details = {}
            for row in results:
                session_id = row['session_id']
                if session_id not in iteration_details:
                    iteration_details[session_id] = []
                
                iteration_details[session_id].append({
                    'iteration': row['iteration'],
                    'score': row['score'],
                    'operation': row['operation'],
                    'eval_time': row['eval_time']
                })
            
            return iteration_details
            
        except Exception as e:
            self.print_error(f"Failed to get iteration details: {e}")
            return {}
    
    def _get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a single session."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_explorations,
                    COUNT(DISTINCT iteration) as unique_iterations,
                    MIN(evaluation_score) as min_score,
                    MAX(evaluation_score) as max_score,
                    AVG(evaluation_score) as avg_score,
                    SUM(evaluation_time) as total_time,
                    AVG(evaluation_time) as avg_time
                FROM exploration_history
                WHERE session_id = ?
            """
            
            result = self.session_service.repository.fetch_one(query, [session_id])
            
            if result and result['total_explorations'] > 0:
                return {
                    'total_explorations': result['total_explorations'],
                    'unique_iterations': result['unique_iterations'],
                    'min_score': result['min_score'],
                    'max_score': result['max_score'],
                    'avg_score': result['avg_score'],
                    'total_time': result['total_time'],
                    'avg_time': result['avg_time']
                }
            
            return {}
            
        except Exception as e:
            self.print_error(f"Failed to get session statistics: {e}")
            return {}
    
    def _output_sessions_rich(self, sessions: List[Dict[str, Any]], 
                             summary_stats: Dict[str, Any], args) -> None:
        """Output sessions using rich tables."""
        console = Console()
        
        # Count unique session names instead of individual runs
        unique_sessions = set()
        total_completed = 0
        total_active = 0
        total_interrupted = 0
        
        for session in sessions:
            session_name = session['session_name'] or "Unnamed"
            unique_sessions.add(session_name)
            
            if session['status'] == 'completed':
                total_completed += 1
            elif session['status'] == 'active':
                total_active += 1
            elif session['status'] == 'interrupted':
                total_interrupted += 1
        
        summary_text = (
            f"[bold]Total sessions:[/bold] {len(unique_sessions)} | "
            f"[bold]Total runs:[/bold] {len(sessions)} | "
            f"[green]Completed: {total_completed}[/green] | "
            f"[yellow]Active: {total_active}[/yellow] | "
            f"[red]Interrupted: {total_interrupted}[/red]"
        )
        
        console.print(Panel(summary_text, title="📊 MCTS Discovery Sessions", box=box.ROUNDED))
        console.print()
        
        # Group sessions by name
        session_groups = {}
        for session in sessions:
            session_name = session['session_name'] or "Unnamed"
            if session_name not in session_groups:
                session_groups[session_name] = []
            session_groups[session_name].append(session)
        
        # Display each session group with aggregated statistics
        group_num = 1
        for session_name, group_sessions in session_groups.items():
            # Sort sessions in group by start time
            group_sessions.sort(key=lambda s: s['start_time'])
            
            # Calculate group statistics
            completed_in_group = [s for s in group_sessions if s['status'] == 'completed']
            active_in_group = [s for s in group_sessions if s['status'] == 'active']
            
            # Get all scores from completed sessions
            all_scores = []
            total_iterations = 0
            total_explorations = 0
            
            for session in completed_in_group:
                stats = self._get_session_statistics(session['session_id'])
                if stats:
                    all_scores.append(stats['min_score'])
                    all_scores.append(stats['avg_score'])
                    all_scores.append(stats['max_score'])
                    total_iterations += stats['unique_iterations']
                    total_explorations += stats['total_explorations']
            
            # Get strategy and metric from first session in group (should be same for all)
            first_session = group_sessions[0]
            strategy = first_session['strategy']
            metric = self.extract_target_metric(first_session['config_snapshot'])
            
            # Create session group header
            header_text = f"[bold]{group_num}. {session_name}[/bold]"
            
            # Calculate session duration (from earliest start to latest end/now)
            start_times = [s['start_time'] for s in group_sessions]
            earliest_start = min(start_times)
            
            # Get latest end time or use current time for active sessions
            end_times = []
            for s in group_sessions:
                if s['end_time']:
                    end_times.append(s['end_time'])
            
            if end_times:
                latest_end = max(end_times)
                total_duration = self.format_duration(str(earliest_start), str(latest_end))
            else:
                # All sessions are still active
                total_duration = self.format_duration(str(earliest_start))
            
            # Build group summary
            details = []
            details.append(f"Strategy: [blue]{strategy}[/blue] | Metric: [magenta]{metric.upper()}[/magenta]")
            details.append(f"Started: {str(earliest_start)[:16]} | Duration: [cyan]{total_duration}[/cyan]")
            details.append(f"Total runs: [cyan]{len(group_sessions)}[/cyan] | "
                          f"Completed: [green]{len(completed_in_group)}[/green] | "
                          f"Active: [yellow]{len(active_in_group)}[/yellow]")
            
            if all_scores:
                min_score = min(all_scores)
                max_score = max(all_scores)
                avg_score = sum(all_scores) / len(all_scores)
                details.append(f"Overall - Min: [dim]{min_score:.5f}[/dim] | "
                              f"Avg: [yellow]{avg_score:.5f}[/yellow] | "
                              f"[bold green]Best: {max_score:.5f}[/bold green]")
                details.append(f"Total iterations: [magenta]{total_iterations}[/magenta] | "
                              f"Total explorations: [cyan]{total_explorations}[/cyan]")
            
            # Add individual run details
            details.append("\n[dim]Individual runs:[/dim]")
            for idx, session in enumerate(group_sessions, 1):
                session_id = session['session_id']
                session_short = session_id[:8]
                status = session['status']
                
                if status == 'completed':
                    stats = self._get_session_statistics(session_id)
                    if stats:
                        run_info = (f"  [{idx}] {session_short} - [green]completed[/green] | "
                                   f"Iter: {stats['unique_iterations']} | "
                                   f"Best: [green]{stats['max_score']:.5f}[/green]")
                    else:
                        run_info = f"  [{idx}] {session_short} - [green]completed[/green]"
                else:
                    run_info = f"  [{idx}] {session_short} - [yellow]{status}[/yellow]"
                
                details.append(run_info)
            
            # Create panel for this session group
            panel_content = "\n".join(details)
            console.print(Panel(panel_content, title=header_text, box=box.ROUNDED))
            
            if group_num < len(session_groups):
                console.print()  # Add space between groups
            
            group_num += 1
        
        # Show filter info
        filters = []
        status_filter = getattr(args, 'status', 'all')
        strategy_filter = getattr(args, 'strategy', None)
        
        if status_filter != 'all':
            filters.append(f"status={status_filter}")
        if strategy_filter:
            filters.append(f"strategy={strategy_filter}")
        
        if filters:
            console.print(f"\n[dim]🔍 Filters applied: {', '.join(filters)}[/dim]")
        
        # Quick actions
        console.print("\n[bold cyan]💡 Quick Actions:[/bold cyan]")
        console.print("   [white]Show details:[/white] [dim]python manager.py sessions --show <SESSION_ID>[/dim]")
        console.print("   [white]Compare sessions:[/white] [dim]python manager.py sessions --compare <ID1> <ID2>[/dim]")
        if sessions:
            example_id = sessions[0]['session_id'][:8]
            console.print(f"   [white]Example:[/white] [dim]python manager.py sessions --show {example_id}[/dim]")
    
    def _output_sessions_table(self, sessions: List[Dict[str, Any]], 
                              summary_stats: Dict[str, Any], args) -> None:
        """Output sessions in formatted table view."""
        print("📅 MCTS DISCOVERY SESSIONS")
        print("=" * 60)
        
        # Show summary statistics
        if summary_stats:
            total = summary_stats.get('total_sessions', 0)
            avg_score = summary_stats.get('avg_score', 0)
            max_score = summary_stats.get('max_score', 0)
            completed = summary_stats.get('completed_sessions', 0)
            active = summary_stats.get('active_sessions', 0)
            interrupted = summary_stats.get('interrupted_sessions', 0)
            
            avg_score_display = f"{avg_score:.5f}" if avg_score else "N/A"
            max_score_display = f"{max_score:.5f}" if max_score else "N/A"
            print(f"📊 Total sessions: {total}, Completed: {completed}, Active: {active}, Interrupted: {interrupted}")
            print(f"📈 Average score: {avg_score_display} | Best score: {max_score_display}")
            print()
        
        # Group sessions by name for better display
        session_groups = {}
        for session in sessions:
            name_key = session['session_name'] or "Unnamed"
            if name_key not in session_groups:
                session_groups[name_key] = []
            session_groups[name_key].append(session)
        
        # Removed iteration details - not displaying them anymore
        
        # Show sessions grouped by name
        session_counter = 1
        for group_name, group_sessions in session_groups.items():
            # Sort sessions within group by start time
            group_sessions.sort(key=lambda s: s['start_time'], reverse=True)
            
            # Show group header if multiple sessions share the same name
            if len(group_sessions) > 1:
                print(f"\n📁 Session Group: {group_name} ({len(group_sessions)} sessions)")
                print("-" * 60)
            
            for session in group_sessions:
                session_short = session['session_id'][:8]
                name_display = session['session_name'] or "Unnamed"
                duration = "Running" if not session['end_time'] else self.format_duration(
                    session['start_time'], session['end_time']
                )
                
                # Extract metric
                metric = self.extract_target_metric(session['config_snapshot'])
                
                test_flag = " [TEST]" if session['is_test_mode'] else ""
                
                # Session header
                print(f"\n{session_counter:2}. {session_short}... | {name_display}{test_flag}")
                print(f"    Session ID: {session['session_id']}")
                print(f"    Status: {session['status']} | Strategy: {session['strategy']}")
                print(f"    Started: {session['start_time']} | Duration: {duration}")
                print(f"    Total iterations: {session['total_iterations']} | Best score: {self.format_score(session['best_score'], metric)} | Metric: {metric.upper()}")
                
                session_counter += 1
        
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