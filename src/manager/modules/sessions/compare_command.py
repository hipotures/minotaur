"""
Compare Command - Compare performance between multiple sessions.

Provides comprehensive session comparison including:
- Side-by-side performance metrics
- Strategy and configuration differences
- Statistical analysis and rankings
- Detailed exploration statistics comparison
"""

from typing import Dict, Any, List
from .base import BaseSessionsCommand


class CompareCommand(BaseSessionsCommand):
    """Handle --compare command for sessions."""
    
    def execute(self, args) -> None:
        """Execute the session comparison command."""
        try:
            session_ids = args.compare
            
            if len(session_ids) < 2:
                self.print_error("At least 2 sessions required for comparison")
                self.print_info("Usage: python manager.py sessions --compare <ID1> <ID2> [ID3...]")
                return
            
            # Get comparison data for all sessions
            comparison_data = self._get_comparison_data(session_ids)
            
            if len(comparison_data) < 2:
                self.print_error("Not enough valid sessions found for comparison")
                return
            
            # Output results
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(comparison_data)
            else:
                self._output_comparison_table(comparison_data)
                
        except Exception as e:
            self.print_error(f"Failed to compare sessions: {e}")
    
    def _get_comparison_data(self, session_ids: List[str]) -> List[Dict[str, Any]]:
        """Get comparison data for multiple sessions."""
        comparison_data = []
        
        for session_id in session_ids:
            try:
                # Get session basic info
                session_info = self.get_session_by_identifier(session_id)
                
                if not session_info:
                    self.print_warning(f"Session not found: {session_id[:8]}...")
                    continue
                
                # Get exploration statistics
                exploration_stats = self.get_exploration_statistics(session_info['session_id'])
                
                # Get detailed metrics
                detailed_stats = self._get_detailed_session_metrics(session_info['session_id'])
                
                # Combine all data
                session_data = {
                    **session_info,
                    'exploration_stats': exploration_stats or {},
                    'detailed_stats': detailed_stats or {}
                }
                
                comparison_data.append(session_data)
                
            except Exception as e:
                self.print_warning(f"Failed to get data for session {session_id[:8]}...: {e}")
                continue
        
        return comparison_data
    
    def _get_detailed_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get detailed metrics for a session."""
        try:
            query = """
                SELECT 
                    COUNT(*) as exploration_count,
                    AVG(evaluation_score) as avg_score,
                    SUM(evaluation_time) as total_time,
                    COUNT(DISTINCT operation_applied) as unique_operations,
                    COUNT(DISTINCT DATE(created_at)) as active_days
                FROM exploration_history 
                WHERE session_id = ?
            """
            result = self.session_service.repository.fetch_one(query, [session_id])
            
            if result:
                return {
                    'exploration_count': result['exploration_count'],
                    'avg_score': result['avg_score'],
                    'total_time': result['total_time'],
                    'unique_operations': result['unique_operations'],
                    'active_days': result['active_days']
                }
            
            return {}
            
        except Exception as e:
            self.print_error(f"Failed to get detailed metrics: {e}")
            return {}
    
    def _output_comparison_table(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Output session comparison in formatted table view."""
        print(f"📊 COMPARING {len(comparison_data)} SESSIONS")
        print("=" * 80)
        
        # Basic comparison table
        print("📋 BASIC COMPARISON:")
        headers = ['Session', 'Name', 'Strategy', 'Iterations', 'Best Score', 'Status', 'Duration']
        rows = []
        
        for data in comparison_data:
            session_short = data['session_id'][:8]
            name_short = (data['session_name'] or "Unnamed")[:12]
            strategy_short = data['strategy'][:10]
            iterations = data['total_iterations']
            
            # Format score with metric
            metric = self.extract_target_metric(data['config_snapshot'])
            best_score = data['best_score']
            score_display = f"{best_score:.5f}" if best_score else "N/A"
            if metric != "unknown" and best_score:
                metric_short = metric[:3] if len(metric) > 3 else metric
                score_display += f" ({metric_short})"
            
            status = data['status']
            duration = self.format_duration(data['start_time'], data['end_time']) if data['end_time'] else "Running"
            
            rows.append([
                session_short,
                name_short,
                strategy_short,
                str(iterations),
                score_display,
                status,
                duration
            ])
        
        self.print_table(headers, rows)
        print()
        
        # Performance comparison
        print("📈 PERFORMANCE COMPARISON:")
        headers = ['Session', 'Explorations', 'Avg Score', 'Total Time', 'Unique Ops', 'Score/Hour']
        rows = []
        
        for data in comparison_data:
            session_short = data['session_id'][:8]
            exploration_stats = data['exploration_stats']
            detailed_stats = data['detailed_stats']
            
            explorations = exploration_stats.get('total_explorations', 0)
            avg_score = exploration_stats.get('avg_score', 0)
            total_time = exploration_stats.get('total_eval_time', 0)
            unique_ops = exploration_stats.get('unique_operations', 0)
            
            # Calculate score per hour
            score_per_hour = "N/A"
            if total_time and total_time > 0 and avg_score:
                hours = total_time / 3600
                score_per_hour = f"{avg_score / hours:.3f}" if hours > 0 else "N/A"
            
            rows.append([
                session_short,
                str(explorations),
                f"{avg_score:.5f}" if avg_score else "N/A",
                f"{total_time:.1f}s" if total_time else "N/A",
                str(unique_ops),
                score_per_hour
            ])
        
        self.print_table(headers, rows)
        print()
        
        # Strategy analysis
        self._show_strategy_analysis(comparison_data)
        
        # Performance ranking
        self._show_performance_ranking(comparison_data)
        
        # Recommendations
        self._show_recommendations(comparison_data)
    
    def _show_strategy_analysis(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Show strategy analysis comparison."""
        print("🔍 STRATEGY ANALYSIS:")
        
        strategies = {}
        for data in comparison_data:
            strategy = data['strategy']
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(data)
        
        for strategy, sessions in strategies.items():
            session_count = len(sessions)
            avg_best_score = sum(s['best_score'] or 0 for s in sessions) / session_count
            avg_iterations = sum(s['total_iterations'] for s in sessions) / session_count
            
            print(f"   • {strategy}: {session_count} session(s)")
            print(f"     Average best score: {avg_best_score:.5f}")
            print(f"     Average iterations: {avg_iterations:.1f}")
        
        print()
    
    def _show_performance_ranking(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Show performance ranking."""
        print("🏆 PERFORMANCE RANKING:")
        
        # Sort by best score (descending)
        valid_sessions = [d for d in comparison_data if d['best_score']]
        sorted_sessions = sorted(valid_sessions, key=lambda x: x['best_score'], reverse=True)
        
        for i, data in enumerate(sorted_sessions, 1):
            session_short = data['session_id'][:8]
            name = data['session_name'] or "Unnamed"
            score = data['best_score']
            strategy = data['strategy']
            
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"   {medal} {session_short}... ({name})")
            print(f"      Score: {score:.5f} | Strategy: {strategy}")
        
        if not sorted_sessions:
            print("   No sessions with valid scores found")
        
        print()
    
    def _show_recommendations(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Show recommendations based on comparison."""
        print("💡 RECOMMENDATIONS:")
        
        if len(comparison_data) < 2:
            print("   Not enough data for recommendations")
            return
        
        # Find best performing session
        best_session = None
        best_score = 0
        
        for data in comparison_data:
            if data['best_score'] and data['best_score'] > best_score:
                best_score = data['best_score']
                best_session = data
        
        if best_session:
            session_short = best_session['session_id'][:8]
            strategy = best_session['strategy']
            print(f"   • Best performer: {session_short}... using {strategy} strategy")
            print(f"   • Consider using {strategy} strategy for future sessions")
            
            # Check if test mode
            if best_session['is_test_mode']:
                print(f"   • Best session was in test mode - consider full run")
        
        # Strategy recommendations
        strategies_performance = {}
        for data in comparison_data:
            strategy = data['strategy']
            score = data['best_score'] or 0
            
            if strategy not in strategies_performance:
                strategies_performance[strategy] = []
            strategies_performance[strategy].append(score)
        
        if len(strategies_performance) > 1:
            best_strategy = max(strategies_performance.keys(), 
                              key=lambda s: sum(strategies_performance[s]) / len(strategies_performance[s]))
            print(f"   • Most effective strategy: {best_strategy}")
        
        # Time efficiency
        most_efficient = None
        best_efficiency = 0
        
        for data in comparison_data:
            exploration_stats = data['exploration_stats']
            if exploration_stats:
                total_time = exploration_stats.get('total_eval_time', 0)
                avg_score = exploration_stats.get('avg_score', 0)
                
                if total_time > 0 and avg_score > 0:
                    efficiency = avg_score / (total_time / 3600)  # Score per hour
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        most_efficient = data
        
        if most_efficient:
            session_short = most_efficient['session_id'][:8]
            print(f"   • Most time-efficient: {session_short}... ({best_efficiency:.3f} score/hour)")
    
    def _output_json(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Output comparison data in JSON format."""
        self.print_json(comparison_data, "Session Comparison")