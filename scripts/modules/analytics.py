"""
Analytics Module - Statistical analysis and reporting

Provides commands for generating reports, visualizations, and statistical analysis of MCTS performance.
"""

import argparse
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from . import ModuleInterface

logger = logging.getLogger(__name__)

class AnalyticsModule(ModuleInterface):
    """Module for analytics and reporting."""
    
    @property
    def name(self) -> str:
        return "analytics"
    
    @property
    def description(self) -> str:
        return "Generate statistical reports and performance analytics"
    
    @property
    def commands(self) -> Dict[str, str]:
        return {
            "--summary": "Generate overall performance summary",
            "--trends": "Show performance trends over time",
            "--operations": "Analyze operation effectiveness",
            "--convergence": "Analyze convergence patterns",
            "--report": "Generate comprehensive HTML report",
            "--compare": "Compare performance across time periods",
            "--help": "Show detailed help for analytics module"
        }
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add analytics-specific arguments."""
        analytics_group = parser.add_argument_group('Analytics Module')
        analytics_group.add_argument('--summary', action='store_true',
                                   help='Generate performance summary')
        analytics_group.add_argument('--trends', action='store_true',
                                   help='Show performance trends')
        analytics_group.add_argument('--operations', action='store_true',
                                   help='Analyze operation effectiveness')
        analytics_group.add_argument('--convergence', type=str, metavar='SESSION_ID',
                                   help='Analyze convergence for session')
        analytics_group.add_argument('--report', action='store_true',
                                   help='Generate comprehensive HTML report')
        analytics_group.add_argument('--compare', nargs=2, metavar=('START_DATE', 'END_DATE'),
                                   help='Compare periods (YYYY-MM-DD format)')
        analytics_group.add_argument('--days', type=int, default=30,
                                   help='Number of days for analysis')
        analytics_group.add_argument('--format', type=str, choices=['text', 'json', 'html'],
                                   default='text', help='Output format')
    
    def execute(self, args: argparse.Namespace, manager) -> None:
        """Execute analytics module commands."""
        
        if args.summary:
            self._generate_summary(args, manager)
        elif args.trends:
            self._show_trends(args, manager)
        elif args.operations:
            self._analyze_operations(args, manager)
        elif args.convergence:
            self._analyze_convergence(args.convergence, args, manager)
        elif args.report:
            self._generate_html_report(args, manager)
        elif args.compare:
            # Convert days to date ranges
            from datetime import datetime, timedelta
            today = datetime.now().date()
            
            days1 = int(args.compare[0])
            days2 = int(args.compare[1])
            
            # Period 1: last N days
            period1_end = today
            period1_start = today - timedelta(days=days1)
            
            # Period 2: N days before period 1
            period2_end = period1_start - timedelta(days=1)
            period2_start = period2_end - timedelta(days=days2)
            
            self._compare_periods(
                period1_start.strftime('%Y-%m-%d'),
                period1_end.strftime('%Y-%m-%d'),
                period2_start.strftime('%Y-%m-%d'),
                period2_end.strftime('%Y-%m-%d'),
                args, 
                manager
            )
        else:
            print("❌ No analytics command specified. Use --help for options.")
    
    def _generate_summary(self, args: argparse.Namespace, manager) -> None:
        """Generate overall performance summary."""
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Get overall statistics
                overall_stats = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT s.session_id) as total_sessions,
                        COUNT(eh.id) as total_explorations,
                        AVG(eh.evaluation_score) as avg_score,
                        MAX(eh.evaluation_score) as best_score,
                        MIN(eh.evaluation_score) as worst_score,
                        SUM(eh.evaluation_time) as total_time,
                        AVG(eh.evaluation_time) as avg_eval_time,
                        COUNT(DISTINCT eh.operation_applied) as unique_operations
                    FROM sessions s
                    LEFT JOIN exploration_history eh ON s.session_id = eh.session_id
                    WHERE eh.timestamp >= (CURRENT_DATE - INTERVAL '{} days')
                """.format(args.days)).fetchone()
                
                if overall_stats:
                    sessions, explorations, avg_score, best_score, worst_score, total_time, avg_time, unique_ops = overall_stats
                    
                    print("🎯 OVERALL METRICS:")
                    print(f"   Sessions: {sessions or 0}")
                    print(f"   Total Explorations: {explorations or 0}")
                    print(f"   Unique Operations: {unique_ops or 0}")
                    
                    if avg_score:
                        print(f"   Score Range: {worst_score:.5f} - {best_score:.5f}")
                        print(f"   Average Score: {avg_score:.5f}")
                        print(f"   Total Evaluation Time: {total_time/3600:.1f} hours")
                        print(f"   Average Evaluation Time: {avg_time:.1f}s")
                    else:
                        print("   No evaluation data available")
                    print()
                
                # Session status breakdown
                status_breakdown = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM sessions
                    WHERE start_time >= (CURRENT_DATE - INTERVAL '{} days')
                    GROUP BY status
                    ORDER BY count DESC
                """.format(args.days)).fetchall()
                
                if status_breakdown:
                    print("📈 SESSION STATUS:")
                    for status, count in status_breakdown:
                        print(f"   {status.title()}: {count}")
                    print()
                
                # Top performing sessions
                top_sessions = conn.execute("""
                    SELECT 
                        s.session_id,
                        s.session_name,
                        s.best_score,
                        s.total_iterations,
                        s.strategy
                    FROM sessions s
                    WHERE s.best_score > 0
                      AND s.start_time >= (CURRENT_DATE - INTERVAL '{} days')
                    ORDER BY s.best_score DESC
                    LIMIT 5
                """.format(args.days)).fetchall()
                
                if top_sessions:
                    print("🏆 TOP SESSIONS:")
                    for session_id, name, score, iterations, strategy in top_sessions:
                        name_display = name or "Unnamed"
                        print(f"   • {session_id[:8]}... ({name_display}) - {score:.5f} [{strategy}, {iterations} iter]")
                    print()
                
                # Feature impact summary
                feature_impact = conn.execute("""
                    SELECT 
                        COUNT(*) as total_features,
                        COUNT(CASE WHEN impact_delta > 0 THEN 1 END) as positive_impact,
                        AVG(impact_delta) as avg_impact,
                        MAX(impact_delta) as best_impact
                    FROM feature_impact fi
                    JOIN sessions s ON fi.session_id = s.session_id
                    WHERE s.start_time >= (CURRENT_DATE - INTERVAL '{} days')
                """.format(args.days)).fetchone()
                
                if feature_impact and feature_impact[0] > 0:
                    total_feat, positive, avg_impact, best_impact = feature_impact
                    success_rate = (positive / total_feat * 100) if total_feat > 0 else 0
                    
                    print("🧪 FEATURE IMPACT:")
                    print(f"   Features Tested: {total_feat}")
                    print(f"   Positive Impact: {positive} ({success_rate:.1f}%)")
                    print(f"   Average Impact: {avg_impact:.5f}")
                    print(f"   Best Impact: {best_impact:.5f}")
                    print()
                
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
    
    def _show_trends(self, args: argparse.Namespace, manager) -> None:
        """Show performance trends over time."""
        print("📈 PERFORMANCE TRENDS")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Daily performance trends
                daily_trends = conn.execute("""
                    SELECT 
                        eh.timestamp::DATE as day,
                        COUNT(*) as explorations,
                        AVG(eh.evaluation_score) as avg_score,
                        MAX(eh.evaluation_score) as best_score,
                        COUNT(DISTINCT eh.session_id) as active_sessions
                    FROM exploration_history eh
                    WHERE eh.timestamp >= (CURRENT_DATE - INTERVAL '{} days')
                    GROUP BY eh.timestamp::DATE
                    ORDER BY day DESC
                """.format(args.days)).fetchall()
                
                if daily_trends:
                    print("📅 DAILY TRENDS:")
                    print(f"{'Date':<12} {'Explorations':<13} {'Avg Score':<12} {'Best Score':<12} {'Sessions':<10}")
                    print("-" * 65)
                    
                    for day, explorations, avg_score, best_score, sessions in daily_trends:
                        avg_str = f"{avg_score:.5f}" if avg_score else "N/A"
                        best_str = f"{best_score:.5f}" if best_score else "N/A"
                        print(f"{day:<12} {explorations:<13} {avg_str:<12} {best_str:<12} {sessions:<10}")
                    print()
                
                # Operation trends
                operation_trends = conn.execute("""
                    SELECT 
                        eh.operation_applied,
                        COUNT(*) as usage_count,
                        AVG(eh.evaluation_score) as avg_score,
                        MAX(eh.evaluation_score) as best_score,
                        AVG(eh.evaluation_time) as avg_time
                    FROM exploration_history eh
                    WHERE eh.timestamp >= (CURRENT_DATE - INTERVAL '{} days')
                    GROUP BY eh.operation_applied
                    HAVING usage_count >= 3
                    ORDER BY avg_score DESC
                    LIMIT 10
                """.format(args.days)).fetchall()
                
                if operation_trends:
                    print("🔧 OPERATION TRENDS:")
                    print(f"{'Operation':<25} {'Usage':<7} {'Avg Score':<12} {'Best Score':<12} {'Avg Time':<10}")
                    print("-" * 75)
                    
                    for operation, usage, avg_score, best_score, avg_time in operation_trends:
                        op_short = operation[:24]
                        avg_str = f"{avg_score:.5f}" if avg_score else "N/A"
                        best_str = f"{best_score:.5f}" if best_score else "N/A"
                        time_str = f"{avg_time:.1f}s" if avg_time else "N/A"
                        print(f"{op_short:<25} {usage:<7} {avg_str:<12} {best_str:<12} {time_str:<10}")
                    print()
                
                # Score improvement trends
                improvement_trends = conn.execute("""
                    SELECT 
                        s.session_id,
                        s.session_name,
                        MIN(eh.evaluation_score) as starting_score,
                        MAX(eh.evaluation_score) as final_score,
                        MAX(eh.evaluation_score) - MIN(eh.evaluation_score) as improvement,
                        COUNT(eh.id) as explorations
                    FROM sessions s
                    JOIN exploration_history eh ON s.session_id = eh.session_id
                    WHERE s.start_time >= (CURRENT_DATE - INTERVAL '{} days')
                      AND s.status = 'completed'
                    GROUP BY s.session_id, s.session_name
                    HAVING explorations >= 5
                    ORDER BY improvement DESC
                    LIMIT 10
                """.format(args.days)).fetchall()
                
                if improvement_trends:
                    print("📊 IMPROVEMENT TRENDS:")
                    print(f"{'Session':<10} {'Name':<15} {'Start':<10} {'Final':<10} {'Improvement':<12} {'Explorations':<12}")
                    print("-" * 80)
                    
                    for session_id, name, start_score, final_score, improvement, explorations in improvement_trends:
                        session_short = session_id[:8]
                        name_short = (name or "Unnamed")[:14]
                        start_str = f"{start_score:.5f}" if start_score else "N/A"
                        final_str = f"{final_score:.5f}" if final_score else "N/A"
                        improve_str = f"{improvement:.5f}" if improvement else "N/A"
                        
                        print(f"{session_short:<10} {name_short:<15} {start_str:<10} {final_str:<10} {improve_str:<12} {explorations:<12}")
                    print()
                
        except Exception as e:
            print(f"❌ Error showing trends: {e}")
    
    def _analyze_operations(self, args: argparse.Namespace, manager) -> None:
        """Analyze operation effectiveness."""
        print("🔧 OPERATION EFFECTIVENESS ANALYSIS")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Get operation performance statistics
                operation_stats = conn.execute("""
                    SELECT 
                        eh.operation_applied,
                        COUNT(*) as total_usage,
                        AVG(eh.evaluation_score) as avg_score,
                        STDDEV(eh.evaluation_score) as score_stddev,
                        MAX(eh.evaluation_score) as best_score,
                        MIN(eh.evaluation_score) as worst_score,
                        AVG(eh.evaluation_time) as avg_time,
                        COUNT(CASE WHEN eh.is_best_so_far = TRUE THEN 1 END) as best_count,
                        COUNT(DISTINCT eh.session_id) as session_count
                    FROM exploration_history eh
                    WHERE eh.timestamp >= (CURRENT_DATE - INTERVAL '{} days')
                    GROUP BY eh.operation_applied
                    HAVING total_usage >= 3
                    ORDER BY avg_score DESC
                """.format(args.days)).fetchall()
                
                if operation_stats:
                    print("📈 OPERATION PERFORMANCE:")
                    print(f"{'Operation':<25} {'Usage':<7} {'Avg Score':<11} {'Best':<11} {'Std Dev':<9} {'Best Count':<11} {'Sessions':<9}")
                    print("-" * 90)
                    
                    for row in operation_stats:
                        operation, usage, avg_score, stddev, best_score, worst_score, avg_time, best_count, sessions = row
                        
                        op_short = operation[:24]
                        avg_str = f"{avg_score:.5f}" if avg_score else "N/A"
                        best_str = f"{best_score:.5f}" if best_score else "N/A"
                        std_str = f"{stddev:.5f}" if stddev else "N/A"
                        
                        print(f"{op_short:<25} {usage:<7} {avg_str:<11} {best_str:<11} {std_str:<9} {best_count:<11} {sessions:<9}")
                    print()
                
                # Operation categories analysis
                category_stats = conn.execute("""
                    SELECT 
                        CASE 
                            WHEN eh.operation_applied LIKE '%npk%' THEN 'NPK Interactions'
                            WHEN eh.operation_applied LIKE '%stress%' OR eh.operation_applied LIKE '%environmental%' THEN 'Environmental'
                            WHEN eh.operation_applied LIKE '%statistical%' OR eh.operation_applied LIKE '%aggregate%' THEN 'Statistical'
                            WHEN eh.operation_applied LIKE '%transformation%' OR eh.operation_applied LIKE '%transform%' THEN 'Transformations'
                            WHEN eh.operation_applied LIKE '%selection%' OR eh.operation_applied LIKE '%filter%' THEN 'Selection'
                            ELSE 'Other'
                        END as category,
                        COUNT(*) as usage_count,
                        AVG(eh.evaluation_score) as avg_score,
                        MAX(eh.evaluation_score) as best_score,
                        AVG(eh.evaluation_time) as avg_time
                    FROM exploration_history eh
                    WHERE eh.timestamp >= (CURRENT_DATE - INTERVAL '{} days')
                    GROUP BY category
                    ORDER BY avg_score DESC
                """.format(args.days)).fetchall()
                
                if category_stats:
                    print("📁 BY CATEGORY:")
                    print(f"{'Category':<17} {'Usage':<7} {'Avg Score':<11} {'Best Score':<11} {'Avg Time':<10}")
                    print("-" * 65)
                    
                    for category, usage, avg_score, best_score, avg_time in category_stats:
                        avg_str = f"{avg_score:.5f}" if avg_score else "N/A"
                        best_str = f"{best_score:.5f}" if best_score else "N/A"
                        time_str = f"{avg_time:.1f}s" if avg_time else "N/A"
                        
                        print(f"{category:<17} {usage:<7} {avg_str:<11} {best_str:<11} {time_str:<10}")
                    print()
                
                # Success rate analysis using subquery
                success_analysis = conn.execute("""
                    WITH score_changes AS (
                        SELECT 
                            operation_applied,
                            evaluation_score,
                            LAG(evaluation_score) OVER (PARTITION BY session_id ORDER BY iteration) as prev_score
                        FROM exploration_history
                        WHERE timestamp >= (CURRENT_DATE - INTERVAL '{} days')
                    )
                    SELECT 
                        operation_applied,
                        COUNT(*) as total_attempts,
                        COUNT(CASE WHEN evaluation_score > prev_score THEN 1 END) as improvements,
                        ROUND(
                            100.0 * COUNT(CASE WHEN evaluation_score > prev_score THEN 1 END) 
                            / COUNT(*), 2
                        ) as success_rate
                    FROM score_changes
                    WHERE prev_score IS NOT NULL
                    GROUP BY operation_applied
                    HAVING COUNT(*) >= 5
                    ORDER BY success_rate DESC
                    LIMIT 10
                """.format(args.days)).fetchall()
                
                if success_analysis:
                    print("🎯 SUCCESS RATES (% of times operation improved score):")
                    print(f"{'Operation':<25} {'Attempts':<9} {'Improvements':<13} {'Success Rate':<12}")
                    print("-" * 65)
                    
                    for operation, attempts, improvements, success_rate in success_analysis:
                        op_short = operation[:24]
                        print(f"{op_short:<25} {attempts:<9} {improvements:<13} {success_rate:.1f}%")
                    print()
                
        except Exception as e:
            print(f"❌ Error analyzing operations: {e}")
    
    def _analyze_convergence(self, session_id: str, args: argparse.Namespace, manager) -> None:
        """Analyze convergence patterns for a specific session."""
        print(f"📉 CONVERGENCE ANALYSIS: {session_id[:8]}...")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Get session exploration history
                exploration_data = conn.execute("""
                    SELECT 
                        iteration,
                        evaluation_score,
                        evaluation_time,
                        operation_applied,
                        is_best_so_far,
                        timestamp
                    FROM exploration_history
                    WHERE session_id = ?
                    ORDER BY iteration
                """, [session_id]).fetchall()
                
                if not exploration_data:
                    print(f"❌ No exploration data found for session: {session_id}")
                    return
                
                # Calculate convergence metrics
                scores = [row[1] for row in exploration_data]
                iterations = [row[0] for row in exploration_data]
                
                initial_score = scores[0] if scores else 0
                final_score = scores[-1] if scores else 0
                best_score = max(scores) if scores else 0
                
                # Find when best score was achieved
                best_iteration = None
                for i, score in enumerate(scores):
                    if score == best_score:
                        best_iteration = iterations[i]
                        break
                
                print("📊 CONVERGENCE METRICS:")
                print(f"   Total Iterations: {len(exploration_data)}")
                print(f"   Initial Score: {initial_score:.5f}")
                print(f"   Final Score: {final_score:.5f}")
                print(f"   Best Score: {best_score:.5f}")
                print(f"   Total Improvement: {final_score - initial_score:.5f}")
                print(f"   Best Achieved at Iteration: {best_iteration}")
                
                if best_iteration:
                    convergence_point = (best_iteration / len(exploration_data)) * 100
                    print(f"   Convergence Point: {convergence_point:.1f}% through session")
                print()
                
                # Show score progression in chunks
                chunk_size = max(1, len(exploration_data) // 10)
                print("📈 SCORE PROGRESSION:")
                print(f"{'Iteration Range':<15} {'Avg Score':<12} {'Best in Range':<15} {'Improvements':<12}")
                print("-" * 60)
                
                for i in range(0, len(exploration_data), chunk_size):
                    chunk = exploration_data[i:i+chunk_size]
                    chunk_scores = [row[1] for row in chunk]
                    
                    start_iter = chunk[0][0]
                    end_iter = chunk[-1][0]
                    avg_score = sum(chunk_scores) / len(chunk_scores)
                    best_in_chunk = max(chunk_scores)
                    
                    # Count improvements in this chunk
                    improvements = 0
                    for j in range(1, len(chunk)):
                        if chunk_scores[j] > chunk_scores[j-1]:
                            improvements += 1
                    
                    range_str = f"{start_iter}-{end_iter}"
                    print(f"{range_str:<15} {avg_score:<12.5f} {best_in_chunk:<15.5f} {improvements:<12}")
                
                print()
                
                # Stagnation analysis
                stagnation_threshold = 0.0001  # Minimal improvement threshold
                stagnation_periods = []
                current_stagnation = 0
                
                for i in range(1, len(scores)):
                    improvement = scores[i] - scores[i-1]
                    if abs(improvement) < stagnation_threshold:
                        current_stagnation += 1
                    else:
                        if current_stagnation >= 3:  # 3+ iterations without improvement
                            stagnation_periods.append(current_stagnation)
                        current_stagnation = 0
                
                if stagnation_periods:
                    print("⏸️  STAGNATION ANALYSIS:")
                    print(f"   Stagnation Periods: {len(stagnation_periods)}")
                    print(f"   Longest Stagnation: {max(stagnation_periods)} iterations")
                    print(f"   Average Stagnation: {sum(stagnation_periods)/len(stagnation_periods):.1f} iterations")
                    print()
                
                # Recent performance (last 20% of iterations)
                recent_threshold = int(len(exploration_data) * 0.8)
                if recent_threshold < len(exploration_data):
                    recent_data = exploration_data[recent_threshold:]
                    recent_scores = [row[1] for row in recent_data]
                    recent_avg = sum(recent_scores) / len(recent_scores)
                    
                    print("🔄 RECENT PERFORMANCE (last 20% of iterations):")
                    print(f"   Recent Average Score: {recent_avg:.5f}")
                    print(f"   vs Overall Average: {sum(scores)/len(scores):.5f}")
                    
                    trend = "improving" if recent_avg > sum(scores)/len(scores) else "declining"
                    print(f"   Trend: {trend}")
                    print()
                
        except Exception as e:
            print(f"❌ Error analyzing convergence: {e}")
    
    def _generate_html_report(self, args: argparse.Namespace, manager) -> None:
        """Generate comprehensive HTML report."""
        print("📄 GENERATING HTML REPORT")
        print("=" * 50)
        
        try:
            from datetime import datetime
            from pathlib import Path
            import yaml
            
            # Try to find the most recent session and use its session directory if available
            output_dir = manager.project_root / 'reports'  # Default location
            session_output_dir = None
            
            try:
                # Get the most recent session info
                with manager._connect() as conn:
                    recent_session = conn.execute("""
                        SELECT session_name, session_id, config_snapshot FROM sessions 
                        WHERE config_snapshot IS NOT NULL 
                        ORDER BY start_time DESC 
                        LIMIT 1
                    """).fetchone()
                    
                    if recent_session:
                        session_name, session_id, config_snapshot = recent_session
                        
                        # Check if session has dedicated output directory
                        potential_session_dir = manager.project_root / 'outputs' / 'sessions' / session_name
                        if potential_session_dir.exists():
                            session_output_dir = potential_session_dir / 'reports'
                            session_output_dir.mkdir(parents=True, exist_ok=True)
                            output_dir = session_output_dir
                            logger.info(f"Using session-based output directory: {output_dir}")
                        else:
                            # Fallback to config-based directory
                            import json
                            config = json.loads(config_snapshot)
                            export_config = config.get('export', {})
                            output_dir_str = export_config.get('output_dir', 'reports')
                            output_dir = manager.project_root / output_dir_str
                            logger.info(f"Using config-based output directory: {output_dir}")
                    else:
                        # Fallback: load from base config file
                        config_path = manager.project_root / 'config' / 'mcts_config.yaml'
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                config = yaml.safe_load(f)
                                export_config = config.get('export', {})
                                output_dir_str = export_config.get('output_dir', 'reports')
                                output_dir = manager.project_root / output_dir_str
                                logger.info(f"Using base config output directory: {output_dir}")
                
            except Exception as e:
                logger.warning(f"Could not load session directory, using default: {e}")
                # Use default reports directory
                output_dir = manager.project_root / 'reports'
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"mcts_report_{timestamp}.html"
            
            # Generate HTML content (simplified version)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MCTS Feature Discovery Report - {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>MCTS Feature Discovery Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Analysis Period: Last {args.days} days</p>
                </div>
            """
            
            # Add summary data
            with manager._connect() as conn:
                # Get summary statistics
                summary_stats = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT s.session_id) as sessions,
                        COUNT(eh.id) as explorations,
                        AVG(eh.evaluation_score) as avg_score,
                        MAX(eh.evaluation_score) as best_score
                    FROM sessions s
                    LEFT JOIN exploration_history eh ON s.session_id = eh.session_id
                    WHERE s.start_time >= (CURRENT_DATE - INTERVAL '{} days')
                """.format(args.days)).fetchone()
                
                if summary_stats:
                    sessions, explorations, avg_score, best_score = summary_stats
                    
                    html_content += f"""
                    <div class="section">
                        <h2>Summary Statistics</h2>
                        <div class="metric">Sessions: {sessions or 0}</div>
                        <div class="metric">Explorations: {explorations or 0}</div>
                        <div class="metric">Average Score: {f'{avg_score:.5f}' if avg_score else 'N/A'}</div>
                        <div class="metric">Best Score: {f'{best_score:.5f}' if best_score else 'N/A'}</div>
                    </div>
                    """
                
                # Add top sessions table
                top_sessions = conn.execute("""
                    SELECT session_id, session_name, best_score, total_iterations, strategy
                    FROM sessions
                    WHERE best_score > 0 AND start_time >= (CURRENT_DATE - INTERVAL '{} days')
                    ORDER BY best_score DESC
                    LIMIT 10
                """.format(args.days)).fetchall()
                
                if top_sessions:
                    html_content += """
                    <div class="section">
                        <h2>Top Performing Sessions</h2>
                        <table>
                            <tr><th>Session ID</th><th>Name</th><th>Best Score</th><th>Iterations</th><th>Strategy</th></tr>
                    """
                    
                    for session_id, name, score, iterations, strategy in top_sessions:
                        html_content += f"""
                        <tr>
                            <td>{session_id[:8]}...</td>
                            <td>{name or 'Unnamed'}</td>
                            <td>{score:.5f}</td>
                            <td>{iterations}</td>
                            <td>{strategy}</td>
                        </tr>
                        """
                    
                    html_content += "</table></div>"
            
            html_content += """
                <div class="section">
                    <p><small>Generated by MCTS Feature Discovery Analytics Module</small></p>
                </div>
            </body>
            </html>
            """
            
            # Write HTML file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            print(f"✅ HTML report generated: {output_file}")
            print(f"📂 File size: {output_file.stat().st_size / 1024:.1f} KB")
            
        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
    
    def _compare_periods(self, period1_start: str, period1_end: str, period2_start: str, period2_end: str, args: argparse.Namespace, manager) -> None:
        """Compare performance between two time periods."""
        print(f"⚖️  COMPARING PERIODS:")
        print(f"   Period 1: {period1_start} to {period1_end}")
        print(f"   Period 2: {period2_start} to {period2_end}")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Get stats for period 1 (recent)
                period1_stats = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT s.session_id) as sessions,
                        COUNT(eh.id) as explorations,
                        AVG(eh.evaluation_score) as avg_score,
                        MAX(eh.evaluation_score) as best_score,
                        SUM(eh.evaluation_time) as total_time
                    FROM sessions s
                    LEFT JOIN exploration_history eh ON s.session_id = eh.session_id
                    WHERE DATE(s.start_time) BETWEEN ? AND ?
                """, [period1_start, period1_end]).fetchone()
                
                # Get stats for period 2 (older)
                period2_stats = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT s.session_id) as sessions,
                        COUNT(eh.id) as explorations,
                        AVG(eh.evaluation_score) as avg_score,
                        MAX(eh.evaluation_score) as best_score,
                        SUM(eh.evaluation_time) as total_time
                    FROM sessions s
                    LEFT JOIN exploration_history eh ON s.session_id = eh.session_id
                    WHERE DATE(s.start_time) BETWEEN ? AND ?
                """, [period2_start, period2_end]).fetchone()
                
                if period1_stats and period2_stats:
                    curr_sessions, curr_exp, curr_avg, curr_best, curr_time = period1_stats
                    prev_sessions, prev_exp, prev_avg, prev_best, prev_time = period2_stats
                    
                    print(f"📊 COMPARISON RESULTS:")
                    print(f"{'Metric':<20} {'Current Period':<15} {'Previous Period':<16} {'Change':<15}")
                    print("-" * 70)
                    
                    # Sessions
                    session_change = ((curr_sessions or 0) - (prev_sessions or 0))
                    session_pct = (session_change / (prev_sessions or 1)) * 100 if prev_sessions else 0
                    print(f"{'Sessions':<20} {curr_sessions or 0:<15} {prev_sessions or 0:<16} {session_change:+} ({session_pct:+.1f}%)")
                    
                    # Explorations
                    exp_change = ((curr_exp or 0) - (prev_exp or 0))
                    exp_pct = (exp_change / (prev_exp or 1)) * 100 if prev_exp else 0
                    print(f"{'Explorations':<20} {curr_exp or 0:<15} {prev_exp or 0:<16} {exp_change:+} ({exp_pct:+.1f}%)")
                    
                    # Average score
                    if curr_avg and prev_avg:
                        avg_change = curr_avg - prev_avg
                        avg_pct = (avg_change / prev_avg) * 100
                        print(f"{'Avg Score':<20} {curr_avg:.5f:<15} {prev_avg:.5f:<16} {avg_change:+.5f} ({avg_pct:+.1f}%)")
                    
                    # Best score
                    if curr_best and prev_best:
                        best_change = curr_best - prev_best
                        best_pct = (best_change / prev_best) * 100
                        print(f"{'Best Score':<20} {curr_best:.5f:<15} {prev_best:.5f:<16} {best_change:+.5f} ({best_pct:+.1f}%)")
                    
                    # Total time
                    if curr_time and prev_time:
                        time_change = curr_time - prev_time
                        time_pct = (time_change / prev_time) * 100
                        print(f"{'Total Time (h)':<20} {curr_time/3600:.1f:<15} {prev_time/3600:.1f:<16} {time_change/3600:+.1f} ({time_pct:+.1f}%)")
                    
                    print()
                    
                    # Overall assessment
                    improvements = 0
                    if curr_sessions and prev_sessions and curr_sessions > prev_sessions:
                        improvements += 1
                    if curr_avg and prev_avg and curr_avg > prev_avg:
                        improvements += 1
                    if curr_best and prev_best and curr_best > prev_best:
                        improvements += 1
                    
                    if improvements >= 2:
                        assessment = "📈 Significant improvement"
                    elif improvements >= 1:
                        assessment = "📊 Mixed results"
                    else:
                        assessment = "📉 Performance decline"
                    
                    print(f"🎯 ASSESSMENT: {assessment}")
                    
                else:
                    print("❌ Insufficient data for comparison")
                
        except Exception as e:
            print(f"❌ Error comparing periods: {e}")