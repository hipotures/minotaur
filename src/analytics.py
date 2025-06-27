"""
Analytics and Visualization Module for MCTS Feature Discovery

Generates comprehensive reports, charts, and dashboards for performance analysis,
timing visualization, and MCTS exploration insights.
"""

import json
import logging
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Optional visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Charts will be disabled.")

class AnalyticsGenerator:
    """Generate analytics reports and visualizations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize analytics generator."""
        self.config = config
        self.output_dir = Path(config.get('export', {}).get('output_dir', 'reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot configuration
        self.plot_config = config.get('analytics', {})
        self.figure_size = self.plot_config.get('figure_size', (12, 8))
        self.dpi = self.plot_config.get('dpi', 100)
        self.save_format = self.plot_config.get('format', 'png')
        
        logger.info(f"Initialized AnalyticsGenerator, output: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    db_path: str, 
                                    timing_data: str = None,
                                    session_id: str = None) -> Dict[str, str]:
        """
        Generate comprehensive analytics report with all visualizations.
        
        Args:
            db_path: Path to SQLite database
            timing_data: Path to timing JSON file
            session_id: Specific session ID to analyze
            
        Returns:
            Dict[str, str]: Generated report file paths
        """
        logger.info("Generating comprehensive analytics report...")
        
        # Load data
        session_data = self._load_session_data(db_path, session_id)
        timing_stats = self._load_timing_data(timing_data) if timing_data else None
        
        report_files = {}
        
        # Generate HTML summary report
        html_report = self._generate_html_report(session_data, timing_stats)
        html_path = self.output_dir / "mcts_analytics_report.html"
        with open(html_path, 'w') as f:
            f.write(html_report)
        report_files['html_report'] = str(html_path)
        
        # Generate individual charts (if plotting available)
        if PLOTTING_AVAILABLE:
            chart_paths = self._generate_all_charts(session_data, timing_stats)
            report_files.update(chart_paths)
        
        # Generate data exports
        data_exports = self._export_analysis_data(session_data, timing_stats)
        report_files.update(data_exports)
        
        logger.info(f"Generated comprehensive report with {len(report_files)} files")
        return report_files
    
    def _load_session_data(self, db_path: str, session_id: str = None) -> Dict[str, pd.DataFrame]:
        """Load session data from SQLite database."""
        logger.debug(f"Loading session data from: {db_path}")
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Get session info
            sessions_query = "SELECT * FROM sessions"
            if session_id:
                sessions_query += f" WHERE session_id = '{session_id}'"
            sessions_query += " ORDER BY start_time DESC"
            
            sessions_df = pd.read_sql_query(sessions_query, conn)
            
            # Get exploration history
            history_query = "SELECT * FROM exploration_history"
            if session_id:
                history_query += f" WHERE session_id = '{session_id}'"
            history_query += " ORDER BY timestamp"
            
            history_df = pd.read_sql_query(history_query, conn)
            
            # Get operation performance (if table exists)
            operation_df = pd.DataFrame()
            try:
                op_query = "SELECT * FROM operation_performance"
                operation_df = pd.read_sql_query(op_query, conn)
            except:
                logger.debug("operation_performance table not found")
            
            conn.close()
            
            return {
                'sessions': sessions_df,
                'exploration_history': history_df,
                'operation_performance': operation_df
            }
            
        except Exception as e:
            logger.error(f"Failed to load session data: {e}")
            return {'sessions': pd.DataFrame(), 'exploration_history': pd.DataFrame(), 'operation_performance': pd.DataFrame()}
    
    def _load_timing_data(self, timing_path: str) -> Dict[str, Any]:
        """Load timing data from JSON file."""
        try:
            with open(timing_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load timing data: {e}")
            return {}
    
    def _generate_html_report(self, session_data: Dict[str, pd.DataFrame], timing_stats: Dict[str, Any] = None) -> str:
        """Generate comprehensive HTML report."""
        logger.debug("Generating HTML report...")
        
        sessions_df = session_data['sessions']
        history_df = session_data['exploration_history']
        operation_df = session_data['operation_performance']
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(session_data, timing_stats)
        
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCTS Feature Discovery Analytics Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin: 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .data-table th, .data-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .data-table th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        .data-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }}
        .success {{
            background-color: #d4edda;
            border-left-color: #28a745;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }}
        .error {{
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }}
        .timestamp {{
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 MCTS Feature Discovery Analytics Report</h1>
        <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="highlight success">
            <h3>📊 Executive Summary</h3>
            <p><strong>Overall Performance:</strong> {summary_stats.get('overall_assessment', 'Analysis complete')}</p>
            <p><strong>Key Insight:</strong> {summary_stats.get('key_insight', 'MCTS exploration successfully discovered feature improvements')}</p>
        </div>
        
        <h2>📈 Key Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{summary_stats.get('total_sessions', 0)}</div>
                <div class="metric-label">Total Sessions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary_stats.get('total_iterations', 0)}</div>
                <div class="metric-label">Total Iterations</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary_stats.get('best_score', 0):.4f}</div>
                <div class="metric-label">Best Score Achieved</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary_stats.get('total_runtime_minutes', 0):.1f}</div>
                <div class="metric-label">Total Runtime (min)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary_stats.get('avg_score_improvement', 0):.3f}</div>
                <div class="metric-label">Avg Score Improvement</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary_stats.get('operations_per_minute', 0):.1f}</div>
                <div class="metric-label">Operations/Minute</div>
            </div>
        </div>
        
        <h2>🎯 Session Performance</h2>
        {self._generate_sessions_table(sessions_df)}
        
        <h2>🔄 Operation Analysis</h2>
        {self._generate_operations_analysis(history_df)}
        
        <h2>⏱️ Timing Performance</h2>
        {self._generate_timing_analysis(timing_stats)}
        
        <h2>📊 Charts and Visualizations</h2>
        <div class="chart-container">
            {'<p>📈 Interactive charts available when matplotlib/seaborn are installed</p>' if not PLOTTING_AVAILABLE else '<p>Charts generated as separate image files</p>'}
        </div>
        
        <h2>🔍 Detailed Exploration History</h2>
        {self._generate_history_table(history_df)}
        
        <div class="highlight">
            <h3>💡 Recommendations</h3>
            {self._generate_recommendations(summary_stats, history_df)}
        </div>
        
        <hr>
        <p class="timestamp">
            Report generated by MCTS Feature Discovery Analytics Module<br>
            For more details, check the exported data files and charts.
        </p>
    </div>
</body>
</html>
        '''
        
        return html_content
    
    def _calculate_summary_stats(self, session_data: Dict[str, pd.DataFrame], timing_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate summary statistics for the report."""
        sessions_df = session_data['sessions']
        history_df = session_data['exploration_history']
        
        stats = {}
        
        if not sessions_df.empty:
            stats['total_sessions'] = len(sessions_df)
            stats['best_score'] = sessions_df['best_score'].max()
            stats['total_iterations'] = sessions_df['total_iterations'].sum()
            
            # Calculate runtime
            if 'start_time' in sessions_df.columns and 'end_time' in sessions_df.columns:
                # Handle datetime conversion
                sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'])
                sessions_df['end_time'] = pd.to_datetime(sessions_df['end_time'])
                runtimes = (sessions_df['end_time'] - sessions_df['start_time']).dt.total_seconds() / 60
                stats['total_runtime_minutes'] = runtimes.sum()
                stats['avg_runtime_minutes'] = runtimes.mean()
            else:
                stats['total_runtime_minutes'] = 0
                stats['avg_runtime_minutes'] = 0
        else:
            stats.update({
                'total_sessions': 0,
                'best_score': 0,
                'total_iterations': 0,
                'total_runtime_minutes': 0,
                'avg_runtime_minutes': 0
            })
        
        if not history_df.empty:
            # Score improvement analysis
            scores = history_df['evaluation_score']
            stats['avg_score_improvement'] = scores.mean()
            stats['score_std'] = scores.std()
            stats['total_evaluations'] = len(history_df)
            
            # Operations per minute
            if stats['total_runtime_minutes'] > 0:
                stats['operations_per_minute'] = stats['total_evaluations'] / stats['total_runtime_minutes']
            else:
                stats['operations_per_minute'] = 0
        else:
            stats.update({
                'avg_score_improvement': 0,
                'score_std': 0,
                'total_evaluations': 0,
                'operations_per_minute': 0
            })
        
        # Add timing stats if available
        if timing_stats and 'session' in timing_stats:
            session_timing = timing_stats['session']
            stats['timing_total_operations'] = session_timing.get('total_operations', 0)
            stats['timing_operations_per_minute'] = session_timing.get('operations_per_minute', 0)
        
        # Assessment
        if stats['best_score'] > 0.4:
            stats['overall_assessment'] = "Excellent performance achieved"
        elif stats['best_score'] > 0.3:
            stats['overall_assessment'] = "Good performance with room for improvement"
        else:
            stats['overall_assessment'] = "System functioning, consider optimization"
        
        stats['key_insight'] = f"Best feature operations achieved {stats['best_score']:.4f} MAP@3 score"
        
        return stats
    
    def _generate_sessions_table(self, sessions_df: pd.DataFrame) -> str:
        """Generate HTML table for sessions."""
        if sessions_df.empty:
            return "<p>No session data available.</p>"
        
        table_html = '<table class="data-table"><thead><tr>'
        table_html += '<th>Session ID</th><th>Start Time</th><th>Iterations</th><th>Best Score</th><th>Status</th>'
        table_html += '</tr></thead><tbody>'
        
        for _, row in sessions_df.iterrows():
            session_id_short = str(row.get('session_id', ''))[:8] + '...'
            start_time = str(row.get('start_time', ''))[:19]  # Remove microseconds
            iterations = row.get('total_iterations', 0)
            best_score = f"{row.get('best_score', 0):.4f}"
            status = row.get('status', 'unknown')
            
            table_html += f'<tr><td>{session_id_short}</td><td>{start_time}</td>'
            table_html += f'<td>{iterations}</td><td>{best_score}</td><td>{status}</td></tr>'
        
        table_html += '</tbody></table>'
        return table_html
    
    def _generate_operations_analysis(self, history_df: pd.DataFrame) -> str:
        """Generate operations analysis section."""
        if history_df.empty:
            return "<p>No operation data available.</p>"
        
        # Analyze operation performance
        op_stats = history_df.groupby('operation_applied').agg({
            'evaluation_score': ['count', 'mean', 'max', 'std']
        }).round(4)
        
        op_stats.columns = ['Count', 'Avg Score', 'Max Score', 'Std Dev']
        op_stats = op_stats.sort_values('Avg Score', ascending=False)
        
        table_html = '<table class="data-table"><thead><tr>'
        table_html += '<th>Operation</th><th>Usage Count</th><th>Avg Score</th><th>Max Score</th><th>Consistency</th>'
        table_html += '</tr></thead><tbody>'
        
        for operation, row in op_stats.head(10).iterrows():
            consistency = "High" if row['Std Dev'] < 0.01 else "Medium" if row['Std Dev'] < 0.05 else "Low"
            table_html += f'<tr><td>{operation}</td><td>{row["Count"]}</td>'
            table_html += f'<td>{row["Avg Score"]:.4f}</td><td>{row["Max Score"]:.4f}</td><td>{consistency}</td></tr>'
        
        table_html += '</tbody></table>'
        return table_html
    
    def _generate_timing_analysis(self, timing_stats: Dict[str, Any]) -> str:
        """Generate timing analysis section."""
        if not timing_stats:
            return "<p>No timing data available.</p>"
        
        operations = timing_stats.get('operations', {})
        if not operations:
            return "<p>No operation timing data available.</p>"
        
        # Create timing summary table
        table_html = '<table class="data-table"><thead><tr>'
        table_html += '<th>Operation</th><th>Count</th><th>Avg Time (s)</th><th>Total Time (s)</th><th>Max Time (s)</th>'
        table_html += '</tr></thead><tbody>'
        
        # Sort by total time
        sorted_ops = sorted(operations.items(), key=lambda x: x[1].get('total_time', 0), reverse=True)
        
        for op_name, op_stats in sorted_ops[:10]:
            count = op_stats.get('count', 0)
            avg_time = op_stats.get('avg_time', 0)
            total_time = op_stats.get('total_time', 0)
            max_time = op_stats.get('max_time', 0)
            
            table_html += f'<tr><td>{op_name}</td><td>{count}</td>'
            table_html += f'<td>{avg_time:.3f}</td><td>{total_time:.3f}</td><td>{max_time:.3f}</td></tr>'
        
        table_html += '</tbody></table>'
        return table_html
    
    def _generate_history_table(self, history_df: pd.DataFrame) -> str:
        """Generate exploration history table."""
        if history_df.empty:
            return "<p>No exploration history available.</p>"
        
        # Show recent 20 entries
        recent_history = history_df.tail(20)
        
        table_html = '<table class="data-table"><thead><tr>'
        table_html += '<th>Iteration</th><th>Operation</th><th>Score</th><th>Eval Time (s)</th><th>Best So Far</th>'
        table_html += '</tr></thead><tbody>'
        
        for _, row in recent_history.iterrows():
            iteration = row.get('iteration', 0)
            operation = str(row.get('operation_applied', ''))[:30] + ('...' if len(str(row.get('operation_applied', ''))) > 30 else '')
            score = f"{row.get('evaluation_score', 0):.4f}"
            eval_time = f"{row.get('evaluation_time', 0):.3f}"
            is_best = "✅" if row.get('is_best_so_far', False) else ""
            
            table_html += f'<tr><td>{iteration}</td><td>{operation}</td>'
            table_html += f'<td>{score}</td><td>{eval_time}</td><td>{is_best}</td></tr>'
        
        table_html += '</tbody></table>'
        return table_html
    
    def _generate_recommendations(self, stats: Dict[str, Any], history_df: pd.DataFrame) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Performance recommendations
        if stats.get('best_score', 0) < 0.3:
            recommendations.append("🎯 Consider increasing exploration time or trying different feature operation combinations")
        
        if stats.get('operations_per_minute', 0) < 5:
            recommendations.append("⚡ Performance could be improved - consider using mock evaluator for faster testing")
        
        if stats.get('total_iterations', 0) < 50:
            recommendations.append("🔄 Consider running more iterations for better exploration coverage")
        
        # Operation-specific recommendations
        if not history_df.empty:
            top_operations = history_df.groupby('operation_applied')['evaluation_score'].mean().sort_values(ascending=False).head(3)
            if len(top_operations) > 0:
                best_op = top_operations.index[0]
                recommendations.append(f"⭐ '{best_op}' operation shows best performance - consider prioritizing similar operations")
        
        if not recommendations:
            recommendations.append("✅ System is performing well - continue with current strategy")
        
        return '<br>'.join([f'<p>{rec}</p>' for rec in recommendations])
    
    def _generate_all_charts(self, session_data: Dict[str, pd.DataFrame], timing_stats: Dict[str, Any] = None) -> Dict[str, str]:
        """Generate all visualization charts."""
        if not PLOTTING_AVAILABLE:
            return {}
        
        chart_paths = {}
        
        try:
            # Score progression chart
            score_chart = self._create_score_progression_chart(session_data['exploration_history'])
            if score_chart:
                chart_paths['score_progression'] = score_chart
            
            # Operation performance chart
            op_chart = self._create_operation_performance_chart(session_data['exploration_history'])
            if op_chart:
                chart_paths['operation_performance'] = op_chart
            
            # Timing analysis chart
            if timing_stats:
                timing_chart = self._create_timing_analysis_chart(timing_stats)
                if timing_chart:
                    chart_paths['timing_analysis'] = timing_chart
            
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
        
        return chart_paths
    
    def _create_score_progression_chart(self, history_df: pd.DataFrame) -> Optional[str]:
        """Create score progression over time chart."""
        if history_df.empty:
            return None
        
        try:
            plt.figure(figsize=self.figure_size)
            
            # Plot score progression
            plt.plot(history_df['iteration'], history_df['evaluation_score'], 'o-', alpha=0.7, linewidth=2)
            
            # Add best score line
            best_scores = history_df['evaluation_score'].cummax()
            plt.plot(history_df['iteration'], best_scores, 'r-', linewidth=3, label='Best Score So Far')
            
            plt.title('MCTS Score Progression Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Evaluation Score (MAP@3)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save chart
            chart_path = self.output_dir / f"score_progression.{self.save_format}"
            plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to create score progression chart: {e}")
            plt.close()
            return None
    
    def _create_operation_performance_chart(self, history_df: pd.DataFrame) -> Optional[str]:
        """Create operation performance comparison chart."""
        if history_df.empty:
            return None
        
        try:
            # Calculate operation statistics
            op_stats = history_df.groupby('operation_applied')['evaluation_score'].agg(['mean', 'count']).reset_index()
            op_stats = op_stats[op_stats['count'] >= 3]  # Only operations with 3+ uses
            op_stats = op_stats.sort_values('mean', ascending=True).tail(15)  # Top 15
            
            if op_stats.empty:
                return None
            
            plt.figure(figsize=(12, 8))
            
            # Create horizontal bar chart
            bars = plt.barh(range(len(op_stats)), op_stats['mean'], alpha=0.8)
            
            # Color bars based on performance
            for i, (bar, score) in enumerate(zip(bars, op_stats['mean'])):
                if score > 0.4:
                    bar.set_color('#2ecc71')  # Green for high performance
                elif score > 0.3:
                    bar.set_color('#f39c12')  # Orange for medium performance
                else:
                    bar.set_color('#e74c3c')  # Red for low performance
            
            plt.yticks(range(len(op_stats)), [op[:25] + '...' if len(op) > 25 else op for op in op_stats['operation_applied']])
            plt.xlabel('Average Evaluation Score', fontsize=12)
            plt.title('Feature Operation Performance Comparison', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (score, count) in enumerate(zip(op_stats['mean'], op_stats['count'])):
                plt.text(score + 0.005, i, f'{score:.3f} ({count}x)', 
                        va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"operation_performance.{self.save_format}"
            plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to create operation performance chart: {e}")
            plt.close()
            return None
    
    def _create_timing_analysis_chart(self, timing_stats: Dict[str, Any]) -> Optional[str]:
        """Create timing analysis chart."""
        try:
            operations = timing_stats.get('operations', {})
            if not operations:
                return None
            
            # Prepare data
            op_names = []
            total_times = []
            avg_times = []
            counts = []
            
            for op_name, op_data in operations.items():
                op_names.append(op_name[:20] + '...' if len(op_name) > 20 else op_name)
                total_times.append(op_data.get('total_time', 0))
                avg_times.append(op_data.get('avg_time', 0))
                counts.append(op_data.get('count', 0))
            
            # Sort by total time
            sorted_data = sorted(zip(op_names, total_times, avg_times, counts), key=lambda x: x[1], reverse=True)
            if len(sorted_data) > 15:
                sorted_data = sorted_data[:15]
            
            op_names, total_times, avg_times, counts = zip(*sorted_data)
            
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Total time chart
            bars1 = ax1.barh(range(len(op_names)), total_times, alpha=0.8, color='skyblue')
            ax1.set_yticks(range(len(op_names)))
            ax1.set_yticklabels(op_names)
            ax1.set_xlabel('Total Time (seconds)')
            ax1.set_title('Total Time per Operation', fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Average time chart
            bars2 = ax2.barh(range(len(op_names)), avg_times, alpha=0.8, color='lightcoral')
            ax2.set_yticks(range(len(op_names)))
            ax2.set_yticklabels(op_names)
            ax2.set_xlabel('Average Time (seconds)')
            ax2.set_title('Average Time per Operation', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"timing_analysis.{self.save_format}"
            plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to create timing analysis chart: {e}")
            plt.close()
            return None
    
    def _export_analysis_data(self, session_data: Dict[str, pd.DataFrame], timing_stats: Dict[str, Any] = None) -> Dict[str, str]:
        """Export analysis data in various formats."""
        export_paths = {}
        
        try:
            # Export session data as CSV
            if not session_data['exploration_history'].empty:
                csv_path = self.output_dir / "exploration_history.csv"
                session_data['exploration_history'].to_csv(csv_path, index=False)
                export_paths['exploration_csv'] = str(csv_path)
            
            # Export summary statistics as JSON
            summary_stats = self._calculate_summary_stats(session_data, timing_stats)
            json_path = self.output_dir / "summary_statistics.json"
            with open(json_path, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            export_paths['summary_json'] = str(json_path)
            
            # Export timing data if available
            if timing_stats:
                timing_path = self.output_dir / "timing_analysis.json"
                with open(timing_path, 'w') as f:
                    json.dump(timing_stats, f, indent=2, default=str)
                export_paths['timing_json'] = str(timing_path)
                
        except Exception as e:
            logger.error(f"Failed to export analysis data: {e}")
        
        return export_paths

def generate_quick_report(db_path: str, config: Dict[str, Any] = None) -> str:
    """Quick function to generate analytics report."""
    if not config:
        config = {'export': {'output_dir': 'reports'}, 'analytics': {}}
    
    analytics = AnalyticsGenerator(config)
    report_files = analytics.generate_comprehensive_report(db_path)
    
    return report_files.get('html_report', 'Report generation failed')