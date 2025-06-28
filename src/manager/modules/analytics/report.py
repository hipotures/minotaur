"""
Report command for analytics module
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from .base import BaseAnalyticsCommand
from manager.core.utils import format_number, format_duration, format_percentage, format_datetime


class ReportCommand(BaseAnalyticsCommand):
    """Generate comprehensive HTML report."""
    
    def execute(self) -> None:
        """Execute report generation command."""
        # Get comprehensive report data from service
        report_data = self.service.generate_report(
            days=self.args.days,
            dataset=self.args.dataset
        )
        
        # Force HTML format for reports
        original_format = self.format
        self.format = 'html'
        
        # Set default output path if not specified
        if not self.output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_path = f"analytics_report_{timestamp}.html"
        
        # Output report
        self.output(report_data, title="Analytics Report")
        
        # Restore original format
        self.format = original_format
    
    def _format_text(self, data: Dict[str, Any], title: Optional[str] = None) -> str:
        """Reports are always in HTML format."""
        return self._format_html(data, title)
    
    def _format_html(self, data: Dict[str, Any], title: Optional[str] = None) -> str:
        """Generate comprehensive HTML report."""
        html_parts = ['<!DOCTYPE html><html><head>']
        html_parts.append('<meta charset="utf-8">')
        html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append(f'<title>{title or "Analytics Report"}</title>')
        
        # Enhanced CSS for professional report
        html_parts.append('<style>')
        html_parts.append(self._get_report_css())
        html_parts.append('</style>')
        
        # JavaScript for interactivity
        html_parts.append('<script>')
        html_parts.append(self._get_report_js())
        html_parts.append('</script>')
        
        html_parts.append('</head><body>')
        
        # Header
        metadata = data.get('report_metadata', {})
        html_parts.append('<header>')
        html_parts.append('<h1>📊 Analytics Report</h1>')
        html_parts.append(f'<p class="subtitle">Generated on {format_datetime(metadata.get("generated_at", datetime.now().isoformat()), "long")}</p>')
        if metadata.get('dataset_filter'):
            html_parts.append(f'<p class="filter-info">Dataset: {metadata["dataset_filter"]}</p>')
        html_parts.append(f'<p class="filter-info">Period: Last {metadata.get("period_days", 30)} days</p>')
        html_parts.append('</header>')
        
        # Navigation
        html_parts.append('<nav>')
        html_parts.append('<ul>')
        html_parts.append('<li><a href="#summary">Summary</a></li>')
        html_parts.append('<li><a href="#trends">Trends</a></li>')
        html_parts.append('<li><a href="#operations">Operations</a></li>')
        html_parts.append('<li><a href="#features">Top Features</a></li>')
        html_parts.append('<li><a href="#resources">Resources</a></li>')
        html_parts.append('</ul>')
        html_parts.append('</nav>')
        
        html_parts.append('<main>')
        
        # Summary Section
        summary = data.get('summary', {})
        if summary:
            html_parts.append('<section id="summary">')
            html_parts.append('<h2>Executive Summary</h2>')
            html_parts.append(self._format_summary_section(summary))
            html_parts.append('</section>')
        
        # Trends Section
        trends = data.get('trends', {})
        if trends:
            html_parts.append('<section id="trends">')
            html_parts.append('<h2>Performance Trends</h2>')
            html_parts.append(self._format_trends_section(trends))
            html_parts.append('</section>')
        
        # Operations Section
        operations = data.get('operations', {})
        if operations:
            html_parts.append('<section id="operations">')
            html_parts.append('<h2>Operations Analysis</h2>')
            html_parts.append(self._format_operations_section(operations))
            html_parts.append('</section>')
        
        # Top Features Section
        features = data.get('top_features', [])
        if features:
            html_parts.append('<section id="features">')
            html_parts.append('<h2>Top Features</h2>')
            html_parts.append(self._format_features_table(features))
            html_parts.append('</section>')
        
        # Resource Usage Section
        resources = data.get('resource_usage', {})
        if resources:
            html_parts.append('<section id="resources">')
            html_parts.append('<h2>Resource Usage</h2>')
            html_parts.append(self._format_resources_section(resources))
            html_parts.append('</section>')
        
        html_parts.append('</main>')
        
        # Footer
        html_parts.append('<footer>')
        html_parts.append('<p>Generated by Minotaur Analytics System</p>')
        html_parts.append('</footer>')
        
        html_parts.append('</body></html>')
        return '\n'.join(html_parts)
    
    def _get_report_css(self) -> str:
        """Get CSS for the report."""
        return '''
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .subtitle { opacity: 0.9; font-size: 1.1rem; }
        .filter-info { opacity: 0.8; font-size: 0.9rem; }
        
        nav {
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        nav ul {
            list-style: none;
            display: flex;
            justify-content: center;
            padding: 1rem;
        }
        
        nav li { margin: 0 1rem; }
        
        nav a {
            text-decoration: none;
            color: #667eea;
            font-weight: 500;
            transition: color 0.3s;
        }
        
        nav a:hover { color: #764ba2; }
        
        main {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        
        section {
            background: white;
            border-radius: 8px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h2 {
            color: #667eea;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.2s;
        }
        
        .metric-card:hover { transform: translateY(-2px); }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }
        
        tr:hover { background: #f8f9fa; }
        
        .chart-container {
            margin: 2rem 0;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .progress-bar {
            background: #e0e0e0;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.3s;
        }
        
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .badge-success { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-danger { background: #f8d7da; color: #721c24; }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            nav ul { flex-direction: column; align-items: center; }
            nav li { margin: 0.5rem 0; }
            .metric-grid { grid-template-columns: 1fr; }
        }
        '''
    
    def _get_report_js(self) -> str:
        """Get JavaScript for the report."""
        return '''
        // Smooth scrolling for navigation
        document.querySelectorAll('nav a').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                target.scrollIntoView({ behavior: 'smooth' });
            });
        });
        
        // Animate numbers on scroll
        const animateValue = (element, start, end, duration) => {
            let startTimestamp = null;
            const step = (timestamp) => {
                if (!startTimestamp) startTimestamp = timestamp;
                const progress = Math.min((timestamp - startTimestamp) / duration, 1);
                element.textContent = Math.floor(progress * (end - start) + start).toLocaleString();
                if (progress < 1) {
                    window.requestAnimationFrame(step);
                }
            };
            window.requestAnimationFrame(step);
        };
        
        // Observe metric cards for animation
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const element = entry.target;
                    const value = parseInt(element.textContent.replace(/,/g, ''));
                    if (!isNaN(value)) {
                        animateValue(element, 0, value, 1000);
                        observer.unobserve(element);
                    }
                }
            });
        });
        
        document.querySelectorAll('.metric-value').forEach(el => observer.observe(el));
        '''
    
    def _format_summary_section(self, summary: Dict[str, Any]) -> str:
        """Format summary section as HTML."""
        overall = summary.get('overall_metrics', {})
        status = summary.get('status_breakdown', {})
        
        html = '<div class="metric-grid">'
        
        # Key metrics
        metrics = [
            ('Total Sessions', overall.get('total_sessions', 0)),
            ('Success Rate', format_percentage(overall.get('success_rate', 0))),
            ('Avg Score', f"{overall.get('avg_score', 0):.5f}"),
            ('Best Score', f"{overall.get('best_score', 0):.5f}"),
            ('Total Time', format_duration(overall.get('total_time', 0))),
            ('Total Features', format_number(overall.get('total_features', 0)))
        ]
        
        for label, value in metrics:
            html += f'''
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            '''
        
        html += '</div>'
        
        # Status breakdown
        if status:
            html += '<h3>Session Status Breakdown</h3>'
            html += '<div class="chart-container">'
            
            total = sum(status.values())
            for stat, count in status.items():
                percentage = (count / total * 100) if total > 0 else 0
                badge_class = {
                    'completed': 'badge-success',
                    'failed': 'badge-danger',
                    'running': 'badge-warning'
                }.get(stat, '')
                
                html += f'''
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span class="badge {badge_class}">{stat.title()}</span>
                        <span>{count} ({percentage:.1f}%)</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {percentage}%"></div>
                    </div>
                </div>
                '''
            
            html += '</div>'
        
        return html
    
    def _format_trends_section(self, trends: Dict[str, Any]) -> str:
        """Format trends section as HTML."""
        summary = trends.get('summary', {})
        daily = trends.get('daily_metrics', [])
        
        html = '<div class="metric-grid">'
        
        # Trend summary
        trend_icon = '📈' if summary.get('score_trend') == 'improving' else '📉'
        html += f'''
        <div class="metric-card">
            <div class="metric-label">Score Trend</div>
            <div class="metric-value">{trend_icon} {summary.get("score_trend", "Unknown").title()}</div>
        </div>
        '''
        
        if summary.get('trend_percentage'):
            html += f'''
            <div class="metric-card">
                <div class="metric-label">Change</div>
                <div class="metric-value">{format_percentage(summary["trend_percentage"], show_sign=True)}</div>
            </div>
            '''
        
        html += '</div>'
        
        # Daily metrics table
        if daily:
            html += '<h3>Daily Performance</h3>'
            html += '<table>'
            html += '<thead><tr>'
            html += '<th>Date</th><th>Sessions</th><th>Avg Score</th><th>Max Score</th><th>Avg Nodes</th>'
            html += '</tr></thead><tbody>'
            
            for day in daily[-14:]:  # Last 14 days
                html += f'''
                <tr>
                    <td>{day["date"]}</td>
                    <td>{day["session_count"]}</td>
                    <td>{day["avg_score"]:.5f}</td>
                    <td>{day["max_score"]:.5f}</td>
                    <td>{format_number(day.get("avg_nodes", 0))}</td>
                </tr>
                '''
            
            html += '</tbody></table>'
        
        return html
    
    def _format_operations_section(self, operations: Dict[str, Any]) -> str:
        """Format operations section as HTML."""
        summary = operations.get('summary', {})
        
        html = '<div class="metric-grid">'
        
        if summary:
            html += f'''
            <div class="metric-card">
                <div class="metric-label">Total Operations</div>
                <div class="metric-value">{format_number(summary.get("total_operations", 0))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Most Used</div>
                <div class="metric-value" style="font-size: 1.2rem;">{summary.get("most_used", "N/A")}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Most Impactful</div>
                <div class="metric-value" style="font-size: 1.2rem;">{summary.get("most_impactful", "N/A")}</div>
            </div>
            '''
        
        html += '</div>'
        
        # Operation impact table
        op_impact = operations.get('operation_impact', [])
        if op_impact:
            html += '<h3>Operation Impact Analysis</h3>'
            html += '<table>'
            html += '<thead><tr>'
            html += '<th>Operation</th><th>Features</th><th>Avg Impact</th><th>Total Impact</th>'
            html += '</tr></thead><tbody>'
            
            for op in op_impact[:10]:
                html += f'''
                <tr>
                    <td>{op["operation"]}</td>
                    <td>{op["features"]}</td>
                    <td>{op["avg_impact"]:.5f}</td>
                    <td>{op["total_impact"]:.5f}</td>
                </tr>
                '''
            
            html += '</tbody></table>'
        
        return html
    
    def _format_features_table(self, features: List[Dict[str, Any]]) -> str:
        """Format features table as HTML."""
        html = '<table>'
        html += '<thead><tr>'
        html += '<th>Rank</th><th>Feature</th><th>Avg Impact</th><th>Success Rate</th><th>Sessions</th><th>Uses</th>'
        html += '</tr></thead><tbody>'
        
        for feature in features:
            success_badge = 'badge-success' if feature['success_rate'] > 0.7 else 'badge-warning'
            html += f'''
            <tr>
                <td>{feature["rank"]}</td>
                <td>{feature["feature_name"]}</td>
                <td>{feature["avg_impact"]:.5f}</td>
                <td><span class="badge {success_badge}">{format_percentage(feature["success_rate"])}</span></td>
                <td>{feature["session_count"]}</td>
                <td>{feature["total_uses"]}</td>
            </tr>
            '''
        
        html += '</tbody></table>'
        return html
    
    def _format_resources_section(self, resources: Dict[str, Any]) -> str:
        """Format resources section as HTML."""
        html = '<div class="metric-grid">'
        
        html += f'''
        <div class="metric-card">
            <div class="metric-label">Database Size</div>
            <div class="metric-value">{resources["database_size_mb"]:.1f} MB</div>
        </div>
        '''
        
        html += '</div>'
        
        # Table sizes
        table_sizes = resources.get('table_sizes', [])
        if table_sizes:
            html += '<h3>Table Sizes</h3>'
            html += '<table>'
            html += '<thead><tr><th>Table</th><th>Size (MB)</th></tr></thead><tbody>'
            
            for table in table_sizes[:10]:
                html += f'''
                <tr>
                    <td>{table["table_name"]}</td>
                    <td>{table["size_mb"]:.2f}</td>
                </tr>
                '''
            
            html += '</tbody></table>'
        
        return html