"""
Trends command for analytics module
"""

from typing import Dict, Any, Optional
from .base import BaseAnalyticsCommand
from manager.core.utils import format_number, format_percentage


class TrendsCommand(BaseAnalyticsCommand):
    """Show performance trends over time."""
    
    def execute(self) -> None:
        """Execute trends command."""
        # Get trends data from service
        trends_data = self.service.get_performance_trends(
            days=self.args.days,
            dataset=self.args.dataset
        )
        
        # Output in requested format
        self.output(trends_data, title="Performance Trends")
    
    def _format_text(self, data: Dict[str, Any], title: Optional[str] = None) -> str:
        """Format trends data as text."""
        lines = []
        
        lines.append("📈 PERFORMANCE TRENDS")
        lines.append("=" * 50)
        
        summary = data.get('summary', {})
        if summary:
            lines.append(f"\nPeriod: Last {summary.get('total_days', 0)} days")
            lines.append(f"Score Trend: {summary.get('score_trend', 'Unknown').upper()}")
            
            trend_pct = summary.get('trend_percentage', 0)
            if trend_pct != 0:
                sign = '+' if trend_pct > 0 else ''
                lines.append(f"Change: {sign}{format_percentage(trend_pct)}")
            
            if summary.get('best_day'):
                lines.append(f"Best Day: {summary['best_day']}")
            if summary.get('most_active_day'):
                lines.append(f"Most Active: {summary['most_active_day']}")
        
        # Daily metrics table
        daily_metrics = data.get('daily_metrics', [])
        if daily_metrics:
            lines.append("\n📊 DAILY METRICS:")
            lines.append("-" * 70)
            lines.append(f"{'Date':<12} {'Sessions':<10} {'Avg Score':<12} {'Max Score':<12} {'Nodes':<10}")
            lines.append("-" * 70)
            
            for day in daily_metrics[-14:]:  # Show last 14 days
                lines.append(
                    f"{day['date']:<12} "
                    f"{format_number(day['session_count']):<10} "
                    f"{day['avg_score']:<12.5f} "
                    f"{day['max_score']:<12.5f} "
                    f"{format_number(day.get('avg_nodes', 0)):<10}"
                )
        
        # Trend visualization (simple ASCII chart)
        if len(daily_metrics) >= 7:
            lines.append("\n📉 SCORE TREND (7-day moving average):")
            lines.append(self._create_ascii_chart(daily_metrics))
        
        return '\n'.join(lines)
    
    def _create_ascii_chart(self, metrics: list, width: int = 50, height: int = 10) -> str:
        """Create simple ASCII chart for score trends."""
        if not metrics:
            return "No data available"
        
        # Extract scores
        scores = [m['avg_score'] for m in metrics if m['avg_score'] is not None]
        if not scores:
            return "No score data available"
        
        # Calculate range
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            return f"Flat trend at {min_score:.5f}"
        
        # Create chart
        chart_lines = []
        
        # Scale scores to chart height
        scaled_scores = []
        for score in scores:
            scaled = int((score - min_score) / score_range * (height - 1))
            scaled_scores.append(scaled)
        
        # Build chart from top to bottom
        for row in range(height - 1, -1, -1):
            line = []
            for col, scaled_score in enumerate(scaled_scores):
                if scaled_score >= row:
                    line.append('█')
                else:
                    line.append(' ')
            
            # Add axis label
            if row == height - 1:
                label = f"{max_score:.5f} |"
            elif row == 0:
                label = f"{min_score:.5f} |"
            else:
                label = "         |"
            
            chart_lines.append(label + ''.join(line))
        
        # Add bottom axis
        chart_lines.append("         " + "+" + "-" * len(scores))
        
        # Add date labels (first and last)
        if metrics:
            first_date = metrics[0]['date']
            last_date = metrics[-1]['date']
            chart_lines.append(f"         {first_date}    ->    {last_date}")
        
        return '\n'.join(chart_lines)