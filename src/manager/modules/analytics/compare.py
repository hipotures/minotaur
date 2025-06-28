"""
Compare command for analytics module
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from .base import BaseAnalyticsCommand
from manager.core.utils import format_number, format_percentage, format_duration


class CompareCommand(BaseAnalyticsCommand):
    """Compare performance across time periods."""
    
    def execute(self, period1: str, period2: str) -> None:
        """Execute comparison command.
        
        Args:
            period1: First period (days or date range)
            period2: Second period (days or date range)
        """
        # Parse periods
        period1_start, period1_end = self._parse_period(period1)
        period2_start, period2_end = self._parse_period(period2)
        
        # Get comparison data from service
        comparison_data = self.service.compare_periods(
            period1_start, period1_end,
            period2_start, period2_end
        )
        
        # Output in requested format
        self.output(comparison_data, title="Period Comparison")
    
    def _parse_period(self, period: str) -> tuple:
        """Parse period string to start and end dates.
        
        Args:
            period: Period string (e.g., "7" for last 7 days, or "2024-01-01:2024-01-31")
            
        Returns:
            Tuple of (start_date, end_date) as strings
        """
        # Check if it's a date range
        if ':' in period:
            parts = period.split(':')
            return parts[0], parts[1]
        
        # Otherwise treat as number of days
        try:
            days = int(period)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            return start_date.isoformat(), end_date.isoformat()
        except ValueError:
            # Try to parse as single date
            try:
                date = datetime.strptime(period, '%Y-%m-%d').date()
                return date.isoformat(), date.isoformat()
            except:
                raise ValueError(f"Invalid period format: {period}")
    
    def _format_text(self, data: Dict[str, Any], title: Optional[str] = None) -> str:
        """Format comparison data as text."""
        lines = []
        
        lines.append("📊 PERIOD COMPARISON")
        lines.append("=" * 50)
        
        # Period information
        period1 = data.get('period1', {})
        period2 = data.get('period2', {})
        
        lines.append(f"\n📅 PERIOD 1: {period1.get('start')} to {period1.get('end')}")
        lines.append(f"📅 PERIOD 2: {period2.get('start')} to {period2.get('end')}")
        
        # Metrics comparison
        metrics1 = period1.get('metrics', {})
        metrics2 = period2.get('metrics', {})
        differences = data.get('differences', {})
        
        lines.append("\n📈 METRICS COMPARISON:")
        lines.append("-" * 70)
        lines.append(f"{'Metric':<25} {'Period 1':<15} {'Period 2':<15} {'Change':<15} {'Trend':<10}")
        lines.append("-" * 70)
        
        # Format each metric
        metric_info = [
            ('Total Sessions', 'total_sessions', format_number),
            ('Completed', 'completed_sessions', format_number),
            ('Success Rate', 'success_rate', lambda x: format_percentage(x)),
            ('Avg Score', 'avg_score', lambda x: f"{x:.5f}"),
            ('Max Score', 'max_score', lambda x: f"{x:.5f}"),
            ('Total Nodes', 'total_nodes', format_number),
            ('Total Features', 'total_features', format_number),
            ('Total Time', 'total_time', lambda x: format_duration(x))
        ]
        
        for display_name, key, formatter in metric_info:
            val1 = metrics1.get(key, 0)
            val2 = metrics2.get(key, 0)
            diff_info = differences.get(key, {})
            
            # Format values
            val1_str = formatter(val1) if val1 else "0"
            val2_str = formatter(val2) if val2 else "0"
            
            # Format change
            if isinstance(diff_info, dict):
                pct_change = diff_info.get('percentage', 0)
                improved = diff_info.get('improved', False)
                
                if abs(pct_change) < 0.01:
                    change_str = "No change"
                else:
                    sign = '+' if pct_change > 0 else ''
                    change_str = f"{sign}{pct_change:.1f}%"
                
                # Trend indicator
                if improved:
                    trend = "✅ Better"
                elif pct_change == 0:
                    trend = "➖ Same"
                else:
                    trend = "❌ Worse"
            else:
                change_str = "N/A"
                trend = "N/A"
            
            lines.append(
                f"{display_name:<25} "
                f"{val1_str:<15} "
                f"{val2_str:<15} "
                f"{change_str:<15} "
                f"{trend:<10}"
            )
        
        # Summary
        summary = data.get('summary', {})
        if summary:
            lines.append("\n🎯 SUMMARY:")
            
            overall = summary.get('overall_improvement', False)
            efficiency = summary.get('efficiency_change', 0)
            
            if overall:
                lines.append("   ✅ Overall performance IMPROVED between periods")
            else:
                lines.append("   ❌ Overall performance DECLINED between periods")
            
            if abs(efficiency) > 0.01:
                sign = '+' if efficiency > 0 else ''
                lines.append(f"   Efficiency change: {sign}{efficiency:.1f}%")
        
        # Key insights
        lines.append("\n💡 KEY INSIGHTS:")
        
        # Find biggest improvements and declines
        improvements = []
        declines = []
        
        for key, diff_info in differences.items():
            if isinstance(diff_info, dict) and 'percentage' in diff_info:
                pct = diff_info['percentage']
                if pct > 10:
                    improvements.append((key, pct))
                elif pct < -10:
                    declines.append((key, abs(pct)))
        
        if improvements:
            improvements.sort(key=lambda x: x[1], reverse=True)
            lines.append("   Biggest improvements:")
            for key, pct in improvements[:3]:
                display_key = key.replace('_', ' ').title()
                lines.append(f"   • {display_key}: +{pct:.1f}%")
        
        if declines:
            declines.sort(key=lambda x: x[1], reverse=True)
            lines.append("   Biggest declines:")
            for key, pct in declines[:3]:
                display_key = key.replace('_', ' ').title()
                lines.append(f"   • {display_key}: -{pct:.1f}%")
        
        # Activity comparison
        if metrics1.get('total_sessions', 0) > 0 and metrics2.get('total_sessions', 0) > 0:
            lines.append("\n📊 ACTIVITY ANALYSIS:")
            
            # Sessions per day
            period1_days = self._calculate_days(period1.get('start'), period1.get('end'))
            period2_days = self._calculate_days(period2.get('start'), period2.get('end'))
            
            if period1_days > 0 and period2_days > 0:
                sessions_per_day1 = metrics1['total_sessions'] / period1_days
                sessions_per_day2 = metrics2['total_sessions'] / period2_days
                
                lines.append(f"   Sessions/day Period 1: {sessions_per_day1:.1f}")
                lines.append(f"   Sessions/day Period 2: {sessions_per_day2:.1f}")
                
                activity_change = ((sessions_per_day1 - sessions_per_day2) / sessions_per_day2 * 100) if sessions_per_day2 > 0 else 0
                if abs(activity_change) > 0.01:
                    sign = '+' if activity_change > 0 else ''
                    lines.append(f"   Activity change: {sign}{activity_change:.1f}%")
        
        return '\n'.join(lines)
    
    def _calculate_days(self, start_date: str, end_date: str) -> int:
        """Calculate number of days between dates."""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            return (end - start).days + 1
        except:
            return 0