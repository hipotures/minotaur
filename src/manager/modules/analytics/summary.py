"""
Summary command for analytics module
"""

from typing import Dict, Any, Optional
from .base import BaseAnalyticsCommand
from manager.core.utils import format_number, format_duration, format_percentage


class SummaryCommand(BaseAnalyticsCommand):
    """Generate overall performance summary."""
    
    def execute(self) -> None:
        """Execute summary command."""
        # Get summary data from service
        summary_data = self.service.get_performance_summary(
            days=self.args.days,
            dataset=self.args.dataset
        )
        
        # Output in requested format
        self.output(summary_data, title="Performance Summary")
    
    def _format_text(self, data: Dict[str, Any], title: Optional[str] = None) -> str:
        """Format summary data as text."""
        lines = []
        
        lines.append("📊 PERFORMANCE SUMMARY")
        lines.append("=" * 50)
        
        # Overall metrics
        overall = data.get('overall_metrics', {})
        if overall:
            lines.append("\n🎯 OVERALL METRICS:")
            lines.append(f"   Sessions: {format_number(overall.get('total_sessions', 0))}")
            lines.append(f"   Completed: {format_number(overall.get('completed_sessions', 0))}")
            lines.append(f"   Success Rate: {format_percentage(overall.get('success_rate', 0))}")
            
            if overall.get('avg_score'):
                lines.append(f"   Average Score: {overall['avg_score']:.5f}")
                lines.append(f"   Best Score: {overall.get('best_score', 0):.5f}")
            
            if overall.get('total_time'):
                lines.append(f"   Total Time: {format_duration(overall['total_time'])}")
                lines.append(f"   Average Time: {format_duration(overall.get('avg_time', 0))}")
        
        # Session status breakdown
        status_breakdown = data.get('status_breakdown', {})
        if status_breakdown:
            lines.append("\n📈 SESSION STATUS:")
            for status, count in sorted(status_breakdown.items(), 
                                       key=lambda x: x[1], reverse=True):
                lines.append(f"   {status.title()}: {format_number(count)}")
        
        # Top sessions
        top_sessions = data.get('top_sessions', [])
        if top_sessions:
            lines.append("\n🏆 TOP SESSIONS:")
            for i, session in enumerate(top_sessions[:5], 1):
                lines.append(f"   {i}. {session['session_id'][:8]}... "
                           f"({session.get('dataset', 'Unknown')}) - "
                           f"Score: {session['score']:.5f} "
                           f"[{format_number(session.get('nodes', 0))} nodes]")
        
        # Feature impact
        feature_stats = data.get('feature_impact', {})
        if feature_stats and feature_stats.get('total_features', 0) > 0:
            lines.append("\n🧪 FEATURE IMPACT:")
            lines.append(f"   Features Tested: {format_number(feature_stats['total_features'])}")
            lines.append(f"   Positive Impact: {format_number(feature_stats.get('positive_features', 0))} "
                        f"({format_percentage(feature_stats.get('success_rate', 0))})")
            lines.append(f"   Average Impact: {feature_stats.get('avg_impact', 0):.5f}")
            lines.append(f"   Best Impact: {feature_stats.get('best_impact', 0):.5f}")
        
        # Recent activity
        recent = data.get('recent_activity', {})
        if recent:
            lines.append("\n📅 RECENT ACTIVITY:")
            lines.append(f"   Last 24h: {format_number(recent.get('sessions_24h', 0))} sessions")
            lines.append(f"   Last 7d: {format_number(recent.get('sessions_7d', 0))} sessions")
            lines.append(f"   Last 30d: {format_number(recent.get('sessions_30d', 0))} sessions")
        
        return '\n'.join(lines)