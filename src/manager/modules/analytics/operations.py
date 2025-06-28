"""
Operations command for analytics module
"""

from typing import Dict, Any, Optional
from .base import BaseAnalyticsCommand
from manager.core.utils import format_number, format_duration


class OperationsCommand(BaseAnalyticsCommand):
    """Analyze operation effectiveness."""
    
    def execute(self) -> None:
        """Execute operations analysis command."""
        # Get operations data from service
        operations_data = self.service.analyze_operations(
            days=self.args.days
        )
        
        # Output in requested format
        self.output(operations_data, title="Operations Analysis")
    
    def _format_text(self, data: Dict[str, Any], title: Optional[str] = None) -> str:
        """Format operations data as text."""
        lines = []
        
        lines.append("⚙️  OPERATIONS ANALYSIS")
        lines.append("=" * 50)
        
        # Summary
        summary = data.get('summary', {})
        if summary:
            lines.append("\n📊 SUMMARY:")
            lines.append(f"   Total Operations: {format_number(summary.get('total_operations', 0))}")
            lines.append(f"   Most Used: {summary.get('most_used', 'N/A')}")
            lines.append(f"   Most Impactful: {summary.get('most_impactful', 'N/A')}")
        
        # Operation metrics
        op_metrics = data.get('operation_metrics', [])
        if op_metrics:
            lines.append("\n⏱️  OPERATION PERFORMANCE:")
            lines.append("-" * 80)
            lines.append(f"{'Operation':<25} {'Count':<10} {'Avg Time':<12} {'Max Time':<12} {'Total Time':<12}")
            lines.append("-" * 80)
            
            for op in op_metrics[:20]:  # Top 20 operations
                lines.append(
                    f"{op['operation_type']:<25} "
                    f"{format_number(op['count']):<10} "
                    f"{format_duration(op['avg_duration']/1000):<12} "
                    f"{format_duration(op['max_duration']/1000):<12} "
                    f"{format_duration(op['total_duration']/1000):<12}"
                )
        
        # Operation impact
        op_impact = data.get('operation_impact', [])
        if op_impact:
            lines.append("\n🎯 OPERATION IMPACT (Features Generated):")
            lines.append("-" * 60)
            lines.append(f"{'Operation':<25} {'Features':<10} {'Avg Impact':<15} {'Total Impact':<15}")
            lines.append("-" * 60)
            
            for op in op_impact[:15]:  # Top 15 by impact
                lines.append(
                    f"{op['operation']:<25} "
                    f"{format_number(op['features']):<10} "
                    f"{op['avg_impact']:<15.5f} "
                    f"{op['total_impact']:<15.5f}"
                )
        
        # Efficiency analysis
        if op_metrics and op_impact:
            lines.append("\n💡 EFFICIENCY ANALYSIS:")
            
            # Find operations that are fast and impactful
            efficient_ops = []
            for metric in op_metrics:
                op_name = metric['operation_type']
                impact_data = next((x for x in op_impact if x['operation'] == op_name), None)
                
                if impact_data and metric['avg_duration'] > 0:
                    efficiency = impact_data['avg_impact'] / (metric['avg_duration'] / 1000)
                    efficient_ops.append({
                        'operation': op_name,
                        'efficiency': efficiency,
                        'impact': impact_data['avg_impact'],
                        'time': metric['avg_duration'] / 1000
                    })
            
            # Sort by efficiency
            efficient_ops.sort(key=lambda x: x['efficiency'], reverse=True)
            
            lines.append("   Most Efficient Operations (Impact/Time):")
            for i, op in enumerate(efficient_ops[:5], 1):
                lines.append(
                    f"   {i}. {op['operation']}: "
                    f"{op['efficiency']:.3f} "
                    f"(impact: {op['impact']:.5f}, time: {op['time']:.2f}s)"
                )
        
        return '\n'.join(lines)