"""
Impact Command - Show detailed impact analysis for specific features.

Provides comprehensive impact analysis including:
- Feature performance across sessions
- Impact distribution and statistics
- Session-by-session breakdown
- Performance trends over time
"""

from typing import Dict, Any, List
from .base import BaseFeaturesCommand


class ImpactCommand(BaseFeaturesCommand):
    """Handle --impact command for features."""
    
    def execute(self, args) -> None:
        """Execute the feature impact analysis command."""
        try:
            feature_name = args.impact
            if not feature_name:
                self.print_error("Feature name is required for impact analysis.")
                return
            
            # Get feature impact data
            impact_data = self._get_feature_impact(feature_name)
            
            if not impact_data:
                self.print_error(f"Feature '{feature_name}' not found or has no impact data.")
                self.print_info("List features: python manager.py features --list")
                return
            
            # Output results
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(impact_data, feature_name)
            else:
                self._output_detailed_analysis(impact_data, feature_name)
                
        except Exception as e:
            self.print_error(f"Failed to analyze feature impact: {e}")
    
    def _get_feature_impact(self, feature_name: str) -> List[Dict[str, Any]]:
        """Get detailed impact data for a specific feature."""
        try:
            query = """
            SELECT 
                fi.session_id,
                fi.dataset_hash,
                fi.impact,
                fi.success_count,
                fi.failure_count,
                s.start_time,
                s.status,
                s.best_score
            FROM feature_impact fi
            LEFT JOIN sessions s ON fi.session_id = s.session_id
            WHERE fi.feature_name = ?
            ORDER BY fi.impact DESC
            """
            
            return self.feature_service.repository.fetch_all(query, (feature_name,))
            
        except Exception as e:
            self.print_error(f"Database query failed: {e}")
            return []
    
    def _output_detailed_analysis(self, impact_data: List[Dict[str, Any]], feature_name: str) -> None:
        """Output detailed impact analysis in formatted view."""
        print(f"\n🧪 FEATURE IMPACT ANALYSIS")
        print("=" * 60)
        print(f"📋 Feature: {feature_name}")
        print(f"📊 Total Sessions: {len(impact_data)}")
        
        if not impact_data:
            return
        
        # Calculate statistics
        impacts = [d.get('impact', 0) for d in impact_data]
        avg_impact = sum(impacts) / len(impacts)
        max_impact = max(impacts)
        min_impact = min(impacts)
        positive_count = len([i for i in impacts if i > 0])
        negative_count = len([i for i in impacts if i < 0])
        
        print(f"\n📈 IMPACT STATISTICS")
        print("-" * 40)
        print(f"📊 Average Impact: {avg_impact:+.5f}")
        print(f"🏆 Best Impact: {max_impact:+.5f}")
        print(f"📉 Worst Impact: {min_impact:+.5f}")
        print(f"✅ Positive Sessions: {positive_count} ({positive_count/len(impacts)*100:.1f}%)")
        print(f"❌ Negative Sessions: {negative_count} ({negative_count/len(impacts)*100:.1f}%)")
        
        # Show top sessions
        print(f"\n🎯 TOP PERFORMING SESSIONS")
        print("-" * 60)
        headers = ['Session ID', 'Impact', 'Dataset', 'Score', 'Status']
        rows = []
        
        for session in impact_data[:10]:  # Top 10
            rows.append([
                session.get('session_id', '')[:8],
                self.format_impact(session.get('impact')),
                session.get('dataset_hash', '')[:8],
                f"{session.get('best_score', 0):.3f}" if session.get('best_score') else "N/A",
                session.get('status', 'Unknown')
            ])
        
        self.print_table(headers, rows)
        
        # Quick actions
        print(f"\n💡 Quick Actions:")
        if impact_data:
            best_session = impact_data[0]['session_id'][:8]
            print(f"   View best session: python manager.py sessions --show {best_session}")
        print(f"   Compare features: python manager.py features --top 10")
        print(f"   Feature catalog: python manager.py features --catalog")
    
    def _output_json(self, impact_data: List[Dict[str, Any]], feature_name: str) -> None:
        """Output impact analysis in JSON format."""
        # Calculate statistics
        impacts = [d.get('impact', 0) for d in impact_data]
        
        output = {
            'feature_name': feature_name,
            'impact_data': impact_data,
            'statistics': {
                'total_sessions': len(impact_data),
                'average_impact': sum(impacts) / len(impacts) if impacts else 0,
                'max_impact': max(impacts) if impacts else 0,
                'min_impact': min(impacts) if impacts else 0,
                'positive_count': len([i for i in impacts if i > 0]),
                'negative_count': len([i for i in impacts if i < 0])
            }
        }
        
        self.print_json(output, f"Impact Analysis: {feature_name}")