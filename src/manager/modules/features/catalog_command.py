"""
Catalog Command - Show feature catalog summary with statistics.

Provides feature catalog overview including:
- Total feature counts by category  
- Usage statistics and performance summaries
- Feature operation breakdowns
- Overall system feature health metrics
"""

from typing import Dict, Any, List
from .base import BaseFeaturesCommand


class CatalogCommand(BaseFeaturesCommand):
    """Handle --catalog command for features."""
    
    def execute(self, args) -> None:
        """Execute the feature catalog summary command."""
        try:
            # Get catalog statistics
            catalog_stats = self._get_catalog_statistics()
            
            if not catalog_stats:
                self.print_info("No feature catalog data available.")
                self.print_info("Features are created during MCTS sessions.")
                return
            
            # Output results
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(catalog_stats)
            else:
                self._output_catalog_summary(catalog_stats)
                
        except Exception as e:
            self.print_error(f"Failed to show feature catalog: {e}")
    
    def _get_catalog_statistics(self) -> Dict[str, Any]:
        """Get comprehensive catalog statistics."""
        try:
            stats = {}
            
            # Total features
            total_query = "SELECT COUNT(*) as total FROM feature_catalog"
            total_result = self.feature_service.repository.fetch_one(total_query)
            stats['total_features'] = total_result.get('total', 0) if total_result else 0
            
            # Features by category
            category_query = """
            SELECT feature_category as category, COUNT(*) as count 
            FROM feature_catalog 
            GROUP BY feature_category 
            ORDER BY count DESC
            """
            stats['by_category'] = self.feature_service.repository.fetch_all(category_query)
            
            # Features by data type
            datatype_query = """
            SELECT data_type, COUNT(*) as count 
            FROM feature_catalog 
            GROUP BY data_type 
            ORDER BY count DESC
            """
            stats['by_data_type'] = self.feature_service.repository.fetch_all(datatype_query)
            
            # Usage statistics
            usage_query = """
            SELECT 
                COUNT(DISTINCT fi.feature_name) as used_features,
                COUNT(*) as total_usages,
                AVG(fi.impact_delta) as avg_impact,
                SUM(CASE WHEN fi.impact_delta > 0 THEN 1 ELSE 0 END) as positive_impacts
            FROM feature_impact fi
            """
            usage_result = self.feature_service.repository.fetch_one(usage_query)
            stats['usage'] = usage_result if usage_result else {}
            
            # Top performing features
            top_query = """
            SELECT 
                fc.feature_name,
                fc.feature_category as category,
                AVG(fi.impact_delta) as avg_impact,
                COUNT(*) as usage_count
            FROM feature_catalog fc
            JOIN feature_impact fi ON fc.feature_name = fi.feature_name
            GROUP BY fc.feature_name, fc.feature_category
            ORDER BY AVG(fi.impact_delta) DESC
            LIMIT 5
            """
            stats['top_features'] = self.feature_service.repository.fetch_all(top_query)
            
            return stats
            
        except Exception as e:
            self.print_error(f"Failed to gather catalog statistics: {e}")
            return {}
    
    def _output_catalog_summary(self, stats: Dict[str, Any]) -> None:
        """Output catalog summary in formatted view."""
        print(f"\n📚 FEATURE CATALOG SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        total_features = stats.get('total_features', 0)
        usage_stats = stats.get('usage', {})
        used_features = usage_stats.get('used_features', 0)
        total_usages = usage_stats.get('total_usages', 0)
        avg_impact = usage_stats.get('avg_impact', 0)
        positive_impacts = usage_stats.get('positive_impacts', 0)
        
        print(f"📊 Total Features: {total_features}")
        print(f"🧪 Used Features: {used_features}")
        print(f"📈 Total Usages: {total_usages}")
        
        if total_usages > 0:
            print(f"📊 Average Impact: {avg_impact:+.5f}")
            success_rate = (positive_impacts / total_usages * 100) if total_usages > 0 else 0
            print(f"✅ Success Rate: {success_rate:.1f}%")
        
        # Features by category
        by_category = stats.get('by_category', [])
        if by_category:
            print(f"\n📂 BY CATEGORY")
            print("-" * 40)
            for cat in by_category:
                category = cat.get('category', 'Unknown')
                count = cat.get('count', 0)
                percentage = (count / total_features * 100) if total_features > 0 else 0
                print(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Features by data type
        by_data_type = stats.get('by_data_type', [])
        if by_data_type:
            print(f"\n🔧 BY DATA TYPE")
            print("-" * 40)
            for dt in by_data_type[:10]:  # Top 10
                data_type = dt.get('data_type', 'Unknown')
                count = dt.get('count', 0)
                print(f"   {data_type}: {count}")
        
        # Top performing features
        top_features = stats.get('top_features', [])
        if top_features:
            print(f"\n🏆 TOP PERFORMING FEATURES")
            print("-" * 50)
            headers = ['Feature', 'Category', 'Avg Impact', 'Usage']
            rows = []
            
            for feature in top_features:
                rows.append([
                    self.truncate_text(feature.get('feature_name', ''), 20),
                    feature.get('category', ''),
                    self.format_impact(feature.get('avg_impact')),
                    str(feature.get('usage_count', 0))
                ])
            
            self.print_table(headers, rows)
        
        # Quick actions
        print(f"\n💡 Quick Actions:")
        print(f"   View all features: python manager.py features --list")
        print(f"   Top performers: python manager.py features --top 10")
        if by_category:
            top_category = by_category[0]['category']
            print(f"   Filter by category: python manager.py features --list --category {top_category}")
    
    def _output_json(self, stats: Dict[str, Any]) -> None:
        """Output catalog summary in JSON format."""
        self.print_json(stats, "Feature Catalog Summary")