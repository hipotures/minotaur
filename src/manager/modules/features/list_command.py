"""
List Command - List all features with performance metrics and filtering.

Provides comprehensive feature listing including:
- All features with usage statistics
- Advanced filtering capabilities
- Performance metrics and impact analysis
- Multiple output formats
"""

from typing import Dict, Any, List
from .base import BaseFeaturesCommand


class ListCommand(BaseFeaturesCommand):
    """Handle --list command for features."""
    
    def execute(self, args) -> None:
        """Execute the list features command."""
        try:
            # Build filters
            filters = self.build_feature_filters(args)
            
            # Get features from repository
            features = self._get_features(filters)
            
            if not features:
                self.print_info("No features found matching the criteria.")
                self.print_info("Try: python manager.py features --catalog")
                return
            
            # Output results
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(features, filters)
            else:
                self._output_table(features, filters)
                
        except Exception as e:
            self.print_error(f"Failed to list features: {e}")
    
    def _get_features(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get features from database with filters."""
        try:
            query = """
            SELECT 
                fc.feature_name,
                fc.category,
                fc.operation,
                COUNT(fi.feature_name) as usage_count,
                AVG(fi.impact) as avg_impact,
                MAX(fi.impact) as max_impact,
                MIN(fi.impact) as min_impact
            FROM feature_catalog fc
            LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
            """
            
            conditions = []
            params = []
            
            # Apply filters (same logic as top_command)
            if 'session_id' in filters:
                conditions.append("fi.session_id = ?")
                params.append(filters['session_id'])
            
            if 'category' in filters:
                conditions.append("fc.category = ?")
                params.append(filters['category'])
            
            if 'dataset_hash' in filters:
                conditions.append("fi.dataset_hash = ?")
                params.append(filters['dataset_hash'])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += """
            GROUP BY fc.feature_name, fc.category, fc.operation
            ORDER BY fc.category, fc.feature_name
            """
            
            if 'limit' in filters:
                query += f" LIMIT {filters['limit']}"
            
            return self.feature_service.repository.fetch_all(query, tuple(params))
            
        except Exception as e:
            self.print_error(f"Database query failed: {e}")
            return []
    
    def _output_table(self, features: List[Dict[str, Any]], filters: Dict[str, Any]) -> None:
        """Output features in table format."""
        headers = ['Feature Name', 'Category', 'Operation', 'Usage', 'Avg Impact']
        rows = []
        
        for feature in features:
            rows.append([
                self.truncate_text(feature.get('feature_name', ''), 30),
                feature.get('category', 'Unknown'),
                self.truncate_text(feature.get('operation', ''), 15),
                str(feature.get('usage_count', 0)),
                self.format_impact(feature.get('avg_impact'))
            ])
        
        self.print_table(headers, rows, f"Features ({len(features)} total)")
        
        # Show summary by category
        categories = {}
        for feature in features:
            cat = feature.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            print(f"\n📊 By Category:")
            for cat, count in sorted(categories.items()):
                print(f"   {cat}: {count} features")
    
    def _output_json(self, features: List[Dict[str, Any]], filters: Dict[str, Any]) -> None:
        """Output features in JSON format."""
        output = {
            'features': features,
            'filters': filters,
            'count': len(features)
        }
        self.print_json(output, "Features List")