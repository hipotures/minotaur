"""
Top Command - Show top performing features.

Displays the highest impact features based on performance metrics:
- Top N features by average impact
- Performance statistics and usage counts
- Filtering by dataset, session, category
- Customizable result limits
"""

from typing import Dict, Any, List
from .base import BaseFeaturesCommand


class TopCommand(BaseFeaturesCommand):
    """Handle --top command for features."""
    
    def execute(self, args) -> None:
        """Execute the show top features command."""
        try:
            # Get number of features to show
            top_n = getattr(args, 'top_n', 10)
            if top_n <= 0:
                top_n = 10
            
            # Build filters
            filters = self.build_feature_filters(args)
            filters['limit'] = top_n  # Override limit with top_n
            
            # Get top features from repository
            features = self._get_top_features(filters)
            
            if not features:
                self.print_info("No features found matching the criteria.")
                self.print_info("Check filters or try: python manager.py features --catalog")
                return
            
            # Output results
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(features, top_n, filters)
            else:
                self._output_table(features, top_n, filters)
            
            # Save to file if requested
            if hasattr(args, 'output_file') and args.output_file:
                self._save_output(features, args.output_file, args.format)
                
        except Exception as e:
            self.print_error(f"Failed to show top features: {e}")
    
    def _get_top_features(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get top features from database."""
        try:
            # Build query based on filters
            query = """
            SELECT 
                fc.feature_name,
                fc.feature_category as category,
                fc.python_code,
                fc.dependencies,
                fc.description,
                COUNT(fi.feature_name) as usage_count,
                AVG(fi.impact_delta) as avg_impact,
                MAX(fi.impact_delta) as max_impact,
                MIN(fi.impact_delta) as min_impact,
                SUM(CASE WHEN fi.impact_delta > 0 THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN fi.impact_delta < 0 THEN 1 ELSE 0 END) as negative_count
            FROM feature_catalog fc
            LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
            """
            
            conditions = []
            params = []
            
            # Apply filters
            if 'session_id' in filters:
                conditions.append("fi.session_id = ?")
                params.append(filters['session_id'])
            
            if 'category' in filters:
                conditions.append("fc.feature_category = ?")
                params.append(filters['category'])
            
            if 'dataset_hash' in filters:
                conditions.append("fi.dataset_hash = ?")
                params.append(filters['dataset_hash'])
            
            if 'min_impact' in filters:
                conditions.append("fi.impact_delta >= ?")
                params.append(filters['min_impact'])
            
            # Add WHERE clause if there are conditions
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Group and order
            query += """
            GROUP BY fc.feature_name, fc.feature_category, fc.python_code, fc.dependencies, fc.description
            HAVING COUNT(fi.feature_name) > 0
            ORDER BY AVG(fi.impact_delta) DESC
            """
            
            # Add limit
            limit = filters.get('limit', 10)
            query += f" LIMIT {limit}"
            
            # Execute query
            return self.feature_service.repository.fetch_all(query, tuple(params))
            
        except Exception as e:
            self.print_error(f"Database query failed: {e}")
            return []
    
    def _output_table(self, features: List[Dict[str, Any]], top_n: int, filters: Dict[str, Any]) -> None:
        """Output top features in table format."""
        title = f"Top {len(features)} Features"
        if len(features) < top_n:
            title += f" (requested {top_n})"
        
        headers = ['Feature Name', 'Category', 'Usage', 'Avg Impact', 'Max Impact', 'Success Rate']
        rows = []
        
        for feature in features:
            # Calculate success rate
            usage_count = feature.get('usage_count', 0)
            positive_count = feature.get('positive_count', 0)
            success_rate = (positive_count / usage_count * 100) if usage_count > 0 else 0
            
            rows.append([
                self.truncate_text(feature.get('feature_name', ''), 25),
                feature.get('category', 'Unknown'),
                str(usage_count),
                self.format_impact(feature.get('avg_impact')),
                self.format_impact(feature.get('max_impact')),
                self.format_percentage(success_rate)
            ])
        
        self.print_table(headers, rows, title)
        
        # Show summary statistics
        if features:
            avg_usage = sum(f.get('usage_count', 0) for f in features) / len(features)
            avg_impact = sum(f.get('avg_impact', 0) for f in features) / len(features)
            best_impact = max(f.get('avg_impact', 0) for f in features)
            
            print(f"\n📈 Summary:")
            print(f"   Average Usage: {avg_usage:.1f} sessions")
            print(f"   Average Impact: {avg_impact:+.5f}")
            print(f"   Best Impact: {best_impact:+.5f}")
        
        # Show applied filters
        active_filters = []
        for key, value in filters.items():
            if key != 'limit' and value is not None:
                active_filters.append(f"{key}={value}")
        
        if active_filters:
            print(f"\n🔍 Applied Filters: {', '.join(active_filters)}")
        
        # Show quick actions
        print(f"\n💡 Quick Actions:")
        if features:
            best_feature = features[0]['feature_name']
            print(f"   Analyze best: python manager.py features --impact {best_feature}")
        print(f"   Show all: python manager.py features --list")
        print(f"   Search: python manager.py features --search TERM")
    
    def _output_json(self, features: List[Dict[str, Any]], top_n: int, filters: Dict[str, Any]) -> None:
        """Output top features in JSON format."""
        # Calculate additional statistics
        for feature in features:
            usage_count = feature.get('usage_count', 0)
            positive_count = feature.get('positive_count', 0)
            feature['success_rate'] = (positive_count / usage_count * 100) if usage_count > 0 else 0
        
        output = {
            'top_features': features,
            'request': {
                'top_n': top_n,
                'filters': filters,
                'results_count': len(features)
            },
            'summary': {
                'average_usage': sum(f.get('usage_count', 0) for f in features) / len(features) if features else 0,
                'average_impact': sum(f.get('avg_impact', 0) for f in features) / len(features) if features else 0,
                'best_impact': max(f.get('avg_impact', 0) for f in features) if features else 0,
                'total_features': len(features)
            }
        }
        
        self.print_json(output, f"Top {top_n} Features")
    
    def _save_output(self, features: List[Dict[str, Any]], filename: str, format_type: str) -> None:
        """Save output to file."""
        try:
            if format_type == 'json':
                import json
                with open(filename, 'w') as f:
                    json.dump({'top_features': features}, f, indent=2, default=str)
            else:
                # Default to CSV-like format
                content = "Feature Name,Category,Usage,Avg Impact,Max Impact\n"
                for feature in features:
                    content += f"{feature.get('feature_name', '')},{feature.get('category', '')},"
                    content += f"{feature.get('usage_count', 0)},{feature.get('avg_impact', 0)},"
                    content += f"{feature.get('max_impact', 0)}\n"
                
                with open(filename, 'w') as f:
                    f.write(content)
            
            self.print_success(f"Results saved to {filename}")
            
        except Exception as e:
            self.print_error(f"Failed to save results: {e}")