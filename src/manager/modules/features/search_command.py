"""
Search Command - Search features by name, category, or description.

Provides flexible feature search including:
- Name-based search (exact and partial matching)
- Category-based filtering
- Description text search
- Operation-based search
"""

from typing import Dict, Any, List
from .base import BaseFeaturesCommand


class SearchCommand(BaseFeaturesCommand):
    """Handle --search command for features."""
    
    def execute(self, args) -> None:
        """Execute the search features command."""
        try:
            query = args.search
            if not query:
                self.print_error("Search query is required.")
                return
            
            # Perform search
            results = self._search_features(query)
            
            if not results:
                self.print_info(f"No features found matching '{query}'.")
                self.print_info("Try a different search term or:")
                self.print_info("  python manager.py features --catalog")
                return
            
            # Output results
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(query, results)
            else:
                self._output_search_results(query, results)
                
        except Exception as e:
            self.print_error(f"Failed to search features: {e}")
    
    def _search_features(self, query: str) -> List[Dict[str, Any]]:
        """Search features using multiple criteria."""
        try:
            search_query = """
            SELECT 
                fc.feature_name,
                fc.category,
                fc.operation,
                fc.description,
                COUNT(fi.feature_name) as usage_count,
                AVG(fi.impact) as avg_impact
            FROM feature_catalog fc
            LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
            WHERE 
                fc.feature_name ILIKE ? OR
                fc.category ILIKE ? OR 
                fc.operation ILIKE ? OR
                fc.description ILIKE ?
            GROUP BY fc.feature_name, fc.category, fc.operation, fc.description
            ORDER BY 
                CASE 
                    WHEN fc.feature_name ILIKE ? THEN 1
                    WHEN fc.category ILIKE ? THEN 2
                    WHEN fc.operation ILIKE ? THEN 3
                    ELSE 4
                END,
                AVG(fi.impact) DESC NULLS LAST
            """
            
            # Prepare search parameters
            like_query = f"%{query}%"
            exact_query = query
            params = (like_query, like_query, like_query, like_query, 
                     exact_query, exact_query, exact_query)
            
            return self.feature_service.repository.fetch_all(search_query, params)
            
        except Exception as e:
            self.print_error(f"Search query failed: {e}")
            return []
    
    def _output_search_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Output search results in formatted view."""
        print(f"\n🔍 SEARCH RESULTS FOR: '{query}'")
        print("=" * 60)
        print(f"📊 Found {len(results)} matching feature(s)")
        
        # Show results
        headers = ['Feature Name', 'Category', 'Operation', 'Usage', 'Avg Impact']
        rows = []
        
        for result in results:
            rows.append([
                self.truncate_text(result.get('feature_name', ''), 25),
                result.get('category', 'Unknown'),
                self.truncate_text(result.get('operation', ''), 15),
                str(result.get('usage_count', 0)),
                self.format_impact(result.get('avg_impact'))
            ])
        
        self.print_table(headers, rows)
        
        # Show match summary by category
        categories = {}
        for result in results:
            cat = result.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        if len(categories) > 1:
            print(f"\n📂 Matches by Category:")
            for cat, count in sorted(categories.items()):
                print(f"   {cat}: {count} features")
        
        # Show quick actions
        if results:
            print(f"\n💡 Quick Actions:")
            
            # Best match
            best_feature = results[0]['feature_name']
            print(f"   Analyze best match: python manager.py features --impact {best_feature}")
            
            # Category filter if multiple categories
            if len(categories) > 1:
                top_category = max(categories.items(), key=lambda x: x[1])[0]
                print(f"   Filter by top category: python manager.py features --list --category {top_category}")
            
            # Refine search
            print(f"   Refine search: python manager.py features --search 'more specific term'")
    
    def _output_json(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Output search results in JSON format."""
        # Group results by category
        by_category = {}
        for result in results:
            cat = result.get('category', 'Unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)
        
        output = {
            'search_query': query,
            'results': results,
            'summary': {
                'total_matches': len(results),
                'categories_found': list(by_category.keys()),
                'by_category': {cat: len(features) for cat, features in by_category.items()}
            }
        }
        
        self.print_json(output, f"Search Results for: {query}")