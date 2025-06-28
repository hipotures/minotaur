"""
Search Command - Search datasets by name, path, or other criteria.

Provides flexible dataset search including:
- Name-based search (exact and partial matching)
- Path-based search
- Metadata search (description, competition)
- Fuzzy matching capabilities
"""

from typing import Dict, Any, List
from .base import BaseDatasetsCommand


class SearchCommand(BaseDatasetsCommand):
    """Handle --search command for datasets."""
    
    def execute(self, args) -> None:
        """Execute the search datasets command."""
        try:
            query = args.search
            if not query:
                self.print_error("Search query is required.")
                return
            
            # Get all datasets
            all_datasets = self.dataset_service.get_all_datasets()
            
            if not all_datasets:
                self.print_info("No datasets registered.")
                return
            
            # Perform search
            results = self._search_datasets(all_datasets, query)
            
            if not results:
                self.print_info(f"No datasets found matching '{query}'.")
                self.print_info("Try a different search term or list all datasets:")
                self.print_info("python manager.py datasets --list")
                return
            
            # Output results
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(query, results)
            else:
                self._output_formatted_results(query, results)
                
        except Exception as e:
            self.print_error(f"Failed to search datasets: {e}")
    
    def _search_datasets(self, datasets: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Search datasets using multiple criteria."""
        query_lower = query.lower()
        results = []
        
        for dataset in datasets:
            score = 0
            match_reasons = []
            
            # Name matching (highest priority)
            name = dataset.get('dataset_name', '').lower()
            if query_lower == name:
                score += 100  # Exact match
                match_reasons.append("exact name match")
            elif query_lower in name:
                score += 50   # Partial match
                match_reasons.append("name contains query")
            elif self._fuzzy_match(query_lower, name):
                score += 25   # Fuzzy match
                match_reasons.append("similar name")
            
            # ID matching
            dataset_id = dataset.get('dataset_id', '').lower()
            if query_lower in dataset_id:
                score += 75
                match_reasons.append("ID match")
            
            # Path matching
            for path_key in ['train_path', 'test_path', 'submission_path']:
                path = dataset.get(path_key, '')
                if path and query_lower in path.lower():
                    score += 30
                    match_reasons.append(f"{path_key} contains query")
                    break  # Don't double-count path matches
            
            # Description matching
            description = dataset.get('description', '').lower()
            if description and query_lower in description:
                score += 20
                match_reasons.append("description contains query")
            
            # Competition name matching
            competition = dataset.get('competition_name', '').lower()
            if competition and query_lower in competition:
                score += 40
                match_reasons.append("competition name match")
            
            # If any match found, add to results
            if score > 0:
                result = dataset.copy()
                result['_search_score'] = score
                result['_match_reasons'] = match_reasons
                results.append(result)
        
        # Sort by relevance score (highest first)
        results.sort(key=lambda r: r['_search_score'], reverse=True)
        
        return results
    
    def _fuzzy_match(self, query: str, target: str, threshold: float = 0.6) -> bool:
        """Simple fuzzy matching using character overlap."""
        if not query or not target:
            return False
        
        # Calculate character overlap
        query_chars = set(query)
        target_chars = set(target)
        
        if not query_chars:
            return False
        
        overlap = len(query_chars.intersection(target_chars))
        similarity = overlap / len(query_chars)
        
        return similarity >= threshold
    
    def _output_formatted_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Output search results in formatted view."""
        print(f"\n🔍 SEARCH RESULTS FOR: '{query}'")
        print("=" * 60)
        print(f"📊 Found {len(results)} matching dataset(s)")
        
        # Show results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📁 {result['dataset_name']}")
            print(f"   🔑 ID: {result['dataset_id'][:8]}")
            
            # Show match reasons
            reasons = result.get('_match_reasons', [])
            if reasons:
                print(f"   🎯 Match: {', '.join(reasons)}")
            
            # Show basic info
            if result.get('description'):
                desc = result['description'][:60]
                if len(result['description']) > 60:
                    desc += "..."
                print(f"   📝 Description: {desc}")
            
            if result.get('competition_name'):
                print(f"   🏆 Competition: {result['competition_name']}")
            
            # Show usage stats if available
            dataset_id = result['dataset_id']
            try:
                sessions = self.session_service.get_sessions_by_dataset(dataset_id)
                if sessions:
                    print(f"   📈 Sessions: {len(sessions)}")
            except Exception:
                pass  # Don't fail search if stats unavailable
        
        # Show quick actions
        if results:
            print(f"\n💡 QUICK ACTIONS")
            print("-" * 40)
            
            best_match = results[0]  # Highest scoring result
            dataset_name = best_match['dataset_name']
            
            print(f"   Show details: python manager.py datasets --details {dataset_name}")
            print(f"   List sessions: python manager.py datasets --sessions {dataset_name}")
            
            if len(results) == 1:
                print(f"   Start session: python mcts.py --config config/mcts_config.yaml")
            else:
                print(f"   Refine search: python manager.py datasets --search 'more specific term'")
    
    def _output_json(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Output search results in JSON format."""
        # Clean up internal search fields
        clean_results = []
        for result in results:
            clean_result = {k: v for k, v in result.items() 
                          if not k.startswith('_')}
            clean_result['search_score'] = result.get('_search_score', 0)
            clean_result['match_reasons'] = result.get('_match_reasons', [])
            clean_results.append(clean_result)
        
        output = {
            'query': query,
            'results': clean_results,
            'summary': {
                'total_matches': len(results),
                'best_match_score': results[0]['_search_score'] if results else 0
            }
        }
        
        self.print_json(output, f"Search Results for: {query}")