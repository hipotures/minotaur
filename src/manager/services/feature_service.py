"""
Service layer for feature-related business logic
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from ..repositories.feature_repository import FeatureRepository
from ..core.utils import format_number, format_percentage


class FeatureService:
    """Handles feature-related business logic."""
    
    def __init__(self, feature_repository: FeatureRepository):
        """Initialize service with repository.
        
        Args:
            feature_repository: Feature repository instance
        """
        self.repository = feature_repository
        self.logger = logging.getLogger(__name__)
    
    def get_top_features(self, limit: int = 20, 
                        min_sessions: int = 3,
                        metric: str = 'avg_impact') -> List[Dict[str, Any]]:
        """Get top performing features.
        
        Args:
            limit: Maximum number of features to return
            min_sessions: Minimum sessions for inclusion
            metric: Ranking metric
            
        Returns:
            List of top feature dictionaries
        """
        return self.repository.get_feature_rankings(
            metric=metric,
            min_sessions=min_sessions,
            limit=limit
        )
    
    def analyze_feature(self, feature_name: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific feature.
        
        Args:
            feature_name: Name of the feature to analyze
            
        Returns:
            Feature analysis data
        """
        feature_data = self.repository.get_feature_by_name(feature_name)
        
        if not feature_data:
            return {'error': f'Feature {feature_name} not found'}
        
        # Get feature combinations
        combinations = self.repository.get_feature_combinations(feature_name, limit=10)
        
        # Format the analysis
        analysis = {
            'feature_name': feature_name,
            'performance': {
                'avg_impact': feature_data['statistics']['avg_impact'],
                'max_impact': feature_data['statistics']['max_impact'],
                'min_impact': feature_data['statistics']['min_impact'],
                'success_rate': format_percentage(feature_data['statistics']['success_rate']),
                'total_uses': format_number(feature_data['statistics']['total_uses']),
                'session_count': format_number(feature_data['statistics']['session_count'])
            },
            'best_combinations': [
                {
                    'paired_feature': combo['paired_feature'],
                    'avg_impact': combo['avg_impact'],
                    'co_occurrences': combo['co_occurrences']
                }
                for combo in combinations
            ],
            'recent_uses': feature_data['recent_uses'][:5]  # Last 5 uses
        }
        
        # Add trend analysis
        if len(feature_data['recent_uses']) >= 2:
            recent_impacts = [use['impact_score'] for use in feature_data['recent_uses'][:10]]
            older_impacts = [use['impact_score'] for use in feature_data['recent_uses'][10:20]]
            
            if older_impacts:
                recent_avg = sum(recent_impacts) / len(recent_impacts)
                older_avg = sum(older_impacts) / len(older_impacts)
                trend = 'improving' if recent_avg > older_avg else 'declining'
            else:
                trend = 'stable'
            
            analysis['trend'] = {
                'direction': trend,
                'recent_avg': sum(recent_impacts) / len(recent_impacts)
            }
        
        return analysis
    
    def get_feature_impact_matrix(self, limit: int = 20) -> Dict[str, Any]:
        """Get feature performance matrix across datasets.
        
        Args:
            limit: Maximum features to include
            
        Returns:
            Feature impact matrix data
        """
        return self.repository.get_feature_performance_matrix(limit)
    
    def search_features(self, pattern: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for features matching a pattern.
        
        Args:
            pattern: Search pattern (supports wildcards)
            limit: Maximum results
            
        Returns:
            List of matching features
        """
        all_features = self.repository.get_all_features(limit=limit)
        
        # Convert pattern to regex-friendly format
        import re
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        regex = re.compile(regex_pattern, re.IGNORECASE)
        
        matching_features = [
            f for f in all_features
            if regex.search(f['feature_name'])
        ]
        
        return matching_features
    
    def get_feature_statistics_by_category(self) -> Dict[str, Any]:
        """Get feature statistics grouped by category.
        
        Returns:
            Statistics grouped by feature category
        """
        all_features = self.repository.get_all_features()
        
        # Group features by category (assuming category is part of feature name)
        categories = {}
        
        for feature in all_features:
            # Extract category from feature name (e.g., "stats_mean_x" -> "stats")
            parts = feature['feature_name'].split('_')
            category = parts[0] if parts else 'other'
            
            if category not in categories:
                categories[category] = {
                    'features': [],
                    'total_uses': 0,
                    'total_impact': 0,
                    'positive_count': 0
                }
            
            categories[category]['features'].append(feature)
            categories[category]['total_uses'] += feature['total_uses']
            categories[category]['total_impact'] += feature['avg_impact'] * feature['total_uses']
            if feature['avg_impact'] > 0:
                categories[category]['positive_count'] += 1
        
        # Calculate category statistics
        category_stats = []
        for category, data in categories.items():
            feature_count = len(data['features'])
            if feature_count > 0:
                avg_impact = data['total_impact'] / max(1, data['total_uses'])
                success_rate = data['positive_count'] / feature_count
                
                category_stats.append({
                    'category': category,
                    'feature_count': feature_count,
                    'total_uses': data['total_uses'],
                    'avg_impact': avg_impact,
                    'success_rate': success_rate,
                    'top_feature': max(data['features'], 
                                     key=lambda x: x['avg_impact'])['feature_name']
                })
        
        # Sort by average impact
        category_stats.sort(key=lambda x: x['avg_impact'], reverse=True)
        
        return {
            'categories': category_stats,
            'summary': {
                'total_categories': len(category_stats),
                'best_category': category_stats[0]['category'] if category_stats else None,
                'worst_category': category_stats[-1]['category'] if category_stats else None
            }
        }
    
    def get_feature_recommendations(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get feature recommendations based on session performance.
        
        Args:
            session_id: Session ID to analyze
            limit: Maximum recommendations
            
        Returns:
            List of recommended features
        """
        # Get features used in this session
        session_features = self.repository.get_features_for_session(session_id)
        used_features = {f['feature_name'] for f in session_features}
        
        # Get top features not used in this session
        all_top_features = self.repository.get_feature_rankings(limit=50)
        
        recommendations = []
        for feature in all_top_features:
            if feature['feature_name'] not in used_features:
                # Check if this feature works well with any features in the session
                combinations = self.repository.get_feature_combinations(
                    feature['feature_name'], 
                    limit=5
                )
                
                synergy_score = 0
                for combo in combinations:
                    if combo['paired_feature'] in used_features:
                        synergy_score += combo['avg_impact']
                
                recommendations.append({
                    'feature_name': feature['feature_name'],
                    'avg_impact': feature['avg_impact'],
                    'success_rate': feature['success_rate'],
                    'synergy_score': synergy_score,
                    'recommendation_score': feature['avg_impact'] + synergy_score * 0.5
                })
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return recommendations[:limit]