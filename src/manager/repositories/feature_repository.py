"""
Repository for feature-related database operations
"""

from typing import Dict, List, Optional, Any, Tuple
from .base import BaseRepository


class FeatureRepository(BaseRepository):
    """Handles all feature-related database operations."""
    
    def get_all_features(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all features with their aggregate statistics.
        
        Args:
            limit: Maximum number of features to return
            
        Returns:
            List of feature dictionaries with statistics
        """
        query = """
        WITH feature_stats AS (
            SELECT 
                fc.feature_name,
                fc.feature_category,
                fc.description,
                COUNT(DISTINCT fi.session_id) as session_count,
                AVG(fi.impact_delta) as avg_impact,
                MAX(fi.impact_delta) as max_impact,
                SUM(CASE WHEN fi.impact_delta > 0 THEN 1 ELSE 0 END) as positive_impacts,
                COUNT(fi.id) as total_uses,
                MAX(fi.last_evaluated) as last_used
            FROM feature_catalog fc
            LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
            WHERE fc.is_active = true
            GROUP BY fc.feature_name, fc.feature_category, fc.description
        )
        SELECT 
            feature_name,
            feature_category,
            description,
            session_count,
            avg_impact,
            max_impact,
            positive_impacts,
            total_uses,
            CAST(positive_impacts AS FLOAT) / NULLIF(total_uses, 0) as success_rate,
            last_used
        FROM feature_stats
        ORDER BY avg_impact DESC NULLS LAST
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.fetch_all(query)
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature dictionary with detailed statistics or None
        """
        # Get feature catalog info
        catalog_query = """
        SELECT 
            feature_name,
            feature_category,
            python_code,
            dependencies,
            description,
            created_by,
            creation_timestamp,
            computational_cost,
            data_type
        FROM feature_catalog
        WHERE feature_name = ? AND is_active = true
        """
        
        catalog = self.fetch_one(catalog_query, (feature_name,))
        
        if not catalog:
            return None
        
        # Get aggregate statistics
        stats_query = """
        SELECT 
            COUNT(DISTINCT session_id) as session_count,
            AVG(impact_delta) as avg_impact,
            MAX(impact_delta) as max_impact,
            MIN(impact_delta) as min_impact,
            AVG(impact_percentage) as avg_impact_pct,
            SUM(CASE WHEN impact_delta > 0 THEN 1 ELSE 0 END) as positive_impacts,
            COUNT(*) as total_uses
        FROM feature_impact
        WHERE feature_name = ?
        """
        
        stats = self.fetch_one(stats_query, (feature_name,))
        
        # Get recent uses
        recent_query = """
        SELECT 
            fi.session_id,
            fi.impact_delta,
            fi.impact_percentage,
            fi.baseline_score,
            fi.with_feature_score,
            fi.last_evaluated,
            s.session_name,
            s.dataset_hash
        FROM feature_impact fi
        JOIN sessions s ON fi.session_id = s.session_id
        WHERE fi.feature_name = ?
        ORDER BY fi.last_evaluated DESC
        LIMIT 10
        """
        
        recent_uses = self.fetch_all(recent_query, (feature_name,))
        
        result = {
            'feature_name': catalog['feature_name'],
            'category': catalog['feature_category'],
            'description': catalog['description'],
            'python_code': catalog['python_code'],
            'dependencies': catalog['dependencies'],
            'created_by': catalog['created_by'],
            'creation_timestamp': catalog['creation_timestamp'],
            'computational_cost': catalog['computational_cost'],
            'data_type': catalog['data_type'],
            'statistics': {
                'session_count': stats['session_count'] if stats else 0,
                'avg_impact': stats['avg_impact'] if stats else 0,
                'max_impact': stats['max_impact'] if stats else 0,
                'min_impact': stats['min_impact'] if stats else 0,
                'avg_impact_pct': stats['avg_impact_pct'] if stats else 0,
                'positive_impacts': stats['positive_impacts'] if stats else 0,
                'total_uses': stats['total_uses'] if stats else 0,
                'success_rate': (stats['positive_impacts'] / stats['total_uses']) if stats and stats['total_uses'] > 0 else 0
            },
            'recent_uses': recent_uses
        }
        
        return result
    
    def get_features_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all features used in a specific session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of feature dictionaries
        """
        query = """
        SELECT 
            fi.feature_name,
            fi.impact_delta,
            fi.impact_percentage,
            fi.baseline_score,
            fi.with_feature_score,
            fi.first_discovered,
            fi.last_evaluated,
            fc.feature_category,
            fc.description
        FROM feature_impact fi
        JOIN feature_catalog fc ON fi.feature_name = fc.feature_name
        WHERE fi.session_id = ?
        ORDER BY fi.impact_delta DESC
        """
        
        return self.fetch_all(query, (session_id,))
    
    def get_feature_rankings(self, metric: str = 'avg_impact', 
                           min_sessions: int = 3,
                           limit: int = 20) -> List[Dict[str, Any]]:
        """Get top features ranked by specified metric.
        
        Args:
            metric: Ranking metric ('avg_impact', 'total_impact', 'success_rate')
            min_sessions: Minimum number of sessions for inclusion
            limit: Maximum number of features to return
            
        Returns:
            List of ranked feature dictionaries
        """
        order_by = {
            'avg_impact': 'avg_impact DESC',
            'total_impact': 'total_impact DESC',
            'success_rate': 'success_rate DESC',
            'usage_count': 'total_uses DESC'
        }.get(metric, 'avg_impact DESC')
        
        query = f"""
        WITH feature_stats AS (
            SELECT 
                fi.feature_name,
                fc.feature_category,
                fc.description,
                COUNT(DISTINCT fi.session_id) as session_count,
                AVG(fi.impact_delta) as avg_impact,
                SUM(fi.impact_delta) as total_impact,
                COUNT(*) as total_uses,
                SUM(CASE WHEN fi.impact_delta > 0 THEN 1 ELSE 0 END) as positive_uses
            FROM feature_impact fi
            JOIN feature_catalog fc ON fi.feature_name = fc.feature_name
            WHERE fc.is_active = true
            GROUP BY fi.feature_name, fc.feature_category, fc.description
            HAVING COUNT(DISTINCT fi.session_id) >= ?
        )
        SELECT 
            feature_name,
            feature_category,
            description,
            session_count,
            avg_impact,
            total_impact,
            total_uses,
            CAST(positive_uses AS FLOAT) / total_uses as success_rate
        FROM feature_stats
        ORDER BY {order_by}
        LIMIT ?
        """
        
        rows = self.fetch_all(query, (min_sessions, limit))
        
        results = []
        for i, row in enumerate(rows):
            results.append({
                'rank': i + 1,
                'feature_name': row['feature_name'],
                'category': row['feature_category'],
                'description': row['description'],
                'session_count': row['session_count'],
                'avg_impact': row['avg_impact'],
                'total_impact': row['total_impact'],
                'total_uses': row['total_uses'],
                'success_rate': row['success_rate']
            })
        
        return results
    
    def get_feature_combinations(self, feature_name: str, 
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Get features that work well with the specified feature.
        
        Args:
            feature_name: Base feature name
            limit: Maximum combinations to return
            
        Returns:
            List of feature combination statistics
        """
        query = """
        WITH base_sessions AS (
            SELECT DISTINCT session_id
            FROM feature_impact
            WHERE feature_name = ?
        ),
        combination_stats AS (
            SELECT 
                fi.feature_name as paired_feature,
                fc.feature_category,
                COUNT(DISTINCT fi.session_id) as session_count,
                AVG(fi.impact_delta) as avg_impact,
                COUNT(*) as co_occurrences
            FROM feature_impact fi
            JOIN base_sessions bs ON fi.session_id = bs.session_id
            JOIN feature_catalog fc ON fi.feature_name = fc.feature_name
            WHERE fi.feature_name != ?
            GROUP BY fi.feature_name, fc.feature_category
        )
        SELECT 
            paired_feature,
            feature_category,
            session_count,
            avg_impact,
            co_occurrences
        FROM combination_stats
        ORDER BY avg_impact DESC
        LIMIT ?
        """
        
        return self.fetch_all(query, (feature_name, feature_name, limit))
    
    def get_feature_catalog_stats(self) -> Dict[str, Any]:
        """Get statistics about the feature catalog.
        
        Returns:
            Dictionary of catalog statistics
        """
        query = """
        SELECT 
            COUNT(*) as total_features,
            COUNT(CASE WHEN is_active = true THEN 1 END) as active_features,
            COUNT(DISTINCT feature_category) as categories,
            AVG(computational_cost) as avg_cost
        FROM feature_catalog
        """
        
        stats = self.fetch_one(query)
        
        # Get category breakdown
        category_query = """
        SELECT 
            feature_category,
            COUNT(*) as count
        FROM feature_catalog
        WHERE is_active = true
        GROUP BY feature_category
        ORDER BY count DESC
        """
        
        categories = self.fetch_all(category_query)
        
        return {
            'total_features': stats['total_features'],
            'active_features': stats['active_features'],
            'categories': stats['categories'],
            'avg_computational_cost': stats['avg_cost'],
            'category_breakdown': categories
        }