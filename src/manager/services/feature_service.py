"""
Service layer for feature-related business logic using SQLAlchemy abstraction layer.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from ..core.utils import format_number, format_percentage


class FeatureService:
    """Handles feature-related business logic using new database abstraction."""
    
    def __init__(self, db_manager):
        """Initialize service with database manager.
        
        Args:
            db_manager: Database manager instance from factory
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Ensure feature tables exist
        self._ensure_feature_tables()
        
        # Legacy compatibility - modules expect a repository attribute
        self.repository = self
    
    def _ensure_feature_tables(self):
        """Ensure feature-related tables exist."""
        self.logger.info("Creating feature tables if not exist...")
        
        # Feature catalog table
        catalog_table = """
        CREATE TABLE IF NOT EXISTS feature_catalog (
            feature_name VARCHAR PRIMARY KEY,
            feature_category VARCHAR,
            python_code TEXT,
            dependencies TEXT,
            description TEXT,
            created_by VARCHAR DEFAULT 'system',
            creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            computational_cost DOUBLE DEFAULT 1.0,
            data_type VARCHAR DEFAULT 'numeric',
            is_active BOOLEAN DEFAULT true
        )
        """
        
        # Feature impact table
        impact_table = """
        CREATE TABLE IF NOT EXISTS feature_impact (
            id INTEGER PRIMARY KEY,
            session_id VARCHAR NOT NULL,
            feature_name VARCHAR NOT NULL,
            impact_delta DOUBLE DEFAULT 0.0,
            impact_percentage DOUBLE DEFAULT 0.0,
            baseline_score DOUBLE DEFAULT 0.0,
            with_feature_score DOUBLE DEFAULT 0.0,
            first_discovered TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_evaluated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        self.db_manager.execute_ddl(catalog_table)
        self.db_manager.execute_ddl(impact_table)
        self.logger.info("Feature tables creation completed")
    
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
        return self.get_feature_rankings(
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
        feature_data = self.get_feature_by_name(feature_name)
        
        if not feature_data:
            return {'error': f'Feature {feature_name} not found'}
        
        # Get feature combinations
        combinations = self.get_feature_combinations(feature_name, limit=10)
        
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
        # Get top features across different datasets
        query = """
        WITH feature_dataset_stats AS (
            SELECT 
                fi.feature_name,
                s.dataset_name,
                AVG(fi.impact_delta) as avg_impact,
                COUNT(*) as usage_count
            FROM feature_impact fi
            JOIN sessions s ON fi.session_id = s.session_id
            GROUP BY fi.feature_name, s.dataset_name
        ),
        ranked_features AS (
            SELECT feature_name,
                   AVG(avg_impact) as overall_impact
            FROM feature_dataset_stats
            GROUP BY feature_name
            ORDER BY overall_impact DESC
            LIMIT :limit
        )
        SELECT 
            fds.feature_name,
            fds.dataset_name,
            fds.avg_impact,
            fds.usage_count
        FROM feature_dataset_stats fds
        JOIN ranked_features rf ON fds.feature_name = rf.feature_name
        ORDER BY rf.overall_impact DESC, fds.dataset_name
        """
        
        results = self.db_manager.execute_query(query, {'limit': limit})
        
        # Organize into matrix format
        matrix = {}
        datasets = set()
        
        for row in results:
            feature_name = row['feature_name']
            dataset_name = row['dataset_name']
            
            if feature_name not in matrix:
                matrix[feature_name] = {}
            
            matrix[feature_name][dataset_name] = {
                'avg_impact': row['avg_impact'],
                'usage_count': row['usage_count']
            }
            datasets.add(dataset_name)
        
        return {
            'matrix': matrix,
            'datasets': sorted(list(datasets)),
            'features': list(matrix.keys())
        }
    
    def search_features(self, pattern: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for features matching a pattern.
        
        Args:
            pattern: Search pattern (supports wildcards)
            limit: Maximum results
            
        Returns:
            List of matching features
        """
        # Convert wildcards to SQL LIKE pattern
        sql_pattern = pattern.replace('*', '%').replace('?', '_')
        
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
            WHERE fc.is_active = true AND fc.feature_name LIKE :pattern
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
        LIMIT :limit
        """
        
        return self.db_manager.execute_query(query, {'pattern': sql_pattern, 'limit': limit})
    
    def get_feature_statistics_by_category(self) -> Dict[str, Any]:
        """Get feature statistics grouped by category.
        
        Returns:
            Statistics grouped by feature category
        """
        all_features = self.get_all_features()
        
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
        session_features = self.get_features_for_session(session_id)
        used_features = {f['feature_name'] for f in session_features}
        
        # Get top features not used in this session
        all_top_features = self.get_feature_rankings(limit=50)
        
        recommendations = []
        for feature in all_top_features:
            if feature['feature_name'] not in used_features:
                # Check if this feature works well with any features in the session
                combinations = self.get_feature_combinations(
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
    
    # Core repository-style methods for database access
    def get_all_features(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all features with their aggregate statistics."""
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
        
        params = {}
        if limit:
            query += " LIMIT :limit"
            params['limit'] = limit
        
        return self.db_manager.execute_query(query, params)
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific feature."""
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
        WHERE feature_name = :feature_name AND is_active = true
        """
        
        catalog_results = self.db_manager.execute_query(catalog_query, {'feature_name': feature_name})
        if not catalog_results:
            return None
        
        catalog = catalog_results[0]
        
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
        WHERE feature_name = :feature_name
        """
        
        stats_results = self.db_manager.execute_query(stats_query, {'feature_name': feature_name})
        stats = stats_results[0] if stats_results else {}
        
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
            s.dataset_name
        FROM feature_impact fi
        JOIN sessions s ON fi.session_id = s.session_id
        WHERE fi.feature_name = :feature_name
        ORDER BY fi.last_evaluated DESC
        LIMIT 10
        """
        
        recent_uses = self.db_manager.execute_query(recent_query, {'feature_name': feature_name})
        
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
                'session_count': stats.get('session_count', 0),
                'avg_impact': stats.get('avg_impact', 0),
                'max_impact': stats.get('max_impact', 0),
                'min_impact': stats.get('min_impact', 0),
                'avg_impact_pct': stats.get('avg_impact_pct', 0),
                'positive_impacts': stats.get('positive_impacts', 0),
                'total_uses': stats.get('total_uses', 0),
                'success_rate': (stats.get('positive_impacts', 0) / stats.get('total_uses', 1)) if stats.get('total_uses', 0) > 0 else 0
            },
            'recent_uses': recent_uses
        }
        
        return result
    
    def get_features_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all features used in a specific session."""
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
        WHERE fi.session_id = :session_id
        ORDER BY fi.impact_delta DESC
        """
        
        return self.db_manager.execute_query(query, {'session_id': session_id})
    
    def get_feature_rankings(self, metric: str = 'avg_impact', 
                           min_sessions: int = 3,
                           limit: int = 20) -> List[Dict[str, Any]]:
        """Get top features ranked by specified metric."""
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
            HAVING COUNT(DISTINCT fi.session_id) >= :min_sessions
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
        LIMIT :limit
        """
        
        rows = self.db_manager.execute_query(query, {'min_sessions': min_sessions, 'limit': limit})
        
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
        """Get features that work well with the specified feature."""
        query = """
        WITH base_sessions AS (
            SELECT DISTINCT session_id
            FROM feature_impact
            WHERE feature_name = :feature_name1
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
            WHERE fi.feature_name != :feature_name2
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
        LIMIT :limit
        """
        
        return self.db_manager.execute_query(query, {
            'feature_name1': feature_name, 
            'feature_name2': feature_name, 
            'limit': limit
        })
    
    # Legacy repository compatibility methods
    def fetch_all(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Legacy compatibility method for raw SQL queries."""
        # Convert list params to dict for SQLAlchemy
        if params:
            # Create numbered parameter placeholders
            sql_params = {}
            for i, param in enumerate(params):
                sql_params[f'param_{i}'] = param
            
            # Replace ? placeholders with :param_n
            formatted_query = query
            for i in range(len(params)):
                formatted_query = formatted_query.replace('?', f':param_{i}', 1)
            
            return self.db_manager.execute_query(formatted_query, sql_params)
        else:
            return self.db_manager.execute_query(query)
    
    def fetch_one(self, query: str, params: Optional[List] = None) -> Optional[Dict[str, Any]]:
        """Legacy compatibility method for single row queries."""
        results = self.fetch_all(query, params)
        return results[0] if results else None