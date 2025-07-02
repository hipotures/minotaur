"""
Service layer for analytics and reporting using SQLAlchemy abstraction layer.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from ..core.utils import format_duration, format_datetime, format_number


def safe_parse_datetime(date_str: Union[str, None]) -> Optional[datetime]:
    """Safely parse datetime string, returning None for invalid/missing values."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


class AnalyticsService:
    """Handles analytics and reporting business logic using new database abstraction."""
    
    def __init__(self, db_manager):
        """Initialize service with database manager.
        
        Args:
            db_manager: Database manager instance from factory
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Ensure analytics tables exist
        self._ensure_analytics_tables()
        
        # Legacy compatibility - modules expect repository attributes
        self.session_repo = self
        self.feature_repo = self 
        self.metrics_repo = self
    
    def get_performance_summary(self, days: int = 30, 
                              dataset: Optional[str] = None) -> Dict[str, Any]:
        """Generate overall performance summary.
        
        Args:
            days: Number of days to analyze
            dataset: Optional dataset filter
            
        Returns:
            Summary data dictionary
        """
        # Get session statistics
        stats = self.get_session_statistics(days)
        
        # Get top sessions
        all_sessions = self.get_all_sessions()
        if dataset:
            all_sessions = [s for s in all_sessions if s.get('dataset_name') == dataset]
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = []
        for s in all_sessions:
            start_dt = safe_parse_datetime(s.get('start_time'))
            if start_dt and start_dt >= cutoff_date:
                recent_sessions.append(s)
        
        # Sort by score for top sessions
        completed_sessions = [s for s in recent_sessions if s['status'] == 'completed']
        top_sessions = sorted(completed_sessions, 
                            key=lambda x: x.get('best_score', 0), 
                            reverse=True)[:10]
        
        # Get feature impact statistics
        all_features = self.get_all_features()
        feature_stats = {
            'total_features': len(all_features),
            'positive_features': sum(1 for f in all_features if f['avg_impact'] > 0),
            'avg_impact': sum(f['avg_impact'] for f in all_features) / max(1, len(all_features)),
            'best_impact': max((f['avg_impact'] for f in all_features), default=0)
        }
        feature_stats['success_rate'] = (feature_stats['positive_features'] / 
                                        max(1, feature_stats['total_features']))
        
        # Recent activity
        now = datetime.now()
        sessions_24h = 0
        sessions_7d = 0
        for s in recent_sessions:
            start_dt = safe_parse_datetime(s.get('start_time'))
            if start_dt:
                days_ago = (now - start_dt).days
                if days_ago < 1:
                    sessions_24h += 1
                if days_ago < 7:
                    sessions_7d += 1
        
        return {
            'overall_metrics': {
                'total_sessions': stats['total_sessions'],
                'completed_sessions': stats['performance']['completed_sessions'],
                'success_rate': (stats['performance']['completed_sessions'] / 
                               max(1, stats['total_sessions'])),
                'avg_score': stats['performance']['avg_score'],
                'best_score': stats['performance']['max_score'],
                'total_time': stats['performance'].get('total_duration', 0),
                'avg_time': stats['performance'].get('avg_duration', 0),
                'total_nodes': stats['performance']['total_nodes'],
                'total_features': stats['performance']['total_features']
            },
            'status_breakdown': stats['status_counts'],
            'top_sessions': [
                {
                    'session_id': s['session_id'],
                    'dataset': s.get('dataset_name', 'Unknown'),
                    'score': s.get('best_score', 0),
                    'iterations': s.get('total_iterations', 0),
                    'status': s.get('status', 'unknown'),
                    'time': (safe_parse_datetime(s.get('end_time')) - safe_parse_datetime(s.get('start_time'))).total_seconds() 
                           if safe_parse_datetime(s.get('start_time')) and safe_parse_datetime(s.get('end_time')) else 0
                }
                for s in top_sessions
            ],
            'feature_impact': feature_stats,
            'recent_activity': {
                'sessions_24h': sessions_24h,
                'sessions_7d': sessions_7d,
                'sessions_30d': len(recent_sessions)
            }
        }
    
    def get_performance_trends(self, days: int = 30,
                             dataset: Optional[str] = None) -> Dict[str, Any]:
        """Get performance trends over time.
        
        Args:
            days: Number of days to analyze
            dataset: Optional dataset filter
            
        Returns:
            Trend data dictionary
        """
        trends = self.get_performance_trends(days, dataset)
        
        # Calculate trend indicators
        if len(trends) >= 7:
            recent_avg = sum(t['avg_score'] for t in trends[-7:]) / 7
            previous_avg = sum(t['avg_score'] for t in trends[-14:-7]) / 7
            
            score_trend = 'improving' if recent_avg > previous_avg else 'declining'
            trend_percentage = ((recent_avg - previous_avg) / previous_avg) if previous_avg > 0 else 0
        else:
            score_trend = 'insufficient_data'
            trend_percentage = 0
        
        return {
            'daily_metrics': trends,
            'summary': {
                'total_days': len(trends),
                'score_trend': score_trend,
                'trend_percentage': trend_percentage,
                'best_day': max(trends, key=lambda x: x['avg_score'])['date'] if trends else None,
                'most_active_day': max(trends, key=lambda x: x['session_count'])['date'] if trends else None
            }
        }
    
    def analyze_convergence(self, session_id: str) -> Dict[str, Any]:
        """Analyze score convergence for a session.
        
        Args:
            session_id: Session ID to analyze
            
        Returns:
            Convergence analysis data
        """
        convergence_data = self.get_convergence_analysis(session_id)
        
        if not convergence_data:
            return {'error': f'No data found for session {session_id}'}
        
        # Calculate convergence metrics
        best_scores = [d['best_score'] for d in convergence_data]
        iterations_to_best = next((i for i, d in enumerate(convergence_data) 
                                 if d['best_score'] == max(best_scores)), 0)
        
        # Find plateaus (where score doesn't improve for N iterations)
        plateau_threshold = 10
        plateaus = []
        current_plateau_start = 0
        current_best = best_scores[0]
        
        for i in range(1, len(best_scores)):
            if best_scores[i] > current_best:
                if i - current_plateau_start >= plateau_threshold:
                    plateaus.append({
                        'start': current_plateau_start,
                        'end': i - 1,
                        'duration': i - current_plateau_start,
                        'score': current_best
                    })
                current_best = best_scores[i]
                current_plateau_start = i
        
        # Check for final plateau
        if len(best_scores) - current_plateau_start >= plateau_threshold:
            plateaus.append({
                'start': current_plateau_start,
                'end': len(best_scores) - 1,
                'duration': len(best_scores) - current_plateau_start,
                'score': current_best
            })
        
        return {
            'session_id': session_id,
            'total_iterations': len(convergence_data),
            'final_score': best_scores[-1] if best_scores else 0,
            'iterations_to_best': iterations_to_best,
            'convergence_data': convergence_data,
            'plateaus': plateaus,
            'improvement_rate': (best_scores[-1] - best_scores[0]) / len(best_scores) 
                              if len(best_scores) > 1 else 0
        }
    
    def analyze_operations(self, days: int = 30) -> Dict[str, Any]:
        """Analyze operation effectiveness.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Operation analysis data
        """
        # Get operation metrics
        op_metrics = self.get_operation_metrics(days=days)
        
        # Get feature rankings to understand which operations create good features
        feature_rankings = self.get_feature_rankings(limit=50)
        
        # Analyze which operations generate high-impact features
        operation_impact = {}
        for feature in feature_rankings:
            # Extract operation from feature name (assumes pattern like "operation_feature")
            parts = feature['feature_name'].split('_')
            if parts:
                operation = parts[0]
                if operation not in operation_impact:
                    operation_impact[operation] = {
                        'features': 0,
                        'total_impact': 0,
                        'avg_impact': 0
                    }
                operation_impact[operation]['features'] += 1
                operation_impact[operation]['total_impact'] += feature['avg_impact']
        
        # Calculate average impact per operation
        for op, stats in operation_impact.items():
            stats['avg_impact'] = stats['total_impact'] / max(1, stats['features'])
        
        return {
            'operation_metrics': op_metrics,
            'operation_impact': sorted(
                [{'operation': k, **v} for k, v in operation_impact.items()],
                key=lambda x: x['avg_impact'],
                reverse=True
            ),
            'summary': {
                'total_operations': len(op_metrics),
                'most_used': op_metrics[0]['operation_name'] if op_metrics else None,
                'most_impactful': max(operation_impact.items(), 
                                    key=lambda x: x[1]['avg_impact'])[0] 
                                if operation_impact else None
            }
        }
    
    def generate_report(self, days: int = 30, 
                       dataset: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive analytics report.
        
        Args:
            days: Number of days to analyze
            dataset: Optional dataset filter
            
        Returns:
            Comprehensive report data
        """
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'period_days': days,
                'dataset_filter': dataset
            },
            'summary': self.get_performance_summary(days, dataset),
            'trends': self.get_performance_trends(days, dataset),
            'operations': self.analyze_operations(days),
            'top_features': self.get_feature_rankings(limit=20),
            'resource_usage': self.get_resource_usage()
        }
    
    def compare_periods(self, period1_start: str, period1_end: str,
                       period2_start: str, period2_end: str) -> Dict[str, Any]:
        """Compare performance between two time periods.
        
        Args:
            period1_start: Start date for period 1 (YYYY-MM-DD)
            period1_end: End date for period 1
            period2_start: Start date for period 2
            period2_end: End date for period 2
            
        Returns:
            Comparison data
        """
        # Get sessions for each period
        all_sessions = self.get_all_sessions()
        
        period1_sessions = []
        period2_sessions = []
        
        for s in all_sessions:
            start_dt = safe_parse_datetime(s.get('start_time'))
            if start_dt:
                date_str = start_dt.date().isoformat()
                if period1_start <= date_str <= period1_end:
                    period1_sessions.append(s)
                if period2_start <= date_str <= period2_end:
                    period2_sessions.append(s)
        
        # Calculate metrics for each period
        def calculate_period_metrics(sessions):
            completed = [s for s in sessions if s['status'] == 'completed']
            return {
                'total_sessions': len(sessions),
                'completed_sessions': len(completed),
                'success_rate': len(completed) / max(1, len(sessions)),
                'avg_score': sum(s.get('best_score', 0) for s in completed) / max(1, len(completed)),
                'max_score': max((s.get('best_score', 0) for s in completed), default=0),
                'total_iterations': sum(s.get('total_iterations', 0) for s in sessions),
                'total_time': sum(
                    (safe_parse_datetime(s.get('end_time')) - safe_parse_datetime(s.get('start_time'))).total_seconds()
                    for s in sessions
                    if safe_parse_datetime(s.get('start_time')) and safe_parse_datetime(s.get('end_time'))
                )
            }
        
        period1_metrics = calculate_period_metrics(period1_sessions)
        period2_metrics = calculate_period_metrics(period2_sessions)
        
        # Calculate differences
        differences = {}
        for key in period1_metrics:
            if isinstance(period1_metrics[key], (int, float)):
                diff = period1_metrics[key] - period2_metrics[key]
                pct_change = (diff / period2_metrics[key] * 100) if period2_metrics[key] > 0 else 0
                differences[key] = {
                    'absolute': diff,
                    'percentage': pct_change,
                    'improved': diff > 0 if 'rate' in key or 'score' in key else diff < 0
                }
        
        return {
            'period1': {
                'start': period1_start,
                'end': period1_end,
                'metrics': period1_metrics
            },
            'period2': {
                'start': period2_start,
                'end': period2_end,
                'metrics': period2_metrics
            },
            'differences': differences,
            'summary': {
                'overall_improvement': differences.get('avg_score', {}).get('improved', False),
                'efficiency_change': differences.get('success_rate', {}).get('percentage', 0)
            }
        }
    
    def _ensure_analytics_tables(self):
        """Ensure analytics-related tables exist."""
        self.logger.info("Creating analytics tables if not exist...")
        
        # Exploration history table for MCTS tracking
        exploration_table = """
        CREATE TABLE IF NOT EXISTS exploration_history (
            id INTEGER PRIMARY KEY,
            session_id VARCHAR NOT NULL,
            node_id VARCHAR,
            iteration INTEGER,
            operation_applied VARCHAR,
            evaluation_score DOUBLE DEFAULT 0.0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            node_visits INTEGER DEFAULT 0,
            mcts_ucb1_score DOUBLE DEFAULT 0.0,
            is_best_so_far BOOLEAN DEFAULT false
        )
        """
        
        # Operation performance table
        operation_performance_table = """
        CREATE TABLE IF NOT EXISTS operation_performance (
            operation_name VARCHAR PRIMARY KEY,
            operation_category VARCHAR,
            total_applications INTEGER DEFAULT 0,
            avg_execution_time DOUBLE DEFAULT 0.0,
            avg_improvement DOUBLE DEFAULT 0.0,
            best_improvement DOUBLE DEFAULT 0.0,
            effectiveness_score DOUBLE DEFAULT 0.0,
            last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # System performance table
        system_performance_table = """
        CREATE TABLE IF NOT EXISTS system_performance (
            id INTEGER PRIMARY KEY,
            session_id VARCHAR NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            memory_usage_mb DOUBLE DEFAULT 0.0,
            cpu_usage_percent DOUBLE DEFAULT 0.0,
            gpu_memory_mb DOUBLE DEFAULT 0.0,
            active_nodes INTEGER DEFAULT 0,
            evaluation_queue_size INTEGER DEFAULT 0
        )
        """
        
        self.db_manager.execute_ddl(exploration_table)
        self.db_manager.execute_ddl(operation_performance_table)
        self.db_manager.execute_ddl(system_performance_table)
        self.logger.info("Analytics tables creation completed")
    
    # Repository-style methods for legacy compatibility
    def get_session_statistics(self, days: int) -> Dict[str, Any]:
        """Get session statistics for the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            COUNT(*) as total_sessions,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions,
            COUNT(CASE WHEN status = 'running' THEN 1 END) as running_sessions,
            AVG(best_score) as avg_score,
            MAX(best_score) as max_score,
            AVG(total_iterations) as avg_iterations,
            SUM(total_iterations) as total_nodes,
            COUNT(DISTINCT dataset_name) as total_features
        FROM sessions
        WHERE start_time >= :cutoff_date
        """
        
        result = self.db_manager.execute_query(query, {'cutoff_date': cutoff_date})
        stats = result[0] if result else {}
        
        # Get status counts separately
        status_query = """
        SELECT 
            status,
            COUNT(*) as count
        FROM sessions 
        WHERE start_time >= :cutoff_date
        GROUP BY status
        """
        
        status_results = self.db_manager.execute_query(status_query, {'cutoff_date': cutoff_date})
        status_counts = {row['status']: row['count'] for row in status_results}
        
        return {
            'total_sessions': stats.get('total_sessions', 0),
            'performance': {
                'completed_sessions': stats.get('completed_sessions', 0),
                'avg_score': stats.get('avg_score', 0.0),
                'max_score': stats.get('max_score', 0.0),
                'avg_duration': 0,  # Would need session duration calculation
                'total_duration': 0,  # Would need session duration calculation
                'total_nodes': stats.get('total_nodes', 0),
                'total_features': stats.get('total_features', 0)
            },
            'status_counts': status_counts
        }
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions."""
        query = """
        SELECT 
            session_id,
            session_name,
            dataset_name,
            start_time,
            end_time,
            status,
            total_iterations,
            best_score,
            config_hash,
            error_message,
            strategy,
            is_test_mode
        FROM sessions
        ORDER BY start_time DESC
        """
        
        return self.db_manager.execute_query(query)
    
    def get_all_features(self) -> List[Dict[str, Any]]:
        """Get all features with basic statistics."""
        query = """
        WITH feature_stats AS (
            SELECT 
                fc.feature_name,
                fc.feature_category,
                fc.description,
                AVG(COALESCE(fi.impact_delta, 0)) as avg_impact,
                COUNT(fi.id) as total_uses
            FROM feature_catalog fc
            LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
            WHERE fc.is_active = true
            GROUP BY fc.feature_name, fc.feature_category, fc.description
        )
        SELECT 
            feature_name,
            feature_category,
            description,
            avg_impact,
            total_uses
        FROM feature_stats
        ORDER BY avg_impact DESC NULLS LAST
        """
        
        return self.db_manager.execute_query(query)
    
    def get_performance_trends(self, days: int = 30, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance trends over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            DATE(start_time) as date,
            COUNT(*) as session_count,
            AVG(best_score) as avg_score,
            MAX(best_score) as max_score,
            AVG(total_iterations) as avg_iterations,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_count,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count
        FROM sessions
        WHERE start_time >= :cutoff_date
        AND status IN ('completed', 'failed')
        """
        
        params = {'cutoff_date': cutoff_date}
        
        if dataset:
            query += " AND dataset_name = :dataset"
            params['dataset'] = dataset
        
        query += " GROUP BY DATE(start_time) ORDER BY date ASC"
        
        return self.db_manager.execute_query(query, params)
    
    def get_convergence_analysis(self, session_id: str) -> List[Dict[str, Any]]:
        """Get score convergence data for a session."""
        query = """
        SELECT 
            id,
            iteration,
            operation_applied,
            evaluation_score,
            timestamp,
            node_visits,
            mcts_ucb1_score,
            is_best_so_far
        FROM exploration_history
        WHERE session_id = :session_id
        ORDER BY iteration ASC
        """
        
        rows = self.db_manager.execute_query(query, {'session_id': session_id})
        
        # Calculate running best score
        best_score = 0
        results = []
        
        for i, row in enumerate(rows):
            current_score = row['evaluation_score'] if row['evaluation_score'] is not None else 0
            best_score = max(best_score, current_score)
            
            results.append({
                'iteration': row['iteration'],
                'node_id': row['id'],
                'operation': row['operation_applied'],
                'score': current_score,
                'best_score': best_score,
                'timestamp': row['timestamp'],
                'visit_count': row['node_visits'],
                'ucb_score': row['mcts_ucb1_score'],
                'is_best': row['is_best_so_far']
            })
        
        return results
    
    def get_operation_metrics(self, operation_type: Optional[str] = None,
                            days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics for operations."""
        query = """
        SELECT 
            operation_name,
            operation_category,
            SUM(total_applications) as count,
            AVG(avg_execution_time) as avg_duration,
            MAX(avg_execution_time) as max_duration,
            MIN(avg_execution_time) as min_duration,
            SUM(total_applications) as total_uses,
            AVG(avg_improvement) as avg_impact,
            MAX(best_improvement) as best_impact,
            AVG(effectiveness_score) as avg_effectiveness
        FROM operation_performance
        WHERE 1=1
        """
        
        params = {}
        
        if operation_type:
            query += " AND operation_name = :operation_type"
            params['operation_type'] = operation_type
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            query += " AND last_used >= :cutoff_date"
            params['cutoff_date'] = cutoff_date
        
        query += " GROUP BY operation_name, operation_category ORDER BY count DESC"
        
        return self.db_manager.execute_query(query, params)
    
    def get_feature_rankings(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top features ranked by performance."""
        query = """
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
        ORDER BY avg_impact DESC
        LIMIT :limit
        """
        
        return self.db_manager.execute_query(query, {'limit': limit})
    
    def get_resource_usage(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get resource usage statistics."""
        # Database size estimation
        db_size_bytes = 0
        try:
            import os
            # Try to get database file size if using DuckDB
            # This is a simplified approach - in practice would need config access
            db_size_bytes = 50 * 1024 * 1024  # Default estimate
        except:
            pass
        
        # Table sizes - simplified for now
        table_sizes_query = """
        SELECT 
            'sessions' as table_name,
            0 as size_mb
        UNION ALL
        SELECT 
            'feature_catalog' as table_name,
            0 as size_mb
        UNION ALL
        SELECT 
            'feature_impact' as table_name,
            0 as size_mb
        """
        table_sizes = self.db_manager.execute_query(table_sizes_query)
        
        # Session-specific metrics if provided
        session_metrics = {}
        if session_id:
            # Get system performance metrics
            perf_query = """
            SELECT 
                MAX(memory_usage_mb) as max_memory,
                AVG(cpu_usage_percent) as avg_cpu,
                MAX(gpu_memory_mb) as max_gpu_memory,
                MAX(active_nodes) as max_active_nodes
            FROM system_performance
            WHERE session_id = :session_id
            """
            
            perf_results = self.db_manager.execute_query(perf_query, {'session_id': session_id})
            perf = perf_results[0] if perf_results else {}
            
            # Get feature and node counts
            counts_query = """
            SELECT 
                COUNT(DISTINCT node_id) as node_count,
                COUNT(DISTINCT operation_applied) as feature_count
            FROM exploration_history
            WHERE session_id = :session_id
            """
            
            counts_results = self.db_manager.execute_query(counts_query, {'session_id': session_id})
            counts = counts_results[0] if counts_results else {}
            
            if perf and counts:
                session_metrics = {
                    'max_memory_mb': perf.get('max_memory', 0),
                    'avg_cpu_percent': perf.get('avg_cpu', 0),
                    'max_gpu_memory_mb': perf.get('max_gpu_memory', 0),
                    'max_active_nodes': perf.get('max_active_nodes', 0),
                    'total_nodes': counts.get('node_count', 0),
                    'total_features': counts.get('feature_count', 0)
                }
        
        return {
            'database_size_bytes': db_size_bytes,
            'database_size_mb': db_size_bytes / (1024 * 1024) if db_size_bytes else 0,
            'table_sizes': table_sizes,
            'session_metrics': session_metrics
        }