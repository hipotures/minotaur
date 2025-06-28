"""
Service layer for analytics and reporting
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from ..repositories import SessionRepository, FeatureRepository, MetricsRepository
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
    """Handles analytics and reporting business logic."""
    
    def __init__(self, session_repo: SessionRepository, 
                 feature_repo: FeatureRepository,
                 metrics_repo: MetricsRepository):
        """Initialize service with repositories.
        
        Args:
            session_repo: Session repository instance
            feature_repo: Feature repository instance
            metrics_repo: Metrics repository instance
        """
        self.session_repo = session_repo
        self.feature_repo = feature_repo
        self.metrics_repo = metrics_repo
        self.logger = logging.getLogger(__name__)
    
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
        stats = self.session_repo.get_session_statistics(days)
        
        # Get top sessions
        all_sessions = self.session_repo.get_all_sessions()
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
        all_features = self.feature_repo.get_all_features()
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
        trends = self.metrics_repo.get_performance_trends(days, dataset)
        
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
        convergence_data = self.metrics_repo.get_convergence_analysis(session_id)
        
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
        op_metrics = self.metrics_repo.get_operation_metrics(days=days)
        
        # Get feature rankings to understand which operations create good features
        feature_rankings = self.feature_repo.get_feature_rankings(limit=50)
        
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
            'top_features': self.feature_repo.get_feature_rankings(limit=20),
            'resource_usage': self.metrics_repo.get_resource_usage()
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
        all_sessions = self.session_repo.get_all_sessions()
        
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