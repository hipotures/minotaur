"""
Stats Command - Show comprehensive dataset usage statistics.

Provides dataset analytics including:
- Overall usage metrics across all datasets
- Performance comparisons between datasets
- Trend analysis and usage patterns
- Resource utilization statistics
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
from .base import BaseDatasetsCommand


class StatsCommand(BaseDatasetsCommand):
    """Handle --stats command for datasets."""
    
    def execute(self, args) -> None:
        """Execute the show dataset statistics command."""
        try:
            datasets = self.dataset_service.get_all_datasets()
            
            if not datasets:
                self.print_info("No datasets registered.")
                return
            
            # Gather comprehensive statistics
            stats = self._gather_statistics(datasets)
            
            # Output in requested format
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(stats)
            else:
                self._output_formatted_stats(stats)
                
        except Exception as e:
            self.print_error(f"Failed to show dataset statistics: {e}")
    
    def _gather_statistics(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gather comprehensive dataset statistics."""
        stats = {
            'overview': self._calculate_overview_stats(datasets),
            'dataset_breakdown': self._calculate_dataset_breakdown(datasets),
            'performance_metrics': self._calculate_performance_metrics(datasets),
            'usage_trends': self._calculate_usage_trends(datasets),
            'generated_at': datetime.now().isoformat()
        }
        
        return stats
    
    def _calculate_overview_stats(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics across all datasets."""
        total_datasets = len(datasets)
        
        # Count active datasets (used in last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        active_count = 0
        
        total_sessions = 0
        total_features = 0
        total_size = 0
        
        for dataset in datasets:
            dataset_id = dataset['dataset_id']
            
            # Get sessions for this dataset
            sessions = self.session_service.get_sessions_by_dataset(dataset_id)
            total_sessions += len(sessions) if sessions else 0
            
            # Check if active
            if sessions:
                recent_session = max(sessions, key=lambda s: s.get('start_time', ''), default=None)
                if recent_session and recent_session.get('start_time'):
                    try:
                        last_used = datetime.fromisoformat(recent_session['start_time'].replace('Z', '+00:00'))
                        if last_used.replace(tzinfo=None) > thirty_days_ago:
                            active_count += 1
                    except (ValueError, TypeError):
                        pass
            
            # Get features for this dataset
            features = self.feature_service.get_features_by_dataset(dataset_id)
            total_features += len(features) if features else 0
            
            # Add size
            total_size += dataset.get('size_bytes', 0)
        
        return {
            'total_datasets': total_datasets,
            'active_datasets': active_count,
            'inactive_datasets': total_datasets - active_count,
            'total_sessions': total_sessions,
            'total_features': total_features,
            'total_size_bytes': total_size,
            'avg_sessions_per_dataset': total_sessions / total_datasets if total_datasets > 0 else 0,
            'avg_features_per_dataset': total_features / total_datasets if total_datasets > 0 else 0
        }
    
    def _calculate_dataset_breakdown(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate detailed breakdown per dataset."""
        breakdown = []
        
        for dataset in datasets:
            dataset_id = dataset['dataset_id']
            
            sessions = self.session_service.get_sessions_by_dataset(dataset_id)
            features = self.feature_service.get_features_by_dataset(dataset_id)
            
            # Calculate success rate
            completed_sessions = [s for s in sessions if s.get('status') == 'completed'] if sessions else []
            success_rate = len(completed_sessions) / len(sessions) * 100 if sessions else 0
            
            # Calculate average score
            scores = [s.get('best_score', 0) for s in completed_sessions if s.get('best_score')] if completed_sessions else []
            avg_score = sum(scores) / len(scores) if scores else 0
            best_score = max(scores) if scores else 0
            
            breakdown.append({
                'dataset_name': dataset['dataset_name'],
                'dataset_id': dataset_id[:8],
                'session_count': len(sessions) if sessions else 0,
                'feature_count': len(features) if features else 0,
                'success_rate': success_rate,
                'avg_score': avg_score,
                'best_score': best_score,
                'size_bytes': dataset.get('size_bytes', 0)
            })
        
        # Sort by session count (most used first)
        breakdown.sort(key=lambda d: d['session_count'], reverse=True)
        return breakdown
    
    def _calculate_performance_metrics(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics across datasets."""
        all_scores = []
        all_success_rates = []
        
        for dataset in datasets:
            dataset_id = dataset['dataset_id']
            sessions = self.session_service.get_sessions_by_dataset(dataset_id)
            
            if sessions:
                completed_sessions = [s for s in sessions if s.get('status') == 'completed']
                if completed_sessions:
                    success_rate = len(completed_sessions) / len(sessions) * 100
                    all_success_rates.append(success_rate)
                    
                    scores = [s.get('best_score', 0) for s in completed_sessions if s.get('best_score')]
                    all_scores.extend(scores)
        
        metrics = {
            'avg_success_rate': sum(all_success_rates) / len(all_success_rates) if all_success_rates else 0,
            'avg_score_overall': sum(all_scores) / len(all_scores) if all_scores else 0,
            'best_score_overall': max(all_scores) if all_scores else 0,
            'total_completed_sessions': sum(1 for dataset in datasets 
                                          for session in (self.session_service.get_sessions_by_dataset(dataset['dataset_id']) or [])
                                          if session.get('status') == 'completed')
        }
        
        return metrics
    
    def _calculate_usage_trends(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate usage trends over time."""
        # This is a simplified version - could be expanded with more detailed trend analysis
        now = datetime.now()
        
        # Count sessions by time period
        last_24h = 0
        last_7d = 0
        last_30d = 0
        
        for dataset in datasets:
            dataset_id = dataset['dataset_id']
            sessions = self.session_service.get_sessions_by_dataset(dataset_id)
            
            if sessions:
                for session in sessions:
                    start_time = session.get('start_time')
                    if start_time:
                        try:
                            session_date = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            session_date = session_date.replace(tzinfo=None)
                            
                            if session_date > now - timedelta(hours=24):
                                last_24h += 1
                            if session_date > now - timedelta(days=7):
                                last_7d += 1
                            if session_date > now - timedelta(days=30):
                                last_30d += 1
                        except (ValueError, TypeError):
                            pass
        
        return {
            'sessions_last_24h': last_24h,
            'sessions_last_7d': last_7d,
            'sessions_last_30d': last_30d
        }
    
    def _output_formatted_stats(self, stats: Dict[str, Any]) -> None:
        """Output statistics in formatted view."""
        overview = stats['overview']
        breakdown = stats['dataset_breakdown']
        performance = stats['performance_metrics']
        trends = stats['usage_trends']
        
        # Overview
        print(f"\n📊 DATASET STATISTICS OVERVIEW")
        print("=" * 50)
        print(f"📁 Total Datasets: {overview['total_datasets']}")
        print(f"✅ Active Datasets: {overview['active_datasets']}")
        print(f"💤 Inactive Datasets: {overview['inactive_datasets']}")
        print(f"📈 Total Sessions: {overview['total_sessions']}")
        print(f"🧪 Total Features: {overview['total_features']}")
        print(f"💾 Total Size: {self.format_size(overview['total_size_bytes'])}")
        
        # Averages
        print(f"\n📊 AVERAGES")
        print("-" * 30)
        print(f"📈 Sessions per Dataset: {overview['avg_sessions_per_dataset']:.1f}")
        print(f"🧪 Features per Dataset: {overview['avg_features_per_dataset']:.1f}")
        
        # Performance
        print(f"\n🏆 PERFORMANCE METRICS")
        print("-" * 30)
        print(f"✅ Average Success Rate: {performance['avg_success_rate']:.1f}%")
        print(f"📊 Average Score: {performance['avg_score_overall']:.5f}")
        print(f"🏆 Best Score Overall: {performance['best_score_overall']:.5f}")
        print(f"✅ Total Completed: {performance['total_completed_sessions']}")
        
        # Usage Trends
        print(f"\n📈 USAGE TRENDS")
        print("-" * 30)
        print(f"🕒 Last 24h: {trends['sessions_last_24h']} sessions")
        print(f"📅 Last 7d: {trends['sessions_last_7d']} sessions")
        print(f"📆 Last 30d: {trends['sessions_last_30d']} sessions")
        
        # Dataset Breakdown
        if breakdown:
            print(f"\n📊 DATASET BREAKDOWN")
            print("-" * 80)
            headers = ['Dataset', 'ID', 'Sessions', 'Features', 'Success%', 'Best Score', 'Size']
            
            rows = []
            for dataset in breakdown[:10]:  # Show top 10
                rows.append([
                    dataset['dataset_name'][:20],
                    dataset['dataset_id'],
                    str(dataset['session_count']),
                    str(dataset['feature_count']),
                    f"{dataset['success_rate']:.1f}%",
                    f"{dataset['best_score']:.3f}" if dataset['best_score'] > 0 else "N/A",
                    self.format_size(dataset['size_bytes'])
                ])
            
            self.print_table(headers, rows)
            
            if len(breakdown) > 10:
                print(f"\n... and {len(breakdown) - 10} more datasets")
    
    def _output_json(self, stats: Dict[str, Any]) -> None:
        """Output statistics in JSON format."""
        self.print_json(stats, "Dataset Statistics")