"""
Repository for metrics and performance-related database operations
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from .base import BaseRepository


class MetricsRepository(BaseRepository):
    """Handles metrics and performance-related database operations."""
    
    def get_performance_trends(self, days: int = 30, 
                             dataset_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance trends over time.
        
        Args:
            days: Number of days to look back
            dataset_hash: Optional dataset filter
            
        Returns:
            List of daily performance metrics
        """
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
        WHERE start_time >= ?
        AND status IN ('completed', 'failed')
        """
        
        params = [cutoff_date.isoformat()]
        
        if dataset_hash:
            query += " AND dataset_hash = ?"
            params.append(dataset_hash)
        
        query += " GROUP BY DATE(start_time) ORDER BY date ASC"
        
        return self.fetch_all(query, tuple(params))
    
    def get_convergence_analysis(self, session_id: str) -> List[Dict[str, Any]]:
        """Get score convergence data for a session.
        
        Args:
            session_id: Session ID to analyze
            
        Returns:
            List of exploration history with running best score
        """
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
        WHERE session_id = ?
        ORDER BY iteration ASC
        """
        
        rows = self.fetch_all(query, (session_id,))
        
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
        """Get metrics for operations.
        
        Args:
            operation_type: Optional operation type filter
            days: Optional days to look back
            
        Returns:
            List of operation metrics
        """
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
        
        params = []
        
        if operation_type:
            query += " AND operation_name = ?"
            params.append(operation_type)
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            query += " AND last_used >= ?"
            params.append(cutoff_date.isoformat())
        
        query += " GROUP BY operation_name, operation_category ORDER BY count DESC"
        
        return self.fetch_all(query, tuple(params))
    
    def get_resource_usage(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get resource usage statistics.
        
        Args:
            session_id: Optional session filter
            
        Returns:
            Dictionary of resource usage metrics
        """
        # Database size - Get file size directly
        db_size_bytes = 0
        try:
            import os
            db_path = self.db_pool.database_path
            if os.path.exists(db_path):
                db_size_bytes = os.path.getsize(db_path)
        except:
            pass
        
        # Table sizes - Just list tables without sizes for now
        table_sizes_query = """
        SELECT 
            table_name,
            0 as size_mb
        FROM information_schema.tables
        WHERE table_schema = 'main'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        table_sizes = self.fetch_all(table_sizes_query)
        
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
            WHERE session_id = ?
            """
            
            perf = self.fetch_one(perf_query, (session_id,))
            
            # Get feature and node counts
            counts_query = """
            SELECT 
                COUNT(DISTINCT node_id) as node_count,
                COUNT(DISTINCT feature_name) as feature_count
            FROM exploration_history
            WHERE session_id = ?
            """
            
            counts = self.fetch_one(counts_query, (session_id,))
            
            if perf and counts:
                session_metrics = {
                    'max_memory_mb': perf['max_memory'],
                    'avg_cpu_percent': perf['avg_cpu'],
                    'max_gpu_memory_mb': perf['max_gpu_memory'],
                    'max_active_nodes': perf['max_active_nodes'],
                    'total_nodes': counts['node_count'],
                    'total_features': counts['feature_count']
                }
        
        return {
            'database_size_bytes': db_size_bytes,
            'database_size_mb': db_size_bytes / (1024 * 1024) if db_size_bytes else 0,
            'table_sizes': table_sizes,
            'session_metrics': session_metrics
        }
    
    def get_feature_performance_matrix(self, limit: int = 20) -> Dict[str, Any]:
        """Get feature performance matrix showing impact by dataset.
        
        Args:
            limit: Maximum features to include
            
        Returns:
            Dictionary with features and their performance by dataset
        """
        # Get top features
        top_features_query = """
        SELECT DISTINCT feature_name
        FROM (
            SELECT 
                feature_name,
                AVG(impact_delta) as avg_impact
            FROM feature_impact
            GROUP BY feature_name
            ORDER BY avg_impact DESC
            LIMIT ?
        )
        """
        
        top_features = [row['feature_name'] for row in self.fetch_all(top_features_query, (limit,))]
        
        if not top_features:
            return {
                'features': [],
                'datasets': [],
                'matrix': {}
            }
        
        # Get performance by dataset
        placeholders = ','.join(['?' for _ in top_features])
        matrix_query = f"""
        SELECT 
            fi.feature_name,
            d.dataset_name,
            AVG(fi.impact_delta) as avg_impact,
            COUNT(*) as use_count,
            MAX(fi.impact_percentage) as max_impact_pct
        FROM feature_impact fi
        JOIN sessions s ON fi.session_id = s.session_id
        JOIN datasets d ON s.dataset_hash = d.dataset_id
        WHERE fi.feature_name IN ({placeholders})
        GROUP BY fi.feature_name, d.dataset_name
        """
        
        matrix_rows = self.fetch_all(matrix_query, tuple(top_features))
        
        # Build matrix
        matrix = {}
        datasets = set()
        
        for row in matrix_rows:
            feature_name = row['feature_name']
            dataset_name = row['dataset_name']
            
            if feature_name not in matrix:
                matrix[feature_name] = {}
            
            matrix[feature_name][dataset_name] = {
                'avg_impact': row['avg_impact'],
                'use_count': row['use_count'],
                'max_impact_pct': row['max_impact_pct']
            }
            
            datasets.add(dataset_name)
        
        return {
            'features': top_features,
            'datasets': sorted(list(datasets)),
            'matrix': matrix
        }
    
    def get_system_performance_summary(self, session_id: str) -> Dict[str, Any]:
        """Get system performance summary for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Performance summary dictionary
        """
        query = """
        SELECT 
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            AVG(memory_usage_mb) as avg_memory,
            MAX(memory_usage_mb) as max_memory,
            AVG(cpu_usage_percent) as avg_cpu,
            MAX(cpu_usage_percent) as max_cpu,
            AVG(gpu_memory_mb) as avg_gpu_memory,
            MAX(gpu_memory_mb) as max_gpu_memory,
            MAX(active_nodes) as max_active_nodes,
            AVG(evaluation_queue_size) as avg_queue_size
        FROM system_performance
        WHERE session_id = ?
        """
        
        result = self.fetch_one(query, (session_id,))
        
        if not result:
            return {}
        
        duration = None
        if result['start_time'] and result['end_time']:
            start = datetime.fromisoformat(result['start_time'])
            end = datetime.fromisoformat(result['end_time'])
            duration = (end - start).total_seconds()
        
        return {
            'duration_seconds': duration,
            'memory': {
                'avg_mb': result['avg_memory'],
                'max_mb': result['max_memory']
            },
            'cpu': {
                'avg_percent': result['avg_cpu'],
                'max_percent': result['max_cpu']
            },
            'gpu': {
                'avg_memory_mb': result['avg_gpu_memory'],
                'max_memory_mb': result['max_gpu_memory']
            },
            'nodes': {
                'max_active': result['max_active_nodes']
            },
            'queue': {
                'avg_size': result['avg_queue_size']
            }
        }