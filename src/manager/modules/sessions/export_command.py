"""
Export Command - Export session data to CSV/JSON format.

Provides comprehensive session data export including:
- Session metadata and configuration
- Performance metrics and statistics
- Exploration history summaries
- Multiple output formats (CSV, JSON)
"""

from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
from .base import BaseSessionsCommand


class ExportCommand(BaseSessionsCommand):
    """Handle --export command for sessions."""
    
    def execute(self, args) -> None:
        """Execute the session export command."""
        try:
            format_type = args.export
            
            # Get sessions data for export
            sessions_data = self._get_export_data(args)
            
            if not sessions_data:
                self.print_info("No sessions found to export.")
                return
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export based on format
            if format_type == 'csv':
                output_file = self._export_csv(sessions_data, timestamp, args)
            elif format_type == 'json':
                output_file = self._export_json(sessions_data, timestamp, args)
            else:
                self.print_error(f"Unsupported export format: {format_type}")
                return
            
            self.print_success(f"Exported {len(sessions_data)} sessions to: {output_file}")
            
        except Exception as e:
            self.print_error(f"Failed to export sessions: {e}")
    
    def _get_export_data(self, args) -> List[Dict[str, Any]]:
        """Get sessions data prepared for export."""
        try:
            # Build filters from args (reuse from list command logic)
            conditions = []
            params = []
            
            # Status filter
            status = getattr(args, 'status', 'all')
            if status != 'all':
                conditions.append("status = ?")
                params.append(status)
            
            # Strategy filter
            strategy = getattr(args, 'strategy', None)
            if strategy:
                conditions.append("strategy = ?")
                params.append(strategy)
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            # Get basic session data
            query = f"""
                SELECT 
                    session_id,
                    session_name,
                    start_time,
                    end_time,
                    total_iterations,
                    best_score,
                    status,
                    strategy,
                    is_test_mode,
                    config_snapshot,
                    notes,
                    dataset_hash
                FROM sessions 
                {where_clause}
                ORDER BY start_time DESC
            """
            
            results = self.session_service.repository.fetch_all(query, params)
            
            # Enhance with exploration statistics
            sessions_data = []
            for row in results:
                session_data = {
                    'session_id': row['session_id'],
                    'session_name': row['session_name'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'total_iterations': row['total_iterations'],
                    'best_score': row['best_score'],
                    'status': row['status'],
                    'strategy': row['strategy'],
                    'is_test_mode': row['is_test_mode'],
                    'config_snapshot': row['config_snapshot'],
                    'notes': row['notes'],
                    'dataset_hash': row['dataset_hash']
                }
                
                # Add exploration statistics
                exploration_stats = self.get_exploration_statistics(row['session_id'])
                if exploration_stats:
                    session_data.update({
                        'total_explorations': exploration_stats['total_explorations'],
                        'min_score': exploration_stats['min_score'],
                        'max_score': exploration_stats['max_score'],
                        'avg_score': exploration_stats['avg_score'],
                        'total_eval_time': exploration_stats['total_eval_time'],
                        'avg_eval_time': exploration_stats['avg_eval_time'],
                        'unique_operations': exploration_stats['unique_operations']
                    })
                else:
                    session_data.update({
                        'total_explorations': 0,
                        'min_score': None,
                        'max_score': None,
                        'avg_score': None,
                        'total_eval_time': None,
                        'avg_eval_time': None,
                        'unique_operations': 0
                    })
                
                # Extract key config values
                if row['config_snapshot']:
                    try:
                        config = json.loads(row['config_snapshot'])
                        session_data.update({
                            'target_metric': config.get('autogluon', {}).get('target_metric'),
                            'train_path': config.get('autogluon', {}).get('train_path'),
                            'max_iterations_config': config.get('session', {}).get('max_iterations'),
                            'exploration_strategy': config.get('session', {}).get('exploration_strategy')
                        })
                    except:
                        session_data.update({
                            'target_metric': None,
                            'train_path': None,
                            'max_iterations_config': None,
                            'exploration_strategy': None
                        })
                else:
                    session_data.update({
                        'target_metric': None,
                        'train_path': None,
                        'max_iterations_config': None,
                        'exploration_strategy': None
                    })
                
                # Calculate derived metrics
                if session_data['start_time'] and session_data['end_time']:
                    duration = self.format_duration(session_data['start_time'], session_data['end_time'])
                    session_data['duration'] = duration
                else:
                    session_data['duration'] = 'Running' if not session_data['end_time'] else 'Unknown'
                
                sessions_data.append(session_data)
            
            return sessions_data
            
        except Exception as e:
            self.print_error(f"Failed to get export data: {e}")
            return []
    
    def _export_csv(self, sessions_data: List[Dict[str, Any]], timestamp: str, args) -> str:
        """Export sessions data to CSV format."""
        try:
            # Determine output file path
            output_file = self._get_output_file_path('csv', timestamp, args)
            
            # Use DuckDB's CSV export for better performance
            # First, create a temporary table with the data
            temp_table_data = []
            for session in sessions_data:
                # Prepare row data for CSV (flatten complex fields)
                row = [
                    session['session_id'],
                    session['session_name'] or '',
                    session['start_time'],
                    session['end_time'] or '',
                    session['total_iterations'],
                    session['best_score'] or 0,
                    session['status'],
                    session['strategy'],
                    session['is_test_mode'],
                    session['notes'] or '',
                    session['dataset_hash'] or '',
                    session['total_explorations'],
                    session['min_score'] or 0,
                    session['max_score'] or 0,
                    session['avg_score'] or 0,
                    session['total_eval_time'] or 0,
                    session['avg_eval_time'] or 0,
                    session['unique_operations'],
                    session['target_metric'] or '',
                    session['train_path'] or '',
                    session['max_iterations_config'] or 0,
                    session['exploration_strategy'] or '',
                    session['duration']
                ]
                temp_table_data.append(row)
            
            # Write CSV manually for better control
            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                headers = [
                    'session_id', 'session_name', 'start_time', 'end_time',
                    'total_iterations', 'best_score', 'status', 'strategy',
                    'is_test_mode', 'notes', 'dataset_hash', 'total_explorations',
                    'min_score', 'max_score', 'avg_score', 'total_eval_time',
                    'avg_eval_time', 'unique_operations', 'target_metric',
                    'train_path', 'max_iterations_config', 'exploration_strategy',
                    'duration'
                ]
                writer.writerow(headers)
                
                # Write data
                for row in temp_table_data:
                    writer.writerow(row)
            
            return str(output_file)
            
        except Exception as e:
            self.print_error(f"Failed to export CSV: {e}")
            raise
    
    def _export_json(self, sessions_data: List[Dict[str, Any]], timestamp: str, args) -> str:
        """Export sessions data to JSON format."""
        try:
            # Determine output file path
            output_file = self._get_output_file_path('json', timestamp, args)
            
            # Prepare export data with metadata
            export_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_sessions': len(sessions_data),
                    'export_format': 'json',
                    'filters_applied': self._get_applied_filters(args)
                },
                'sessions': sessions_data
            }
            
            # Write JSON file
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=2, default=str)
            
            return str(output_file)
            
        except Exception as e:
            self.print_error(f"Failed to export JSON: {e}")
            raise
    
    def _get_output_file_path(self, format_type: str, timestamp: str, args) -> Path:
        """Get the output file path for export."""
        # Check if user specified output file
        output_file = getattr(args, 'output_file', None)
        if output_file:
            return Path(output_file)
        
        # Use project's export directory
        try:
            # Get export configuration from database service
            config = self.session_service.repository.db_pool.config
            exports_dir = Path(config.get_export_config()['export_dir'])
            exports_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"sessions_export_{timestamp}.{format_type}"
            return exports_dir / filename
            
        except Exception:
            # Fallback to current directory
            filename = f"sessions_export_{timestamp}.{format_type}"
            return Path.cwd() / filename
    
    def _get_applied_filters(self, args) -> Dict[str, Any]:
        """Get information about applied filters for metadata."""
        filters = {}
        
        status = getattr(args, 'status', 'all')
        if status != 'all':
            filters['status'] = status
        
        strategy = getattr(args, 'strategy', None)
        if strategy:
            filters['strategy'] = strategy
        
        limit = getattr(args, 'limit', None)
        if limit:
            filters['limit'] = limit
        
        return filters