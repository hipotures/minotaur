"""
List Command - Display all registered datasets with usage statistics.

Provides comprehensive listing of datasets including:
- Basic dataset information (name, ID, size)
- Usage statistics (sessions count, last used)
- Status information (active/inactive)
- Multiple output formats (table, JSON)
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from .base import BaseDatasetsCommand


class ListCommand(BaseDatasetsCommand):
    """Handle --list command for datasets."""
    
    def execute(self, args) -> None:
        """Execute the list datasets command."""
        try:
            # Get datasets directly from repository to avoid service layer issues
            datasets = self.dataset_service.repository.get_all_datasets(include_inactive=True)
            
            if not datasets:
                self.print_info("No datasets registered.")
                self.print_info("Register a dataset: python manager.py datasets --register --help")
                return
            
            # Filter active only if requested
            if getattr(args, 'active_only', False):
                datasets = [d for d in datasets if d.get('is_active', False)]
            
            # Output in requested format
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(datasets)
            else:
                self._output_table(datasets)
                
        except Exception as e:
            self.print_error(f"Failed to list datasets: {e}")
    
    def _is_dataset_active(self, dataset: Dict[str, Any]) -> bool:
        """Check if dataset is considered active (used in last 30 days)."""
        try:
            if not dataset.get('last_used'):
                return False
            
            last_used = dataset['last_used']
            if isinstance(last_used, str):
                last_used = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
            
            thirty_days_ago = datetime.now() - timedelta(days=30)
            return last_used > thirty_days_ago
            
        except (ValueError, TypeError):
            return False
    
    def _enrich_with_stats(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich datasets with usage statistics."""
        enriched = []
        
        for dataset in datasets:
            dataset_id = dataset['dataset_id']
            
            # Get session count
            sessions = self.session_service.get_sessions_by_dataset(dataset_id)
            session_count = len(sessions) if sessions else 0
            
            # Get last used date
            last_used = None
            if sessions:
                # Find most recent session
                recent_session = max(sessions, key=lambda s: s.get('start_time', ''), default=None)
                if recent_session:
                    last_used = recent_session.get('start_time')
            
            # Calculate total size (if available)
            total_size = self._calculate_dataset_size(dataset)
            
            # Add enrichment data
            enriched_dataset = dataset.copy()
            enriched_dataset.update({
                'session_count': session_count,
                'last_used': last_used,
                'total_size': total_size,
                'status': 'Active' if self._is_dataset_active(dataset) else 'Inactive'
            })
            
            enriched.append(enriched_dataset)
        
        # Sort by last used (most recent first), then by name
        enriched.sort(key=lambda d: (
            d['last_used'] or '1970-01-01',
            d['dataset_name']
        ), reverse=True)
        
        return enriched
    
    def _calculate_dataset_size(self, dataset: Dict[str, Any]) -> int:
        """Calculate total dataset size in bytes."""
        try:
            # This would need to be implemented based on your dataset storage
            # For now, return 0 as placeholder
            return dataset.get('size_bytes', 0)
        except Exception:
            return 0
    
    def _output_table(self, datasets: List[Dict[str, Any]]) -> None:
        """Output datasets in table format."""
        headers = ['Name', 'ID', 'Train Records', 'Last Used', 'Size (MB)', 'Status']
        rows = []
        
        for dataset in datasets:
            # Format last used date
            last_used = dataset.get('last_used')
            if last_used:
                try:
                    if isinstance(last_used, str):
                        dt = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
                        last_used_str = dt.strftime('%Y-%m-%d')
                    else:
                        last_used_str = str(last_used)[:10]
                except (ValueError, TypeError):
                    last_used_str = str(last_used)[:10] if last_used else 'Never'
            else:
                last_used_str = 'Never'
            
            # Status
            status = 'Active' if dataset.get('is_active', False) else 'Inactive'
            
            rows.append([
                dataset.get('dataset_name', 'Unknown')[:30],
                dataset.get('dataset_id', 'Unknown')[:8],
                str(dataset.get('train_records', 0)),
                last_used_str,
                f"{dataset.get('data_size_mb') or 0:.1f}",
                status
            ])
        
        self.print_table(headers, rows, "Registered Datasets")
        
        # Summary information
        total_datasets = len(datasets)
        active_datasets = len([d for d in datasets if d.get('is_active', False)])
        
        print(f"\n📈 Summary:")
        print(f"   Total Datasets: {total_datasets}")
        print(f"   Active Datasets: {active_datasets}")
        
        if total_datasets > 0:
            print(f"\n💡 Usage:")
            print(f"   Show details: python manager.py datasets --details DATASET_NAME")
            print(f"   View sessions: python manager.py datasets --sessions DATASET_NAME")
    
    def _output_json(self, datasets: List[Dict[str, Any]]) -> None:
        """Output datasets in JSON format."""
        output = {
            'datasets': datasets,
            'summary': {
                'total_count': len(datasets),
                'active_count': len([d for d in datasets if d.get('status') == 'Active']),
                'total_sessions': sum(d.get('session_count', 0) for d in datasets),
                'generated_at': datetime.now().isoformat()
            }
        }
        
        self.print_json(output, "Registered Datasets")