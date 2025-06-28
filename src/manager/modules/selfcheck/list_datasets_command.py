"""
List Datasets Command - List available test datasets.

Provides dataset registry listing including:
- Dataset metadata and status
- Training and test data information
- Target column and task type details
- Usage recommendations
"""

from typing import Dict, Any, List
from .base import BaseSelfCheckCommand


class ListDatasetsCommand(BaseSelfCheckCommand):
    """Handle --list-datasets command for self-check."""
    
    def execute(self, args) -> None:
        """Execute the list datasets command."""
        try:
            print("📋 AVAILABLE REGISTERED DATASETS")
            print("=" * 45)
            
            # Get all datasets from registry
            datasets = self._get_available_datasets()
            
            if not datasets:
                self.print_error("No datasets found in registry")
                self.print_info("Use 'datasets --register' to add datasets")
                return
            
            # Display datasets
            self._display_datasets(datasets)
            
            # Show usage information
            self._show_usage_info()
            
        except Exception as e:
            self.print_error(f"Failed to list datasets: {e}")
    
    def _get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get all available datasets from registry."""
        try:
            query = """
                SELECT 
                    dataset_name, dataset_id, target_column, 
                    train_records, train_columns, 
                    test_records, test_columns, 
                    is_active
                FROM datasets 
                ORDER BY dataset_name
            """
            results = self.dataset_service.repository.fetch_all(query)
            
            datasets = []
            for row in results:
                datasets.append({
                    'name': row['dataset_name'],
                    'dataset_id': row['dataset_id'],
                    'target_column': row['target_column'],
                    'train_records': row['train_records'],
                    'train_columns': row['train_columns'],
                    'test_records': row['test_records'],
                    'test_columns': row['test_columns'],
                    'is_active': row['is_active']
                })
            
            return datasets
            
        except Exception as e:
            self.print_error(f"Error accessing dataset registry: {e}")
            return []
    
    def _display_datasets(self, datasets: List[Dict[str, Any]]) -> None:
        """Display datasets with detailed information."""
        for dataset in datasets:
            name = dataset['name']
            dataset_id = dataset['dataset_id']
            target_col = dataset['target_column']
            train_rows = dataset['train_records']
            train_cols = dataset['train_columns']
            test_rows = dataset['test_records']
            test_cols = dataset['test_columns']
            is_active = dataset['is_active']
            
            status = "✅" if is_active else "❌"
            train_info = f"{train_rows} rows, {train_cols} columns" if train_rows else "N/A"
            test_info = f"{test_rows} rows, {test_cols} columns" if test_rows else "N/A"
            
            print(f"{status} {name}")
            print(f"   ID: {dataset_id[:8]}...")
            print(f"   Target: {target_col}")
            print(f"   Train: {train_info}")
            print(f"   Test: {test_info}")
            print()
    
    def _show_usage_info(self) -> None:
        """Show usage information."""
        print(f"\\n💡 Usage: --run <dataset_name> --config <config_file>")
        print(f"   Example: python manager.py selfcheck --run titanic")
        print(f"   Example: python manager.py selfcheck --run titanic --quick")
        print(f"   Example: python manager.py selfcheck --run titanic --config custom_config.yaml")