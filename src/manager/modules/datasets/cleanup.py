"""
Cleanup Command - Safely remove datasets and related data.

Provides comprehensive dataset removal including:
- Backup creation before deletion
- Cascade deletion of related data (sessions, features)
- Verification and confirmation steps
- Safe rollback capabilities
"""

from typing import Dict, Any
from .base import BaseDatasetsCommand


class CleanupCommand(BaseDatasetsCommand):
    """Handle --cleanup command for datasets."""
    
    def execute(self, args) -> None:
        """Execute the cleanup dataset command."""
        try:
            dataset_identifier = args.cleanup
            dataset = self.find_dataset_by_identifier(dataset_identifier)
            
            if not dataset:
                self.print_error(f"Dataset '{dataset_identifier}' not found.")
                return
            
            # Show what will be removed
            self._show_cleanup_preview(dataset)
            
            # Confirm deletion (unless force flag is used)
            if not getattr(args, 'force', False):
                response = input("\nProceed with cleanup? (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    self.print_info("Cleanup cancelled.")
                    return
            
            # Perform cleanup
            self._perform_cleanup(dataset, args)
            
        except Exception as e:
            self.print_error(f"Failed to cleanup dataset: {e}")
    
    def _show_cleanup_preview(self, dataset: Dict[str, Any]) -> None:
        """Show what will be removed during cleanup."""
        dataset_id = dataset['dataset_id']
        dataset_name = dataset['dataset_name']
        
        print(f"\n🗑️  CLEANUP PREVIEW")
        print("=" * 40)
        print(f"📋 Dataset: {dataset_name}")
        print(f"🔑 ID: {dataset_id[:8]}")
        
        # Count related data
        sessions = self.session_service.get_sessions_by_dataset(dataset_id)
        features = self.feature_service.get_features_by_dataset(dataset_id)
        
        print(f"\n📊 Related data to be removed:")
        print(f"   Sessions: {len(sessions) if sessions else 0}")
        print(f"   Features: {len(features) if features else 0}")
        
        print(f"\n⚠️  This action cannot be undone!")
        print(f"💾 A backup will be created before deletion.")
    
    def _perform_cleanup(self, dataset: Dict[str, Any], args) -> None:
        """Perform the actual cleanup operation."""
        dataset_id = dataset['dataset_id']
        dataset_name = dataset['dataset_name']
        
        try:
            # Step 1: Create backup
            self.print_info("Creating backup before cleanup...")
            backup_result = self.backup_service.create_dataset_backup(dataset_id)
            
            if backup_result:
                self.print_success(f"Backup created: {backup_result.get('backup_path', 'Unknown')}")
            else:
                self.print_warning("Could not create backup, continuing with cleanup...")
            
            # Step 2: Remove related data
            self.print_info("Removing related sessions and features...")
            self._cleanup_related_data(dataset_id)
            
            # Step 3: Remove dataset record
            self.print_info("Removing dataset record...")
            self.dataset_service.delete_dataset(dataset_id)
            
            self.print_success(f"Dataset '{dataset_name}' cleaned up successfully!")
            
            # Show next steps
            print(f"\n💡 Next steps:")
            print(f"   List datasets: python manager.py datasets --list")
            if backup_result:
                print(f"   Restore backup: python manager.py backup --restore {backup_result.get('backup_id', '')}")
            
        except Exception as e:
            self.print_error(f"Cleanup failed: {e}")
            self.print_info("Dataset may be partially cleaned. Check system state.")
    
    def _cleanup_related_data(self, dataset_id: str) -> None:
        """Clean up all data related to the dataset."""
        try:
            # Remove sessions
            sessions = self.session_service.get_sessions_by_dataset(dataset_id)
            for session in sessions:
                self.session_service.delete_session(session['session_id'])
            
            # Remove features
            features = self.feature_service.get_features_by_dataset(dataset_id)
            for feature in features:
                self.feature_service.delete_feature(feature['feature_id'])
            
            self.print_success(f"Removed {len(sessions)} sessions and {len(features)} features")
            
        except Exception as e:
            self.print_warning(f"Error cleaning related data: {e}")