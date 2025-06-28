"""
Update Command - Update dataset metadata and regenerate features.

Provides dataset maintenance capabilities including:
- Metadata updates (description, competition name)
- Feature regeneration and updates
- Configuration changes
- Integrity verification
"""

from typing import Dict, Any
from .base import BaseDatasetsCommand


class UpdateCommand(BaseDatasetsCommand):
    """Handle --update command for datasets."""
    
    def execute(self, args) -> None:
        """Execute the update dataset command."""
        try:
            dataset_identifier = args.update
            dataset = self.find_dataset_by_identifier(dataset_identifier)
            
            if not dataset:
                self.print_error(f"Dataset '{dataset_identifier}' not found.")
                return
            
            # Check if feature update was requested
            if getattr(args, 'update_features', False):
                self._update_features(dataset, args)
            else:
                self._update_metadata(dataset, args)
                
        except Exception as e:
            self.print_error(f"Failed to update dataset: {e}")
    
    def _update_metadata(self, dataset: Dict[str, Any], args) -> None:
        """Update dataset metadata."""
        dataset_name = dataset['dataset_name']
        dataset_id = dataset['dataset_id']
        
        print(f"\n📝 UPDATING METADATA FOR: {dataset_name}")
        print("=" * 50)
        print(f"🔑 Dataset ID: {dataset_id[:8]}")
        
        # Collect updates
        updates = {}
        
        # Check for new description
        if hasattr(args, 'description') and args.description:
            updates['description'] = args.description
            print(f"📝 New description: {args.description}")
        
        # Check for new competition name
        if hasattr(args, 'competition_name') and args.competition_name:
            updates['competition_name'] = args.competition_name
            print(f"🏆 New competition: {args.competition_name}")
        
        # If no updates specified, show interactive update
        if not updates:
            updates = self._interactive_metadata_update(dataset)
        
        if not updates:
            self.print_info("No updates specified.")
            return
        
        # Perform update
        if getattr(args, 'dry_run', False):
            print(f"\n🔍 DRY RUN - Changes that would be made:")
            for key, value in updates.items():
                old_value = dataset.get(key, 'Not set')
                print(f"   {key}: '{old_value}' → '{value}'")
            print(f"\nNo changes made (dry run mode).")
        else:
            try:
                self.dataset_service.update_dataset_metadata(dataset_id, updates)
                self.print_success(f"Dataset '{dataset_name}' updated successfully!")
                
                # Show what was changed
                print(f"\n✅ Changes made:")
                for key, value in updates.items():
                    print(f"   {key}: {value}")
                
            except Exception as e:
                self.print_error(f"Failed to update metadata: {e}")
    
    def _update_features(self, dataset: Dict[str, Any], args) -> None:
        """Update/regenerate features for dataset."""
        dataset_name = dataset['dataset_name']
        dataset_id = dataset['dataset_id']
        
        print(f"\n🧪 UPDATING FEATURES FOR: {dataset_name}")
        print("=" * 50)
        print(f"🔑 Dataset ID: {dataset_id[:8]}")
        
        # Check current features
        current_features = self.feature_service.get_features_by_dataset(dataset_id)
        print(f"📊 Current features: {len(current_features) if current_features else 0}")
        
        # Check if force update is requested
        force_update = getattr(args, 'force_update', False)
        
        if not force_update:
            # Check if update is needed
            needs_update = self._check_if_features_need_update(dataset, current_features)
            if not needs_update:
                self.print_info("Features are up to date. Use --force-update to regenerate anyway.")
                return
        
        # Perform feature update
        if getattr(args, 'dry_run', False):
            print(f"\n🔍 DRY RUN - Feature regeneration would:")
            print(f"   - Backup existing {len(current_features)} features")
            print(f"   - Regenerate features using current feature space")
            print(f"   - Update feature catalog")
            print(f"\nNo changes made (dry run mode).")
        else:
            try:
                # Perform feature regeneration
                self.print_info("Starting feature regeneration...")
                result = self.feature_service.regenerate_features_for_dataset(dataset_id)
                
                if result:
                    new_feature_count = result.get('feature_count', 0)
                    self.print_success(f"Features regenerated successfully!")
                    print(f"✅ Generated {new_feature_count} features")
                    
                    if current_features:
                        print(f"📊 Previous: {len(current_features)} features")
                        print(f"📊 New: {new_feature_count} features")
                else:
                    self.print_error("Feature regeneration failed")
                
            except Exception as e:
                self.print_error(f"Failed to regenerate features: {e}")
    
    def _interactive_metadata_update(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Interactive metadata update for when no specific updates are provided."""
        updates = {}
        
        print(f"\n📝 INTERACTIVE METADATA UPDATE")
        print("-" * 40)
        print("Press Enter to keep current value, or type new value:")
        
        # Description update
        current_desc = dataset.get('description', '')
        print(f"\nCurrent description: {current_desc or 'Not set'}")
        new_desc = input("New description: ").strip()
        if new_desc:
            updates['description'] = new_desc
        
        # Competition name update
        current_comp = dataset.get('competition_name', '')
        print(f"\nCurrent competition: {current_comp or 'Not set'}")
        new_comp = input("New competition name: ").strip()
        if new_comp:
            updates['competition_name'] = new_comp
        
        return updates
    
    def _check_if_features_need_update(self, dataset: Dict[str, Any], current_features: list) -> bool:
        """Check if features need to be updated."""
        # This is a simplified check - could be expanded with more sophisticated logic
        
        # Check if no features exist
        if not current_features:
            return True
        
        # Check if feature space has been updated (this would need more implementation)
        # For now, assume features are up to date unless forced
        return False