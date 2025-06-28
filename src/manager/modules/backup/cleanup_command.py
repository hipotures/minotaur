"""
Cleanup Command - Remove old backup files.

Provides backup cleanup including:
- Age-based backup removal
- Count-based backup retention
- Interactive confirmation for safety
- Dry-run mode for preview
"""

from typing import List
from pathlib import Path
from datetime import datetime, timedelta
from .base import BaseBackupCommand


class CleanupCommand(BaseBackupCommand):
    """Handle --cleanup command for backup."""
    
    def execute(self, args) -> None:
        """Execute the backup cleanup command."""
        try:
            keep_count = getattr(args, 'keep', 5)
            dry_run = getattr(args, 'dry_run', False)
            
            print("🧹 CLEANING UP OLD BACKUPS")
            print("=" * 40)
            
            # Find all backup files
            backup_files = self.find_backup_files()
            
            if not backup_files:
                self.print_info("No backup files found to clean up.")
                return
            
            # Determine files to remove
            files_to_remove = self._get_files_to_remove(backup_files, keep_count)
            
            if not files_to_remove:
                self.print_info(f"All backups are within retention policy (keep {keep_count} backups).")
                self._show_current_backups(backup_files)
                return
            
            # Show cleanup plan
            self._show_cleanup_plan(backup_files, files_to_remove, keep_count, dry_run)
            
            # Confirm and execute cleanup
            if self._confirm_cleanup(files_to_remove, dry_run):
                if dry_run:
                    self._simulate_cleanup(files_to_remove)
                else:
                    self._perform_cleanup(files_to_remove)
            
        except Exception as e:
            self.print_error(f"Backup cleanup failed: {e}")
    
    def _get_files_to_remove(self, backup_files: List[Path], keep_count: int) -> List[Path]:
        """Determine which backup files should be removed."""
        if len(backup_files) <= keep_count:
            return []
        
        # Sort by modification time (newest first) and keep the newest ones
        sorted_files = sorted(backup_files, key=lambda x: x.stat().st_mtime, reverse=True)
        files_to_remove = sorted_files[keep_count:]
        
        return files_to_remove
    
    def _show_cleanup_plan(self, all_files: List[Path], files_to_remove: List[Path], 
                          keep_count: int, dry_run: bool) -> None:
        """Show the cleanup plan to the user."""
        files_to_keep = [f for f in all_files if f not in files_to_remove]
        
        print(f"Retention policy: Keep {keep_count} newest backups")
        print(f"Total backups found: {len(all_files)}")
        print(f"Backups to keep: {len(files_to_keep)}")
        print(f"Backups to remove: {len(files_to_remove)}")
        print()
        
        if files_to_keep:
            print("📁 Backups to KEEP:")
            for backup_file in files_to_keep:
                info = self.format_backup_info(backup_file)
                print(f"   ✅ {info['name']} ({info['size']}, {info['created']})")
            print()
        
        if files_to_remove:
            print("🗑️  Backups to REMOVE:")
            total_size_to_remove = 0
            for backup_file in files_to_remove:
                info = self.format_backup_info(backup_file)
                size_mb = backup_file.stat().st_size / (1024 * 1024)
                total_size_to_remove += size_mb
                print(f"   ❌ {info['name']} ({info['size']}, {info['created']})")
            
            print(f"\nTotal space to be freed: {total_size_to_remove:.1f} MB")
            print()
        
        if dry_run:
            print("🧪 DRY-RUN MODE: No files will actually be removed")
    
    def _show_current_backups(self, backup_files: List[Path]) -> None:
        """Show current backup status when no cleanup is needed."""
        print(f"\nCurrent backups ({len(backup_files)}):")
        for backup_file in backup_files:
            info = self.format_backup_info(backup_file)
            print(f"   • {info['name']} ({info['size']}, {info['created']})")
    
    def _confirm_cleanup(self, files_to_remove: List[Path], dry_run: bool) -> bool:
        """Confirm cleanup operation with user."""
        if dry_run:
            return True
        
        if not files_to_remove:
            return False
        
        message = f"Remove {len(files_to_remove)} old backup files?"
        return self.confirm_operation(message)
    
    def _simulate_cleanup(self, files_to_remove: List[Path]) -> None:
        """Simulate cleanup for dry-run mode."""
        print("🧪 DRY-RUN MODE: Cleanup simulation")
        print("=" * 40)
        
        total_size = 0
        for backup_file in files_to_remove:
            size_mb = backup_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"✅ Would remove: {backup_file.name} ({size_mb:.1f} MB)")
        
        print(f"\n🧪 DRY-RUN MODE: Would free {total_size:.1f} MB of space")
        print("🧪 DRY-RUN MODE: Cleanup simulation completed successfully")
    
    def _perform_cleanup(self, files_to_remove: List[Path]) -> None:
        """Perform the actual cleanup operation."""
        print("🗑️  Removing old backup files...")
        
        removed_count = 0
        total_size_freed = 0
        failed_removals = []
        
        for backup_file in files_to_remove:
            try:
                size_mb = backup_file.stat().st_size / (1024 * 1024)
                backup_file.unlink()
                
                removed_count += 1
                total_size_freed += size_mb
                print(f"✅ Removed: {backup_file.name} ({size_mb:.1f} MB)")
                
            except Exception as e:
                failed_removals.append((backup_file, str(e)))
                self.print_error(f"Failed to remove {backup_file.name}: {e}")
        
        # Show results
        print(f"\n✅ Cleanup completed:")
        print(f"   Files removed: {removed_count}")
        print(f"   Space freed: {total_size_freed:.1f} MB")
        
        if failed_removals:
            print(f"   Failed removals: {len(failed_removals)}")
            for backup_file, error in failed_removals:
                print(f"      • {backup_file.name}: {error}")
        
        if removed_count > 0:
            self.print_success(f"✅ Successfully removed {removed_count} old backup files")
        else:
            self.print_warning("⚠️  No files were removed")