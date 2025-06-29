"""
Restore Command - Restore database from backup.

Provides database restoration including:
- Backup file validation and verification
- Current database backup before restoration
- Decompression handling for compressed backups
- Post-restoration verification and testing
"""

from datetime import datetime
from pathlib import Path
import shutil
from .base import BaseBackupCommand


class RestoreCommand(BaseBackupCommand):
    """Handle --restore command for backup."""
    
    def execute(self, args) -> None:
        """Execute the database restoration command."""
        try:
            backup_file = args.restore
            dry_run = getattr(args, 'dry_run', False)
            
            print(f"🔄 RESTORING DATABASE FROM BACKUP")
            print("=" * 40)
            
            # Resolve backup file path
            backup_path = self.resolve_backup_path(backup_file)
            
            if not backup_path.exists():
                self.print_error(f"Backup file not found: {backup_path}")
                self._suggest_available_backups()
                return
            
            # Verify backup before restoration
            if not self._verify_backup_before_restore(backup_path):
                return
            
            # Show restoration info and confirm
            if not self._confirm_restoration(backup_path, dry_run):
                return
            
            # Perform restoration
            if dry_run:
                self._simulate_restoration(backup_path)
            else:
                self._perform_restoration(backup_path)
            
        except Exception as e:
            self.print_error(f"Database restoration failed: {e}")
    
    def _verify_backup_before_restore(self, backup_path: Path) -> bool:
        """Verify backup file before restoration."""
        print("🔍 Verifying backup file...")
        
        if not self.verify_backup_integrity(backup_path):
            self.print_error("Backup verification failed - restoration aborted")
            return False
        
        print("✅ Backup file verification successful")
        return True
    
    def _confirm_restoration(self, backup_path: Path, dry_run: bool) -> bool:
        """Confirm restoration operation with user."""
        db_path = self.get_database_path()
        
        print(f"\nThis will replace the current database:")
        print(f"  Current: {db_path}")
        print(f"  Backup:  {backup_path}")
        
        # Show backup info
        backup_info = self.format_backup_info(backup_path)
        print(f"  Backup size: {backup_info['size']}")
        print(f"  Backup type: {backup_info['type']}")
        print(f"  Created: {backup_info['created']}")
        print()
        
        if dry_run:
            print("🧪 DRY-RUN MODE: Restoration will be simulated")
            return True
        
        return self.confirm_operation("Continue with restoration?")
    
    def _simulate_restoration(self, backup_path: Path) -> None:
        """Simulate restoration for dry-run mode."""
        print("🧪 DRY-RUN MODE: Restoration simulation")
        print("=" * 40)
        
        backup_info = self.format_backup_info(backup_path)
        
        print("✅ Would backup current database")
        print("✅ Would extract backup file")
        print(f"✅ Would restore {backup_info['size']} of data")
        print("✅ Would verify restored database")
        print("\n🧪 DRY-RUN MODE: Restoration simulation completed successfully")
    
    def _perform_restoration(self, backup_path: Path) -> None:
        """Perform the actual database restoration."""
        db_path = self.get_database_path()
        
        try:
            # Step 1: Backup current database
            current_backup = self._backup_current_database(db_path)
            
            # Step 2: Restore from backup
            if self._restore_database_file(backup_path, db_path):
                # Step 3: Verify restoration
                if self._verify_restored_database(db_path):
                    self.print_success("✅ Database restoration completed successfully")
                    
                    # Cleanup current backup if restoration was successful
                    if current_backup and current_backup.exists():
                        try:
                            current_backup.unlink()
                            print(f"🗑️  Cleaned up temporary backup: {current_backup.name}")
                        except Exception:
                            pass
                else:
                    self.print_error("Database verification failed after restoration")
                    self._restore_from_current_backup(current_backup, db_path)
            else:
                self.print_error("Database restoration failed")
                self._restore_from_current_backup(current_backup, db_path)
            
        except Exception as e:
            self.print_error(f"Restoration process failed: {e}")
            # Try to restore original if possible
            current_backup = db_path.with_suffix('.duckdb.current_backup')
            if current_backup.exists():
                self._restore_from_current_backup(current_backup, db_path)
    
    def _backup_current_database(self, db_path: Path) -> Path:
        """Create backup of current database before restoration."""
        if not db_path.exists():
            return None
        
        print("💾 Backing up current database...")
        
        current_backup = db_path.with_suffix('.duckdb.current_backup')
        
        try:
            shutil.copy2(db_path, current_backup)
            print(f"✅ Current database backed up to: {current_backup.name}")
            return current_backup
            
        except Exception as e:
            self.print_error(f"Failed to backup current database: {e}")
            raise
    
    def _restore_database_file(self, backup_path: Path, db_path: Path) -> bool:
        """Restore database file from backup."""
        try:
            print("🔄 Restoring database file...")
            
            start_time = datetime.now()
            
            # Extract from backup (handles compression automatically)
            self.extract_from_backup(backup_path, db_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Check restored file
            if db_path.exists():
                restored_size = db_path.stat().st_size / (1024 * 1024)
                print(f"✅ Database restored in {duration:.1f}s")
                print(f"Restored size: {restored_size:.1f} MB")
                return True
            else:
                self.print_error("Restored database file not found")
                return False
            
        except Exception as e:
            self.print_error(f"Failed to restore database file: {e}")
            return False
    
    def _verify_restored_database(self, db_path: Path) -> bool:
        """Verify the restored database."""
        print("🔍 Verifying restored database...")
        
        try:
            db_info = self.verify_database_connectivity(db_path)
            
            if db_info:
                print(f"✅ Database verification successful")
                print(f"✅ Tables found: {db_info['table_count']}")
                if 'session_count' in db_info:
                    print(f"✅ Sessions verified: {db_info['session_count']} sessions")
                return True
            else:
                self.print_error("Database connectivity test failed")
                return False
            
        except Exception as e:
            self.print_error(f"Database verification failed: {e}")
            return False
    
    def _restore_from_current_backup(self, current_backup: Path, db_path: Path) -> None:
        """Restore from current backup if restoration failed."""
        if not current_backup or not current_backup.exists():
            self.print_error("Cannot restore original database - no backup found")
            return
        
        try:
            print("🔄 Restoring original database from backup...")
            shutil.copy2(current_backup, db_path)
            print("✅ Original database restored successfully")
            
            # Clean up backup
            current_backup.unlink()
            
        except Exception as e:
            self.print_error(f"Failed to restore original database: {e}")
    
    def _suggest_available_backups(self) -> None:
        """Suggest available backup files when requested file not found."""
        backup_files = self.find_backup_files()
        
        if backup_files:
            print("\n💡 Available backup files:")
            for backup_file in backup_files[:5]:  # Show first 5
                info = self.format_backup_info(backup_file)
                print(f"   • {backup_file.name} ({info['size']}, {info['created']})")
            
            if len(backup_files) > 5:
                print(f"   ... and {len(backup_files) - 5} more")
            
            print("\n💡 Usage: python manager.py backup --restore <filename>")
        else:
            print("\n💡 No backup files found. Create one with: python manager.py backup --create")