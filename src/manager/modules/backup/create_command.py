"""
Create Command - Create a new database backup.

Provides database backup creation including:
- Full database backup with metadata
- Optional compression for space efficiency
- Progress tracking and performance metrics
- Backup verification and integrity checking
"""

from datetime import datetime
from pathlib import Path
from .base import BaseBackupCommand


class CreateCommand(BaseBackupCommand):
    """Handle --create command for backup."""
    
    def execute(self, args) -> None:
        """Execute the backup creation command."""
        try:
            print("💾 CREATING DATABASE BACKUP")
            print("=" * 40)
            
            # Get database path
            db_path = self.get_database_path()
            
            if not db_path.exists():
                self.print_error(f"Database not found: {db_path}")
                return
            
            # Prepare backup
            compress = getattr(args, 'compress', False)
            backup_dir = self.get_backup_directory()
            backup_filename = self.generate_backup_filename(compress)
            backup_path = backup_dir / backup_filename
            
            # Show backup information
            self._show_backup_info(db_path, backup_path, compress)
            
            # Create the backup
            success = self._create_backup_file(db_path, backup_path, compress)
            
            if success:
                # Verify the backup
                if self._verify_created_backup(backup_path):
                    self.print_success("✅ Backup created and verified successfully")
                else:
                    self.print_warning("⚠️  Backup created but verification failed")
            
        except Exception as e:
            self.print_error(f"Backup creation failed: {e}")
    
    def _show_backup_info(self, source: Path, target: Path, compress: bool) -> None:
        """Display backup information before creation."""
        try:
            source_size = source.stat().st_size
            source_size_mb = source_size / (1024 * 1024)
            
            print(f"Source: {source}")
            print(f"Target: {target}")
            print(f"Source size: {source_size_mb:.1f} MB")
            print(f"Compression: {'Yes' if compress else 'No'}")
            print()
            
        except Exception as e:
            self.print_warning(f"Could not get source file info: {e}")
    
    def _create_backup_file(self, source: Path, target: Path, compress: bool) -> bool:
        """Create the actual backup file."""
        try:
            print("📦 Creating backup...", end="", flush=True)
            
            start_time = datetime.now()
            
            # Copy with optional compression
            self.copy_with_compression(source, target, compress)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\\r✅ Backup file created successfully")
            
            # Show backup results
            self._show_backup_results(source, target, duration, compress)
            
            return True
            
        except Exception as e:
            print("\\r❌ Backup creation failed")
            self.print_error(f"Error creating backup: {e}")
            return False
    
    def _show_backup_results(self, source: Path, target: Path, duration: float, compress: bool) -> None:
        """Show backup creation results."""
        try:
            source_size = source.stat().st_size
            backup_size = target.stat().st_size
            
            source_size_mb = source_size / (1024 * 1024)
            backup_size_mb = backup_size / (1024 * 1024)
            
            print(f"✅ Backup completed in {duration:.1f}s")
            print(f"Backup size: {backup_size_mb:.1f} MB")
            
            if compress and source_size > 0:
                compression_ratio = source_size / backup_size
                space_saved = (source_size - backup_size) / (1024 * 1024)
                print(f"Compression ratio: {compression_ratio:.1f}x")
                print(f"Space saved: {space_saved:.1f} MB")
            
            print(f"Backup location: {target}")
            
        except Exception as e:
            self.print_warning(f"Could not calculate backup statistics: {e}")
    
    def _verify_created_backup(self, backup_path: Path) -> bool:
        """Verify the created backup file."""
        try:
            print("🔍 Verifying backup integrity...")
            
            # Use base class verification
            if not self.verify_backup_integrity(backup_path):
                return False
            
            # Additional verification for uncompressed backups
            if backup_path.suffix != '.gz':
                db_info = self.verify_database_connectivity(backup_path)
                if db_info:
                    print(f"✅ Database verification: {db_info['table_count']} tables found")
                    if 'session_count' in db_info:
                        print(f"✅ Sessions verified: {db_info['session_count']} sessions")
                    return True
                else:
                    return False
            else:
                print("✅ Compressed backup format verified")
                return True
            
        except Exception as e:
            self.print_error(f"Backup verification failed: {e}")
            return False