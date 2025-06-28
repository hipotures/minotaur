"""
Backup Module - Database backup and restoration

Provides commands for creating backups, managing backup files, and database restoration.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from . import ModuleInterface

class BackupModule(ModuleInterface):
    """Module for database backup and restoration."""
    
    @property
    def name(self) -> str:
        return "backup"
    
    @property
    def description(self) -> str:
        return "Create and manage database backups"
    
    @property
    def commands(self) -> Dict[str, str]:
        return {
            "--create": "Create a new database backup",
            "--list": "List available backup files",
            "--restore": "Restore database from backup",
            "--cleanup": "Remove old backup files",
            "--verify": "Verify backup file integrity",
            "--help": "Show detailed help for backup module"
        }
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add backup-specific arguments."""
        backup_group = parser.add_argument_group('Backup Module')
        backup_group.add_argument('--create', action='store_true',
                                help='Create database backup')
        backup_group.add_argument('--list', action='store_true',
                                help='List available backups')
        backup_group.add_argument('--restore', type=str, metavar='BACKUP_FILE',
                                help='Restore from backup file')
        backup_group.add_argument('--cleanup', action='store_true',
                                help='Remove old backup files')
        backup_group.add_argument('--verify', type=str, metavar='BACKUP_FILE',
                                help='Verify backup integrity')
        backup_group.add_argument('--compress', action='store_true',
                                help='Compress backup file')
        backup_group.add_argument('--keep', type=int, default=5,
                                help='Number of backups to keep (default: 5)')
    
    def execute(self, args: argparse.Namespace, manager) -> None:
        """Execute backup module commands."""
        
        if getattr(args, 'create', False):
            self._create_backup(args, manager)
        elif getattr(args, 'list', False):
            self._list_backups(args, manager)
        elif args.restore:
            self._restore_backup(args.restore, args, manager)
        elif getattr(args, 'cleanup', False):
            self._cleanup_backups(args, manager)
        elif args.verify:
            self._verify_backup(args.verify, args, manager)
        else:
            print("❌ No backup command specified. Use --help for options.")
    
    def _create_backup(self, args: argparse.Namespace, manager) -> None:
        """Create a new database backup."""
        print("💾 CREATING DATABASE BACKUP")
        print("=" * 40)
        
        # Load backup configuration
        backup_config = manager.get_backup_config()
        backup_dir = manager.project_root / backup_config['backup_path']
        backup_prefix = backup_config['backup_prefix']
        backup_dir.mkdir(exist_ok=True)
        
        if not manager.duckdb_path.exists():
            print(f"❌ Database not found: {manager.duckdb_path}")
            return
        
        # Generate backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.compress:
            backup_file = backup_dir / f"{backup_prefix}{timestamp}.duckdb.gz"
        else:
            backup_file = backup_dir / f"{backup_prefix}{timestamp}.duckdb"
        
        try:
            # Get source info
            source_size = manager.duckdb_path.stat().st_size
            source_size_mb = source_size / (1024 * 1024)
            
            print(f"Source: {manager.duckdb_path}")
            print(f"Target: {backup_file}")
            print(f"Source size: {source_size_mb:.1f} MB")
            print(f"Compression: {'Yes' if args.compress else 'No'}")
            print()
            
            # Create backup
            start_time = datetime.now()
            
            if args.compress:
                import gzip
                with open(manager.duckdb_path, 'rb') as f_in:
                    with gzip.open(backup_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(manager.duckdb_path, backup_file)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get backup info
            backup_size = backup_file.stat().st_size
            backup_size_mb = backup_size / (1024 * 1024)
            
            print(f"✅ Backup completed in {duration:.1f}s")
            print(f"Backup size: {backup_size_mb:.1f} MB")
            
            if args.compress:
                compression_ratio = source_size / backup_size
                print(f"Compression ratio: {compression_ratio:.1f}x")
            
            print(f"Backup location: {backup_file}")
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
    
    def _list_backups(self, args: argparse.Namespace, manager) -> None:
        """List available backup files."""
        print("📋 AVAILABLE BACKUPS")
        print("=" * 40)
        
        backup_dir = manager.project_root / "data" / "backups"
        
        if not backup_dir.exists():
            print("No backup directory found.")
            return
        
        # Find backup files
        backup_config = manager.get_backup_config()
        backup_prefix = backup_config['backup_prefix']
        backup_files = list(backup_dir.glob(f"{backup_prefix}*.duckdb*"))
        
        if not backup_files:
            print("No backup files found.")
            return
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"{'File':<30} {'Size':<10} {'Created':<20} {'Type':<12}")
        print("-" * 75)
        
        for backup_file in backup_files:
            size_mb = backup_file.stat().st_size / (1024 * 1024)
            created = datetime.fromtimestamp(backup_file.stat().st_mtime)
            backup_type = "Compressed" if backup_file.suffix == '.gz' else "Regular"
            
            file_name = backup_file.name
            if len(file_name) > 29:
                file_name = file_name[:26] + "..."
            
            print(f"{file_name:<30} {size_mb:>7.1f} MB {created.strftime('%Y-%m-%d %H:%M:%S'):<20} {backup_type:<12}")
        
        print(f"\nTotal backups: {len(backup_files)}")
    
    def _restore_backup(self, backup_file: str, args: argparse.Namespace, manager) -> None:
        """Restore database from backup."""
        print(f"🔄 RESTORING DATABASE FROM BACKUP")
        print("=" * 40)
        
        backup_path = Path(backup_file)
        if not backup_path.is_absolute():
            backup_path = manager.project_root / "data" / "backups" / backup_file
        
        if not backup_path.exists():
            print(f"❌ Backup file not found: {backup_path}")
            return
        
        # Confirm restoration (skip in dry-run mode for testing)
        print(f"This will replace the current database:")
        print(f"  Current: {manager.duckdb_path}")
        print(f"  Backup:  {backup_path}")
        print()
        
        # Check for dry-run mode (used in testing)
        if getattr(args, 'dry_run', False):
            print("🧪 DRY-RUN MODE: Restoration simulation completed successfully")
            return
        
        response = input("Continue with restoration? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Restoration cancelled")
            return
        
        try:
            # Create backup of current database
            if manager.duckdb_path.exists():
                current_backup = manager.duckdb_path.with_suffix('.duckdb.current_backup')
                shutil.copy2(manager.duckdb_path, current_backup)
                print(f"💾 Current database backed up to: {current_backup}")
            
            # Restore from backup
            start_time = datetime.now()
            
            if backup_path.suffix == '.gz':
                # Decompress backup
                import gzip
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(manager.duckdb_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Regular copy
                shutil.copy2(backup_path, manager.duckdb_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Verify restoration
            restored_size = manager.duckdb_path.stat().st_size / (1024 * 1024)
            
            print(f"✅ Database restored in {duration:.1f}s")
            print(f"Restored size: {restored_size:.1f} MB")
            
            # Test database connectivity
            try:
                with manager._connect() as conn:
                    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                    print(f"✅ Database verification successful ({session_count} sessions)")
            except Exception as e:
                print(f"⚠️  Database verification failed: {e}")
            
        except Exception as e:
            print(f"❌ Restoration failed: {e}")
            # Try to restore current backup if it exists
            current_backup = manager.duckdb_path.with_suffix('.duckdb.current_backup')
            if current_backup.exists():
                try:
                    shutil.copy2(current_backup, manager.duckdb_path)
                    print("🔄 Original database restored from backup")
                except:
                    print("❌ Failed to restore original database")
    
    def _cleanup_backups(self, args: argparse.Namespace, manager) -> None:
        """Remove old backup files."""
        print("🧹 CLEANING UP BACKUP FILES")
        print("=" * 40)
        
        backup_dir = manager.project_root / "data" / "backups"
        
        if not backup_dir.exists():
            print("No backup directory found.")
            return
        
        # Find backup files
        backup_config = manager.get_backup_config()
        backup_prefix = backup_config['backup_prefix']
        backup_files = list(backup_dir.glob(f"{backup_prefix}*.duckdb*"))
        
        if len(backup_files) <= args.keep:
            print(f"✅ Only {len(backup_files)} backups found (keeping {args.keep}), no cleanup needed.")
            return
        
        # Sort by modification time (oldest first for deletion)
        backup_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Files to delete (all except the last N)
        files_to_delete = backup_files[:-args.keep]
        
        print(f"Found {len(backup_files)} backup files, keeping {args.keep} most recent:")
        print()
        
        total_size_freed = 0
        for backup_file in files_to_delete:
            size_mb = backup_file.stat().st_size / (1024 * 1024)
            total_size_freed += size_mb
            print(f"  🗑️  {backup_file.name} ({size_mb:.1f} MB)")
        
        print(f"\nTotal space to free: {total_size_freed:.1f} MB")
        print()
        
        # Check for dry-run mode (used in testing)
        if getattr(args, 'dry_run', False):
            print("🧪 DRY-RUN MODE: Cleanup simulation completed successfully")
            print(f"Would have deleted {len(files_to_delete)} files, freeing {total_size_freed:.1f} MB")
            return
        
        response = input("Delete these backup files? (yes/no): ")
        if response.lower() == 'yes':
            deleted_count = 0
            for backup_file in files_to_delete:
                try:
                    backup_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️  Failed to delete {backup_file.name}: {e}")
            
            print(f"✅ Deleted {deleted_count} backup files ({total_size_freed:.1f} MB freed)")
        else:
            print("❌ Cleanup cancelled")
    
    def _verify_backup(self, backup_file: str, args: argparse.Namespace, manager) -> None:
        """Verify backup file integrity."""
        print(f"🔍 VERIFYING BACKUP: {backup_file}")
        print("=" * 40)
        
        backup_path = Path(backup_file)
        if not backup_path.is_absolute():
            backup_path = manager.project_root / "data" / "backups" / backup_file
        
        if not backup_path.exists():
            print(f"❌ Backup file not found: {backup_path}")
            return
        
        try:
            # Check file size
            file_size = backup_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            print(f"File: {backup_path}")
            print(f"Size: {file_size_mb:.1f} MB")
            print(f"Type: {'Compressed' if backup_path.suffix == '.gz' else 'Regular'}")
            print()
            
            # For compressed files, try to read headers
            if backup_path.suffix == '.gz':
                import gzip
                try:
                    with gzip.open(backup_path, 'rb') as f:
                        # Read first few bytes to verify it's a valid gzip file
                        header = f.read(100)
                        if header:
                            print("✅ Gzip compression format valid")
                        else:
                            print("❌ Invalid gzip file")
                            return
                except Exception as e:
                    print(f"❌ Gzip verification failed: {e}")
                    return
            
            # Try to open as DuckDB database (for uncompressed) or verify structure
            if backup_path.suffix != '.gz':
                try:
                    import duckdb
                    with duckdb.connect(str(backup_path)) as conn:
                        # Try basic operations
                        tables = conn.execute("SHOW TABLES").fetchall()
                        table_names = [row[0] for row in tables]
                        
                        print(f"✅ DuckDB format valid")
                        print(f"Tables found: {len(table_names)}")
                        
                        # Check key tables
                        expected_tables = ['sessions', 'exploration_history', 'feature_catalog', 'feature_impact']
                        missing_tables = [t for t in expected_tables if t not in table_names]
                        
                        if missing_tables:
                            print(f"⚠️  Missing expected tables: {missing_tables}")
                        else:
                            print("✅ All expected tables present")
                        
                        # Check data counts
                        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                        print(f"Sessions: {session_count}")
                        
                except Exception as e:
                    print(f"❌ DuckDB verification failed: {e}")
                    return
            
            print("✅ Backup verification completed successfully")
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")