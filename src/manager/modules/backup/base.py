"""
Base Backup Command - Common functionality for backup commands.

Provides shared utilities including:
- Backup directory management
- File operations and path handling
- Compression utilities
- Database connectivity testing
- Error handling and validation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import shutil
from src.manager.core.command_base import BaseCommand


class BaseBackupCommand(BaseCommand, ABC):
    """Base class for all backup commands."""
    
    def __init__(self):
        super().__init__()
        self.backup_service = None
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """Inject required services."""
        super().inject_services(services)
        self.backup_service = services.get('backup_service')
        if not self.backup_service:
            raise ValueError("BackupService is required for backup commands")
    
    @abstractmethod
    def execute(self, args) -> None:
        """Execute the command with given arguments."""
        pass
    
    def get_backup_directory(self) -> Path:
        """Get the backup directory path."""
        if hasattr(self.config, 'get_backup_config'):
            backup_config = self.config.get_backup_config()
            backup_dir = Path(backup_config.get('backup_path', 'data/backups'))
        else:
            backup_dir = Path('data/backups')
        
        # Ensure directory exists
        backup_dir.mkdir(parents=True, exist_ok=True)
        return backup_dir
    
    def get_backup_prefix(self) -> str:
        """Get the backup filename prefix."""
        if hasattr(self.config, 'get_backup_config'):
            backup_config = self.config.get_backup_config()
            return backup_config.get('backup_prefix', 'backup_')
        return 'backup_'
    
    def get_database_path(self) -> Path:
        """Get the main database file path."""
        if self.config:
            return Path(self.config.database_path)
        return Path('data/minotaur.duckdb')
    
    def find_backup_files(self) -> List[Path]:
        """Find all backup files in the backup directory."""
        backup_dir = self.get_backup_directory()
        backup_prefix = self.get_backup_prefix()
        
        # Find all backup files (both compressed and uncompressed)
        backup_files = list(backup_dir.glob(f"{backup_prefix}*.duckdb*"))
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return backup_files
    
    def resolve_backup_path(self, backup_file: str) -> Path:
        """Resolve backup file path (absolute or relative to backup directory)."""
        backup_path = Path(backup_file)
        
        if not backup_path.is_absolute():
            backup_dir = self.get_backup_directory()
            backup_path = backup_dir / backup_file
        
        return backup_path
    
    def generate_backup_filename(self, compress: bool = False) -> str:
        """Generate a timestamped backup filename."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = self.get_backup_prefix()
        
        if compress:
            return f"{prefix}{timestamp}.duckdb.gz"
        else:
            return f"{prefix}{timestamp}.duckdb"
    
    def copy_with_compression(self, source: Path, target: Path, compress: bool = False) -> None:
        """Copy file with optional compression."""
        if compress:
            import gzip
            with open(source, 'rb') as f_in:
                with gzip.open(target, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(source, target)
    
    def extract_from_backup(self, source: Path, target: Path) -> None:
        """Extract file from backup (handle compressed and uncompressed)."""
        if source.suffix == '.gz':
            import gzip
            with gzip.open(source, 'rb') as f_in:
                with open(target, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(source, target)
    
    def verify_database_connectivity(self, db_path: Path) -> Optional[Dict[str, Any]]:
        """Test database connectivity and get basic info."""
        try:
            import duckdb
            with duckdb.connect(str(db_path)) as conn:
                # Get basic database info
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [row[0] for row in tables]
                
                info = {
                    'tables': table_names,
                    'table_count': len(table_names)
                }
                
                # Get session count if sessions table exists
                if 'sessions' in table_names:
                    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                    info['session_count'] = session_count
                
                return info
                
        except Exception as e:
            self.print_error(f"Database connectivity test failed: {e}")
            return None
    
    def verify_backup_integrity(self, backup_path: Path) -> bool:
        """Verify backup file integrity."""
        try:
            # Check file exists and has reasonable size
            if not backup_path.exists():
                self.print_error(f"Backup file not found: {backup_path}")
                return False
            
            file_size = backup_path.stat().st_size
            if file_size == 0:
                self.print_error("Backup file is empty")
                return False
            
            # For compressed files, verify gzip format
            if backup_path.suffix == '.gz':
                import gzip
                try:
                    with gzip.open(backup_path, 'rb') as f:
                        # Try to read first chunk
                        chunk = f.read(1024)
                        if not chunk:
                            self.print_error("Compressed backup appears to be empty")
                            return False
                except Exception as e:
                    self.print_error(f"Compressed backup verification failed: {e}")
                    return False
            
            # For uncompressed files, try to open as DuckDB
            else:
                db_info = self.verify_database_connectivity(backup_path)
                if not db_info:
                    return False
            
            return True
            
        except Exception as e:
            self.print_error(f"Backup integrity verification failed: {e}")
            return False
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        return self.format_file_size(size_bytes)  # Use inherited method
    
    def format_backup_info(self, backup_path: Path) -> Dict[str, str]:
        """Get formatted backup file information."""
        try:
            stat = backup_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            created = datetime.fromtimestamp(stat.st_mtime)
            backup_type = "Compressed" if backup_path.suffix == '.gz' else "Regular"
            
            return {
                'name': backup_path.name,
                'size': f"{size_mb:.1f} MB",
                'created': created.strftime('%Y-%m-%d %H:%M:%S'),
                'type': backup_type,
                'path': str(backup_path)
            }
            
        except Exception as e:
            return {
                'name': backup_path.name,
                'size': 'Unknown',
                'created': 'Unknown', 
                'type': 'Unknown',
                'path': str(backup_path)
            }
    
    def confirm_operation(self, message: str, dry_run: bool = False) -> bool:
        """Confirm operation with user (skip in dry-run mode)."""
        if dry_run:
            return True
        
        try:
            response = input(f"{message} (yes/no): ").strip().lower()
            return response == 'yes'
        except (EOFError, KeyboardInterrupt):
            print("\\nOperation cancelled by user")
            return False