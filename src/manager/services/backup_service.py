"""
Service layer for backup and restore operations using new database abstraction.
"""

import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..core.config import Config


class BackupService:
    """Handles backup and restore operations using new database abstraction."""
    
    def __init__(self, config: Config, db_manager):
        """Initialize service with configuration and database manager.
        
        Args:
            config: Configuration instance
            db_manager: Database manager instance from factory
        """
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def create_backup(self, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a backup of the database.
        
        Args:
            description: Optional backup description
            
        Returns:
            Backup result with path and metadata
        """
        try:
            # Ensure backup directory exists
            backup_dir = self.config.backup_path
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{self.config.backup_prefix}{timestamp}.duckdb"
            backup_path = backup_dir / backup_filename
            
            # Get database path from config
            db_path = self.config.database_path
            
            if not db_path.exists():
                raise FileNotFoundError(f"Database file not found: {db_path}")
            
            # Create backup by copying database file
            shutil.copy2(db_path, backup_path)
            
            # Create metadata file
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'description': description or 'Manual backup',
                'original_path': str(db_path),
                'backup_path': str(backup_path),
                'file_size_mb': backup_path.stat().st_size / 1024 / 1024
            }
            
            metadata_path = backup_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Created backup: {backup_path}")
            
            return {
                'success': True,
                'backup_path': str(backup_path),
                'metadata_path': str(metadata_path),
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        backup_dir = self.config.backup_path
        
        if not backup_dir.exists():
            return backups
        
        # Find all backup files
        for backup_file in backup_dir.glob(f"{self.config.backup_prefix}*.duckdb"):
            try:
                # Try to load metadata
                metadata_file = backup_file.with_suffix('.json')
                metadata = {}
                
                if metadata_file.exists():
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                backup_info = {
                    'backup_path': str(backup_file),
                    'filename': backup_file.name,
                    'size_mb': backup_file.stat().st_size / 1024 / 1024,
                    'created_at': datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                    'description': metadata.get('description', 'No description'),
                    'metadata': metadata
                }
                
                backups.append(backup_info)
                
            except Exception as e:
                self.logger.warning(f"Could not read backup info for {backup_file}: {e}")
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        return backups
    
    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restore from a backup file.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Restore result
        """
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Get current database path
            current_db_path = self.config.database_path
            
            # Create backup of current database first
            if current_db_path.exists():
                current_backup_path = current_db_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.duckdb')
                shutil.copy2(current_db_path, current_backup_path)
                self.logger.info(f"Created safety backup: {current_backup_path}")
            
            # Close any existing database connections
            if hasattr(self.db_manager, 'close'):
                self.db_manager.close()
            
            # Restore backup
            shutil.copy2(backup_file, current_db_path)
            
            self.logger.info(f"Restored backup from: {backup_path}")
            
            return {
                'success': True,
                'restored_from': str(backup_path),
                'restored_to': str(current_db_path)
            }
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_backup(self, backup_path: str) -> Dict[str, Any]:
        """Delete a backup file.
        
        Args:
            backup_path: Path to backup file to delete
            
        Returns:
            Deletion result
        """
        try:
            backup_file = Path(backup_path)
            metadata_file = backup_file.with_suffix('.json')
            
            if backup_file.exists():
                backup_file.unlink()
                self.logger.info(f"Deleted backup: {backup_path}")
            
            if metadata_file.exists():
                metadata_file.unlink()
            
            return {
                'success': True,
                'deleted_path': str(backup_path)
            }
            
        except Exception as e:
            self.logger.error(f"Delete backup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_backup(self, backup_path: str) -> Dict[str, Any]:
        """Verify a backup file integrity.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Verification result
        """
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Try to open backup file with DuckDB to verify integrity
            try:
                import duckdb
                conn = duckdb.connect(str(backup_file), read_only=True)
                
                # Test basic query
                result = conn.execute("SELECT COUNT(*) FROM information_schema.tables").fetchone()
                table_count = result[0] if result else 0
                
                conn.close()
                
                return {
                    'success': True,
                    'valid': True,
                    'table_count': table_count,
                    'file_size_mb': backup_file.stat().st_size / 1024 / 1024
                }
                
            except Exception as db_error:
                return {
                    'success': True,
                    'valid': False,
                    'error': str(db_error)
                }
                
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics.
        
        Returns:
            Backup statistics
        """
        backups = self.list_backups()
        
        if not backups:
            return {
                'total_backups': 0,
                'total_size_mb': 0.0,
                'latest_backup': None,
                'oldest_backup': None
            }
        
        total_size = sum(backup['size_mb'] for backup in backups)
        
        return {
            'total_backups': len(backups),
            'total_size_mb': round(total_size, 2),
            'latest_backup': backups[0]['created_at'] if backups else None,
            'oldest_backup': backups[-1]['created_at'] if backups else None,
            'backup_directory': str(self.config.backup_path)
        }