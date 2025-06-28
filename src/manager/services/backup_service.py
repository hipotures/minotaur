"""
Service layer for backup and restore operations
"""

import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..core.config import Config
from ..core.database import DatabasePool


class BackupService:
    """Handles backup and restore operations."""
    
    def __init__(self, config: Config, db_pool: DatabasePool):
        """Initialize service with configuration and database pool.
        
        Args:
            config: Configuration instance
            db_pool: Database connection pool
        """
        self.config = config
        self.db_pool = db_pool
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
            
            # Create metadata file
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'description': description or 'Manual backup',
                'source_path': str(self.config.database_path),
                'backup_path': str(backup_path),
                'file_size': self.config.database_path.stat().st_size
            }
            
            # Close all connections before backup
            self.db_pool.close_all()
            
            try:
                # Copy database file
                shutil.copy2(self.config.database_path, backup_path)
                
                # Write metadata
                metadata_path = backup_path.with_suffix('.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info(f"Backup created: {backup_path}")
                
                return {
                    'success': True,
                    'backup_path': str(backup_path),
                    'metadata': metadata,
                    'message': f'Backup created successfully: {backup_filename}'
                }
                
            finally:
                # Reconnect to database
                with self.db_pool.get_connection():
                    pass  # Just establish connection
                    
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups.
        
        Returns:
            List of backup metadata
        """
        backup_dir = self.config.backup_path
        backups = []
        
        if not backup_dir.exists():
            return backups
        
        # Find all backup files
        for backup_file in sorted(backup_dir.glob(f"{self.config.backup_prefix}*.duckdb"), 
                                reverse=True):
            metadata_file = backup_file.with_suffix('.json')
            
            # Read metadata if available
            if metadata_file.exists():
                import json
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    metadata = {}
            else:
                # Generate basic metadata
                stat = backup_file.stat()
                timestamp = backup_file.stem.replace(self.config.backup_prefix, '')
                
                metadata = {
                    'timestamp': timestamp,
                    'file_size': stat.st_size,
                    'backup_path': str(backup_file)
                }
            
            backups.append({
                'filename': backup_file.name,
                'path': str(backup_file),
                'size': metadata.get('file_size', backup_file.stat().st_size),
                'created': metadata.get('timestamp', 'Unknown'),
                'description': metadata.get('description', 'No description')
            })
        
        return backups
    
    def restore_backup(self, backup_filename: str, confirm: bool = False) -> Dict[str, Any]:
        """Restore database from a backup.
        
        Args:
            backup_filename: Name of backup file to restore
            confirm: Safety confirmation flag
            
        Returns:
            Restore result
        """
        if not confirm:
            return {
                'success': False,
                'error': 'Restore requires confirmation. Use --confirm flag.'
            }
        
        try:
            backup_path = self.config.backup_path / backup_filename
            
            if not backup_path.exists():
                return {
                    'success': False,
                    'error': f'Backup file not found: {backup_filename}'
                }
            
            # Create safety backup of current database
            safety_backup = self.create_backup('Pre-restore safety backup')
            
            if not safety_backup['success']:
                return {
                    'success': False,
                    'error': 'Failed to create safety backup before restore'
                }
            
            # Close all connections
            self.db_pool.close_all()
            
            try:
                # Restore the backup
                shutil.copy2(backup_path, self.config.database_path)
                
                self.logger.info(f"Database restored from: {backup_path}")
                
                return {
                    'success': True,
                    'message': f'Database restored from {backup_filename}',
                    'safety_backup': safety_backup['backup_path']
                }
                
            finally:
                # Reconnect to database
                with self.db_pool.get_connection():
                    pass
                    
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_backup(self, backup_filename: str) -> Dict[str, Any]:
        """Delete a backup file.
        
        Args:
            backup_filename: Name of backup to delete
            
        Returns:
            Deletion result
        """
        try:
            backup_path = self.config.backup_path / backup_filename
            metadata_path = backup_path.with_suffix('.json')
            
            if not backup_path.exists():
                return {
                    'success': False,
                    'error': f'Backup not found: {backup_filename}'
                }
            
            # Delete backup and metadata
            backup_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            return {
                'success': True,
                'message': f'Backup deleted: {backup_filename}'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to delete backup: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_backup_info(self, backup_filename: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a backup.
        
        Args:
            backup_filename: Name of backup file
            
        Returns:
            Backup information or None
        """
        backup_path = self.config.backup_path / backup_filename
        
        if not backup_path.exists():
            return None
        
        metadata_path = backup_path.with_suffix('.json')
        
        # Read metadata if available
        metadata = {}
        if metadata_path.exists():
            import json
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Get file stats
        stat = backup_path.stat()
        
        return {
            'filename': backup_filename,
            'path': str(backup_path),
            'size': stat.st_size,
            'size_formatted': f"{stat.st_size / (1024*1024):.2f} MB",
            'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'metadata': metadata,
            'has_metadata': metadata_path.exists()
        }
    
    def cleanup_old_backups(self, days_to_keep: int = 30, 
                          dry_run: bool = True) -> Dict[str, Any]:
        """Clean up old backup files.
        
        Args:
            days_to_keep: Keep backups newer than this many days
            dry_run: If True, only show what would be deleted
            
        Returns:
            Cleanup results
        """
        backup_dir = self.config.backup_path
        if not backup_dir.exists():
            return {
                'success': True,
                'deleted': 0,
                'message': 'No backups found'
            }
        
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 86400)
        
        to_delete = []
        total_size = 0
        
        for backup_file in backup_dir.glob(f"{self.config.backup_prefix}*.duckdb"):
            stat = backup_file.stat()
            if stat.st_mtime < cutoff_date:
                to_delete.append(backup_file)
                total_size += stat.st_size
        
        if dry_run:
            return {
                'success': True,
                'would_delete': len(to_delete),
                'total_size': total_size,
                'backups': [f.name for f in to_delete],
                'message': f'Would delete {len(to_delete)} backups ({total_size / (1024*1024):.2f} MB)'
            }
        else:
            deleted = 0
            for backup_file in to_delete:
                try:
                    backup_file.unlink()
                    metadata_file = backup_file.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    deleted += 1
                except Exception as e:
                    self.logger.error(f"Failed to delete {backup_file}: {e}")
            
            return {
                'success': True,
                'deleted': deleted,
                'total_size': total_size,
                'message': f'Deleted {deleted} backups ({total_size / (1024*1024):.2f} MB)'
            }