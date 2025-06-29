"""
List Command - List available backup files.

Provides backup file listing including:
- Backup file metadata and details
- File size and creation timestamps
- Backup type identification
- Storage space analysis
"""

from typing import List, Dict, Any
from .base import BaseBackupCommand


class ListCommand(BaseBackupCommand):
    """Handle --list command for backup."""
    
    def execute(self, args) -> None:
        """Execute the backup list command."""
        try:
            print("📋 AVAILABLE BACKUPS")
            print("=" * 40)
            
            # Find backup files
            backup_files = self.find_backup_files()
            
            if not backup_files:
                self.print_info("No backup files found.")
                self._show_backup_directory_info()
                return
            
            # Display backup list
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(backup_files)
            else:
                self._output_table(backup_files)
            
            # Show summary information
            self._show_summary(backup_files)
            
        except Exception as e:
            self.print_error(f"Failed to list backups: {e}")
    
    def _output_table(self, backup_files: List) -> None:
        """Output backup files in table format."""
        # Table headers
        headers = ['File', 'Size', 'Created', 'Type']
        rows = []
        
        for backup_file in backup_files:
            info = self.format_backup_info(backup_file)
            
            # Truncate filename if too long
            file_name = info['name']
            if len(file_name) > 29:
                file_name = file_name[:26] + "..."
            
            rows.append([
                file_name,
                info['size'],
                info['created'],
                info['type']
            ])
        
        self.print_table(headers, rows)
    
    def _output_json(self, backup_files: List) -> None:
        """Output backup files in JSON format."""
        backup_data = []
        
        for backup_file in backup_files:
            info = self.format_backup_info(backup_file)
            backup_data.append({
                'filename': info['name'],
                'path': info['path'],
                'size_mb': float(info['size'].replace(' MB', '')),
                'created': info['created'],
                'type': info['type'],
                'compressed': info['type'] == 'Compressed'
            })
        
        self.print_json(backup_data, "Available Backups")
    
    def _show_summary(self, backup_files: List) -> None:
        """Show summary information about backups."""
        if not backup_files:
            return
        
        total_count = len(backup_files)
        total_size_mb = 0
        compressed_count = 0
        
        for backup_file in backup_files:
            try:
                size_mb = backup_file.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb
                
                if backup_file.suffix == '.gz':
                    compressed_count += 1
                    
            except Exception:
                continue
        
        regular_count = total_count - compressed_count
        
        print(f"\nTotal backups: {total_count}")
        print(f"Regular backups: {regular_count}")
        print(f"Compressed backups: {compressed_count}")
        print(f"Total size: {total_size_mb:.1f} MB")
        
        # Show backup directory
        backup_dir = self.get_backup_directory()
        print(f"Backup directory: {backup_dir}")
        
        # Quick actions
        if backup_files:
            latest_backup = backup_files[0].name  # Already sorted by newest first
            print(f"\n💡 Quick Actions:")
            print(f"   Verify latest: python manager.py backup --verify {latest_backup}")
            print(f"   Create new: python manager.py backup --create")
            print(f"   Cleanup old: python manager.py backup --cleanup --keep 5")
    
    def _show_backup_directory_info(self) -> None:
        """Show information about backup directory when no backups found."""
        backup_dir = self.get_backup_directory()
        
        if backup_dir.exists():
            print(f"Backup directory: {backup_dir}")
            print("💡 Create your first backup: python manager.py backup --create")
        else:
            print(f"Backup directory will be created at: {backup_dir}")
            print("💡 Create your first backup: python manager.py backup --create")