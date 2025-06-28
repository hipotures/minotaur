"""
Backup Module - Database Backup and Restoration

This module provides comprehensive database backup and restoration capabilities including:
- Database backup creation with optional compression
- Backup file listing and management
- Database restoration from backup files
- Backup file cleanup and maintenance
- Backup integrity verification

Architecture:
- Each backup operation is implemented as a separate class
- Base command class provides common functionality
- Repository pattern for data access
- File system operations with error handling
- Compression support for space efficiency
"""

from typing import Dict, Any
from src.manager.core.module_base import ModuleInterface
from .create_command import CreateCommand
from .list_command import ListCommand
from .restore_command import RestoreCommand
from .cleanup_command import CleanupCommand
from .verify_command import VerifyCommand


class BackupModule(ModuleInterface):
    """Main backup module with command routing."""
    
    def __init__(self):
        super().__init__()
        self._name = "backup"
        self._description = "Create and manage database backups"
        self._commands = {}
        self._init_commands()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def commands(self) -> Dict[str, str]:
        return {
            "--create": "Create a new database backup",
            "--list": "List available backup files",
            "--restore": "Restore database from backup",
            "--cleanup": "Remove old backup files",
            "--verify": "Verify backup file integrity"
        }
    
    def _init_commands(self):
        """Initialize all command handlers."""
        self._commands = {
            'create': CreateCommand(),
            'list': ListCommand(),
            'restore': RestoreCommand(),
            'cleanup': CleanupCommand(),
            'verify': VerifyCommand(),
        }
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """Inject services into all commands."""
        super().inject_services(services)
        for command in self._commands.values():
            command.inject_services(services)
    
    def add_arguments(self, parser) -> None:
        """Setup command line arguments."""
        # Main commands (mutually exclusive)
        command_group = parser.add_mutually_exclusive_group(required=True)
        command_group.add_argument('--create', action='store_true',
                                 help='Create database backup')
        command_group.add_argument('--list', action='store_true',
                                 help='List available backups')
        command_group.add_argument('--restore', type=str, metavar='BACKUP_FILE',
                                 help='Restore from backup file')
        command_group.add_argument('--cleanup', action='store_true',
                                 help='Remove old backup files')
        command_group.add_argument('--verify', type=str, metavar='BACKUP_FILE',
                                 help='Verify backup integrity')
        
        # Backup options
        backup_group = parser.add_argument_group('Backup Options')
        backup_group.add_argument('--compress', action='store_true',
                                help='Compress backup file')
        backup_group.add_argument('--keep', type=int, default=5,
                                help='Number of backups to keep (default: 5)')
        
        # Safety options
        safety_group = parser.add_argument_group('Safety Options')
        safety_group.add_argument('--dry-run', action='store_true',
                                help='Simulate operations without making changes')
    
    def execute(self, args, manager) -> None:
        """Execute the appropriate command based on arguments."""
        try:
            if args.create:
                self._commands['create'].execute(args)
            elif args.list:
                self._commands['list'].execute(args)
            elif args.restore:
                self._commands['restore'].execute(args)
            elif args.cleanup:
                self._commands['cleanup'].execute(args)
            elif args.verify:
                self._commands['verify'].execute(args)
            else:
                # Default to list if no specific command
                args.list = True
                self._commands['list'].execute(args)
                
        except Exception as e:
            print(f"❌ Error executing backup command: {e}")
            if hasattr(args, 'debug') and args.debug:
                import traceback
                traceback.print_exc()