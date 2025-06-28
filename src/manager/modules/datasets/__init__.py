"""
Datasets Module - Dataset Registration and Management

This module provides comprehensive dataset lifecycle management including:
- Dataset registration (auto-detection and manual)
- Dataset listing and search functionality
- Detailed dataset information and statistics
- Session tracking per dataset
- Safe dataset cleanup with backup
- Feature regeneration and updates

Architecture:
- Each command is implemented as a separate class
- Base command class provides common functionality
- Repository pattern for data access
- Service layer for business logic
"""

from typing import Dict, Any
from src.manager.core.module_base import ModuleInterface
from .list import ListCommand
from .details import DetailsCommand
from .register import RegisterCommand
from .cleanup import CleanupCommand
from .stats import StatsCommand
from .sessions import SessionsCommand
from .search import SearchCommand
from .update import UpdateCommand


class DatasetsModule(ModuleInterface):
    """Main datasets module with command routing."""
    
    def __init__(self):
        super().__init__()
        self._name = "datasets"
        self._description = "Dataset registration and management"
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
            "--list": "List all registered datasets with usage statistics",
            "--details": "Show detailed information about a specific dataset",
            "--register": "Register a new dataset manually",
            "--cleanup": "Safely remove dataset and all related data",
            "--stats": "Show dataset usage statistics and comparisons",
            "--sessions": "Show sessions using a specific dataset",
            "--search": "Search datasets by name or path",
            "--update": "Update dataset metadata"
        }
    
    def _init_commands(self):
        """Initialize all command handlers."""
        self._commands = {
            'list': ListCommand(),
            'details': DetailsCommand(),
            'register': RegisterCommand(),
            'cleanup': CleanupCommand(),
            'stats': StatsCommand(),
            'sessions': SessionsCommand(),
            'search': SearchCommand(),
            'update': UpdateCommand(),
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
        command_group.add_argument('--list', action='store_true',
                                 help='List all registered datasets')
        command_group.add_argument('--details', type=str, metavar='DATASET',
                                 help='Show detailed dataset information')
        command_group.add_argument('--register', action='store_true',
                                 help='Register a new dataset')
        command_group.add_argument('--cleanup', type=str, metavar='DATASET',
                                 help='Safely remove dataset and related data')
        command_group.add_argument('--stats', action='store_true',
                                 help='Show dataset usage statistics')
        command_group.add_argument('--sessions', type=str, metavar='DATASET',
                                 help='Show sessions using specific dataset')
        command_group.add_argument('--search', type=str, metavar='QUERY',
                                 help='Search datasets by name or path')
        command_group.add_argument('--update', type=str, metavar='DATASET',
                                 help='Update dataset metadata')
        
        # Registration options
        register_group = parser.add_argument_group('Registration Options')
        register_group.add_argument('--auto', action='store_true',
                                  help='Auto-detect dataset files')
        register_group.add_argument('--dataset-name', type=str,
                                  help='Dataset name for registration')
        register_group.add_argument('--dataset-path', type=str,
                                  help='Path to dataset directory')
        register_group.add_argument('--train', type=str,
                                  help='Training data file path')
        register_group.add_argument('--test', type=str,
                                  help='Test data file path')
        register_group.add_argument('--submission', type=str,
                                  help='Submission template file path')
        register_group.add_argument('--validation', type=str,
                                  help='Validation data file path')
        register_group.add_argument('--target-column', type=str,
                                  help='Target column name')
        register_group.add_argument('--id-column', type=str,
                                  help='ID column name')
        register_group.add_argument('--competition-name', type=str,
                                  help='Competition name')
        register_group.add_argument('--description', type=str,
                                  help='Dataset description')
        
        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument('--format', choices=['table', 'json'], default='table',
                                help='Output format')
        output_group.add_argument('--active-only', action='store_true',
                                help='Show only active datasets')
        
        # Update options
        update_group = parser.add_argument_group('Update Options')
        update_group.add_argument('--update-features', action='store_true',
                                help='Regenerate features for dataset')
        update_group.add_argument('--dry-run', action='store_true',
                                help='Show what would be changed')
        update_group.add_argument('--force-update', action='store_true',
                                help='Force feature regeneration')
    
    def execute(self, args, manager) -> None:
        """Execute the appropriate command based on arguments."""
        try:
            if args.list:
                self._commands['list'].execute(args)
            elif args.details:
                self._commands['details'].execute(args)
            elif args.register:
                self._commands['register'].execute(args)
            elif args.cleanup:
                self._commands['cleanup'].execute(args)
            elif args.stats:
                self._commands['stats'].execute(args)
            elif args.sessions:
                self._commands['sessions'].execute(args)
            elif args.search:
                self._commands['search'].execute(args)
            elif args.update:
                self._commands['update'].execute(args)
            else:
                print("No command specified. Use --help for usage information.")
                
        except Exception as e:
            print(f"❌ Error executing datasets command: {e}")
            if hasattr(args, 'debug') and args.debug:
                import traceback
                traceback.print_exc()