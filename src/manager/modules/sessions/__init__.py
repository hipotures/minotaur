"""
Sessions Module - MCTS Session Management

This module provides comprehensive session management capabilities including:
- Session listing with summary statistics and filtering
- Detailed session information and analysis
- Session comparison and performance benchmarking
- Session data export in multiple formats
- Session cleanup and maintenance

Architecture:
- Each command is implemented as a separate class
- Base command class provides common functionality
- Repository pattern for data access
- Service layer for business logic
"""

from typing import Dict, Any
from src.manager.core.module_base import ModuleInterface
from .list_command import ListCommand
from .show_command import ShowCommand
from .compare_command import CompareCommand
from .export_command import ExportCommand
from .cleanup_command import CleanupCommand


class SessionsModule(ModuleInterface):
    """Main sessions module with command routing."""
    
    def __init__(self):
        super().__init__()
        self._name = "sessions"
        self._description = "Manage and analyze MCTS discovery sessions"
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
            "--list": "List recent sessions with summary statistics",
            "--show": "Show detailed information about a specific session",
            "--compare": "Compare performance between multiple sessions",
            "--export": "Export session data to CSV/JSON",
            "--cleanup": "Remove old or incomplete sessions"
        }
    
    def _init_commands(self):
        """Initialize all command handlers."""
        self._commands = {
            'list': ListCommand(),
            'show': ShowCommand(),
            'compare': CompareCommand(),
            'export': ExportCommand(),
            'cleanup': CleanupCommand(),
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
                                 help='List recent sessions with summary statistics')
        command_group.add_argument('--show', type=str, metavar='SESSION_ID',
                                 help='Show detailed information about a specific session')
        command_group.add_argument('--compare', nargs='+', metavar='SESSION_ID',
                                 help='Compare performance between multiple sessions')
        command_group.add_argument('--export', type=str, metavar='FORMAT',
                                 choices=['csv', 'json'], 
                                 help='Export session data to CSV/JSON format')
        command_group.add_argument('--cleanup', action='store_true',
                                 help='Remove old or incomplete sessions')
        
        # Filter options (shared across commands)
        filter_group = parser.add_argument_group('Filter Options')
        filter_group.add_argument('--status', type=str,
                                choices=['active', 'completed', 'interrupted', 'all'],
                                default='all', help='Filter by session status')
        filter_group.add_argument('--limit', type=int, default=10,
                                help='Limit number of sessions to show')
        filter_group.add_argument('--strategy', type=str, metavar='STRATEGY',
                                help='Filter by search strategy')
        
        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument('--format', choices=['table', 'json'], default='table',
                                help='Output format')
        output_group.add_argument('--output-file', type=str, metavar='FILE',
                                help='Save output to file')
        output_group.add_argument('--dry-run', action='store_true',
                                help='Simulate cleanup without making changes')
    
    def execute(self, args, manager) -> None:
        """Execute the appropriate command based on arguments."""
        try:
            if args.list:
                self._commands['list'].execute(args)
            elif args.show:
                self._commands['show'].execute(args)
            elif args.compare:
                self._commands['compare'].execute(args)
            elif args.export:
                self._commands['export'].execute(args)
            elif args.cleanup:
                self._commands['cleanup'].execute(args)
            else:
                # Default to list if no specific command
                args.list = True
                self._commands['list'].execute(args)
                
        except Exception as e:
            print(f"❌ Error executing sessions command: {e}")
            if hasattr(args, 'debug') and args.debug:
                import traceback
                traceback.print_exc()