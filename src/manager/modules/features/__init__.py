"""
Features Module - Feature Analysis and Management

This module provides comprehensive feature analysis capabilities including:
- Feature listing with performance metrics and filtering
- Top performing features identification  
- Detailed feature impact analysis
- Feature catalog overview and statistics
- Feature data export in multiple formats
- Feature search by name, category, or description

Architecture:
- Each command is implemented as a separate class
- Base command class provides common functionality
- Repository pattern for data access
- Service layer for business logic
"""

from typing import Dict, Any
from src.manager.core.module_base import ModuleInterface
from .list_command import ListCommand
from .top_command import TopCommand
from .impact_command import ImpactCommand
from .catalog_command import CatalogCommand
from .export_command import ExportCommand
from .search_command import SearchCommand


class FeaturesModule(ModuleInterface):
    """Main features module with command routing."""
    
    def __init__(self):
        super().__init__()
        self._name = "features"
        self._description = "Feature analysis and management"
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
            "--list": "List all features with performance metrics and filtering",
            "--top": "Show top N performing features (default: 10)",
            "--impact": "Show detailed impact analysis for specific feature",
            "--catalog": "Show feature catalog summary with statistics",
            "--export": "Export feature data to CSV/JSON format",
            "--search": "Search features by name, category, or description"
        }
    
    def _init_commands(self):
        """Initialize all command handlers."""
        self._commands = {
            'list': ListCommand(),
            'top': TopCommand(),
            'impact': ImpactCommand(),
            'catalog': CatalogCommand(),
            'export': ExportCommand(),
            'search': SearchCommand(),
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
                                 help='List all features with performance metrics')
        command_group.add_argument('--top', type=int, metavar='N',
                                 help='Show top N performing features (default: 10)')
        command_group.add_argument('--impact', type=str, metavar='FEATURE',
                                 help='Show detailed impact analysis for specific feature')
        command_group.add_argument('--catalog', action='store_true',
                                 help='Show feature catalog summary')
        command_group.add_argument('--export', type=str, metavar='FORMAT',
                                 choices=['csv', 'json'], 
                                 help='Export feature data (csv/json)')
        command_group.add_argument('--search', type=str, metavar='QUERY',
                                 help='Search features by name/category/description')
        
        # Filter options (shared across commands)
        filter_group = parser.add_argument_group('Filter Options')
        filter_group.add_argument('--session', type=str, metavar='SESSION_ID',
                                help='Filter by session ID')
        filter_group.add_argument('--category', type=str, metavar='CATEGORY',
                                help='Filter by feature category')
        filter_group.add_argument('--dataset', type=str, metavar='HASH_OR_NAME',
                                help='Filter by dataset hash or name')
        filter_group.add_argument('--dataset-name', type=str, metavar='NAME',
                                help='Filter by dataset name')
        filter_group.add_argument('--min-impact', type=float, metavar='THRESHOLD',
                                help='Minimum impact threshold')
        
        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument('--format', choices=['table', 'json'], default='table',
                                help='Output format')
        output_group.add_argument('--output-file', type=str, metavar='FILE',
                                help='Save output to file')
        output_group.add_argument('--limit', type=int, metavar='N',
                                help='Limit number of results')
    
    def execute(self, args, manager) -> None:
        """Execute the appropriate command based on arguments."""
        try:
            if args.list:
                self._commands['list'].execute(args)
            elif hasattr(args, 'top') and args.top is not None:
                # Handle --top N argument
                args.top_n = args.top if isinstance(args.top, int) else 10
                self._commands['top'].execute(args)
            elif args.impact:
                self._commands['impact'].execute(args)
            elif args.catalog:
                self._commands['catalog'].execute(args)
            elif args.export:
                self._commands['export'].execute(args)
            elif args.search:
                self._commands['search'].execute(args)
            else:
                # Default to top 10 if no specific command
                args.top_n = 10
                self._commands['top'].execute(args)
                
        except Exception as e:
            print(f"❌ Error executing features command: {e}")
            if hasattr(args, 'debug') and args.debug:
                import traceback
                traceback.print_exc()