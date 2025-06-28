"""
Self-Check Module - Comprehensive System Validation and Testing

This module provides automated testing of all DuckDB manager modules and MCTS functionality including:
- Dataset validation and registry verification
- Database connectivity and schema validation
- MCTS integration testing with real feature discovery
- Module command testing with comprehensive coverage
- System health checks and performance validation

Architecture:
- Each validation category is implemented as a separate class
- Base command class provides common functionality
- Repository pattern for data access
- MCTS integration with temporary configuration
- Comprehensive test result reporting
"""

from typing import Dict, Any
from src.manager.core.module_base import ModuleInterface
from .run_command import RunCommand
from .list_datasets_command import ListDatasetsCommand


class SelfcheckModule(ModuleInterface):
    """Main self-check module with command routing."""
    
    def __init__(self):
        super().__init__()
        self._name = "selfcheck"
        self._description = "Comprehensive system validation and testing"
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
            "--run": "Run comprehensive self-check tests",
            "--list-datasets": "List available test datasets"
        }
    
    def _init_commands(self):
        """Initialize all command handlers."""
        self._commands = {
            'run': RunCommand(),
            'list_datasets': ListDatasetsCommand(),
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
        command_group.add_argument('--run', type=str, metavar='DATASET', nargs='?',
                                 const='titanic', default='titanic',
                                 help='Run self-check with specified dataset (default: titanic)')
        command_group.add_argument('--list-datasets', action='store_true',
                                 help='List available test datasets')
        
        # Test options
        test_group = parser.add_argument_group('Test Options')
        test_group.add_argument('--quick', action='store_true',
                              help='Quick validation without MCTS test')
        test_group.add_argument('--verbose', action='store_true',
                              help='Enable verbose output')
        test_group.add_argument('--config', type=str, metavar='CONFIG_FILE',
                              help='Use custom MCTS configuration file')
    
    def execute(self, args, manager) -> None:
        """Execute the appropriate command based on arguments."""
        try:
            if args.list_datasets:
                self._commands['list_datasets'].execute(args)
            elif hasattr(args, 'run') and args.run:
                self._commands['run'].execute(args)
            else:
                self.print_error("No self-check command specified. Use --help for options.")
                
        except Exception as e:
            print(f"❌ Error executing self-check command: {e}")
            if hasattr(args, 'debug') and args.debug:
                import traceback
                traceback.print_exc()