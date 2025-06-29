"""
Verification Module - Comprehensive MCTS Session Verification

This module provides comprehensive validation of MCTS discovery sessions including:
- Database integrity verification and consistency checks
- MCTS algorithm correctness validation
- AutoGluon integration verification
- File system and cache validation
- Configuration and environment validation

Architecture:
- Each verification category is implemented as a separate class
- Base command class provides common functionality
- Repository pattern for data access
- Multiple output formats (text, JSON, HTML)
- Batch verification with filtering capabilities
"""

from typing import Dict, Any
from manager.core.module_base import ModuleInterface
from .verify_session_command import VerifySessionCommand
from .verify_all_command import VerifyAllCommand
from .verify_latest_command import VerifyLatestCommand
from .verify_last_command import VerifyLastCommand
from .report_command import ReportCommand


class VerificationModule(ModuleInterface):
    """Main verification module with command routing."""
    
    def __init__(self):
        super().__init__()
        self._name = "verification"
        self._description = "Comprehensive MCTS session verification and validation"
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
            "--verify-session": "Verify specific session integrity and correctness",
            "--verify-all": "Verify all sessions in database",
            "--verify-latest": "Verify most recent session",
            "--last": "Verify last N sessions (default: 1 if no number given)",
            "--report": "Generate detailed verification report"
        }
    
    def _init_commands(self):
        """Initialize all command handlers."""
        self._commands = {
            'verify_session': VerifySessionCommand(),
            'verify_all': VerifyAllCommand(),
            'verify_latest': VerifyLatestCommand(),
            'verify_last': VerifyLastCommand(),
            'report': ReportCommand(),
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
        command_group.add_argument('--verify-session', type=str, metavar='SESSION_ID',
                                 help='Verify specific session by ID or name')
        command_group.add_argument('--verify-all', action='store_true',
                                 help='Verify all sessions in database')
        command_group.add_argument('--verify-latest', action='store_true',
                                 help='Verify most recent session')
        command_group.add_argument('--last', type=int, metavar='N', nargs='?',
                                 const=1, default=None,
                                 help='Verify last N sessions (default: 1 if no number given)')
        command_group.add_argument('--report', type=str, metavar='SESSION_ID',
                                 help='Generate detailed report for specific session')
        
        # Filter options (for batch operations)
        filter_group = parser.add_argument_group('Filter Options')
        filter_group.add_argument('--failed-only', action='store_true',
                                help='Show only sessions with FAIL status')
        filter_group.add_argument('--warn-only', action='store_true',
                                help='Show only sessions with WARN status')
        filter_group.add_argument('--pass-only', action='store_true',
                                help='Show only sessions with PASS status')
        
        # Verification options
        verification_group = parser.add_argument_group('Verification Options')
        verification_group.add_argument('--quick', action='store_true',
                                      help='Quick verification (skip detailed analysis)')
        verification_group.add_argument('--verbose', action='store_true',
                                      help='Verbose output with detailed diagnostics')
        
        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument('--format', choices=['text', 'json', 'html'], default='text',
                                help='Output format (default: text)')
        output_group.add_argument('--output-file', type=str, metavar='FILE',
                                help='Save output to file')
    
    def execute(self, args, manager) -> None:
        """Execute the appropriate command based on arguments."""
        try:
            if args.verify_session:
                self._commands['verify_session'].execute(args)
            elif args.verify_all:
                self._commands['verify_all'].execute(args)
            elif args.verify_latest:
                self._commands['verify_latest'].execute(args)
            elif hasattr(args, 'last') and args.last is not None:
                self._commands['verify_last'].execute(args)
            elif args.report:
                self._commands['report'].execute(args)
            else:
                # Default to latest session verification
                args.verify_latest = True
                self._commands['verify_latest'].execute(args)
                
        except Exception as e:
            print(f"❌ Error executing verification command: {e}")
            if hasattr(args, 'debug') and args.debug:
                import traceback
                traceback.print_exc()