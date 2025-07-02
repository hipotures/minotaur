"""
Analytics Module - Statistical analysis and reporting

Provides commands for generating reports, visualizations, and statistical 
analysis of MCTS performance.
"""

import argparse
from typing import Dict
try:
    from src.manager.core.module_base import ModuleInterface
    from src.manager.core.database import DatabaseConnection
    from src.manager.services.analytics_service import AnalyticsService
except ImportError as e:
    print(f"DEBUG: Import failed: {e}")
    # Skip if dependencies not available
    ModuleInterface = object
    DatabaseConnection = None
    AnalyticsService = None

# Import command handlers
try:
    from .summary import SummaryCommand
    from .trends import TrendsCommand
    from .operations import OperationsCommand
    from .convergence import ConvergenceCommand
    from .report import ReportCommand
    from .compare import CompareCommand
except ImportError:
    # Fallback for when relative imports fail
    SummaryCommand = None
    TrendsCommand = None
    OperationsCommand = None
    ConvergenceCommand = None
    ReportCommand = None
    CompareCommand = None


class AnalyticsModule(ModuleInterface):
    """Module for analytics and reporting."""
    
    @property
    def name(self) -> str:
        return "analytics"
    
    @property
    def description(self) -> str:
        return "Generate statistical reports and performance analytics"
    
    @property
    def commands(self) -> Dict[str, str]:
        return {
            "--summary": "Generate overall performance summary",
            "--trends": "Show performance trends over time",
            "--operations": "Analyze operation effectiveness",
            "--convergence": "Analyze convergence patterns",
            "--report": "Generate comprehensive HTML report",
            "--compare": "Compare performance across time periods",
            "--help": "Show detailed help for analytics module"
        }
    
    @property
    def required_services(self) -> Dict[str, type]:
        return {
            'analytics_service': AnalyticsService
        }
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add analytics-specific arguments."""
        analytics_group = parser.add_argument_group('Analytics Module')
        analytics_group.add_argument('--summary', action='store_true',
                                   help='Generate performance summary')
        analytics_group.add_argument('--trends', action='store_true',
                                   help='Show performance trends')
        analytics_group.add_argument('--operations', action='store_true',
                                   help='Analyze operation effectiveness')
        analytics_group.add_argument('--convergence', type=str, metavar='SESSION_ID',
                                   help='Analyze convergence for session')
        analytics_group.add_argument('--report', action='store_true',
                                   help='Generate comprehensive HTML report')
        analytics_group.add_argument('--compare', nargs=2, metavar=('PERIOD1', 'PERIOD2'),
                                   help='Compare periods (days or YYYY-MM-DD format)')
        analytics_group.add_argument('--days', type=int, default=30,
                                   help='Number of days for analysis')
        analytics_group.add_argument('--format', type=str, choices=['text', 'json', 'html', 'csv'],
                                   default='text', help='Output format')
        analytics_group.add_argument('--output', type=str, help='Output file path')
        analytics_group.add_argument('--dataset', type=str, help='Filter by dataset name')
    
    def execute(self, args: argparse.Namespace, manager) -> None:
        """Execute analytics module commands."""
        # Get analytics service
        analytics_service = self.get_service('analytics_service')
        
        # Create command context
        context = {
            'args': args,
            'manager': manager,
            'service': analytics_service,
            'format': args.format,
            'output': args.output
        }
        
        # Execute appropriate command
        if args.summary:
            if SummaryCommand:
                command = SummaryCommand(context)
                command.execute()
            else:
                print("❌ Summary command not available")
        elif args.trends:
            if TrendsCommand:
                command = TrendsCommand(context)
                command.execute()
            else:
                print("❌ Trends command not available")
        elif args.operations:
            if OperationsCommand:
                command = OperationsCommand(context)
                command.execute()
            else:
                print("❌ Operations command not available")
        elif args.convergence:
            if ConvergenceCommand:
                command = ConvergenceCommand(context)
                command.execute(args.convergence)
            else:
                print("❌ Convergence command not available")
        elif args.report:
            if ReportCommand:
                command = ReportCommand(context)
                command.execute()
            else:
                print("❌ Report command not available")
        elif args.compare:
            if CompareCommand:
                command = CompareCommand(context)
                command.execute(args.compare[0], args.compare[1])
            else:
                print("❌ Compare command not available")
        else:
            print("❌ No analytics command specified. Use --help for options.")