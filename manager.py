#!/usr/bin/env python3
"""
Modular DuckDB Manager - Universal database management with dynamic modules

Dynamic module system for comprehensive database management operations.
Uses a modern architecture with separate layers for data access, business logic,
and presentation.

Usage:
    ./manager.py                    # Show available modules
    ./manager.py sessions --list    # Use sessions module
    ./manager.py features --top 5   # Use features module
    ./manager.py analytics --summary # Use analytics module
"""

import argparse
import importlib
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.manager.core import Config, DatabaseConnection, DatabasePool, ModuleInterface
from src.manager.repositories import (
    SessionRepository, FeatureRepository, 
    DatasetRepository, MetricsRepository
)
from src.manager.services import (
    SessionService, FeatureService,
    DatasetService, AnalyticsService, BackupService
)


from rich.console import Console
from rich.table import Table

class ModularDuckDBManager:
    """Universal database manager with modern architecture."""
    
    def __init__(self):
        """Initialize the manager with configuration and services."""
        self.project_root = Path(__file__).parent
        
        # Initialize configuration
        self.config = Config()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize database connection pool
        self.db_pool = DatabasePool(
            self.config.database_path,
            self.config.duckdb_settings,
            max_connections=3
        )
        
        # Initialize repositories
        self._init_repositories()
        
        # Initialize services
        self._init_services()
        
        # Load available modules
        self.console = Console()
        self.modules = {}
        self._discover_modules()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration using main application logging."""
        try:
            import yaml
            from logging_utils import setup_main_logging, set_session_context
            
            # Load config for logging
            config_path = self.config.config_path
            try:
                with open(config_path, 'r') as f:
                    main_config = yaml.safe_load(f)
                setup_main_logging(main_config)
                set_session_context('manager')
            except Exception as e:
                # Fallback to basic logging
                logging.basicConfig(level=logging.INFO,
                                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logging.getLogger(__name__).warning(f"Could not setup main logging: {e}")
        except ImportError:
            # Fallback to basic logging if logging_utils not available  
            logging.basicConfig(
                level=getattr(logging, self.config.get('logging.level', 'INFO')),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Reduce noise from some libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    def _init_repositories(self) -> None:
        """Initialize repository instances."""
        # Repositories will get connections from the pool as needed
        self.repositories = {
            'session': SessionRepository(self.db_pool),
            'feature': FeatureRepository(self.db_pool),
            'dataset': DatasetRepository(self.db_pool),
            'metrics': MetricsRepository(self.db_pool)
        }
    
    def _init_services(self) -> None:
        """Initialize service instances."""
        self.services = {
            'session_service': SessionService(self.repositories['session']),
            'feature_service': FeatureService(self.repositories['feature']),
            'dataset_service': DatasetService(self.repositories['dataset']),
            'analytics_service': AnalyticsService(
                self.repositories['session'],
                self.repositories['feature'],
                self.repositories['metrics']
            ),
            'backup_service': BackupService(self.config, self.db_pool)
        }
    
    def _discover_modules(self) -> None:
        """Discover and load available modules."""
        modules_path = Path(__file__).parent / 'src' / 'manager' / 'modules'
        
        for module_dir in modules_path.iterdir():
            if module_dir.is_dir() and not module_dir.name.startswith('_'):
                try:
                    # Import the module
                    module_name = f'manager.modules.{module_dir.name}'
                    module = importlib.import_module(module_name)
                    
                    # Find the module class (should match directory name)
                    class_name = f"{module_dir.name.title()}Module"
                    if hasattr(module, class_name):
                        module_class = getattr(module, class_name)
                        instance = module_class()
                        
                        # Inject required services
                        if hasattr(instance, 'inject_services'):
                            # Include config and db_pool in services
                            services_with_core = {
                                **self.services,
                                'config': self.config,
                                'db_pool': self.db_pool
                            }
                            instance.inject_services(services_with_core)
                        
                        self.modules[instance.name] = instance
                        logging.debug(f"Loaded module: {instance.name}")
                except Exception as e:
                    logging.error(f"Failed to load module {module_dir.name}: {e}")
    
    def get_module(self, name: str) -> Optional[ModuleInterface]:
        """Get a module by name."""
        return self.modules.get(name)
    
    def list_modules(self) -> List[Dict[str, str]]:
        """List all available modules."""
        return [
            {
                'name': module.name,
                'description': module.description
            }
            for module in self.modules.values()
        ]
    
    def run(self, argv: List[str]) -> int:
        """Run the manager with command line arguments."""
        # Check if we should show module-specific help BEFORE any parsing
        if len(argv) >= 2 and '--help' in argv:
            potential_module = argv[1] if not argv[1].startswith('-') else None
            if potential_module and potential_module in self.modules:
                # This is module-specific help, handle it directly
                module = self.get_module(potential_module)
                if module:
                    module_parser = argparse.ArgumentParser(
                        prog=f"manager.py {module.name}",
                        description=module.description
                    )
                    module.add_arguments(module_parser)
                    module_parser.print_help()
                    return 0
        
        parser = argparse.ArgumentParser(
            description="Modular DuckDB Manager - Database management and analytics",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_epilog(),
            add_help=False  # Disable automatic help to handle it manually
        )
        
        # Global arguments
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug logging')
        parser.add_argument('--version', action='version',
                          version='%(prog)s 2.0.0')
        parser.add_argument('--help', '-h', action='store_true',
                          help='show this help message and exit')
        
        # Add module as positional argument
        parser.add_argument('module', nargs='?',
                          help='Module to execute')
        
        # Parse initial args to get module
        args, remaining = parser.parse_known_args(argv)
        
        # Handle main help
        if args.help and not args.module:
            parser.print_help()
            return 0
        
        # Enable debug if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # If no module specified, show available modules
        if not args.module:
            self._show_modules()
            return 0
        
        # Get the requested module
        module = self.get_module(args.module)
        if not module:
            print(f"❌ Unknown module: {args.module}")
            self._show_modules()
            return 1
        
        # Create new parser for module-specific args
        module_parser = argparse.ArgumentParser(
            prog=f"{parser.prog} {module.name}",
            description=module.description
        )
        
        # Add module arguments
        module.add_arguments(module_parser)
        
        # Parse module arguments
        try:
            module_args = module_parser.parse_args(remaining)
        except SystemExit:
            return 1
        
        # Execute module
        try:
            module.execute(module_args, self)
            return 0
        except Exception as e:
            logging.error(f"Module execution failed: {e}", exc_info=args.debug)
            print(f"❌ Error: {e}")
            return 1
    
    def _show_modules(self) -> None:
        """Display available modules using rich."""
        self.console.print("\n[bold cyan]🔧 MODULAR DUCKDB MANAGER[/bold cyan]")
        self.console.print("[cyan]" + "=" * 50 + "[/cyan]")

        table = Table(title="[bold blue]Available modules[/bold blue]", show_header=True, header_style="bold magenta")
        table.add_column("Module", style="bold yellow", width=15)
        table.add_column("Description", style="white")

        for module in self.modules.values():
            table.add_row(module.name, module.description)

        self.console.print(table)

        self.console.print("\n[bold green]Usage:[/bold green]")
        self.console.print("  [dim]./manager.py <module> [options][/dim]")
        self.console.print("  [dim]./manager.py <module> --help[/dim]")

        self.console.print("\n[bold green]Examples:[/bold green]")
        self.console.print("  [bold magenta]# Dataset management[/bold magenta]")
        self.console.print("  [dim]./manager.py datasets --list[/dim]")
        self.console.print("  [dim]./manager.py datasets --register --dataset-name titanic --dataset-path datasets/Titanic/ --target-column Survived --auto[/dim]")
        self.console.print("  [dim]./manager.py datasets --details titanic[/dim]")
        self.console.print("")
        self.console.print("  [bold magenta]# Session analysis[/bold magenta]")
        self.console.print("  [dim]./manager.py sessions --list[/dim]")
        self.console.print("  [dim]./manager.py sessions --details [session_id][/dim]")
        self.console.print("  [dim]./manager.py sessions --best 5[/dim]")
        self.console.print("")
        self.console.print("  [bold magenta]# Feature analysis[/bold magenta]")
        self.console.print("  [dim]./manager.py features --top 10[/dim]")
        self.console.print("  [dim]./manager.py features --session [session_id][/dim]")
        self.console.print("")
        self.console.print("  [bold magenta]# System maintenance[/bold magenta]")
        self.console.print("  [dim]./manager.py analytics --summary --days 7[/dim]")
        self.console.print("  [dim]./manager.py backup --create[/dim]")
        self.console.print("  [dim]./manager.py selfcheck --validate[/dim]")
    
    def _get_epilog(self) -> str:
        """Get epilog text for help."""
        return """
Examples:
  %(prog)s                          # Show available modules
  %(prog)s datasets --list          # List registered datasets
  %(prog)s datasets --register --dataset-name titanic --dataset-path datasets/Titanic/ --target-column Survived --auto
  %(prog)s sessions --list          # List all MCTS sessions
  %(prog)s features --top 10        # Show top 10 performing features
  %(prog)s analytics --summary      # Show performance summary
  %(prog)s backup --create          # Create database backup
  %(prog)s selfcheck --validate     # Run system validation

For module-specific help:
  %(prog)s <module> --help
"""
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'db_pool'):
            self.db_pool.close_all()


def main():
    """Main entry point."""
    manager = ModularDuckDBManager()
    
    try:
        return manager.run(sys.argv[1:])
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        manager.cleanup()


if __name__ == "__main__":
    sys.exit(main())