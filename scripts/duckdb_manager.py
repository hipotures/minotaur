#!/usr/bin/env python3
"""
Modular DuckDB Manager - Universal database management with dynamic modules

Dynamic module system for comprehensive database management operations.
Automatically discovers and loads modules from the modules/ directory.

Usage:
    ./scripts/duckdb_manager_modular.py                    # Show available modules
    ./scripts/duckdb_manager_modular.py sessions --list    # Use sessions module
    ./scripts/duckdb_manager_modular.py features --top 5   # Use features module
    ./scripts/duckdb_manager_modular.py analytics --summary # Use analytics module
"""

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pkgutil

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class ModularDuckDBManager:
    """Universal database manager with dynamic module loading."""
    
    def __init__(self):
        """Initialize the manager with project paths."""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.modules_dir = self.script_dir / "modules"
        
        # Load database path from config
        self.duckdb_path = self._load_database_path()
        
        # Try to import duckdb
        try:
            import duckdb
            self.duckdb_available = True
            self._connect = lambda: duckdb.connect(str(self.duckdb_path))
        except ImportError:
            self.duckdb_available = False
            print("⚠️  Warning: DuckDB not available. Some functionality will be limited.")
            self._connect = None
        
        # Load available modules
        self.available_modules = {}
        self.module_instances = {}
        self._discover_modules()
    
    def _load_database_path(self) -> Path:
        """Load database path from main config."""
        try:
            import yaml
            config_path = self.project_root / 'config' / 'mcts_config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            db_path = config['database']['path']
            return self.project_root / db_path
        except Exception as e:
            print(f"⚠️  Could not load database path from config: {e}")
            return self.project_root / "data" / "minotaur.duckdb"  # fallback
    
    def get_backup_config(self) -> dict:
        """Load backup configuration from main config."""
        try:
            import yaml
            config_path = self.project_root / 'config' / 'mcts_config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            backup_config = config['database']
            return {
                'backup_path': backup_config.get('backup_path', 'data/backups/'),
                'backup_prefix': backup_config.get('backup_prefix', 'minotaur_backup_')
            }
        except Exception as e:
            print(f"⚠️  Could not load backup config: {e}")
            return {'backup_path': 'data/backups/', 'backup_prefix': 'minotaur_backup_'}
    
    def get_export_config(self) -> dict:
        """Load export configuration from main config."""
        # DuckDB manager should have its own dedicated export directory
        # Don't mix with MCTS reports!
        return {
            'export_dir': 'outputs/duckdb_exports'  # Dedicated directory for DuckDB manager
        }
    
    def _create_database_with_migrations(self) -> None:
        """Create database and run migrations using the same system as run_feature_discovery.py"""
        try:
            # Import the database service from the main system
            import sys
            import os
            
            # Add src to Python path
            src_path = str(self.project_root / 'src')
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            # Change to project root for imports
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            # Import database service
            import src.db_service as db_service_module
            DatabaseService = db_service_module.DatabaseService
            
            # Load base configuration
            import yaml
            config_path = self.project_root / 'config' / 'mcts_config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Create database service (this will auto-run migrations)
            db_service = DatabaseService(config)
            
            # Clean up
            os.chdir(original_cwd)
            if src_path in sys.path:
                sys.path.remove(src_path)
                
        except Exception as e:
            print(f"❌ Failed to create database: {e}")
            print("💡 Try running: python run_feature_discovery.py --list-sessions")
            raise
    
    def _discover_modules(self) -> None:
        """Discover and load all available modules."""
        if not self.modules_dir.exists():
            print(f"❌ Modules directory not found: {self.modules_dir}")
            return
        
        # Import modules package
        sys.path.insert(0, str(self.script_dir))
        
        try:
            import modules
            from modules import ModuleInterface
            
            # Discover all modules in the modules directory
            for importer, modname, ispkg in pkgutil.iter_modules(modules.__path__, modules.__name__ + "."):
                if ispkg:
                    continue
                
                try:
                    # Import the module
                    module = importlib.import_module(modname)
                    
                    # Find classes that inherit from ModuleInterface
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, ModuleInterface) and 
                            obj != ModuleInterface and 
                            not inspect.isabstract(obj)):
                            
                            # Create instance and register
                            instance = obj()
                            module_name = instance.name
                            
                            self.available_modules[module_name] = {
                                'instance': instance,
                                'description': instance.description,
                                'commands': instance.commands,
                                'dependencies': instance.dependencies,
                                'module_path': modname
                            }
                            self.module_instances[module_name] = instance
                            
                            print(f"✅ Loaded module: {module_name}")
                            
                except Exception as e:
                    print(f"⚠️  Failed to load module {modname}: {e}")
        
        except ImportError as e:
            print(f"❌ Failed to import modules package: {e}")
        
        finally:
            # Clean up path
            if str(self.script_dir) in sys.path:
                sys.path.remove(str(self.script_dir))
    
    def validate_dependencies(self) -> bool:
        """Validate that all module dependencies are satisfied."""
        for module_name, module_info in self.available_modules.items():
            dependencies = module_info['dependencies']
            
            for dep in dependencies:
                if dep not in self.available_modules:
                    print(f"❌ Module '{module_name}' depends on missing module '{dep}'")
                    return False
        
        return True
    
    def show_available_modules(self) -> None:
        """Display all available modules and their capabilities."""
        print("🧩 AVAILABLE MODULES")
        print("=" * 60)
        
        if not self.available_modules:
            print("No modules found. Check the modules/ directory.")
            return
        
        print(f"Database: {self.duckdb_path}")
        print(f"Database exists: {'✅' if self.duckdb_path.exists() else '❌'}")
        print(f"DuckDB available: {'✅' if self.duckdb_available else '❌'}")
        print()
        
        for module_name, module_info in self.available_modules.items():
            instance = module_info['instance']
            commands = module_info['commands']
            dependencies = module_info['dependencies']
            
            print(f"📦 {module_name.upper()}")
            print(f"   {instance.description}")
            
            if dependencies:
                print(f"   Dependencies: {', '.join(dependencies)}")
            
            print("   Commands:")
            for command, description in commands.items():
                print(f"      {command}: {description}")
            print()
        
        print("💡 Usage Examples:")
        print(f"   {sys.argv[0]} sessions --list")
        print(f"   {sys.argv[0]} sessions --help")
        print(f"   {sys.argv[0]} sessions --compare session1 session2")
        print(f"   {sys.argv[0]} features --top 10")
        print(f"   {sys.argv[0]} features --dataset Titanic")
        print(f"   {sys.argv[0]} features --search npk")
        print(f"   {sys.argv[0]} datasets --list")
        print(f"   {sys.argv[0]} datasets --show Titanic")
        print(f"   {sys.argv[0]} datasets --stats")
        print(f"   {sys.argv[0]} datasets --search fertilizer")
        print(f"   {sys.argv[0]} analytics --summary")
        print(f"   {sys.argv[0]} backup --create")
        print(f"   {sys.argv[0]} verification --verify-session session_20250628_123028")
        print(f"   {sys.argv[0]} verification --verify-latest --report json")
        print(f"   {sys.argv[0]} verification --verify-all --quick")
        print(f"   {sys.argv[0]} verification --last 5 --warn-only")
        print(f"   {sys.argv[0]} verification --last --pass-only")
        print(f"   {sys.argv[0]} verification --verify-all --failed-only")
        print()
    
    def create_parser(self, target_module: str = None) -> argparse.ArgumentParser:
        """Create argument parser, optionally for specific module."""
        
        if target_module:
            # Module-specific parser with custom usage
            parser = argparse.ArgumentParser(
                prog=f"duckdb_manager.py {target_module}",
                description="Modular DuckDB Manager for MCTS Feature Discovery",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Available Modules:
""" + "\n".join([f"  {name}: {info['description']}" 
                 for name, info in self.available_modules.items()])
            )
            
            # Add module as positional but make it optional in parsing
            parser.add_argument('module', nargs='?', default=target_module, 
                              help=argparse.SUPPRESS)  # Hide from help
            
            # Add module-specific arguments
            if target_module in self.available_modules:
                instance = self.available_modules[target_module]['instance']
                try:
                    instance.add_arguments(parser)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to add arguments for {target_module}: {e}")
        else:
            # General parser
            parser = argparse.ArgumentParser(
                description="Modular DuckDB Manager for MCTS Feature Discovery",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Available Modules:
""" + "\n".join([f"  {name}: {info['description']}" 
                 for name, info in self.available_modules.items()])
            )
            
            # Add module selection
            if self.available_modules:
                module_choices = list(self.available_modules.keys())
                parser.add_argument('module', nargs='?', choices=module_choices,
                              help='Module to use')
        
        # Global options
        if not target_module:
            # Only add --list-modules for general parser
            parser.add_argument('--list-modules', action='store_true',
                              help='List all available modules and exit')
            parser.add_argument('--self-check', type=str, metavar='DATASET', nargs='?',
                              const='titanic', help='Run comprehensive self-check (default: titanic)')
            parser.add_argument('--quick', action='store_true',
                              help='Quick validation without MCTS test (for --self-check)')
            parser.add_argument('--verbose', action='store_true',
                              help='Enable verbose output (for --self-check)')
            parser.add_argument('--config', type=str, metavar='CONFIG_FILE',
                              help='Use custom MCTS configuration file (for --self-check)')
        
        parser.add_argument('--validate', action='store_true',
                          help='Validate module dependencies and exit')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug output')
        
        return parser
    
    def execute_module(self, module_name: str, args: argparse.Namespace) -> None:
        """Execute a specific module with given arguments."""
        if module_name not in self.available_modules:
            print(f"❌ Module not found: {module_name}")
            print(f"Available modules: {', '.join(self.available_modules.keys())}")
            return
        
        if not self.duckdb_available:
            print("❌ DuckDB not available. Please install DuckDB to use modules.")
            return
        
        if not self.duckdb_path.exists():
            print(f"⚠️  Database not found: {self.duckdb_path}")
            print("🔧 Creating database and running migrations...")
            self._create_database_with_migrations()
            print(f"✅ Database created successfully: {self.duckdb_path}")
        
        # Check dependencies
        module_info = self.available_modules[module_name]
        dependencies = module_info['dependencies']
        
        for dep in dependencies:
            if dep not in self.available_modules:
                print(f"❌ Missing dependency for {module_name}: {dep}")
                return
        
        # Execute module
        try:
            instance = module_info['instance']
            
            if args.debug:
                print(f"🔧 Executing module: {module_name}")
                print(f"   Arguments: {vars(args)}")
            
            instance.execute(args, self)
            
        except Exception as e:
            print(f"❌ Error executing module {module_name}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    def _run_self_check(self, dataset: str, args: argparse.Namespace) -> None:
        """Run comprehensive self-check using selfcheck module."""
        if 'selfcheck' not in self.available_modules:
            print("❌ Self-check module not available")
            return
        
        # Create arguments for selfcheck module
        selfcheck_args = argparse.Namespace()
        selfcheck_args.module = 'selfcheck'
        selfcheck_args.run = dataset
        selfcheck_args.list_datasets = False
        selfcheck_args.quick = getattr(args, 'quick', False)
        selfcheck_args.verbose = getattr(args, 'verbose', False)
        selfcheck_args.debug = getattr(args, 'debug', False)
        selfcheck_args.config = getattr(args, 'config', None)
        
        # Execute selfcheck module
        self.execute_module('selfcheck', selfcheck_args)
    
    def _transform_arguments(self, args: List[str]) -> List[str]:
        """Transform user-friendly arguments to internal prefixed arguments."""
        if not args or len(args) < 2:
            return args
            
        module_name = args[0]
        if module_name not in self.available_modules:
            return args
            
        # Transform arguments based on module
        transformed = [module_name]  # Keep module name
        
        # Argument mapping for each module
        module_mappings = {
            'sessions': {
                # Sessions module now uses simple argument names without prefixes
            },
            'features': {
                # Features module uses simple argument names without prefixes
            },
            'backup': {
                # Backup module uses simple argument names without prefixes
            }
        }
        
        mapping = module_mappings.get(module_name, {})
        
        i = 1
        while i < len(args):
            arg = args[i]
            if arg in mapping:
                transformed.append(mapping[arg])
            else:
                transformed.append(arg)
            i += 1
            
        return transformed
    
    def run(self, args: List[str] = None) -> None:
        """Main entry point for the manager."""
        # Transform user arguments to internal format
        if args:
            args = self._transform_arguments(args)
        
        # First, parse with basic parser to get module name
        basic_parser = self.create_parser()
        
        # Check for global options first (but not --help with module)
        if args and ('--list-modules' in args or '--validate' in args or '--self-check' in args):
            parsed_args = basic_parser.parse_args(args)
            
            if parsed_args.list_modules:
                self.show_available_modules()
                return
            
            if parsed_args.validate:
                if self.validate_dependencies():
                    print("✅ All module dependencies satisfied")
                else:
                    print("❌ Module dependency validation failed")
                return
            
            if hasattr(parsed_args, 'self_check') and parsed_args.self_check is not None:
                self._run_self_check(parsed_args.self_check, parsed_args)
                return
        
        # Handle --help without module
        if args and args == ['--help']:
            basic_parser.parse_args(args)
            return
        
        # Parse to get module name
        if not args or len(args) == 0:
            self.show_available_modules()
            return
            
        # Get module from first argument
        module_name = args[0] if args and args[0] in self.available_modules else None
        
        if not module_name:
            self.show_available_modules()
            return
        
        # Create parser with module-specific arguments
        parser = self.create_parser(module_name)
        parsed_args = parser.parse_args(args)
        
        # Execute specified module
        self.execute_module(parsed_args.module, parsed_args)


def main():
    """Main entry point."""
    try:
        manager = ModularDuckDBManager()
        manager.run(sys.argv[1:])  # Pass command line arguments
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()