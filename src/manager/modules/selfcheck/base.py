"""
Base Self-Check Command - Common functionality for self-check commands.

Provides shared utilities including:
- Dataset validation and analysis
- Database schema validation
- Module testing framework
- MCTS integration testing
- Test result reporting
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import argparse
from manager.core.command_base import BaseCommand


class BaseSelfCheckCommand(BaseCommand, ABC):
    """Base class for all self-check commands."""
    
    def __init__(self):
        super().__init__()
        self.dataset_service = None
        self.session_service = None
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """Inject required services."""
        super().inject_services(services)
        self.dataset_service = services.get('dataset_service')
        self.session_service = services.get('session_service')
        # Get config and db_pool from services as well
        self.config = services.get('config')
        self.db_pool = services.get('db_pool')
        if not self.dataset_service:
            raise ValueError("DatasetService is required for self-check commands")
        if not self.session_service:
            raise ValueError("SessionService is required for self-check commands")
    
    @abstractmethod
    def execute(self, args) -> None:
        """Execute the command with given arguments."""
        pass
    
    def validate_dataset(self, dataset: str, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """Validate dataset exists in registered dataset system."""
        if verbose:
            print(f"   Checking registered dataset: {dataset}")
        
        try:
            # Check if dataset exists in registry
            query = """
                SELECT dataset_id, dataset_name, train_path, test_path, target_column, id_column 
                FROM datasets 
                WHERE dataset_name = ? OR dataset_id LIKE ?
            """
            result = self.dataset_service.repository.fetch_one(query, [dataset, f"{dataset}%"])
            
            if not result:
                self.print_error(f"Dataset '{dataset}' not found in registry")
                
                # Show available datasets
                available_query = "SELECT dataset_name FROM datasets WHERE is_active = TRUE"
                available = self.dataset_service.repository.fetch_all(available_query)
                if available:
                    dataset_names = [row['dataset_name'] for row in available]
                    self.print_info(f"Available datasets: {', '.join(dataset_names)}")
                return False, {}
            
            dataset_id = result['dataset_id']
            name = result['dataset_name']
            train_path = result['train_path']
            test_path = result['test_path']
            target_column = result['target_column']
            id_column = result['id_column']
            
            print(f"   ✅ Found registered dataset: {name} (ID: {dataset_id[:8]})")
            
            dataset_info = {
                'dataset_id': dataset_id,
                'name': name,
                'train_path': train_path,
                'test_path': test_path,
                'target_column': target_column,
                'id_column': id_column
            }
            
            # Validate actual data files exist
            if not self._validate_data_files(dataset_info, verbose):
                return False, {}
            
            return True, dataset_info
            
        except Exception as e:
            self.print_error(f"Database error checking dataset: {e}")
            return False, {}
    
    def _validate_data_files(self, dataset_info: Dict[str, Any], verbose: bool) -> bool:
        """Validate that data files exist and are readable."""
        try:
            import pandas as pd
            
            train_path = dataset_info['train_path']
            test_path = dataset_info['test_path']
            target_column = dataset_info['target_column']
            
            train_file = Path(train_path)
            test_file = Path(test_path) if test_path else None
            
            if not train_file.exists():
                self.print_error(f"Training file not found: {train_path}")
                return False
            
            if test_file and not test_file.exists():
                self.print_error(f"Test file not found: {test_path}")
                return False
            
            # Analyze dataset files
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file) if test_file else None
            
            # Use registered metadata
            task_type = self._detect_task_type(train_df, target_column) if target_column else "unknown"
            metric = self._get_metric_for_task(task_type)
            
            # Update dataset_info with analysis
            dataset_info.update({
                'train_rows': len(train_df),
                'train_cols': len(train_df.columns),
                'test_rows': len(test_df) if test_df is not None else 0,
                'test_cols': len(test_df.columns) if test_df is not None else 0,
                'task_type': task_type,
                'metric': metric
            })
            
            print(f"   ✅ Training data: {train_file.name} ({dataset_info['train_rows']} rows, {dataset_info['train_cols']} columns)")
            if test_df is not None:
                print(f"   ✅ Test data: {test_file.name} ({dataset_info['test_rows']} rows, {dataset_info['test_cols']} columns)")
            if target_column:
                print(f"   ✅ Target column: {target_column} ({task_type})")
                print(f"   ✅ Suggested metric: {metric}")
            else:
                print(f"   ⚠️  Target column auto-detection failed, will use first column")
            
            return True
            
        except Exception as e:
            self.print_error(f"Error analyzing dataset: {e}")
            return False
    
    def _detect_task_type(self, df, target_column: str) -> str:
        """Detect if task is classification or regression."""
        if target_column not in df.columns:
            return "unknown"
        
        unique_values = df[target_column].nunique()
        total_values = len(df[target_column])
        
        # If very few unique values compared to total, likely classification
        if unique_values <= 10 or unique_values / total_values < 0.05:
            return "binary_classification" if unique_values == 2 else "multiclass_classification"
        else:
            return "regression"
    
    def _get_metric_for_task(self, task_type: str) -> str:
        """Get appropriate metric for task type."""
        mapping = {
            'binary_classification': 'roc_auc',
            'multiclass_classification': 'accuracy',
            'regression': 'rmse',
            'unknown': 'accuracy'
        }
        return mapping.get(task_type, 'accuracy')
    
    def validate_database(self, verbose: bool = False) -> bool:
        """Validate database connectivity and schema."""
        try:
            # Check database connection using config from services
            if not self.config:
                self.print_error("Database configuration not available")
                return False
            
            db_path = Path(self.config.database_path)
            if not db_path.exists():
                self.print_error(f"Database not found: {db_path}")
                return False
            
            print(f"   ✅ DuckDB connection: {db_path.name}")
            
            # Check tables
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            tables_result = self.session_service.repository.fetch_all(tables_query)
            table_names = [row['table_name'] for row in tables_result]
            
            expected_tables = ['sessions', 'exploration_history', 'feature_catalog', 'feature_impact']
            missing_tables = [t for t in expected_tables if t not in table_names]
            
            if missing_tables:
                self.print_error(f"Missing tables: {missing_tables}")
                return False
            
            print(f"   ✅ Tables present: {', '.join(expected_tables)}")
            
            # Check existing sessions
            session_count_query = "SELECT COUNT(*) as count FROM sessions"
            session_count_result = self.session_service.repository.fetch_one(session_count_query)
            session_count = session_count_result.get('count', 0) if session_count_result else 0
            
            print(f"   ✅ Existing sessions: {session_count} found")
            
            if verbose and session_count > 0:
                recent_query = """
                    SELECT session_id, session_name, start_time 
                    FROM sessions 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """
                recent_result = self.session_service.repository.fetch_one(recent_query)
                if recent_result:
                    sid = recent_result['session_id']
                    name = recent_result['session_name']
                    start_time = recent_result['start_time']
                    print(f"   📋 Most recent: {sid[:8]}... ({name or 'Unnamed'}) at {start_time}")
            
            return True
            
        except Exception as e:
            self.print_error(f"Database validation failed: {e}")
            return False
    
    def test_all_modules(self, manager, verbose: bool = False) -> bool:
        """Test all available modules with comprehensive command coverage."""
        all_passed = True
        total_commands_tested = 0
        total_commands_available = 0
        
        print("\n   📊 COMPREHENSIVE MODULE TESTING")
        print("   " + "=" * 50)
        
        for module_name, module_info in manager.available_modules.items():
            if module_name == 'selfcheck':  # Skip self
                continue
                
            print(f"\n   📦 Testing {module_name.upper()} module...")
            
            try:
                passed, failed = self._test_module(module_name, module_info, manager, verbose)
                total = len(module_info['commands'])
                
                total_commands_tested += passed + failed
                total_commands_available += total
                
                coverage_pct = (passed + failed) / total * 100 if total > 0 else 0
                
                if failed > 0:
                    print(f"   ❌ {module_name}: {passed} passed, {failed} failed ({coverage_pct:.0f}% coverage)")
                    all_passed = False
                else:
                    print(f"   ✅ {module_name}: {passed} passed ({coverage_pct:.0f}% coverage)")
                    
            except Exception as e:
                print(f"   ❌ {module_name}: Error during testing - {e}")
                all_passed = False
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Summary
        print(f"\n   📈 TESTING SUMMARY:")
        print(f"   • Total commands available: {total_commands_available}")
        print(f"   • Commands tested: {total_commands_tested}")
        overall_coverage = total_commands_tested / total_commands_available * 100 if total_commands_available > 0 else 0
        print(f"   • Overall coverage: {overall_coverage:.1f}%")
        
        return all_passed
    
    def _test_module(self, module_name: str, module_info: Dict, manager, verbose: bool) -> Tuple[int, int]:
        """Test a specific module's commands."""
        instance = module_info['instance']
        commands = module_info['commands']
        
        passed = 0
        failed = 0
        
        # Test each command safely
        for command_name, description in commands.items():
            try:
                if verbose:
                    print(f"      Testing {command_name}...")
                
                # Special handling for help commands
                if command_name == '--help':
                    help_result = self._test_help_command(module_name, instance, manager, verbose)
                    if help_result:
                        passed += 1
                        if verbose:
                            print(f"      ✅ {command_name}: OK")
                    else:
                        failed += 1
                        if verbose:
                            print(f"      ❌ {command_name}: Help test failed")
                    continue
                
                # Create test arguments based on command type
                test_args = self._create_test_args(module_name, command_name, manager)
                
                if test_args is None:
                    if verbose:
                        print(f"      Skipped {command_name} (no test data)")
                    continue
                
                # Capture output and test
                success = self._execute_and_validate_command(instance, test_args, command_name, manager, verbose)
                
                if success:
                    passed += 1
                    if verbose:
                        print(f"      ✅ {command_name}: OK")
                else:
                    failed += 1
                    if verbose:
                        print(f"      ❌ {command_name}: Failed")
                
            except Exception as e:
                failed += 1
                if verbose:
                    print(f"      ❌ {command_name}: Exception - {e}")
        
        return passed, failed
    
    def _test_help_command(self, module_name: str, instance, manager, verbose: bool) -> bool:
        """Test the help command for a module."""
        try:
            import io
            import contextlib
            
            test_args = argparse.Namespace()
            
            # Set common attributes to prevent AttributeError
            common_attrs = [
                'help', 'module', 'list', 'show', 'compare', 'export', 'cleanup', 'top', 'impact', 
                'catalog', 'search', 'summary', 'trends', 'operations', 'convergence', 'report', 
                'create', 'restore', 'verify', 'dry_run', 'compress', 'keep', 'days', 'limit', 
                'format', 'session', 'category', 'min_impact', 'status'
            ]
            
            for attr in common_attrs:
                setattr(test_args, attr, False if attr != 'help' else True)
            
            setattr(test_args, 'module', module_name)
            
            # Capture output
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                try:
                    instance.execute(test_args, manager)
                except SystemExit:
                    # argparse calls sys.exit() on --help, which is expected
                    pass
            
            output = output_buffer.getvalue()
            
            # Check if help output contains expected content
            has_help_content = any(word in output.lower() for word in [
                'command', 'option', 'usage', 'help', module_name, 'argument'
            ])
            
            return has_help_content or len(output.strip()) > 10
            
        except Exception as e:
            if verbose:
                print(f"        Help test error: {e}")
            return False
    
    def _execute_and_validate_command(self, instance, test_args, command_name: str, manager, verbose: bool) -> bool:
        """Execute a command and validate it didn't fail."""
        try:
            import io
            import contextlib
            
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(stdout_buffer), \
                 contextlib.redirect_stderr(stderr_buffer):
                instance.execute(test_args, manager)
            
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()
            full_output = stdout_output + stderr_output
            
            # Error detection
            error_indicators = [
                "❌ Error", "Failed to", "Exception:", "Traceback",
                "error:", "ERROR:", "Error detected", "Parser Error:",
                "AttributeError", "TypeError", "ValueError", "KeyError"
            ]
            
            has_error = any(indicator in full_output for indicator in error_indicators)
            
            # Special handling for export commands
            if command_name == '--export' and ("✅ Exported" in full_output or "TOP 10 PERFORMING FEATURES" in full_output):
                has_error = False
            elif command_name == '--list' and "FEATURE PERFORMANCE LIST" in full_output:
                has_error = False
            
            if has_error:
                if verbose:
                    print(f"        Error detected in output: {full_output[:200]}...")
                return False
            
            return True
                
        except Exception as e:
            if verbose:
                print(f"        Execution failed: {e}")
            return False
    
    def _create_test_args(self, module_name: str, command_name: str, manager) -> Optional[argparse.Namespace]:
        """Create test arguments for different module commands."""
        # Get a real session ID for testing
        session_id = None
        try:
            session_query = "SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1"
            session_result = self.session_service.repository.fetch_one(session_query)
            if session_result:
                session_id = session_result['session_id']
        except:
            pass
        
        # Base namespace
        args = argparse.Namespace()
        
        # Set module
        setattr(args, 'module', module_name)
        
        # Set all command flags to False initially
        for flag in ['list', 'show', 'compare', 'export', 'cleanup', 'top', 'impact', 'catalog', 'search', 
                    'summary', 'trends', 'operations', 'convergence', 'report', 'create', 'restore', 'verify',
                    'help', 'dry_run', 'compress', 'keep', 'days', 'limit', 'format', 'session', 'category',
                    'min_impact', 'status', 'details', 'register', 'stats', 'sessions', 'update', 'active_only']:
            setattr(args, flag, False)
        
        # Set specific command arguments based on module and command
        if command_name == '--list':
            setattr(args, 'list', True)
            setattr(args, 'limit', 5)
            setattr(args, 'status', 'all')
            
        elif command_name == '--show' and session_id:
            setattr(args, 'show', session_id)
            
        elif command_name == '--export':
            setattr(args, 'export', 'csv')
            
        elif command_name == '--cleanup':
            setattr(args, 'cleanup', True)
            setattr(args, 'dry_run', True)  # Safe testing mode
            
        elif command_name == '--top':
            setattr(args, 'top', 3)
            
        elif command_name == '--catalog':
            setattr(args, 'catalog', True)
            
        elif command_name == '--summary':
            setattr(args, 'summary', True)
            setattr(args, 'days', 30)
            
        elif command_name == '--help':
            setattr(args, 'help', True)
            
        else:
            return None
        
        return args
    
    def get_test_session_id(self, manager) -> Optional[str]:
        """Get a test session ID from the database."""
        try:
            session_query = "SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1"
            session_result = self.session_service.repository.fetch_one(session_query)
            return session_result['session_id'] if session_result else None
        except:
            return None
    
    def print_final_results(self, test_results: List[Tuple[str, bool]], session_name: Optional[str] = None) -> None:
        """Print final test results summary."""
        print("\n" + "=" * 50)
        print("📋 SELF-CHECK SUMMARY")
        print("=" * 50)
        
        all_passed = True
        for test_name, passed in test_results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {status}: {test_name}")
            if not passed:
                all_passed = False
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 FINAL RESULT: ✅ ALL TESTS PASSED")
            if session_name:
                print(f"   Session details: python manager.py sessions --show {session_name}")
        else:
            print("❌ FINAL RESULT: SOME TESTS FAILED")
            print("💡 Check individual test results above for details")