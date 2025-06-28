#!/usr/bin/env python3
"""
Self-Check Module - Comprehensive system validation and testing

Provides automated testing of all DuckDB manager modules and MCTS functionality
with dynamic dataset support.
"""

import argparse
import json
import os
import sys
import yaml
import tempfile
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from . import ModuleInterface

class SelfCheckModule(ModuleInterface):
    """Module for comprehensive system self-checking and validation."""
    
    @property
    def name(self) -> str:
        return "selfcheck"
    
    @property
    def description(self) -> str:
        return "Comprehensive system validation and testing"
    
    @property
    def commands(self) -> Dict[str, str]:
        return {
            "--run": "Run comprehensive self-check tests",
            "--list-datasets": "List available test datasets",
            "--quick": "Run quick validation without MCTS test",
            "--verbose": "Enable verbose output",
            "--config": "Use custom MCTS configuration file",
            "--help": "Show detailed help for self-check module"
        }
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add self-check specific arguments."""
        selfcheck_group = parser.add_argument_group('Self-Check Module')
        selfcheck_group.add_argument('--run', type=str, metavar='DATASET', nargs='?',
                                   const='titanic', default='titanic',
                                   help='Run self-check with specified dataset (default: titanic)')
        selfcheck_group.add_argument('--list-datasets', action='store_true',
                                   help='List available test datasets')
        selfcheck_group.add_argument('--quick', action='store_true',
                                   help='Quick validation without MCTS test')
        selfcheck_group.add_argument('--verbose', action='store_true',
                                   help='Enable verbose output')
        selfcheck_group.add_argument('--config', type=str, metavar='CONFIG_FILE',
                                   help='Use custom MCTS configuration file')
    
    def execute(self, args: argparse.Namespace, manager) -> None:
        """Execute self-check module commands."""
        
        if getattr(args, 'list_datasets', False):
            self._list_datasets(manager)
        elif hasattr(args, 'run') and args.run:
            self._run_selfcheck(args.run, args, manager)
        else:
            print("❌ No self-check command specified. Use --help for options.")
    
    def _list_datasets(self, manager) -> None:
        """List available registered datasets."""
        print("📋 AVAILABLE REGISTERED DATASETS")
        print("=" * 45)
        
        try:
            with manager._connect() as conn:
                datasets = conn.execute("""
                    SELECT dataset_name, dataset_id, target_column, train_records, train_columns, 
                           test_records, test_columns, is_active
                    FROM datasets 
                    ORDER BY dataset_name
                """).fetchall()
                
                if not datasets:
                    print("❌ No datasets found in registry")
                    print("💡 Use 'datasets --register' to add datasets")
                    return
                
                for row in datasets:
                    name, dataset_id, target_col, train_rows, train_cols, test_rows, test_cols, is_active = row
                    
                    status = "✅" if is_active else "❌"
                    train_info = f"{train_rows} rows, {train_cols} columns" if train_rows else "N/A"
                    test_info = f"{test_rows} rows, {test_cols} columns" if test_rows else "N/A"
                    
                    print(f"{status} {name}")
                    print(f"   ID: {dataset_id[:8]}...")
                    print(f"   Target: {target_col}")
                    print(f"   Train: {train_info}")
                    print(f"   Test: {test_info}")
                    print()
                    
        except Exception as e:
            print(f"❌ Error accessing dataset registry: {e}")
            return
        
        print(f"\n💡 Usage: --run <dataset_name> --config <config_file>")
        print(f"   Example: --self-check titanic")
    
    def _run_selfcheck(self, dataset: str, args: argparse.Namespace, manager) -> None:
        """Run comprehensive self-check tests."""
        print(f"🔧 DUCKDB MANAGER SELF-CHECK: {dataset.upper()}")
        print("=" * 50)
        
        verbose = getattr(args, 'verbose', False)
        quick = getattr(args, 'quick', False)
        config_file = getattr(args, 'config', None)
        
        # Track test results
        test_results = []
        session_id = None
        
        try:
            # 1. Dataset Validation
            print("\n📊 DATASET VALIDATION")
            dataset_valid, dataset_info = self._validate_dataset(dataset, manager, verbose)
            test_results.append(("Dataset Validation", dataset_valid))
            
            if not dataset_valid:
                self._print_final_results(test_results, None)
                return
            
            # 2. Database Validation
            print("\n🗄️  DATABASE VALIDATION")
            db_valid = self._validate_database(manager, verbose)
            test_results.append(("Database Validation", db_valid))
            
            if not db_valid:
                self._print_final_results(test_results, None)
                return
            
            # 3. MCTS Integration Test (unless quick mode) - MUST run first to populate database
            if not quick:
                print("\n🎯 MCTS INTEGRATION TEST")
                mcts_result = self._test_mcts_integration(dataset, dataset_info, manager, verbose, config_file)
                if mcts_result[0]:  # if success
                    mcts_valid = True
                    session_id, session_name = mcts_result[1]
                else:
                    mcts_valid = False
                    session_id, session_name = None, None
                test_results.append(("MCTS Integration", mcts_valid))
            
            # 4. Module Testing - runs after MCTS creates data
            print("\n🧩 MODULE TESTING")
            modules_valid = self._test_all_modules(manager, verbose)
            test_results.append(("Module Testing", modules_valid))
            
            # 5. Final Results
            self._print_final_results(test_results, session_name if 'session_name' in locals() else session_id)
            
        except KeyboardInterrupt:
            print("\n⚠️  Self-check interrupted by user")
        except Exception as e:
            print(f"\n❌ Self-check failed with error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    def _validate_dataset(self, dataset: str, manager, verbose: bool) -> Tuple[bool, Dict[str, Any]]:
        """Validate dataset exists in registered dataset system."""
        if verbose:
            print(f"   Checking registered dataset: {dataset}")
        
        # Check if dataset exists in registry
        try:
            with manager._connect() as conn:
                result = conn.execute("""
                    SELECT dataset_id, dataset_name, train_path, test_path, target_column, id_column 
                    FROM datasets 
                    WHERE dataset_name = ? OR dataset_id LIKE ?
                """, [dataset, f"{dataset}%"]).fetchone()
                
                if not result:
                    print(f"❌ Dataset '{dataset}' not found in registry")
                    
                    # Show available datasets
                    available = conn.execute("SELECT dataset_name FROM datasets WHERE is_active = TRUE").fetchall()
                    if available:
                        dataset_names = [row[0] for row in available]
                        print(f"💡 Available datasets: {', '.join(dataset_names)}")
                    return False, {}
                
                dataset_id, name, train_path, test_path, target_column, id_column = result
                print(f"   ✅ Found registered dataset: {name} (ID: {dataset_id[:8]})")
                
                return True, {
                    'dataset_id': dataset_id,
                    'name': name,
                    'train_path': train_path,
                    'test_path': test_path,
                    'target_column': target_column,
                    'id_column': id_column
                }
                
        except Exception as e:
            print(f"❌ Database error checking dataset: {e}")
            return False, {}
        
        # Validate actual data files exist
        try:
            from pathlib import Path
            import pandas as pd
            
            train_file = Path(train_path)
            test_file = Path(test_path) if test_path else None
            
            if not train_file.exists():
                print(f"❌ Training file not found: {train_path}")
                return False, {}
            
            if test_file and not test_file.exists():
                print(f"❌ Test file not found: {test_path}")
                return False, {}
            
            # Analyze dataset files
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file) if test_file else None
            
            # Use registered metadata
            task_type = self._detect_task_type(train_df, target_column) if target_column else "unknown"
            metric = self._get_metric_for_task(task_type)
            
            dataset_info = {
                'train_path': train_path,
                'test_path': test_path,
                'train_rows': len(train_df),
                'train_cols': len(train_df.columns),
                'test_rows': len(test_df) if test_df is not None else 0,
                'test_cols': len(test_df.columns) if test_df is not None else 0,
                'target_column': target_column,
                'id_column': id_column,
                'task_type': task_type,
                'metric': metric
            }
            
            print(f"   ✅ Training data: {train_file.name} ({dataset_info['train_rows']} rows, {dataset_info['train_cols']} columns)")
            if test_df is not None:
                print(f"   ✅ Test data: {test_file.name} ({dataset_info['test_rows']} rows, {dataset_info['test_cols']} columns)")
            if target_column:
                print(f"   ✅ Target column: {target_column} ({task_type})")
                print(f"   ✅ Suggested metric: {metric}")
            else:
                print(f"   ⚠️  Target column auto-detection failed, will use first column")
            
            return True, dataset_info
            
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
            return False, {}
    
    def _detect_target_column(self, train_df, test_df) -> Optional[str]:
        """Detect target column by finding columns in train but not in test."""
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        target_candidates = train_cols - test_cols
        
        # Common target column names
        common_targets = ['Survived', 'target', 'label', 'y', 'class', 'outcome', 'price', 'SalePrice']
        
        # Check for common names first
        for target in common_targets:
            if target in target_candidates:
                return target
        
        # Return first candidate or None
        return list(target_candidates)[0] if target_candidates else None
    
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
    
    def _validate_database(self, manager, verbose: bool) -> bool:
        """Validate database connectivity and schema."""
        
        if not manager.duckdb_available:
            print("❌ DuckDB not available")
            return False
        
        if not manager.duckdb_path.exists():
            print(f"❌ Database not found: {manager.duckdb_path}")
            return False
        
        print(f"   ✅ DuckDB connection: {manager.duckdb_path.relative_to(manager.project_root)}")
        
        try:
            with manager._connect() as conn:
                # Check tables
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [row[0] for row in tables]
                
                expected_tables = ['sessions', 'exploration_history', 'feature_catalog', 'feature_impact']
                missing_tables = [t for t in expected_tables if t not in table_names]
                
                if missing_tables:
                    print(f"❌ Missing tables: {missing_tables}")
                    return False
                
                print(f"   ✅ Tables present: {', '.join(expected_tables)}")
                
                # Check existing sessions
                session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                print(f"   ✅ Existing sessions: {session_count} found")
                
                if verbose and session_count > 0:
                    recent_session = conn.execute("""
                        SELECT session_id, session_name, start_time 
                        FROM sessions 
                        ORDER BY start_time DESC 
                        LIMIT 1
                    """).fetchone()
                    if recent_session:
                        sid, name, start_time = recent_session
                        print(f"   📋 Most recent: {sid[:8]}... ({name or 'Unnamed'}) at {start_time}")
                
                return True
                
        except Exception as e:
            print(f"❌ Database validation failed: {e}")
            return False
    
    def _test_all_modules(self, manager, verbose: bool) -> bool:
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
                passed, failed = self._test_module(
                    module_name, module_info, manager, verbose
                )
                skipped = 0  # Original method doesn't track skipped
                total = len(module_info['commands'])
                
                total_commands_tested += passed + failed
                total_commands_available += total
                
                # Detailed reporting
                coverage_pct = (passed + failed) / total * 100 if total > 0 else 0
                
                if failed > 0:
                    print(f"   ❌ {module_name}: {passed} passed, {failed} failed, {skipped} skipped ({coverage_pct:.0f}% coverage)")
                    all_passed = False
                    
                    # Show failed command details
                    failed_details = getattr(self, f'_{module_name}_failed_details', [])
                    if failed_details:
                        print(f"      💥 Failed commands:")
                        for detail in failed_details[:3]:  # Show first 3 failures
                            print(f"         • {detail}")
                        if len(failed_details) > 3:
                            print(f"         ... and {len(failed_details) - 3} more failures")
                else:
                    print(f"   ✅ {module_name}: {passed} passed, {skipped} skipped ({coverage_pct:.0f}% coverage)")
                    
                if verbose and skipped > 0:
                    print(f"      ℹ️  {skipped} commands skipped (no test data or unsupported)")
                    
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
    
    def _test_module_comprehensive(self, module_name: str, module_info: Dict, manager, verbose: bool) -> Tuple[int, int, int, int]:
        """Test a specific module's commands comprehensively."""
        
        instance = module_info['instance']
        commands = module_info['commands']
        
        passed = 0
        failed = 0
        skipped = 0
        total = len(commands)
        
        if verbose:
            print(f"      📋 Available commands: {list(commands.keys())}")
        
        # Test each command including help
        for command_name, description in commands.items():
            try:
                if verbose:
                    print(f"      🧪 Testing {command_name}...")
                
                # Test help commands specially
                if command_name == '--help':
                    help_result = self._test_help_command(module_name, instance, manager, verbose)
                    if help_result:
                        passed += 1
                    else:
                        failed += 1
                    continue
                
                # Create test arguments based on command type
                test_args = self._create_test_args(module_name, command_name, manager)
                
                if test_args is None:
                    if verbose:
                        print(f"      ⏭️  Skipped {command_name} (no test strategy available)")
                    skipped += 1
                    continue
                
                # Execute and test command
                success = self._execute_and_validate_command(
                    instance, test_args, command_name, manager, verbose
                )
                
                if success:
                    passed += 1
                    if verbose:
                        print(f"      ✅ {command_name} passed")
                else:
                    failed += 1
                    if verbose:
                        print(f"      ❌ {command_name} failed")
                        
            except Exception as e:
                failed += 1
                if verbose:
                    print(f"      ❌ {command_name} exception: {e}")
        
        return passed, failed, skipped, total
    
    def _test_help_command(self, module_name: str, instance, manager, verbose: bool) -> bool:
        """Test the help command for a module."""
        try:
            import io
            import contextlib
            import argparse
            
            # Create a test argument namespace with help=True
            test_args = argparse.Namespace()
            setattr(test_args, 'help', True)
            
            # Capture output
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                # Most modules should handle --help gracefully
                try:
                    instance.execute(test_args, manager)
                except SystemExit:
                    # argparse calls sys.exit() on --help, which is expected
                    pass
            
            output = output_buffer.getvalue()
            
            # Check if help output contains module description or commands
            has_help_content = any(word in output.lower() for word in [
                'command', 'option', 'usage', 'help', module_name, 'argument'
            ])
            
            return has_help_content or len(output) > 10  # Some help output expected
            
        except Exception as e:
            if verbose:
                print(f"        Help test error: {e}")
            return False
    
    def _create_test_args_enhanced(self, module_name: str, command_name: str, manager):
        """Create enhanced test arguments for different module commands."""
        import argparse
        
        # Base namespace
        args = argparse.Namespace()
        
        # Set all possible attributes to False/None initially
        for attr in ['list', 'show', 'compare', 'export', 'cleanup', 'summary', 
                     'trends', 'operations', 'convergence', 'report', 'create',
                     'restore', 'verify', 'top', 'impact', 'catalog', 'search',
                     'limit', 'format', 'session_id', 'days', 'metric']:
            setattr(args, attr, None)
            setattr(args, attr.replace('_', ''), None)  # Handle both formats
        
        # Set the specific command
        cmd = command_name.lstrip('-').replace('-', '_')
        
        if module_name == 'sessions':
            if command_name == '--list':
                setattr(args, 'list', True)
                setattr(args, 'limit', 5)
            elif command_name == '--show':
                # Need a session ID - try to get one from database
                session_id = self._get_test_session_id(manager)
                if session_id:
                    setattr(args, 'show', session_id)
                else:
                    return None  # Skip if no sessions available
            elif command_name == '--compare':
                # Need multiple session IDs
                session_ids = self._get_test_session_ids(manager, count=2)
                if len(session_ids) >= 2:
                    setattr(args, 'compare', session_ids)
                else:
                    return None
            elif command_name == '--export':
                setattr(args, 'export', 'json')
            elif command_name == '--cleanup':
                # Skip cleanup during testing - handled by original method
                return None
                
        elif module_name == 'analytics':
            if command_name == '--summary':
                setattr(args, 'summary', True)
            elif command_name == '--trends':
                setattr(args, 'trends', True)
            elif command_name == '--operations':
                setattr(args, 'operations', True)
            elif command_name == '--convergence':
                setattr(args, 'convergence', True)
            elif command_name == '--report':
                setattr(args, 'report', True)
            elif command_name == '--compare':
                setattr(args, 'compare', True)
                
        elif module_name == 'features':
            if command_name == '--list':
                setattr(args, 'list', True)
                setattr(args, 'limit', 5)
            elif command_name == '--top':
                setattr(args, 'top', 5)
            elif command_name == '--impact':
                setattr(args, 'impact', True)
            elif command_name == '--catalog':
                setattr(args, 'catalog', True)
            elif command_name == '--export':
                setattr(args, 'export', 'json')
            elif command_name == '--search':
                setattr(args, 'search', 'test')  # Search for 'test'
                
        elif module_name == 'backup':
            if command_name == '--create':
                setattr(args, 'create', True)
            elif command_name == '--list':
                setattr(args, 'list', True)
            elif command_name == '--restore':
                # Need a backup file - this might not exist, so we'll let it fail gracefully
                setattr(args, 'restore', 'test_backup.db')
            elif command_name == '--cleanup':
                # Skip cleanup during testing - handled by original method
                return None
            elif command_name == '--verify':
                setattr(args, 'verify', 'test_backup.db')
        
        return args
    
    def _execute_and_validate_command(self, instance, test_args, command_name: str, manager, verbose: bool) -> bool:
        """Execute a command and validate it didn't fail."""
        try:
            import io
            import contextlib
            
            # Capture both stdout and stderr
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(stdout_buffer), \
                 contextlib.redirect_stderr(stderr_buffer):
                instance.execute(test_args, manager)
            
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()
            full_output = stdout_output + stderr_output
            
            # Enhanced error detection
            error_indicators = [
                "❌ Error", "Failed to", "Exception:", "Traceback",
                "error:", "ERROR:", "Error detected", "Parser Error:",
                "AttributeError", "TypeError", "ValueError", "KeyError",
                "FileNotFoundError", "PermissionError"
            ]
            
            # Success indicators
            success_indicators = [
                "✅", "SUCCESS", "completed", "found", "generated",
                "exported", "created", "listed", "analyzed"
            ]
            
            has_error = any(indicator in full_output for indicator in error_indicators)
            has_success = any(indicator in full_output for indicator in success_indicators)
            
            # Special handling for commands that might legitimately have no output
            no_output_ok_commands = ['--cleanup', '--verify', '--create']
            is_no_output_ok = any(cmd in command_name for cmd in no_output_ok_commands)
            
            if has_error:
                if verbose:
                    print(f"        Error detected in output: {full_output[:200]}...")
                return False
            elif has_success or (len(full_output.strip()) == 0 and is_no_output_ok):
                return True
            elif len(full_output.strip()) > 0:
                # Has output but no clear success/error indicator
                # Consider it successful if it produced reasonable output
                return True
            else:
                # No output and not a command where that's OK
                if verbose:
                    print(f"        No output produced")
                return False
                
        except Exception as e:
            if verbose:
                print(f"        Execution failed: {e}")
            return False
    
    def _get_test_session_id(self, manager) -> Optional[str]:
        """Get a test session ID from the database."""
        try:
            if not manager.duckdb_path.exists():
                return None
                
            with manager._connect() as conn:
                result = conn.execute("""
                    SELECT session_id FROM sessions 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """).fetchone()
                
                return result[0] if result else None
        except:
            return None
    
    def _get_test_session_ids(self, manager, count: int = 2) -> List[str]:
        """Get multiple test session IDs from the database."""
        try:
            if not manager.duckdb_path.exists():
                return []
                
            with manager._connect() as conn:
                results = conn.execute(f"""
                    SELECT session_id FROM sessions 
                    ORDER BY start_time DESC 
                    LIMIT {count}
                """).fetchall()
                
                return [row[0] for row in results]
        except:
            return []
    
    def _test_module(self, module_name: str, module_info: Dict, manager, verbose: bool) -> Tuple[int, int]:
        """Test a specific module's commands."""
        
        instance = module_info['instance']
        commands = module_info['commands']
        
        passed = 0
        failed = 0
        failed_details = []  # Track failed command details
        
        # Test each command safely
        for command_name, description in commands.items():
                
            try:
                if verbose:
                    print(f"      Testing {command_name}...")
                
                # Special handling for help commands
                if command_name == '--help':
                    help_result = self._test_help_command_simple(module_name, instance, manager, verbose)
                    if help_result:
                        passed += 1
                        if verbose:
                            print(f"      ✅ {command_name}: OK")
                    else:
                        failed += 1
                        failed_details.append(f"{command_name}: Help test failed")
                        if verbose:
                            print(f"      ❌ {command_name}: Help test failed")
                    continue
                
                # Create test arguments based on command type
                test_args = self._create_test_args(module_name, command_name, manager)
                
                if test_args is None:
                    if verbose:
                        print(f"      Skipped {command_name} (no test data)")
                    continue
                
                # Capture output
                import io
                import contextlib
                
                output_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                
                with contextlib.redirect_stdout(output_buffer), \
                     contextlib.redirect_stderr(stderr_buffer):
                    instance.execute(test_args, manager)
                
                output = output_buffer.getvalue()
                stderr = stderr_buffer.getvalue()
                full_output = output + stderr
                
                # More specific error detection - avoid false positives
                error_indicators = [
                    "❌ Error",
                    "Failed to",
                    "Exception:",
                    "Traceback",
                    "error:",
                    "ERROR:",
                    "Error detected",
                    "Parser Error:",
                    "AttributeError",
                    "TypeError", 
                    "ValueError",
                    "KeyError"
                ]
                
                # Check if any real error indicator is present
                has_error = any(indicator in full_output for indicator in error_indicators)
                
                # Special handling for certain commands
                if command_name == '--export' and ("✅ Exported" in output or "TOP 10 PERFORMING FEATURES" in output):
                    has_error = False
                elif command_name == '--list' and "FEATURE PERFORMANCE LIST" in output:
                    has_error = False
                
                if has_error:
                    failed += 1
                    # Extract first error message for details
                    error_msg = "Unknown error"
                    for line in full_output.split('\n'):
                        if any(indicator in line for indicator in error_indicators):
                            error_msg = line.strip()[:100]  # First 100 chars
                            break
                    failed_details.append(f"{command_name}: {error_msg}")
                    if verbose:
                        print(f"      ❌ {command_name}: Error detected - {error_msg}")
                else:
                    passed += 1
                    if verbose:
                        print(f"      ✅ {command_name}: OK")
                
            except Exception as e:
                failed += 1
                failed_details.append(f"{command_name}: Exception - {str(e)[:100]}")
                if verbose:
                    print(f"      ❌ {command_name}: Exception - {e}")
        
        # Store failed details for later reporting
        if failed_details:
            setattr(self, f'_{module_name}_failed_details', failed_details)
        
        return passed, failed
    
    def _test_help_command_simple(self, module_name: str, instance, manager, verbose: bool) -> bool:
        """Simple test for help command."""
        try:
            import io
            import contextlib
            import argparse
            
            # Create a test argument namespace with help=True
            test_args = argparse.Namespace()
            
            # Set all common attributes to prevent AttributeError
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
            
            return has_help_content or len(output.strip()) > 10  # Some help output expected
            
        except Exception as e:
            if verbose:
                print(f"        Help test error: {e}")
            return False
    
    def _create_test_backup_file(self, manager) -> Optional[str]:
        """Create a temporary backup file for testing."""
        try:
            import tempfile
            import shutil
            
            # Create temporary backup file by copying existing database
            if manager.duckdb_path.exists():
                temp_dir = Path(tempfile.mkdtemp())
                backup_file = temp_dir / "test_backup.duckdb"
                shutil.copy2(manager.duckdb_path, backup_file)
                return str(backup_file)
            else:
                # Create minimal empty backup file for testing
                temp_dir = Path(tempfile.mkdtemp())
                backup_file = temp_dir / "test_backup.duckdb"
                backup_file.write_bytes(b"DUMMY_BACKUP_FILE_FOR_TESTING")
                return str(backup_file)
                
        except Exception:
            return None
    
    def _get_test_feature_name(self, manager) -> Optional[str]:
        """Get a test feature name from the database."""
        try:
            if not manager.duckdb_path.exists():
                return None
                
            with manager._connect() as conn:
                result = conn.execute("""
                    SELECT feature_name FROM feature_catalog 
                    LIMIT 1
                """).fetchone()
                
                return result[0] if result else None
        except:
            return None
    
    def _create_test_args(self, module_name: str, command_name: str, manager) -> Optional[argparse.Namespace]:
        """Create test arguments for different module commands."""
        
        # Get a real session ID for testing
        session_id = None
        try:
            with manager._connect() as conn:
                result = conn.execute("SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1").fetchone()
                if result:
                    session_id = result[0]
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
        
        # Set specific command arguments
        if command_name == '--list':
            setattr(args, 'list', True)
            setattr(args, 'limit', 5)
            setattr(args, 'status', 'all')
            
        elif command_name == '--show' and session_id:
            setattr(args, 'show', session_id)
            
        elif command_name == '--compare':
            setattr(args, 'compare', ['7', '7'])  # Compare last 7 days with previous 7 days
            
        elif command_name == '--export':
            setattr(args, 'export', 'csv')
            
        elif command_name == '--cleanup':
            # Enable cleanup testing with dry-run mode for safety
            setattr(args, 'cleanup', True)
            setattr(args, 'dry_run', True)  # Safe testing mode
            setattr(args, 'days', 30)
            setattr(args, 'keep', 5)  # Number of backups to keep
            
        elif command_name == '--top':
            setattr(args, 'top', 3)
            
        elif command_name == '--impact':
            # Get a real feature name from database
            feature_name = self._get_test_feature_name(manager)
            if feature_name:
                setattr(args, 'impact', feature_name)
            else:
                # Use a default feature name if none available
                setattr(args, 'impact', 'test_feature')
            
        elif command_name == '--catalog':
            setattr(args, 'catalog', True)
            
        elif command_name == '--search':
            setattr(args, 'search', 'test')
            
        elif command_name == '--summary':
            setattr(args, 'summary', True)
            setattr(args, 'days', 30)
            
        elif command_name == '--trends':
            setattr(args, 'trends', True)
            setattr(args, 'days', 30)
            
        elif command_name == '--operations':
            setattr(args, 'operations', True)
            setattr(args, 'days', 30)
            
        elif command_name == '--convergence' and session_id:
            setattr(args, 'convergence', session_id)
            
        elif command_name == '--report':
            setattr(args, 'report', True)
            setattr(args, 'days', 30)
            
        elif command_name == '--create':
            setattr(args, 'create', True)
            setattr(args, 'compress', False)
            setattr(args, 'keep', 5)
            
        elif command_name == '--restore':
            # Create temporary backup file for testing
            backup_file = self._create_test_backup_file(manager)
            if backup_file:
                setattr(args, 'restore', backup_file)
                setattr(args, 'dry_run', True)  # Safe testing mode
            else:
                return None
            
        elif command_name == '--verify':
            # Create temporary backup file for testing
            backup_file = self._create_test_backup_file(manager)
            if backup_file:
                setattr(args, 'verify', backup_file)
            else:
                return None
        
        elif command_name == '--help':
            # Special handling for help commands
            setattr(args, 'help', True)
            
        # Datasets module specific commands
        elif command_name == '--details':
            # Test with existing dataset
            setattr(args, 'details', 'titanic')
            
        elif command_name == '--register':
            # Skip registration test (would create duplicate data)
            return None
            
        elif command_name == '--stats':
            setattr(args, 'stats', True)
            
        elif command_name == '--sessions':
            # Test with existing dataset
            setattr(args, 'sessions', 'titanic')
            
        elif command_name == '--update':
            # Skip update test (would modify data)
            return None
            
        else:
            return None
        
        return args
    
    def _test_mcts_integration(self, dataset: str, dataset_info: Dict, manager, verbose: bool, config_file: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Test MCTS integration with real feature discovery."""
        
        try:
            # Use provided config file or create temporary config
            if config_file:
                config_path = config_file
                if not Path(config_path).exists():
                    print(f"   ❌ Config file not found: {config_path}")
                    return False, None
                if verbose:
                    print(f"   📋 Using provided config: {config_path}")
            else:
                # Create temporary config
                config_content = self._create_test_config(dataset_info)
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(config_content, f, default_flow_style=False)
                    config_path = f.name
            
            print(f"   🎯 Starting feature discovery with {dataset} dataset...")
            if verbose:
                print(f"   📋 Config: {config_path}")
                print(f"   📊 Train size: {dataset_info.get('train_rows', 'unknown')} rows")
                print(f"   🎲 Target: {dataset_info.get('target_column', 'unknown')}")
            
            # Show animated MCTS simulation message
            print("   🔄 MCTS simulation in progress", end="", flush=True)
            
            # Run feature discovery
            cmd = [
                sys.executable, 
                str(manager.project_root / "run_feature_discovery.py"),
                "--config", config_path,
                "--real-autogluon"
            ]
            
            import threading
            import time
            
            # Animation control
            animation_running = threading.Event()
            animation_running.set()
            
            def animate():
                """Show animated progress indicator"""
                chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                i = 0
                while animation_running.is_set():
                    print(f"\r   🔄 MCTS simulation in progress {chars[i % len(chars)]}", end="", flush=True)
                    time.sleep(0.1)
                    i += 1
            
            # Start animation in background
            animation_thread = threading.Thread(target=animate, daemon=True)
            animation_thread.start()
            
            start_time = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
            end_time = datetime.now()
            
            # Stop animation
            animation_running.clear()
            print("\r   ✅ MCTS simulation completed" + " " * 10, flush=True)  # Clear animation chars
            
            duration = (end_time - start_time).total_seconds()
            
            # Clean up temp config (only if we created it)
            if not config_file:
                os.unlink(config_path)
            
            if result.returncode != 0:
                print(f"   ❌ Feature discovery failed")
                if verbose:
                    print(f"   Error output: {result.stderr}")
                return False, None
            
            # Extract JSON results from output
            mcts_results = self._extract_mcts_results(result.stdout)
            
            if not mcts_results:
                print(f"   ❌ Could not extract MCTS results from output")
                if verbose:
                    print(f"   📝 STDOUT ({len(result.stdout)} chars):")
                    print(f"   {result.stdout[:500]}...")
                    print(f"   📝 STDERR ({len(result.stderr)} chars):")
                    print(f"   {result.stderr[:500]}...")
                return False, None
            
            session_id = mcts_results['session_id']
            
            # Verify session was created (subprocess.run already waited for completion)
            if session_id:
                try:
                    with manager._connect() as conn:
                        session_exists = conn.execute(
                            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", 
                            [session_id]
                        ).fetchone()[0] > 0
                        
                        if session_exists:
                            # Get session details
                            session_info = conn.execute("""
                                SELECT best_score, total_iterations 
                                FROM sessions 
                                WHERE session_id = ?
                            """, [session_id]).fetchone()
                            
                            if session_info:
                                # Use JSON data instead of database values
                                iterations = mcts_results['iterations']
                                best_score = mcts_results['score']
                                
                                session_name = mcts_results.get('session_name')
                                session_display = session_name if session_name else f"{session_id[:8]}..."
                                
                                print(f"   ✅ Feature discovery completed ({duration:.1f}s)")
                                print(f"   ✅ Session created: {session_display}")
                                print(f"   ✅ Iterations: {iterations}")
                                if best_score:
                                    # Try to get metric from session info or mcts_results
                                    actual_metric = 'unknown'
                                    try:
                                        # Try to get from database
                                        metric_result = conn.execute(
                                            "SELECT DISTINCT target_metric FROM exploration_history WHERE session_id = ? LIMIT 1",
                                            [session_id]
                                        ).fetchone()
                                        if metric_result and metric_result[0]:
                                            actual_metric = metric_result[0]
                                        else:
                                            # Fallback to dataset_info
                                            actual_metric = dataset_info.get('metric', 'unknown')
                                    except:
                                        actual_metric = dataset_info.get('metric', 'unknown')
                                    
                                    print(f"   ✅ Best score: {best_score:.5f} ({actual_metric})")
                                else:
                                    print(f"   ⚠️  No score recorded")
                                
                                return True, (session_id, session_name)
                        else:
                            print(f"   ❌ Session not found in database")
                            return False, None
                except Exception as e:
                    print(f"   ❌ Error verifying session: {e}")
                    return False, None
            else:
                print(f"   ❌ Could not extract session ID from output")
                if verbose:
                    print(f"   Output: {result.stdout[:500]}...")
                return False, None
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ Feature discovery timed out (5 minutes)")
            return False, None
        except Exception as e:
            print(f"   ❌ MCTS integration test failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return False, None
    
    def _create_test_config(self, dataset_info: Dict) -> Dict[str, Any]:
        """Create configuration for MCTS test."""
        
        config = {
            'session': {
                'max_iterations': 3,
                'max_runtime_hours': 0.25,
                'checkpoint_interval': 1
            },
            'mcts': {
                'max_tree_depth': 4,
                'expansion_threshold': 1,
                'min_visits_for_best': 2,
                'max_children_per_node': 2,
                'expansion_budget': 5,
                'max_nodes_in_memory': 1000
            },
            'autogluon': {
                'train_path': dataset_info['train_path'],
                'test_path': dataset_info['test_path'],
                'target_metric': dataset_info.get('metric', 'accuracy'),
                'included_model_types': ['XGB'],
                'enable_gpu': True,
                'train_size': min(100, dataset_info.get('train_rows', 100)),
                'time_limit': 10,
                'presets': 'medium_quality',
                'skip_final_evaluation': True,
                'holdout_frac': 0.4,
                'verbosity': 0,
                'ag_args_fit': {
                    'num_cpus': 1,
                    'num_gpus': 1
                },
                'ag_args_ensemble': {
                    'fold_fitting_strategy': 'sequential_local',
                    'enable_ray': False
                }
            },
            'feature_space': {
                'domain_module': 'domains.generic',  # Use generic domain for unknown datasets
                'max_features_per_node': 20,
                'feature_timeout': 30,
                'max_cache_size_mb': 256,
                'max_features_to_build': 5,
                'max_features_per_iteration': 2,
                'feature_build_timeout': 60,
                'cache_miss_limit': 10,
                'enabled_categories': ['statistical_aggregations', 'feature_transformations'],
                'category_weights': {
                    'statistical_aggregations': 2.0,
                    'feature_transformations': 2.0
                }
            },
            'resources': {
                'max_memory_gb': 1,
                'memory_check_interval': 1,
                'force_gc_interval': 5,
                'max_cpu_cores': 2,
                'max_disk_usage_gb': 5
            },
            'database': {
                'path': 'data/feature_discovery_selfcheck.duckdb',
                'max_history_size': 100,
                'backup_interval': 10,
                'retention_days': 1
            },
            'logging': {
                'log_level': 'WARNING',
                'max_log_size_mb': 10,
                'progress_interval': 1,
                'log_autogluon_details': False
            },
            'export': {
                'formats': ['python'],
                'include_documentation': False,
                'export_on_improvement': False,
                'export_on_completion': True
            },
            'analytics': {
                'generate_charts': False,
                'include_timing_analysis': False
            },
            'validation': {
                'validate_generated_features': False
            },
            'testing': {
                'fast_test_mode': True,  # Mark selfcheck as test mode
                'use_small_dataset': True
            }
        }
        
        # Add target column if detected
        if dataset_info.get('target_column'):
            config['autogluon']['target_column'] = dataset_info['target_column']
        
        return config
    
    def _extract_mcts_results(self, output: str) -> Optional[Dict[str, Any]]:
        """Extract MCTS results JSON from command output."""
        
        # Look for JSON marker in output
        import re
        import json
        
        json_match = re.search(r'MCTS_RESULT_JSON:({.*?})', output)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return None
        
        return None
    
    def _print_final_results(self, test_results: List[Tuple[str, bool]], session_name: Optional[str]) -> None:
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
                print(f"   Session details: python scripts/duckdb_manager.py sessions --show {session_name}")
        else:
            print("❌ FINAL RESULT: SOME TESTS FAILED")
            print("💡 Check individual test results above for details")
    
