#!/usr/bin/env python3
"""
Verification Module - Comprehensive MCTS session verification

Provides comprehensive validation of MCTS discovery sessions including:
- Database integrity verification
- MCTS algorithm correctness validation
- AutoGluon integration verification
- File system and cache validation
"""

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from . import ModuleInterface

class VerificationModule(ModuleInterface):
    """Module for comprehensive MCTS session verification."""
    
    @property
    def name(self) -> str:
        return "verification"
    
    @property
    def description(self) -> str:
        return "Comprehensive MCTS session verification and validation"
    
    @property
    def commands(self) -> Dict[str, str]:
        return {
            "--verify-session": "Verify specific session integrity and correctness",
            "--verify-all": "Verify all sessions in database",
            "--verify-latest": "Verify most recent session",
            "--last": "Verify last N sessions (default: 1 if no number given)",
            "--failed-only": "Show only sessions with FAIL status",
            "--warn-only": "Show only sessions with WARN status", 
            "--pass-only": "Show only sessions with PASS status",
            "--report": "Generate detailed verification report",
            "--quick": "Run quick verification (skip detailed analysis)",
            "--help": "Show detailed help for verification module"
        }
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add verification-specific arguments."""
        verification_group = parser.add_argument_group('Verification Module')
        verification_group.add_argument('--verify-session', type=str, metavar='SESSION_ID',
                                      help='Verify specific session by ID or name')
        verification_group.add_argument('--verify-all', action='store_true',
                                      help='Verify all sessions in database')
        verification_group.add_argument('--verify-latest', action='store_true',
                                      help='Verify most recent session')
        verification_group.add_argument('--last', type=int, metavar='N', nargs='?',
                                      const=1, default=None,
                                      help='Verify last N sessions (default: 1 if no number given)')
        verification_group.add_argument('--failed-only', action='store_true',
                                      help='Show only sessions with FAIL status')
        verification_group.add_argument('--warn-only', action='store_true',
                                      help='Show only sessions with WARN status')
        verification_group.add_argument('--pass-only', action='store_true',
                                      help='Show only sessions with PASS status')
        verification_group.add_argument('--report', type=str, metavar='FORMAT',
                                      choices=['json', 'text', 'html'],
                                      default='text',
                                      help='Report format (default: text)')
        verification_group.add_argument('--quick', action='store_true',
                                      help='Quick verification (skip detailed analysis)')
        verification_group.add_argument('--output', type=str, metavar='FILE',
                                      help='Output report to file')
        verification_group.add_argument('--verbose', action='store_true',
                                      help='Verbose output with detailed diagnostics')
    
    def execute(self, args: argparse.Namespace, manager) -> None:
        """Execute verification module commands."""
        
        if getattr(args, 'verify_session', None):
            self._verify_session(args.verify_session, args, manager)
        elif getattr(args, 'verify_all', False):
            self._verify_all_sessions(args, manager)
        elif getattr(args, 'verify_latest', False):
            self._verify_latest_session(args, manager)
        elif getattr(args, 'last', None) is not None:
            self._verify_last_sessions(args.last, args, manager)
        else:
            print("❌ No verification command specified. Use --help for options.")
    
    def _verify_session(self, session_identifier: str, args: argparse.Namespace, manager) -> None:
        """Verify a specific session comprehensively."""
        print(f"🔍 COMPREHENSIVE SESSION VERIFICATION")
        print("=" * 60)
        print(f"Session: {session_identifier}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Initialize verification results
        verification_results = {
            'session_id': session_identifier,
            'verification_time': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'categories': {}
        }
        
        try:
            # Get session info
            session_info = self._get_session_info(session_identifier, manager)
            if not session_info:
                print(f"❌ Session not found: {session_identifier}")
                return
            
            verification_results['session_info'] = session_info
            
            # Run verification categories
            print("📊 Running verification checks...")
            print()
            
            # 1. Database Integrity Verification
            db_results = self._verify_database_integrity(session_info, args, manager)
            verification_results['categories']['database'] = db_results
            self._print_category_results("Database Integrity", db_results)
            
            # 2. MCTS Algorithm Correctness
            mcts_results = self._verify_mcts_correctness(session_info, args, manager)
            verification_results['categories']['mcts'] = mcts_results
            self._print_category_results("MCTS Algorithm", mcts_results)
            
            # 3. AutoGluon Integration Verification
            autogluon_results = self._verify_autogluon_integration(session_info, args, manager)
            verification_results['categories']['autogluon'] = autogluon_results
            self._print_category_results("AutoGluon Integration", autogluon_results)
            
            # 4. File System and Cache Verification
            fs_results = self._verify_filesystem_cache(session_info, args, manager)
            verification_results['categories']['filesystem'] = fs_results
            self._print_category_results("File System & Cache", fs_results)
            
            # 5. Configuration and Environment Validation
            config_results = self._verify_configuration(session_info, args, manager)
            verification_results['categories']['configuration'] = config_results
            self._print_category_results("Configuration", config_results)
            
            # Calculate overall status
            verification_results['overall_status'] = self._calculate_overall_status(verification_results)
            
            # Print summary
            self._print_verification_summary(verification_results)
            
            # Generate report if requested
            if args.output or args.report != 'text':
                self._generate_report(verification_results, args)
                
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            verification_results['overall_status'] = 'ERROR'
            verification_results['error'] = str(e)
    
    def _verify_latest_session(self, args: argparse.Namespace, manager) -> None:
        """Verify the most recent session."""
        try:
            with manager._connect() as conn:
                latest_session = conn.execute("""
                    SELECT session_id, session_name 
                    FROM sessions 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """).fetchone()
                
                if not latest_session:
                    print("❌ No sessions found in database")
                    return
                
                session_id, session_name = latest_session
                print(f"🔍 Verifying latest session: {session_name} ({session_id[:8]}...)")
                print()
                
                self._verify_session(session_name or session_id, args, manager)
                
        except Exception as e:
            print(f"❌ Error finding latest session: {e}")
    
    def _verify_last_sessions(self, count: int, args: argparse.Namespace, manager) -> None:
        """Verify last N sessions."""
        print(f"🔍 VERIFYING LAST {count} SESSION{'S' if count != 1 else ''}")
        print("=" * 60)
        
        try:
            with manager._connect() as conn:
                sessions = conn.execute("""
                    SELECT session_id, session_name, status 
                    FROM sessions 
                    ORDER BY start_time DESC
                    LIMIT ?
                """, [count]).fetchall()
                
                if not sessions:
                    print("❌ No sessions found in database")
                    return
                
                # Reverse to show oldest to newest (but only the last N)
                sessions = list(reversed(sessions))
                
                print(f"Found {len(sessions)} session{'s' if len(sessions) != 1 else ''} to verify")
                print()
                
                # Use the same table format as verify_all
                self._verify_sessions_batch(sessions, args, manager)
                
        except Exception as e:
            print(f"❌ Error verifying last sessions: {e}")
    
    def _verify_all_sessions(self, args: argparse.Namespace, manager) -> None:
        """Verify all sessions in database."""
        print("🔍 BATCH SESSION VERIFICATION")
        print("=" * 60)
        
        try:
            with manager._connect() as conn:
                sessions = conn.execute("""
                    SELECT session_id, session_name, status 
                    FROM sessions 
                    ORDER BY start_time ASC
                """).fetchall()
                
                if not sessions:
                    print("❌ No sessions found in database")
                    return
                
                print(f"Found {len(sessions)} sessions to verify (oldest to newest)")
                print()
                
                # Use common batch verification logic
                self._verify_sessions_batch(sessions, args, manager)
                
        except Exception as e:
            print(f"❌ Batch verification failed: {e}")
    
    def _verify_sessions_batch(self, sessions: List[tuple], args: argparse.Namespace, manager) -> None:
        """Common batch verification logic with filtering support."""
        
        # Print table header
        print(f"{'#':<4} {'Session Name':<25} {'Status':<8} {'Result':<6}")
        print("-" * 50)
        
        batch_results = []
        displayed_count = 0
        
        for i, (session_id, session_name, status) in enumerate(sessions, 1):
            # Run quick verification for batch mode
            original_quick = getattr(args, 'quick', False)
            args.quick = True
            
            try:
                verification_result = self._verify_session_quick(session_name or session_id, args, manager)
                batch_results.append(verification_result)
                
                # Format result with icon
                result_status = verification_result['overall_status']
                if result_status == 'PASS':
                    result_display = "✅ PASS"
                elif result_status == 'WARN':
                    result_display = "⚠️ WARN"
                elif result_status == 'FAIL':
                    result_display = "❌ FAIL"
                else:
                    result_display = "💥 ERROR"
                
            except Exception as e:
                batch_results.append({
                    'session_id': session_id,
                    'overall_status': 'ERROR',
                    'error': str(e)
                })
                result_display = "💥 ERROR"
                result_status = 'ERROR'
            
            # Apply filters
            should_display = self._should_display_result(result_status, args)
            
            if should_display:
                # Print table row
                session_display = (session_name or session_id[:8])[:24]
                status_display = status[:7] if status else "unknown"
                print(f"{i:<4} {session_display:<25} {status_display:<8} {result_display}")
                displayed_count += 1
            
            # Restore original quick setting
            args.quick = original_quick
        
        print()
        
        # Print filtering info if filters were applied
        if displayed_count != len(sessions):
            filter_info = self._get_filter_description(args)
            print(f"📋 Showing {displayed_count}/{len(sessions)} sessions ({filter_info})")
            print()
        
        # Print batch summary
        self._print_batch_summary(batch_results)
    
    def _should_display_result(self, result_status: str, args: argparse.Namespace) -> bool:
        """Check if result should be displayed based on filters."""
        # If any filter is specified, use it
        if getattr(args, 'failed_only', False):
            return result_status in ['FAIL', 'ERROR']
        elif getattr(args, 'warn_only', False):
            return result_status == 'WARN'
        elif getattr(args, 'pass_only', False):
            return result_status == 'PASS'
        else:
            # No filter specified, show all
            return True
    
    def _get_filter_description(self, args: argparse.Namespace) -> str:
        """Get description of active filters."""
        if getattr(args, 'failed_only', False):
            return "failed only"
        elif getattr(args, 'warn_only', False):
            return "warnings only"
        elif getattr(args, 'pass_only', False):
            return "passed only"
        else:
            return "all"
    
    def _get_session_info(self, session_identifier: str, manager) -> Optional[Dict[str, Any]]:
        """Get comprehensive session information."""
        try:
            with manager._connect() as conn:
                # Try to find session by ID or name
                session_data = conn.execute("""
                    SELECT * FROM sessions 
                    WHERE session_id = ? OR session_name = ?
                """, [session_identifier, session_identifier]).fetchone()
                
                if not session_data:
                    return None
                
                # Parse session data
                session_info = {
                    'session_id': session_data[0],
                    'session_name': session_data[1],
                    'start_time': session_data[2],
                    'end_time': session_data[3],
                    'total_iterations': session_data[4],
                    'best_score': session_data[5],
                    'config_snapshot': session_data[6],
                    'status': session_data[7],
                    'strategy': session_data[8],
                    'is_test_mode': session_data[9],
                    'notes': session_data[10]
                }
                
                # Parse config if available
                if session_info['config_snapshot']:
                    try:
                        session_info['config'] = json.loads(session_info['config_snapshot'])
                    except json.JSONDecodeError:
                        session_info['config'] = None
                
                return session_info
                
        except Exception as e:
            print(f"❌ Error getting session info: {e}")
            return None
    
    def _verify_database_integrity(self, session_info: Dict[str, Any], args: argparse.Namespace, manager) -> Dict[str, Any]:
        """Verify database integrity for the session."""
        results = {
            'status': 'UNKNOWN',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            with manager._connect() as conn:
                session_id = session_info['session_id']
                
                # Check 1: Session record completeness
                required_fields = ['session_id', 'session_name', 'start_time', 'status']
                missing_fields = [field for field in required_fields if not session_info.get(field)]
                
                if missing_fields:
                    results['errors'].append(f"Missing required session fields: {missing_fields}")
                    results['checks']['session_completeness'] = 'FAIL'
                else:
                    results['checks']['session_completeness'] = 'PASS'
                
                # Check 2: Exploration history consistency
                exploration_count = conn.execute(
                    "SELECT COUNT(*) FROM exploration_history WHERE session_id = ?", 
                    [session_id]
                ).fetchone()[0]
                
                expected_explorations = session_info.get('total_iterations', 0) * 2  # Rough estimate
                if exploration_count == 0 and session_info.get('total_iterations', 0) > 0:
                    results['errors'].append("No exploration history found despite completed iterations")
                    results['checks']['exploration_history'] = 'FAIL'
                elif exploration_count < session_info.get('total_iterations', 0):
                    results['warnings'].append(f"Fewer explorations ({exploration_count}) than iterations ({session_info.get('total_iterations', 0)})")
                    results['checks']['exploration_history'] = 'WARN'
                else:
                    results['checks']['exploration_history'] = 'PASS'
                
                # Check 3: Score consistency
                max_exploration_score = conn.execute(
                    "SELECT MAX(evaluation_score) FROM exploration_history WHERE session_id = ?",
                    [session_id]
                ).fetchone()[0]
                
                session_best_score = session_info.get('best_score')
                if max_exploration_score and session_best_score:
                    if abs(max_exploration_score - session_best_score) > 0.001:
                        results['warnings'].append(f"Score mismatch: session={session_best_score:.5f}, exploration_max={max_exploration_score:.5f}")
                        results['checks']['score_consistency'] = 'WARN'
                    else:
                        results['checks']['score_consistency'] = 'PASS'
                else:
                    results['checks']['score_consistency'] = 'SKIP'
                
                # Check 4: Timing consistency
                if session_info.get('start_time') and session_info.get('end_time'):
                    try:
                        start_time_str = str(session_info['start_time']).replace('Z', '+00:00')
                        end_time_str = str(session_info['end_time']).replace('Z', '+00:00')
                        
                        start_time = datetime.fromisoformat(start_time_str)
                        end_time = datetime.fromisoformat(end_time_str)
                        
                        if end_time <= start_time:
                            results['errors'].append("End time is before or equal to start time")
                            results['checks']['timing_consistency'] = 'FAIL'
                        else:
                            duration = (end_time - start_time).total_seconds()
                            if duration > 86400:  # More than 24 hours
                                results['warnings'].append(f"Unusually long session duration: {duration/3600:.1f} hours")
                            results['checks']['timing_consistency'] = 'PASS'
                    except Exception as e:
                        results['warnings'].append(f"Error parsing timestamps: {e}")
                        results['checks']['timing_consistency'] = 'WARN'
                else:
                    results['warnings'].append("Missing start or end time")
                    results['checks']['timing_consistency'] = 'WARN'
                
                # Check 5: Feature impact data
                feature_impact_count = conn.execute(
                    "SELECT COUNT(*) FROM feature_impact WHERE session_id = ?",
                    [session_id]
                ).fetchone()[0]
                
                if feature_impact_count == 0 and exploration_count > 0:
                    results['warnings'].append("No feature impact data found")
                    results['checks']['feature_impact'] = 'WARN'
                else:
                    results['checks']['feature_impact'] = 'PASS'
                
        except Exception as e:
            results['errors'].append(f"Database verification error: {e}")
            results['status'] = 'ERROR'
            return results
        
        # Determine overall status
        if results['errors']:
            results['status'] = 'FAIL'
        elif results['warnings']:
            results['status'] = 'WARN'
        else:
            results['status'] = 'PASS'
        
        return results
    
    def _verify_mcts_correctness(self, session_info: Dict[str, Any], args: argparse.Namespace, manager) -> Dict[str, Any]:
        """Verify MCTS algorithm correctness."""
        results = {
            'status': 'UNKNOWN',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            with manager._connect() as conn:
                session_id = session_info['session_id']
                
                # Check 1: Operation sequence validity
                explorations = conn.execute("""
                    SELECT operation_applied, evaluation_score, iteration 
                    FROM exploration_history 
                    WHERE session_id = ? 
                    ORDER BY timestamp
                """, [session_id]).fetchall()
                
                if not explorations:
                    results['warnings'].append("No exploration data found")
                    results['checks']['operation_sequence'] = 'WARN'
                else:
                    # Check for valid operations
                    valid_operations = set()
                    for op, score, iteration in explorations:
                        if op:
                            valid_operations.add(op)
                    
                    if len(valid_operations) == 0:
                        results['errors'].append("No valid operations found")
                        results['checks']['operation_sequence'] = 'FAIL'
                    else:
                        results['checks']['operation_sequence'] = 'PASS'
                
                # Check 2: Score progression analysis
                if explorations:
                    scores = [score for _, score, _ in explorations if score is not None]
                    if scores:
                        best_score_progression = []
                        current_best = 0
                        for score in scores:
                            current_best = max(current_best, score)
                            best_score_progression.append(current_best)
                        
                        # Check if best score improved or stayed same (monotonic improvement)
                        is_monotonic = all(best_score_progression[i] >= best_score_progression[i-1] 
                                         for i in range(1, len(best_score_progression)))
                        
                        if not is_monotonic:
                            results['errors'].append("Best score progression is not monotonic")
                            results['checks']['score_progression'] = 'FAIL'
                        else:
                            results['checks']['score_progression'] = 'PASS'
                        
                        # Check for reasonable score improvement
                        final_best = best_score_progression[-1]
                        initial_score = scores[0] if scores else 0
                        improvement = final_best - initial_score
                        
                        if improvement < 0:
                            results['errors'].append(f"Negative improvement: {improvement:.5f}")
                            results['checks']['improvement'] = 'FAIL'
                        elif improvement == 0 and len(scores) > 1:
                            results['warnings'].append("No score improvement observed")
                            results['checks']['improvement'] = 'WARN'
                        else:
                            results['checks']['improvement'] = 'PASS'
                    else:
                        results['warnings'].append("No valid scores found")
                        results['checks']['score_progression'] = 'WARN'
                
                # Check 3: Iteration consistency
                expected_iterations = session_info.get('total_iterations', 0)
                if explorations:
                    valid_iterations = [iteration for _, _, iteration in explorations if iteration is not None]
                    if valid_iterations:
                        max_iteration = max(valid_iterations)
                        # Check if iterations are reasonable (allowing some flexibility)
                        if max_iteration > expected_iterations + 2:  # Allow some tolerance
                            results['warnings'].append(f"Iteration mismatch: max_found={max_iteration}, expected_max={expected_iterations}")
                            results['checks']['iteration_consistency'] = 'WARN'
                        else:
                            results['checks']['iteration_consistency'] = 'PASS'
                    else:
                        results['warnings'].append("No valid iteration numbers found")
                        results['checks']['iteration_consistency'] = 'WARN'
                else:
                    results['checks']['iteration_consistency'] = 'SKIP'
                
        except Exception as e:
            results['errors'].append(f"MCTS verification error: {e}")
            results['status'] = 'ERROR'
            return results
        
        # Determine overall status
        if results['errors']:
            results['status'] = 'FAIL'
        elif results['warnings']:
            results['status'] = 'WARN'
        else:
            results['status'] = 'PASS'
        
        return results
    
    def _verify_autogluon_integration(self, session_info: Dict[str, Any], args: argparse.Namespace, manager) -> Dict[str, Any]:
        """Verify AutoGluon integration correctness."""
        results = {
            'status': 'UNKNOWN',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            config = session_info.get('config', {})
            autogluon_config = config.get('autogluon', {}) if config else {}
            
            # Check 1: Configuration validation
            required_config = ['target_metric', 'included_model_types']
            missing_config = [key for key in required_config if key not in autogluon_config]
            
            if missing_config:
                results['errors'].append(f"Missing AutoGluon config: {missing_config}")
                results['checks']['config_validation'] = 'FAIL'
            else:
                results['checks']['config_validation'] = 'PASS'
            
            # Check 2: Target metric consistency
            target_metric = autogluon_config.get('target_metric')
            session_best_score = session_info.get('best_score')
            
            if target_metric and session_best_score is not None:
                # Basic sanity check for score ranges
                if target_metric.lower() in ['accuracy'] and (session_best_score < 0 or session_best_score > 1):
                    results['errors'].append(f"Invalid accuracy score: {session_best_score}")
                    results['checks']['metric_consistency'] = 'FAIL'
                elif target_metric.lower() in ['map@3'] and (session_best_score < 0 or session_best_score > 1):
                    results['errors'].append(f"Invalid MAP@3 score: {session_best_score}")
                    results['checks']['metric_consistency'] = 'FAIL'
                else:
                    results['checks']['metric_consistency'] = 'PASS'
            else:
                results['checks']['metric_consistency'] = 'SKIP'
            
            # Check 3: Model type validation
            model_types = autogluon_config.get('included_model_types', [])
            valid_model_types = ['XGB', 'GBM', 'CAT', 'RF', 'LR', 'KNN', 'FASTAI', 'NN_TORCH']
            
            invalid_models = [model for model in model_types if model not in valid_model_types]
            if invalid_models:
                results['warnings'].append(f"Unknown model types: {invalid_models}")
                results['checks']['model_types'] = 'WARN'
            elif not model_types:
                results['warnings'].append("No model types specified")
                results['checks']['model_types'] = 'WARN'
            else:
                results['checks']['model_types'] = 'PASS'
            
            # Check 4: Training data validation
            train_path = autogluon_config.get('train_path')
            if train_path:
                train_path_obj = Path(train_path)
                if not train_path_obj.exists():
                    results['errors'].append(f"Training data not found: {train_path}")
                    results['checks']['data_validation'] = 'FAIL'
                else:
                    results['checks']['data_validation'] = 'PASS'
            else:
                results['warnings'].append("No training data path specified")
                results['checks']['data_validation'] = 'WARN'
            
            # Check 5: Performance analysis
            with manager._connect() as conn:
                session_id = session_info['session_id']
                avg_eval_time = conn.execute(
                    "SELECT AVG(evaluation_time) FROM exploration_history WHERE session_id = ? AND evaluation_time IS NOT NULL",
                    [session_id]
                ).fetchone()[0]
                
                if avg_eval_time:
                    if avg_eval_time > 300:  # More than 5 minutes per evaluation
                        results['warnings'].append(f"High average evaluation time: {avg_eval_time:.1f}s")
                        results['checks']['performance'] = 'WARN'
                    else:
                        results['checks']['performance'] = 'PASS'
                else:
                    results['checks']['performance'] = 'SKIP'
            
        except Exception as e:
            results['errors'].append(f"AutoGluon verification error: {e}")
            results['status'] = 'ERROR'
            return results
        
        # Determine overall status
        if results['errors']:
            results['status'] = 'FAIL'
        elif results['warnings']:
            results['status'] = 'WARN'
        else:
            results['status'] = 'PASS'
        
        return results
    
    def _verify_filesystem_cache(self, session_info: Dict[str, Any], args: argparse.Namespace, manager) -> Dict[str, Any]:
        """Verify file system and cache integrity."""
        results = {
            'status': 'UNKNOWN',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check 1: Log file existence and integrity
            log_files = list(Path('.').glob('**/mcts_discovery.log'))
            if not log_files:
                results['warnings'].append("MCTS discovery log file not found")
                results['checks']['log_files'] = 'WARN'
            else:
                # Check if session is mentioned in logs
                session_name = session_info.get('session_name', '')
                found_in_logs = False
                
                for log_file in log_files:
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()
                            if session_name in content:
                                found_in_logs = True
                                break
                    except Exception:
                        continue
                
                if found_in_logs:
                    results['checks']['log_files'] = 'PASS'
                else:
                    results['warnings'].append(f"Session {session_name} not found in log files")
                    results['checks']['log_files'] = 'WARN'
            
            # Check 2: Cache directory structure
            cache_dirs = list(Path('.').glob('cache/*'))
            if not cache_dirs:
                results['warnings'].append("No cache directories found")
                results['checks']['cache_structure'] = 'WARN'
            else:
                results['checks']['cache_structure'] = 'PASS'
            
            # Check 3: Database file integrity
            db_path = manager.duckdb_path
            if not db_path.exists():
                results['errors'].append(f"Main database not found: {db_path}")
                results['checks']['database_files'] = 'FAIL'
            else:
                # Check database size
                db_size = db_path.stat().st_size
                if db_size == 0:
                    results['errors'].append("Database file is empty")
                    results['checks']['database_files'] = 'FAIL'
                else:
                    results['checks']['database_files'] = 'PASS'
            
            # Check 4: Export/output files
            config = session_info.get('config', {})
            export_config = config.get('export', {}) if config else {}
            
            if export_config:
                python_output = export_config.get('python_output')
                if python_output:
                    output_path = Path(python_output)
                    if not output_path.exists():
                        results['warnings'].append(f"Expected output file not found: {python_output}")
                        results['checks']['export_files'] = 'WARN'
                    else:
                        results['checks']['export_files'] = 'PASS'
                else:
                    results['checks']['export_files'] = 'SKIP'
            else:
                results['checks']['export_files'] = 'SKIP'
            
            # Check 5: Backup files (if session completed)
            if session_info.get('status') == 'completed':
                backup_dir = manager.project_root / "data" / "backups"
                if backup_dir.exists():
                    backup_files = list(backup_dir.glob("*.duckdb*"))
                    if backup_files:
                        results['checks']['backup_files'] = 'PASS'
                    else:
                        results['warnings'].append("No backup files found for completed session")
                        results['checks']['backup_files'] = 'WARN'
                else:
                    results['checks']['backup_files'] = 'SKIP'
            else:
                results['checks']['backup_files'] = 'SKIP'
            
        except Exception as e:
            results['errors'].append(f"Filesystem verification error: {e}")
            results['status'] = 'ERROR'
            return results
        
        # Determine overall status
        if results['errors']:
            results['status'] = 'FAIL'
        elif results['warnings']:
            results['status'] = 'WARN'
        else:
            results['status'] = 'PASS'
        
        return results
    
    def _verify_configuration(self, session_info: Dict[str, Any], args: argparse.Namespace, manager) -> Dict[str, Any]:
        """Verify configuration and environment consistency."""
        results = {
            'status': 'UNKNOWN',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            config = session_info.get('config', {})
            
            # Check 1: Required configuration sections
            required_sections = ['session', 'autogluon', 'feature_space', 'mcts']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                results['errors'].append(f"Missing config sections: {missing_sections}")
                results['checks']['config_completeness'] = 'FAIL'
            else:
                results['checks']['config_completeness'] = 'PASS'
            
            # Check 2: Session configuration validation
            session_config = config.get('session', {})
            max_iterations = session_config.get('max_iterations')
            actual_iterations = session_info.get('total_iterations', 0)
            
            if max_iterations and actual_iterations > max_iterations:
                results['errors'].append(f"Exceeded max iterations: {actual_iterations} > {max_iterations}")
                results['checks']['iteration_limits'] = 'FAIL'
            else:
                results['checks']['iteration_limits'] = 'PASS'
            
            # Check 3: Feature space configuration
            feature_config = config.get('feature_space', {})
            domain_module = feature_config.get('custom_domain_module')
            
            if domain_module:
                # Check if domain module exists
                try:
                    import importlib
                    import sys
                    
                    # Add src directory to path for domain modules
                    src_path = str(Path('.').resolve() / 'src')
                    if src_path not in sys.path:
                        sys.path.insert(0, src_path)
                    
                    importlib.import_module(domain_module)
                    results['checks']['domain_module'] = 'PASS'
                except ImportError as e:
                    # Check if it's the expected titanic domain
                    if domain_module == 'domains.titanic':
                        # Check if file exists
                        domain_file = Path('src/domains/titanic.py')
                        if domain_file.exists():
                            results['warnings'].append(f"Domain module file exists but import failed: {e}")
                            results['checks']['domain_module'] = 'WARN'
                        else:
                            results['warnings'].append(f"Domain module file not found: {domain_file}")
                            results['checks']['domain_module'] = 'WARN'
                    else:
                        results['errors'].append(f"Domain module not found: {domain_module}")
                        results['checks']['domain_module'] = 'FAIL'
            else:
                results['checks']['domain_module'] = 'SKIP'
            
            # Check 4: Resource constraints
            resources_config = config.get('resources', {})
            max_memory = resources_config.get('max_memory_gb')
            
            if max_memory:
                try:
                    # Get current Python process memory usage (not system-wide)
                    import psutil
                    import os
                    
                    # Get memory usage of current Python process
                    process = psutil.Process(os.getpid())
                    current_memory_mb = process.memory_info().rss / (1024**2)
                    current_memory_gb = current_memory_mb / 1024
                    
                    # For test sessions, be more lenient with memory limits
                    is_test = session_info.get('is_test_mode', False)
                    memory_multiplier = 10 if is_test else 2  # More lenient for test sessions
                    
                    if current_memory_gb > max_memory * memory_multiplier:
                        results['warnings'].append(f"Python process memory ({current_memory_gb:.1f}GB) may exceed limits ({max_memory}GB)")
                        results['checks']['resource_constraints'] = 'WARN'
                    else:
                        results['checks']['resource_constraints'] = 'PASS'
                        
                    # Add debug info for very low memory processes
                    if current_memory_gb < 0.1:  # Less than 100MB
                        if getattr(args, 'verbose', False):
                            results['warnings'].append(f"Python process using only {current_memory_mb:.1f}MB - memory check may be inaccurate")
                            
                except ImportError:
                    results['warnings'].append("psutil not available for memory check")
                    results['checks']['resource_constraints'] = 'SKIP'
            else:
                results['checks']['resource_constraints'] = 'SKIP'
            
            # Check 5: Test mode consistency
            test_mode = session_info.get('is_test_mode', False)
            testing_config = config.get('testing', {})
            fast_test_mode = testing_config.get('fast_test_mode', False)
            
            if test_mode != fast_test_mode:
                results['warnings'].append(f"Test mode mismatch: session={test_mode}, config={fast_test_mode}")
                results['checks']['test_mode_consistency'] = 'WARN'
            else:
                results['checks']['test_mode_consistency'] = 'PASS'
            
        except Exception as e:
            results['errors'].append(f"Configuration verification error: {e}")
            results['status'] = 'ERROR'
            return results
        
        # Determine overall status
        if results['errors']:
            results['status'] = 'FAIL'
        elif results['warnings']:
            results['status'] = 'WARN'
        else:
            results['status'] = 'PASS'
        
        return results
    
    def _verify_session_quick(self, session_identifier: str, args: argparse.Namespace, manager) -> Dict[str, Any]:
        """Quick verification for batch processing."""
        session_info = self._get_session_info(session_identifier, manager)
        if not session_info:
            return {'session_id': session_identifier, 'overall_status': 'NOT_FOUND'}
        
        # Run only critical checks
        results = {
            'session_id': session_identifier,
            'session_name': session_info.get('session_name'),
            'checks_run': 'quick',
            'categories': {}
        }
        
        # Quick database check
        db_results = self._verify_database_integrity(session_info, args, manager)
        results['categories']['database'] = {'status': db_results['status']}
        
        # Quick MCTS check
        mcts_results = self._verify_mcts_correctness(session_info, args, manager)
        results['categories']['mcts'] = {'status': mcts_results['status']}
        
        # Calculate overall status
        results['overall_status'] = self._calculate_overall_status(results)
        
        return results
    
    def _calculate_overall_status(self, verification_results: Dict[str, Any]) -> str:
        """Calculate overall verification status."""
        categories = verification_results.get('categories', {})
        
        if not categories:
            return 'UNKNOWN'
        
        statuses = [category.get('status', 'UNKNOWN') for category in categories.values()]
        
        if 'ERROR' in statuses:
            return 'ERROR'
        elif 'FAIL' in statuses:
            return 'FAIL'
        elif 'WARN' in statuses:
            return 'WARN'
        elif all(status == 'PASS' for status in statuses):
            return 'PASS'
        else:
            return 'PARTIAL'
    
    def _print_category_results(self, category_name: str, results: Dict[str, Any]) -> None:
        """Print results for a verification category."""
        status = results.get('status', 'UNKNOWN')
        status_icon = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️', 'ERROR': '💥', 'UNKNOWN': '❓'}.get(status, '❓')
        
        print(f"{status_icon} {category_name}: {status}")
        
        # Print individual checks
        checks = results.get('checks', {})
        for check_name, check_status in checks.items():
            check_icon = {'PASS': '  ✓', 'FAIL': '  ✗', 'WARN': '  ⚠', 'SKIP': '  ○'}.get(check_status, '  ?')
            print(f"{check_icon} {check_name.replace('_', ' ').title()}")
        
        # Print errors and warnings
        for error in results.get('errors', []):
            print(f"  ❌ {error}")
        
        for warning in results.get('warnings', []):
            print(f"  ⚠️  {warning}")
        
        print()
    
    def _print_verification_summary(self, verification_results: Dict[str, Any]) -> None:
        """Print verification summary."""
        overall_status = verification_results.get('overall_status', 'UNKNOWN')
        status_icon = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️', 'ERROR': '💥', 'UNKNOWN': '❓'}.get(overall_status, '❓')
        
        print("=" * 60)
        print(f"🎯 VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {status_icon} {overall_status}")
        print(f"Session: {verification_results.get('session_id', 'Unknown')}")
        print(f"Verification Time: {verification_results.get('verification_time', 'Unknown')}")
        print()
        
        # Count checks by status
        all_checks = {}
        for category_results in verification_results.get('categories', {}).values():
            for check_name, check_status in category_results.get('checks', {}).items():
                all_checks[check_status] = all_checks.get(check_status, 0) + 1
        
        print("📊 Check Summary:")
        for status in ['PASS', 'WARN', 'FAIL', 'ERROR', 'SKIP']:
            count = all_checks.get(status, 0)
            if count > 0:
                icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'ERROR': '💥', 'SKIP': '○'}.get(status, '?')
                print(f"   {icon} {status}: {count}")
        
        print()
        
        # Recommendations
        if overall_status in ['FAIL', 'ERROR']:
            print("🔧 RECOMMENDATIONS:")
            print("   • Review error messages above for specific issues")
            print("   • Check database integrity and session completeness")
            print("   • Verify configuration and environment setup")
            print("   • Consider re-running the session if critical errors found")
        elif overall_status == 'WARN':
            print("💡 RECOMMENDATIONS:")
            print("   • Review warnings for potential improvements")
            print("   • Session completed successfully with minor issues")
            print("   • Consider optimizing configuration for future runs")
        else:
            print("🎉 Session verification completed successfully!")
            print("   • All critical checks passed")
            print("   • Session integrity verified")
            print("   • MCTS algorithm performed correctly")
        
        print()
    
    def _print_batch_summary(self, batch_results: List[Dict[str, Any]]) -> None:
        """Print summary for batch verification."""
        total_sessions = len(batch_results)
        status_counts = {}
        
        for result in batch_results:
            status = result.get('overall_status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("=" * 60)
        print("📊 BATCH VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total Sessions Verified: {total_sessions}")
        print()
        
        for status in ['PASS', 'WARN', 'FAIL', 'ERROR', 'NOT_FOUND']:
            count = status_counts.get(status, 0)
            if count > 0:
                percentage = (count / total_sessions) * 100
                icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'ERROR': '💥', 'NOT_FOUND': '❓'}.get(status, '?')
                print(f"{icon} {status}: {count} ({percentage:.1f}%)")
        
        print()
    
    def _generate_report(self, verification_results: Dict[str, Any], args: argparse.Namespace) -> None:
        """Generate verification report in specified format."""
        report_format = args.report
        output_file = args.output
        
        if report_format == 'json':
            # Convert datetime objects to strings for JSON serialization
            json_results = self._serialize_for_json(verification_results)
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(json_results, f, indent=2)
                print(f"📄 JSON report saved to: {output_file}")
            else:
                print("📄 JSON Report:")
                print(json.dumps(json_results, indent=2))
        
        elif report_format == 'html':
            html_content = self._generate_html_report(verification_results)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(html_content)
                print(f"📄 HTML report saved to: {output_file}")
            else:
                print("📄 HTML report generated (use --output to save)")
        
        print()
    
    def _serialize_for_json(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _generate_html_report(self, verification_results: Dict[str, Any]) -> str:
        """Generate HTML verification report."""
        session_id = verification_results.get('session_id', 'Unknown')
        overall_status = verification_results.get('overall_status', 'UNKNOWN')
        verification_time = verification_results.get('verification_time', 'Unknown')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MCTS Session Verification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .status-pass {{ color: #28a745; }}
                .status-warn {{ color: #ffc107; }}
                .status-fail {{ color: #dc3545; }}
                .status-error {{ color: #6f42c1; }}
                .category {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .check {{ margin: 10px 0; padding: 5px; }}
                .error {{ background: #f8d7da; color: #721c24; padding: 10px; border-radius: 3px; }}
                .warning {{ background: #fff3cd; color: #856404; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MCTS Session Verification Report</h1>
                <p><strong>Session:</strong> {session_id}</p>
                <p><strong>Overall Status:</strong> <span class="status-{overall_status.lower()}">{overall_status}</span></p>
                <p><strong>Verification Time:</strong> {verification_time}</p>
            </div>
        """
        
        # Add category results
        for category_name, category_results in verification_results.get('categories', {}).items():
            status = category_results.get('status', 'UNKNOWN')
            html += f"""
            <div class="category">
                <h2>{category_name.title()} <span class="status-{status.lower()}">[{status}]</span></h2>
            """
            
            # Add checks
            for check_name, check_status in category_results.get('checks', {}).items():
                html += f"""
                <div class="check">
                    ✓ {check_name.replace('_', ' ').title()}: <span class="status-{check_status.lower()}">{check_status}</span>
                </div>
                """
            
            # Add errors and warnings
            for error in category_results.get('errors', []):
                html += f'<div class="error">❌ {error}</div>'
            
            for warning in category_results.get('warnings', []):
                html += f'<div class="warning">⚠️ {warning}</div>'
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html