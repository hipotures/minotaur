"""
Base Verification Command - Common functionality for verification commands.

Provides shared utilities including:
- Session information retrieval
- Verification engine core functionality
- Results formatting and display
- Report generation utilities
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path
from manager.core.command_base import BaseCommand


class BaseVerificationCommand(BaseCommand, ABC):
    """Base class for all verification commands."""
    
    def __init__(self):
        super().__init__()
        self.session_service = None
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """Inject required services."""
        super().inject_services(services)
        self.session_service = services.get('session_service')
        if not self.session_service:
            raise ValueError("SessionService is required for verification commands")
    
    @abstractmethod
    def execute(self, args) -> None:
        """Execute the command with given arguments."""
        pass
    
    def get_session_info(self, session_identifier: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session information."""
        try:
            # Get basic session data
            query = """
                SELECT 
                    session_id, session_name, start_time, end_time, 
                    total_iterations, best_score, config_snapshot, 
                    status, strategy, is_test_mode, notes, dataset_hash
                FROM sessions 
                WHERE session_id = ? OR session_name = ?
            """
            result = self.session_service.repository.fetch_one(query, [session_identifier, session_identifier])
            
            if not result:
                return None
            
            # Parse session data
            session_info = {
                'session_id': result['session_id'],
                'session_name': result['session_name'],
                'start_time': result['start_time'],
                'end_time': result['end_time'],
                'total_iterations': result['total_iterations'],
                'best_score': result['best_score'],
                'config_snapshot': result['config_snapshot'],
                'status': result['status'],
                'strategy': result['strategy'],
                'is_test_mode': result['is_test_mode'],
                'notes': result['notes'],
                'dataset_hash': result['dataset_hash']
            }
            
            # Parse config if available
            if session_info['config_snapshot']:
                try:
                    session_info['config'] = json.loads(session_info['config_snapshot'])
                except json.JSONDecodeError:
                    session_info['config'] = None
            
            return session_info
            
        except Exception as e:
            self.print_error(f"Failed to get session info: {e}")
            return None
    
    def verify_database_integrity(self, session_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verify database integrity for the session."""
        results = {
            'status': 'UNKNOWN',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
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
            exploration_query = "SELECT COUNT(*) as count FROM exploration_history WHERE session_id = ?"
            exploration_result = self.session_service.repository.fetch_one(exploration_query, [session_id])
            exploration_count = exploration_result.get('count', 0) if exploration_result else 0
            
            expected_explorations = session_info.get('total_iterations', 0)
            if exploration_count == 0 and expected_explorations > 0:
                results['errors'].append("No exploration history found despite completed iterations")
                results['checks']['exploration_history'] = 'FAIL'
            elif exploration_count < expected_explorations:
                results['warnings'].append(f"Fewer explorations ({exploration_count}) than iterations ({expected_explorations})")
                results['checks']['exploration_history'] = 'WARN'
            else:
                results['checks']['exploration_history'] = 'PASS'
            
            # Check 3: Score consistency
            max_score_query = "SELECT MAX(evaluation_score) as max_score FROM exploration_history WHERE session_id = ?"
            max_score_result = self.session_service.repository.fetch_one(max_score_query, [session_id])
            max_exploration_score = max_score_result.get('max_score') if max_score_result else None
            
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
            feature_impact_query = "SELECT COUNT(*) as count FROM feature_impact WHERE session_id = ?"
            feature_impact_result = self.session_service.repository.fetch_one(feature_impact_query, [session_id])
            feature_impact_count = feature_impact_result.get('count', 0) if feature_impact_result else 0
            
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
    
    def verify_mcts_correctness(self, session_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verify MCTS algorithm correctness."""
        results = {
            'status': 'UNKNOWN',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            session_id = session_info['session_id']
            
            # Get exploration data
            exploration_query = """
                SELECT operation_applied, evaluation_score, iteration 
                FROM exploration_history 
                WHERE session_id = ? 
                ORDER BY timestamp
            """
            explorations = self.session_service.repository.fetch_all(exploration_query, [session_id])
            
            if not explorations:
                results['warnings'].append("No exploration data found")
                results['checks']['operation_sequence'] = 'WARN'
            else:
                # Check for valid operations
                valid_operations = set()
                for exploration in explorations:
                    op = exploration.get('operation_applied')
                    if op:
                        valid_operations.add(op)
                
                if len(valid_operations) == 0:
                    results['errors'].append("No valid operations found")
                    results['checks']['operation_sequence'] = 'FAIL'
                else:
                    results['checks']['operation_sequence'] = 'PASS'
            
            # Check score progression
            if explorations:
                scores = [exp.get('evaluation_score') for exp in explorations if exp.get('evaluation_score') is not None]
                if scores:
                    best_score_progression = []
                    current_best = 0
                    for score in scores:
                        current_best = max(current_best, score)
                        best_score_progression.append(current_best)
                    
                    # Check monotonic improvement
                    is_monotonic = all(best_score_progression[i] >= best_score_progression[i-1] 
                                     for i in range(1, len(best_score_progression)))
                    
                    if not is_monotonic:
                        results['errors'].append("Best score progression is not monotonic")
                        results['checks']['score_progression'] = 'FAIL'
                    else:
                        results['checks']['score_progression'] = 'PASS'
                    
                    # Check improvement
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
            
            # Check iteration consistency
            expected_iterations = session_info.get('total_iterations', 0)
            if explorations:
                valid_iterations = [exp.get('iteration') for exp in explorations if exp.get('iteration') is not None]
                if valid_iterations:
                    max_iteration = max(valid_iterations)
                    if max_iteration > expected_iterations + 2:  # Allow tolerance
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
    
    def verify_autogluon_integration(self, session_info: Dict[str, Any]) -> Dict[str, Any]:
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
            
            # Check configuration validation
            required_config = ['target_metric', 'included_model_types']
            missing_config = [key for key in required_config if key not in autogluon_config]
            
            if missing_config:
                results['errors'].append(f"Missing AutoGluon config: {missing_config}")
                results['checks']['config_validation'] = 'FAIL'
            else:
                results['checks']['config_validation'] = 'PASS'
            
            # Check target metric consistency
            target_metric = autogluon_config.get('target_metric')
            session_best_score = session_info.get('best_score')
            
            if target_metric and session_best_score is not None:
                if target_metric.lower() in ['accuracy', 'map@3'] and (session_best_score < 0 or session_best_score > 1):
                    results['errors'].append(f"Invalid {target_metric} score: {session_best_score}")
                    results['checks']['metric_consistency'] = 'FAIL'
                else:
                    results['checks']['metric_consistency'] = 'PASS'
            else:
                results['checks']['metric_consistency'] = 'SKIP'
            
            # Check model types
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
            
            # Check training data
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
            
            # Check performance
            session_id = session_info['session_id']
            avg_time_query = """
                SELECT AVG(evaluation_time) as avg_time 
                FROM exploration_history 
                WHERE session_id = ? AND evaluation_time IS NOT NULL
            """
            avg_time_result = self.session_service.repository.fetch_one(avg_time_query, [session_id])
            avg_eval_time = avg_time_result.get('avg_time') if avg_time_result else None
            
            if avg_eval_time:
                if avg_eval_time > 300:  # More than 5 minutes
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
    
    def calculate_overall_status(self, verification_results: Dict[str, Any]) -> str:
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
    
    def print_category_results(self, category_name: str, results: Dict[str, Any]) -> None:
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
    
    def should_display_result(self, result_status: str, args) -> bool:
        """Check if result should be displayed based on filters."""
        if getattr(args, 'failed_only', False):
            return result_status in ['FAIL', 'ERROR']
        elif getattr(args, 'warn_only', False):
            return result_status == 'WARN'
        elif getattr(args, 'pass_only', False):
            return result_status == 'PASS'
        else:
            return True
    
    def get_filter_description(self, args) -> str:
        """Get description of active filters."""
        if getattr(args, 'failed_only', False):
            return "failed only"
        elif getattr(args, 'warn_only', False):
            return "warnings only"
        elif getattr(args, 'pass_only', False):
            return "passed only"
        else:
            return "all"