"""
Verify Session Command - Verify specific session integrity and correctness.

Provides comprehensive session verification including:
- Database integrity verification
- MCTS algorithm correctness validation
- AutoGluon integration verification
- File system and cache validation
- Configuration and environment validation
"""

from typing import Dict, Any
from datetime import datetime
from .base import BaseVerificationCommand


class VerifySessionCommand(BaseVerificationCommand):
    """Handle --verify-session command for verification."""
    
    def execute(self, args) -> None:
        """Execute the session verification command."""
        try:
            session_identifier = args.verify_session
            
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
            
            # Get session info
            session_info = self.get_session_info(session_identifier)
            if not session_info:
                self.print_error(f"Session not found: {session_identifier}")
                self.print_info("💡 Use either full UUID or session name")
                self.print_info("   List sessions: python manager.py sessions --list")
                return
            
            verification_results['session_info'] = session_info
            
            # Run verification categories
            print("📊 Running verification checks...")
            print()
            
            # 1. Database Integrity Verification
            db_results = self.verify_database_integrity(session_info)
            verification_results['categories']['database'] = db_results
            self.print_category_results("Database Integrity", db_results)
            
            # 2. MCTS Algorithm Correctness
            mcts_results = self.verify_mcts_correctness(session_info)
            verification_results['categories']['mcts'] = mcts_results
            self.print_category_results("MCTS Algorithm", mcts_results)
            
            # 3. AutoGluon Integration Verification
            autogluon_results = self.verify_autogluon_integration(session_info)
            verification_results['categories']['autogluon'] = autogluon_results
            self.print_category_results("AutoGluon Integration", autogluon_results)
            
            # 4. File System and Cache Verification (simplified)
            fs_results = self._verify_filesystem_cache(session_info)
            verification_results['categories']['filesystem'] = fs_results
            self.print_category_results("File System & Cache", fs_results)
            
            # 5. Configuration and Environment Validation
            config_results = self._verify_configuration(session_info, args)
            verification_results['categories']['configuration'] = config_results
            self.print_category_results("Configuration", config_results)
            
            # Calculate overall status
            verification_results['overall_status'] = self.calculate_overall_status(verification_results)
            
            # Print summary
            self._print_verification_summary(verification_results)
            
            # Generate report if requested
            if getattr(args, 'output_file', None) or getattr(args, 'format', 'text') != 'text':
                self._generate_report(verification_results, args)
                
        except Exception as e:
            self.print_error(f"Verification failed: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
    
    def _verify_filesystem_cache(self, session_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verify file system and cache integrity (simplified version)."""
        results = {
            'status': 'UNKNOWN',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            from pathlib import Path
            
            # Check 1: Database file integrity
            try:
                db_config = self.db_pool.config if self.db_pool else None
                if db_config:
                    db_path = Path(db_config.database_path)
                    if not db_path.exists():
                        results['errors'].append(f"Main database not found: {db_path}")
                        results['checks']['database_files'] = 'FAIL'
                    else:
                        db_size = db_path.stat().st_size
                        if db_size == 0:
                            results['errors'].append("Database file is empty")
                            results['checks']['database_files'] = 'FAIL'
                        else:
                            results['checks']['database_files'] = 'PASS'
                else:
                    results['checks']['database_files'] = 'SKIP'
            except Exception as e:
                results['warnings'].append(f"Could not verify database files: {e}")
                results['checks']['database_files'] = 'WARN'
            
            # Check 2: Log file existence
            log_files = list(Path('.').glob('**/mcts_discovery.log'))
            if not log_files:
                results['warnings'].append("MCTS discovery log file not found")
                results['checks']['log_files'] = 'WARN'
            else:
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
            
            # Check 3: Cache directory structure
            cache_dirs = list(Path('.').glob('cache/*'))
            if not cache_dirs:
                results['warnings'].append("No cache directories found")
                results['checks']['cache_structure'] = 'WARN'
            else:
                results['checks']['cache_structure'] = 'PASS'
            
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
    
    def _verify_configuration(self, session_info: Dict[str, Any], args) -> Dict[str, Any]:
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
            
            # Check 3: Test mode consistency
            test_mode = session_info.get('is_test_mode', False)
            testing_config = config.get('testing', {})
            fast_test_mode = testing_config.get('fast_test_mode', False)
            
            if test_mode != fast_test_mode:
                results['warnings'].append(f"Test mode mismatch: session={test_mode}, config={fast_test_mode}")
                results['checks']['test_mode_consistency'] = 'WARN'
            else:
                results['checks']['test_mode_consistency'] = 'PASS'
            
            # Check 4: Resource constraints (simplified)
            resources_config = config.get('resources', {})
            max_memory = resources_config.get('max_memory_gb')
            
            if max_memory:
                try:
                    import psutil
                    import os
                    
                    process = psutil.Process(os.getpid())
                    current_memory_gb = process.memory_info().rss / (1024**3)
                    
                    # Be more lenient for test sessions
                    is_test = session_info.get('is_test_mode', False)
                    memory_multiplier = 10 if is_test else 2
                    
                    if current_memory_gb > max_memory * memory_multiplier:
                        results['warnings'].append(f"Process memory ({current_memory_gb:.1f}GB) may exceed limits ({max_memory}GB)")
                        results['checks']['resource_constraints'] = 'WARN'
                    else:
                        results['checks']['resource_constraints'] = 'PASS'
                        
                except ImportError:
                    results['warnings'].append("psutil not available for memory check")
                    results['checks']['resource_constraints'] = 'SKIP'
            else:
                results['checks']['resource_constraints'] = 'SKIP'
            
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
    
    def _generate_report(self, verification_results: Dict[str, Any], args) -> None:
        """Generate verification report in specified format."""
        format_type = getattr(args, 'format', 'text')
        output_file = getattr(args, 'output_file', None)
        
        if format_type == 'json':
            # Convert datetime objects to strings for JSON serialization
            json_results = self._serialize_for_json(verification_results)
            
            if output_file:
                import json
                with open(output_file, 'w') as f:
                    json.dump(json_results, f, indent=2)
                self.print_success(f"JSON report saved to: {output_file}")
            else:
                self.print_json(json_results, "Verification Report")
        
        elif format_type == 'html':
            html_content = self._generate_html_report(verification_results)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(html_content)
                self.print_success(f"HTML report saved to: {output_file}")
            else:
                self.print_info("HTML report generated (use --output-file to save)")
    
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