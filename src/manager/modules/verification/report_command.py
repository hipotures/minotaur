"""
Report Command - Generate detailed verification report for specific session.

Provides comprehensive reporting capabilities including:
- Multiple output formats (text, JSON, HTML)
- Detailed verification results
- File output support
- Report customization options
"""

from typing import Dict, Any
from .verify_session_command import VerifySessionCommand


class ReportCommand(VerifySessionCommand):
    """Handle --report command for verification."""
    
    def execute(self, args) -> None:
        """Execute the verification report command."""
        try:
            session_identifier = args.report
            
            print(f"📄 GENERATING VERIFICATION REPORT")
            print("=" * 60)
            print(f"Session: {session_identifier}")
            print(f"Format: {getattr(args, 'format', 'text')}")
            if getattr(args, 'output_file', None):
                print(f"Output: {args.output_file}")
            print()
            
            # Get session info
            session_info = self.get_session_info(session_identifier)
            if not session_info:
                self.print_error(f"Session not found: {session_identifier}")
                self.print_info("💡 Use either full UUID or session name")
                self.print_info("   List sessions: python manager.py sessions --list")
                return
            
            # Run comprehensive verification (same as verify-session but focus on reporting)
            verification_results = self._run_comprehensive_verification(session_info, args)
            
            # Generate report in requested format
            self._generate_comprehensive_report(verification_results, args)
            
        except Exception as e:
            self.print_error(f"Report generation failed: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
    
    def _run_comprehensive_verification(self, session_info: Dict[str, Any], args) -> Dict[str, Any]:
        """Run comprehensive verification and collect all results."""
        from datetime import datetime
        
        verification_results = {
            'session_id': session_info['session_id'],
            'session_name': session_info['session_name'],
            'verification_time': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'session_info': session_info,
            'categories': {}
        }
        
        print("📊 Running comprehensive verification checks...")
        
        # 1. Database Integrity Verification
        print("   • Database integrity...")
        db_results = self.verify_database_integrity(session_info)
        verification_results['categories']['database'] = db_results
        
        # 2. MCTS Algorithm Correctness
        print("   • MCTS algorithm correctness...")
        mcts_results = self.verify_mcts_correctness(session_info)
        verification_results['categories']['mcts'] = mcts_results
        
        # 3. AutoGluon Integration Verification
        print("   • AutoGluon integration...")
        autogluon_results = self.verify_autogluon_integration(session_info)
        verification_results['categories']['autogluon'] = autogluon_results
        
        # 4. File System and Cache Verification
        print("   • File system and cache...")
        fs_results = self._verify_filesystem_cache(session_info)
        verification_results['categories']['filesystem'] = fs_results
        
        # 5. Configuration and Environment Validation
        print("   • Configuration and environment...")
        config_results = self._verify_configuration(session_info, args)
        verification_results['categories']['configuration'] = config_results
        
        # Calculate overall status
        verification_results['overall_status'] = self.calculate_overall_status(verification_results)
        
        print("   ✅ Verification complete")
        print()
        
        return verification_results
    
    def _generate_comprehensive_report(self, verification_results: Dict[str, Any], args) -> None:
        """Generate comprehensive report in requested format."""
        format_type = getattr(args, 'format', 'text')
        output_file = getattr(args, 'output_file', None)
        
        if format_type == 'text':
            self._generate_text_report(verification_results, output_file)
        elif format_type == 'json':
            self._generate_json_report(verification_results, output_file)
        elif format_type == 'html':
            self._generate_html_report_to_file(verification_results, output_file)
    
    def _generate_text_report(self, verification_results: Dict[str, Any], output_file: str = None) -> None:
        """Generate detailed text report."""
        from datetime import datetime
        import io
        
        # Create string buffer to capture output
        output = io.StringIO()
        
        # Report header
        session_id = verification_results.get('session_id', 'Unknown')
        session_name = verification_results.get('session_name', 'Unnamed')
        overall_status = verification_results.get('overall_status', 'UNKNOWN')
        verification_time = verification_results.get('verification_time', 'Unknown')
        
        output.write("=" * 80 + "\n")
        output.write("MCTS SESSION VERIFICATION REPORT\n")
        output.write("=" * 80 + "\n")
        output.write(f"Session ID: {session_id}\n")
        output.write(f"Session Name: {session_name}\n")
        output.write(f"Overall Status: {overall_status}\n")
        output.write(f"Verification Time: {verification_time}\n")
        output.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.write("\n")
        
        # Session information
        session_info = verification_results.get('session_info', {})
        if session_info:
            output.write("SESSION INFORMATION:\n")
            output.write("-" * 40 + "\n")
            output.write(f"Start Time: {session_info.get('start_time', 'Unknown')}\n")
            output.write(f"End Time: {session_info.get('end_time', 'Not completed')}\n")
            output.write(f"Total Iterations: {session_info.get('total_iterations', 0)}\n")
            output.write(f"Best Score: {session_info.get('best_score', 'No score')}\n")
            output.write(f"Status: {session_info.get('status', 'Unknown')}\n")
            output.write(f"Strategy: {session_info.get('strategy', 'Unknown')}\n")
            output.write(f"Test Mode: {'Yes' if session_info.get('is_test_mode') else 'No'}\n")
            output.write("\n")
        
        # Verification results by category
        categories = verification_results.get('categories', {})
        for category_name, category_results in categories.items():
            output.write(f"{category_name.upper()} VERIFICATION:\n")
            output.write("-" * 40 + "\n")
            output.write(f"Status: {category_results.get('status', 'UNKNOWN')}\n")
            output.write("\n")
            
            # Individual checks
            checks = category_results.get('checks', {})
            if checks:
                output.write("Checks:\n")
                for check_name, check_status in checks.items():
                    check_display = check_name.replace('_', ' ').title()
                    output.write(f"  - {check_display}: {check_status}\n")
                output.write("\n")
            
            # Errors
            errors = category_results.get('errors', [])
            if errors:
                output.write("Errors:\n")
                for error in errors:
                    output.write(f"  ❌ {error}\n")
                output.write("\n")
            
            # Warnings
            warnings = category_results.get('warnings', [])
            if warnings:
                output.write("Warnings:\n")
                for warning in warnings:
                    output.write(f"  ⚠️  {warning}\n")
                output.write("\n")
            
            output.write("\n")
        
        # Summary statistics
        all_checks = {}
        for category_results in categories.values():
            for check_name, check_status in category_results.get('checks', {}).items():
                all_checks[check_status] = all_checks.get(check_status, 0) + 1
        
        output.write("SUMMARY STATISTICS:\n")
        output.write("-" * 40 + "\n")
        for status in ['PASS', 'WARN', 'FAIL', 'ERROR', 'SKIP']:
            count = all_checks.get(status, 0)
            if count > 0:
                output.write(f"{status}: {count}\n")
        
        output.write("\n")
        output.write("=" * 80 + "\n")
        output.write(f"Report completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Get the report content
        report_content = output.getvalue()
        output.close()
        
        # Output to file or console
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            self.print_success(f"Text report saved to: {output_file}")
        else:
            print(report_content)
    
    def _generate_json_report(self, verification_results: Dict[str, Any], output_file: str = None) -> None:
        """Generate JSON report."""
        json_results = self._serialize_for_json(verification_results)
        
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            self.print_success(f"JSON report saved to: {output_file}")
        else:
            self.print_json(json_results, "Verification Report")
    
    def _generate_html_report_to_file(self, verification_results: Dict[str, Any], output_file: str = None) -> None:
        """Generate HTML report to file."""
        html_content = self._generate_html_report(verification_results)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)
            self.print_success(f"HTML report saved to: {output_file}")
        else:
            # Generate a default filename
            from datetime import datetime
            session_id = verification_results.get('session_id', 'unknown')[:8]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_filename = f"verification_report_{session_id}_{timestamp}.html"
            
            with open(default_filename, 'w') as f:
                f.write(html_content)
            self.print_success(f"HTML report saved to: {default_filename}")
        
        self.print_info("💡 Open the HTML file in a web browser to view the formatted report")