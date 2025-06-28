"""
Show Command - Show detailed information about a specific session.

Provides comprehensive session analysis including:
- Basic session information and metadata
- Exploration statistics and performance metrics
- Top performing operations and their usage
- Configuration details and parameters
- Dataset information and context
"""

from typing import Dict, Any
from .base import BaseSessionsCommand


class ShowCommand(BaseSessionsCommand):
    """Handle --show command for sessions."""
    
    def execute(self, args) -> None:
        """Execute the session show command."""
        try:
            session_identifier = args.show
            
            # Get session information
            session_info = self.get_session_by_identifier(session_identifier)
            
            if not session_info:
                self.print_error(f"Session not found: {session_identifier}")
                self.print_info("💡 Use either full UUID (e.g., 8208bed6-...) or session name")
                self.print_info("   List sessions: python manager.py sessions --list")
                return
            
            # Get exploration statistics
            exploration_stats = self.get_exploration_statistics(session_info['session_id'])
            
            # Get top operations
            top_operations = self.get_top_operations(session_info['session_id'])
            
            # Output results
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(session_info, exploration_stats, top_operations)
            else:
                self._output_session_details(session_info, exploration_stats, top_operations)
                
        except Exception as e:
            self.print_error(f"Failed to show session details: {e}")
    
    def _output_session_details(self, session_info: Dict[str, Any], 
                               exploration_stats: Dict[str, Any] = None,
                               top_operations: list = None) -> None:
        """Output detailed session information in formatted view."""
        print("🔍 SESSION DETAILS")
        print("=" * 60)
        
        # Basic information
        print("📋 BASIC INFORMATION:")
        print(f"   Session ID: {session_info['session_id']}")
        print(f"   Name: {session_info['session_name'] or 'Unnamed'}")
        print(f"   Status: {session_info['status']}")
        print(f"   Strategy: {session_info['strategy']}")
        print(f"   Test Mode: {'Yes' if session_info['is_test_mode'] else 'No'}")
        print(f"   Started: {session_info['start_time']}")
        print(f"   Ended: {session_info['end_time'] or 'Still running'}")
        
        duration_display = "Running"
        if session_info['end_time']:
            duration_display = self.format_duration(session_info['start_time'], session_info['end_time'])
        
        print(f"   Duration: {duration_display}")
        print(f"   Total Iterations: {session_info['total_iterations']}")
        
        # Format score with metric
        metric = self.extract_target_metric(session_info['config_snapshot'])
        score_display = self.format_score(session_info['best_score'], metric)
        print(f"   Best Score: {score_display}")
        
        # Dataset information
        dataset_hash = session_info.get('dataset_hash')
        if dataset_hash:
            print(f"   Dataset Hash: {dataset_hash[:8]}...")
        else:
            print(f"   Dataset Hash: Not set")
        
        # Notes if available
        if session_info.get('notes'):
            print(f"   Notes: {session_info['notes']}")
        
        print()
        
        # Exploration statistics
        if exploration_stats:
            print("📊 EXPLORATION STATISTICS:")
            total_exp = exploration_stats['total_explorations']
            min_score = exploration_stats['min_score']
            max_score = exploration_stats['max_score']
            avg_score = exploration_stats['avg_score']
            total_time = exploration_stats['total_eval_time']
            avg_time = exploration_stats['avg_eval_time']
            unique_ops = exploration_stats['unique_operations']
            
            print(f"   Total Explorations: {total_exp}")
            print(f"   Score Range: {min_score:.5f} - {max_score:.5f}")
            print(f"   Average Score: {avg_score:.5f}")
            print(f"   Total Evaluation Time: {total_time:.1f}s ({total_time/60:.1f}m)")
            print(f"   Average Evaluation Time: {avg_time:.1f}s")
            print(f"   Unique Operations: {unique_ops}")
            print()
        
        # Top operations
        if top_operations:
            print("🏆 TOP OPERATIONS:")
            for op_data in top_operations:
                operation = op_data['operation_applied']
                count = op_data['usage_count']
                avg_score = op_data['avg_score']
                best_score = op_data['best_score']
                print(f"   • {operation}: {count} uses, avg: {avg_score:.5f}, best: {best_score:.5f}")
            print()
        
        # Configuration details
        config_json = session_info.get('config_snapshot')
        if config_json:
            self._show_configuration_details(config_json)
        
        # Quick actions
        print("💡 Quick Actions:")
        session_id_short = session_info['session_id'][:8]
        print(f"   Compare with others: python manager.py sessions --compare {session_id_short} <OTHER_ID>")
        print(f"   List all sessions: python manager.py sessions --list")
        if exploration_stats:
            print(f"   Export session data: python manager.py sessions --export json")
    
    def _show_configuration_details(self, config_json: str) -> None:
        """Show configuration details from session."""
        try:
            import json
            config_data = json.loads(config_json) if isinstance(config_json, str) else config_json
            
            print("⚙️  CONFIGURATION:")
            
            # AutoGluon configuration
            autogluon_config = config_data.get('autogluon', {})
            if autogluon_config:
                train_path = autogluon_config.get('train_path', 'N/A')
                target_metric = autogluon_config.get('target_metric', 'N/A')
                model_types = autogluon_config.get('included_model_types', 'N/A')
                time_limit = autogluon_config.get('time_limit', 'N/A')
                
                print(f"   Train Path: {train_path}")
                print(f"   Target Metric: {target_metric}")
                print(f"   Model Types: {model_types}")
                print(f"   Time Limit: {time_limit}")
            
            # Session configuration
            session_config = config_data.get('session', {})
            if session_config:
                max_iterations = session_config.get('max_iterations', 'N/A')
                exploration_strategy = session_config.get('exploration_strategy', 'N/A')
                print(f"   Max Iterations: {max_iterations}")
                print(f"   Exploration Strategy: {exploration_strategy}")
            
            # MCTS configuration
            mcts_config = config_data.get('mcts', {})
            if mcts_config:
                c_param = mcts_config.get('c_param', 'N/A')
                max_depth = mcts_config.get('max_depth', 'N/A')
                print(f"   MCTS C Parameter: {c_param}")
                print(f"   MCTS Max Depth: {max_depth}")
            
            print()
            
        except (json.JSONDecodeError, AttributeError, TypeError):
            print("⚙️  Configuration data available but not readable as JSON")
            print()
    
    def _output_json(self, session_info: Dict[str, Any], 
                     exploration_stats: Dict[str, Any] = None,
                     top_operations: list = None) -> None:
        """Output session details in JSON format."""
        output_data = {
            'session_info': session_info,
            'exploration_stats': exploration_stats,
            'top_operations': top_operations
        }
        self.print_json(output_data, "Session Details")