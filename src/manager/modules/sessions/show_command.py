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
        print("\033[96m🔍 SESSION DETAILS\033[0m")
        print("\033[96m" + "=" * 60 + "\033[0m")
        
        # Basic information
        print("\033[94m📋 BASIC INFORMATION:\033[0m")
        print(f"   \033[93mSession ID:\033[0m {session_info['session_id']}")
        print(f"   \033[93mName:\033[0m {session_info['session_name'] or 'Unnamed'}")
        
        # Status with color coding
        status = session_info['status']
        status_color = "\033[92m" if status == "completed" else "\033[91m" if status == "failed" else "\033[93m"
        print(f"   \033[93mStatus:\033[0m {status_color}{status}\033[0m")
        
        print(f"   \033[93mStrategy:\033[0m {session_info['strategy']}")
        
        # Test mode with color
        test_mode = 'Yes' if session_info['is_test_mode'] else 'No'
        test_color = "\033[93m" if session_info['is_test_mode'] else "\033[92m"
        print(f"   \033[93mTest Mode:\033[0m {test_color}{test_mode}\033[0m")
        
        print(f"   \033[93mStarted:\033[0m {session_info['start_time']}")
        print(f"   \033[93mEnded:\033[0m {session_info['end_time'] or 'Still running'}")
        
        duration_display = "Running"
        if session_info['end_time']:
            duration_display = self.format_duration(session_info['start_time'], session_info['end_time'])
        
        print(f"   \033[93mDuration:\033[0m {duration_display}")
        print(f"   \033[93mTotal Iterations:\033[0m \033[96m{session_info['total_iterations']}\033[0m")
        
        # Format score with metric
        metric = self.extract_target_metric(session_info['config_snapshot'])
        score_display = self.format_score(session_info['best_score'], metric)
        print(f"   \033[93mBest Score:\033[0m \033[92m{score_display}\033[0m")
        
        # Dataset information
        dataset_hash = session_info.get('dataset_hash')
        if dataset_hash:
            print(f"   \033[93mDataset Hash:\033[0m {dataset_hash[:8]}...")
        else:
            print(f"   \033[93mDataset Hash:\033[0m \033[91mNot set\033[0m")
        
        # Notes if available
        if session_info.get('notes'):
            print(f"   \033[93mNotes:\033[0m {session_info['notes']}")
        
        print()
        
        # Exploration statistics
        if exploration_stats:
            print("\033[95m📊 EXPLORATION STATISTICS:\033[0m")
            total_exp = exploration_stats['total_explorations']
            min_score = exploration_stats['min_score']
            max_score = exploration_stats['max_score']
            avg_score = exploration_stats['avg_score']
            total_time = exploration_stats['total_eval_time']
            avg_time = exploration_stats['avg_eval_time']
            unique_ops = exploration_stats['unique_operations']
            
            print(f"   \033[93mTotal Explorations:\033[0m \033[96m{total_exp}\033[0m")
            print(f"   \033[93mScore Range:\033[0m \033[92m{min_score:.5f}\033[0m - \033[92m{max_score:.5f}\033[0m")
            print(f"   \033[93mAverage Score:\033[0m \033[92m{avg_score:.5f}\033[0m")
            print(f"   \033[93mTotal Evaluation Time:\033[0m \033[96m{total_time:.1f}s\033[0m (\033[96m{total_time/60:.1f}m\033[0m)")
            print(f"   \033[93mAverage Evaluation Time:\033[0m \033[96m{avg_time:.1f}s\033[0m")
            print(f"   \033[93mUnique Operations:\033[0m \033[96m{unique_ops}\033[0m")
            print()
        
        # Top operations
        if top_operations:
            print("\033[93m🏆 TOP OPERATIONS:\033[0m")
            for op_data in top_operations:
                operation = op_data['operation_applied']
                count = op_data['usage_count']
                avg_score = op_data['avg_score']
                best_score = op_data['best_score']
                print(f"   \033[97m•\033[0m \033[96m{operation}\033[0m: \033[94m{count}\033[0m uses, avg: \033[92m{avg_score:.5f}\033[0m, best: \033[92m{best_score:.5f}\033[0m")
            print()
        
        # Configuration details
        config_json = session_info.get('config_snapshot')
        if config_json:
            self._show_configuration_details(config_json)
        
        # Quick actions
        print("\033[92m💡 Quick Actions:\033[0m")
        session_id_short = session_info['session_id'][:8]
        print(f"   \033[97mCompare with others:\033[0m \033[90mpython manager.py sessions --compare {session_id_short} <OTHER_ID>\033[0m")
        print(f"   \033[97mList all sessions:\033[0m \033[90mpython manager.py sessions --list\033[0m")
        if exploration_stats:
            print(f"   \033[97mExport session data:\033[0m \033[90mpython manager.py sessions --export json\033[0m")
    
    def _show_configuration_details(self, config_json: str) -> None:
        """Show configuration details from session."""
        try:
            import json
            config_data = json.loads(config_json) if isinstance(config_json, str) else config_json
            
            print("\033[94m⚙️  CONFIGURATION:\033[0m")
            
            # AutoGluon configuration
            autogluon_config = config_data.get('autogluon', {})
            if autogluon_config:
                train_path = autogluon_config.get('train_path', 'N/A')
                target_metric = autogluon_config.get('target_metric', 'N/A')
                model_types = autogluon_config.get('included_model_types', 'N/A')
                time_limit = autogluon_config.get('time_limit', 'N/A')
                
                print(f"   \033[93mTrain Path:\033[0m {train_path}")
                print(f"   \033[93mTarget Metric:\033[0m \033[96m{target_metric}\033[0m")
                print(f"   \033[93mModel Types:\033[0m \033[96m{model_types}\033[0m")
                print(f"   \033[93mTime Limit:\033[0m \033[96m{time_limit}\033[0m")
            
            # Session configuration
            session_config = config_data.get('session', {})
            if session_config:
                max_iterations = session_config.get('max_iterations', 'N/A')
                exploration_strategy = session_config.get('exploration_strategy', 'N/A')
                print(f"   \033[93mMax Iterations:\033[0m \033[96m{max_iterations}\033[0m")
                print(f"   \033[93mExploration Strategy:\033[0m \033[96m{exploration_strategy}\033[0m")
            
            # MCTS configuration
            mcts_config = config_data.get('mcts', {})
            if mcts_config:
                c_param = mcts_config.get('c_param', 'N/A')
                max_depth = mcts_config.get('max_depth', 'N/A')
                print(f"   \033[93mMCTS C Parameter:\033[0m \033[96m{c_param}\033[0m")
                print(f"   \033[93mMCTS Max Depth:\033[0m \033[96m{max_depth}\033[0m")
            
            print()
            
        except (json.JSONDecodeError, AttributeError, TypeError):
            print("\033[94m⚙️  Configuration data available but not readable as JSON\033[0m")
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