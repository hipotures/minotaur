#!/usr/bin/env python3
"""
MCTS Feature Discovery - Main Runner Script

Orchestrates the complete MCTS-driven automated feature engineering process.
Integrates all components: MCTS engine, AutoGluon evaluation, feature space, and database logging.
"""

import os
import sys
import yaml
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import (
    FeatureDiscoveryDB,
    MCTSEngine, 
    FeatureSpace,
    AutoGluonEvaluator,
    MockAutoGluonEvaluator,
    initialize_timing,
    get_timing_collector,
    performance_monitor,
    AnalyticsGenerator,
    AUTOGLUON_AVAILABLE
)

# Setup logging
def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config['logging']
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    from logging.handlers import RotatingFileHandler
    
    file_handler = RotatingFileHandler(
        log_config['log_file'],
        maxBytes=log_config['max_log_size_mb'] * 1024 * 1024,
        backupCount=log_config['backup_count']
    )
    
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler
        ]
    )
    
    # Reduce verbosity of some libraries
    logging.getLogger('autogluon').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration initialized")

class FeatureDiscoveryRunner:
    """Main runner for MCTS feature discovery system."""
    
    def __init__(self, config_path: str, config_overrides: Dict[str, Any] = None):
        """Initialize the runner with configuration."""
        self.config_path = config_path
        self.config = load_config_with_overrides(config_path)
        
        # Apply any config overrides
        if config_overrides:
            self._apply_config_overrides(config_overrides)
        
        # Initialize components
        self.db: Optional[FeatureDiscoveryDB] = None
        self.mcts_engine: Optional[MCTSEngine] = None
        self.feature_space: Optional[FeatureSpace] = None
        self.evaluator: Optional[AutoGluonEvaluator] = None
        self.timing_collector = None
        
        # Runtime tracking
        self.start_time = time.time()
        self.interrupted = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Initialized FeatureDiscoveryRunner with config: {config_path}")
    
    
    def _apply_config_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides recursively."""
        def merge_dict(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    merge_dict(target[key], value)
                else:
                    target[key] = value
        
        merge_dict(self.config, overrides)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        logger = logging.getLogger(__name__)
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.interrupted = True
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        logger = logging.getLogger(__name__)
        logger.info("Initializing system components...")
        
        try:
            # Initialize timing collector first
            logger.info("Initializing timing system...")
            self.timing_collector = initialize_timing(self.config)
            
            # Initialize database
            logger.info("Initializing database...")
            self.db = FeatureDiscoveryDB(self.config)
            
            # Initialize feature space
            logger.info("Initializing feature space...")
            self.feature_space = FeatureSpace(self.config)
            
            # Initialize evaluator (AutoGluon or Mock)
            use_mock = self.config.get('testing', {}).get('use_mock_evaluator', False)
            logger.info(f"DEBUG: use_mock={use_mock}, AUTOGLUON_AVAILABLE={AUTOGLUON_AVAILABLE}")
            
            if use_mock or not AUTOGLUON_AVAILABLE:
                logger.info("Initializing Mock evaluator...")
                self.evaluator = MockAutoGluonEvaluator(self.config)
                if not AUTOGLUON_AVAILABLE:
                    logger.warning("AutoGluon not available, using mock evaluator")
                else:
                    logger.info("Using mock evaluator for testing mode")
            else:
                logger.info("Initializing AutoGluon evaluator...")
                self.evaluator = AutoGluonEvaluator(self.config)
            
            # Initialize MCTS engine
            logger.info("Initializing MCTS engine...")
            self.mcts_engine = MCTSEngine(self.config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def _get_initial_features(self) -> set:
        """Get initial feature set for MCTS tree."""
        # Basic features that should always be available
        base_features = {
            'Nitrogen', 'Phosphorous', 'Potassium', 
            'Temperature', 'Humidity', 'Moisture',
            'Soil Type', 'Crop Type'
        }
        
        # Add derived basic features
        derived_features = {
            'NP_ratio', 'NK_ratio', 'PK_ratio',
            'soil_crop', 'nutrient_balance', 'moisture_stress',
            'low_Nitrogen', 'low_Phosphorous', 'low_Potassium'
        }
        
        return base_features.union(derived_features)
    
    def run_discovery(self) -> Dict[str, Any]:
        """Run the complete feature discovery process."""
        logger = logging.getLogger(__name__)
        logger.info("Starting MCTS feature discovery process")
        
        try:
            # Initialize all components
            self._initialize_components()
            
            # Get initial features
            initial_features = self._get_initial_features()
            logger.info(f"Starting with {len(initial_features)} initial features")
            
            # Run MCTS search
            search_results = self.mcts_engine.run_search(
                evaluator=self.evaluator,
                feature_space=self.feature_space,
                db=self.db,
                initial_features=initial_features
            )
            
            # Get best discovered features
            best_path = self.mcts_engine.get_best_path()
            logger.info(f"Best discovery path: {best_path}")
            
            # Perform final evaluation if we found improvements
            final_results = None
            if search_results['best_score'] > 0:
                logger.info("Performing final thorough evaluation...")
                best_features_df = self.feature_space.generate_features_for_node(self.mcts_engine.best_node)
                final_results = self.evaluator.evaluate_final_features(best_features_df)
            
            # Get timing statistics
            timing_stats = None
            if self.timing_collector:
                timing_stats = self.timing_collector.get_stats()
                timing_export = self.timing_collector.export_timings()
                
                # Save timing data
                timing_file = f"timing_data_{self.db.session_id[:8]}.json"
                with open(timing_file, 'w') as f:
                    f.write(timing_export)
                logger.info(f"Exported timing data to: {timing_file}")
            
            # Generate comprehensive results
            results = {
                'search_results': search_results,
                'final_evaluation': final_results,
                'best_features_path': best_path,  # best_path is already a list of operation names
                'tree_statistics': self.mcts_engine.get_tree_statistics(),
                'operation_stats': self.feature_space.get_operation_stats(),
                'evaluator_stats': self.evaluator.get_evaluation_statistics(),
                'session_progress': self.db.get_session_progress(),
                'timing_statistics': timing_stats,
                'total_runtime': time.time() - self.start_time
            }
            
            # Export results
            self._export_results(results)
            
            # Generate analytics report
            self._generate_analytics_report(timing_file if timing_stats else None)
            
            logger.info(f"Feature discovery completed successfully in {results['total_runtime']:.2f}s")
            logger.info(f"Best score achieved: {search_results['best_score']:.5f}")
            logger.info(f"Total iterations: {search_results['total_iterations']}")
            logger.info(f"Total evaluations: {search_results['total_evaluations']}")
            
            return results
            
        except KeyboardInterrupt:
            logger.info("Discovery process interrupted by user")
            return self._handle_interruption()
            
        except Exception as e:
            logger.error(f"Discovery process failed: {e}")
            raise
        
        finally:
            self._cleanup()
    
    def _export_results(self, results: Dict[str, Any]) -> None:
        """Export discovery results to various formats."""
        logger = logging.getLogger(__name__)
        export_config = self.config['export']
        
        try:
            # Export best features code
            if 'python' in export_config['formats']:
                output_file = export_config['python_output']
                self.db.export_best_features_code(output_file)
                logger.info(f"Exported best features code to: {output_file}")
            
            # Export session data
            if 'json' in export_config['formats']:
                import json
                session_file = f"discovery_session_{self.db.session_id[:8]}.json"
                with open(session_file, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    json_results = self._serialize_for_json(results)
                    json.dump(json_results, f, indent=2)
                logger.info(f"Exported session data to: {session_file}")
            
            # Generate HTML report if configured
            if 'html' in export_config['formats'] and export_config.get('include_plots', False):
                self._generate_html_report(results)
                
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
    
    def _serialize_for_json(self, obj):
        """Convert objects to JSON-serializable format."""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _generate_html_report(self, results: Dict[str, Any]) -> None:
        """Generate HTML report with plots and analysis."""
        logger = logging.getLogger(__name__)
        
        try:
            # This would integrate with analytics.py for report generation
            # For now, just create a simple HTML summary
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MCTS Feature Discovery Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>MCTS Feature Discovery Report</h1>
                <div class="section">
                    <h2>Summary</h2>
                    <div class="metric">Best Score: {results['search_results']['best_score']:.5f}</div>
                    <div class="metric">Total Iterations: {results['search_results']['total_iterations']}</div>
                    <div class="metric">Total Runtime: {results['total_runtime']:.2f} seconds</div>
                    <div class="metric">Session ID: {self.db.session_id}</div>
                </div>
                
                <div class="section">
                    <h2>Best Feature Path</h2>
                    <ul>
                        {''.join([f'<li>{op}</li>' for op in results['best_features_path']])}
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Tree Statistics</h2>
                    <div class="metric">Total Nodes: {results['tree_statistics'].get('total_nodes', 0)}</div>
                    <div class="metric">Max Depth: {results['tree_statistics'].get('max_depth', 0)}</div>
                    <div class="metric">Average Depth: {results['tree_statistics'].get('avg_depth', 0):.2f}</div>
                </div>
            </body>
            </html>
            """
            
            report_file = self.config['export']['html_report']
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
    
    def _generate_analytics_report(self, timing_file: str = None) -> None:
        """Generate comprehensive analytics report."""
        logger = logging.getLogger(__name__)
        
        try:
            export_config = self.config.get('export', {})
            if not export_config.get('include_analytics', True):
                return
            
            logger.info("Generating comprehensive analytics report...")
            
            # Initialize analytics generator
            analytics = AnalyticsGenerator(self.config)
            
            # Generate report
            db_path = self.config['database']['path']
            session_id = self.db.session_id if self.db else None
            
            report_files = analytics.generate_comprehensive_report(
                db_path=db_path,
                timing_data=timing_file,
                session_id=session_id
            )
            
            # Log generated files
            logger.info("Analytics report generated:")
            for report_type, file_path in report_files.items():
                logger.info(f"  {report_type}: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
    
    def _handle_interruption(self) -> Dict[str, Any]:
        """Handle graceful interruption of the discovery process."""
        logger = logging.getLogger(__name__)
        logger.info("Handling process interruption...")
        
        # Get current state
        results = {
            'interrupted': True,
            'partial_results': True,
            'runtime_until_interruption': time.time() - self.start_time
        }
        
        # Try to get partial results
        try:
            if self.mcts_engine and self.mcts_engine.best_node:
                results['best_score_so_far'] = self.mcts_engine.best_score
                results['iterations_completed'] = self.mcts_engine.current_iteration
                
                # Export partial results
                best_features = self.db.get_best_features(10)
                if best_features:
                    self.db.export_best_features_code('partial_best_features.py', 10)
                    logger.info("Exported partial results")
            
        except Exception as e:
            logger.error(f"Failed to save partial results: {e}")
        
        return results
    
    def _cleanup(self) -> None:
        """Cleanup all resources."""
        logger = logging.getLogger(__name__)
        logger.info("Cleaning up resources...")
        
        try:
            # Close database session
            if self.db:
                self.db.close_session('completed' if not self.interrupted else 'interrupted')
            
            # Cleanup evaluator resources
            if self.evaluator:
                self.evaluator.cleanup()
            
            # Cleanup feature space cache
            if self.feature_space:
                self.feature_space.cleanup()
                
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

def list_sessions(config_path: str, limit: int = 10) -> None:
    """List recent sessions from database."""
    try:
        # Load config with overrides to get database path
        config = load_config_with_overrides(config_path)
        
        db_path = config['database']['path']
        
        if not os.path.exists(db_path):
            print("❌ No database found. No sessions exist yet.")
            print(f"Expected database at: {db_path}")
            return
        
        # Connect to database and ensure migrations
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if strategy column exists and add if missing
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'strategy' not in columns:
            print("🔧 Migrating database: adding strategy column...")
            cursor.execute("ALTER TABLE sessions ADD COLUMN strategy TEXT DEFAULT 'default'")
            conn.commit()
            print("✅ Database migration completed")
        
        # Query recent sessions
        query = '''
        SELECT 
            session_id,
            session_name,
            start_time,
            end_time,
            status,
            total_iterations,
            best_score,
            strategy
        FROM sessions 
        ORDER BY start_time DESC 
        LIMIT ?
        '''
        
        cursor.execute(query, (limit,))
        sessions = cursor.fetchall()
        conn.close()
        
        if not sessions:
            print("❌ No sessions found in database.")
            return
        
        print(f"\n📋 Recent Sessions (showing {len(sessions)} of max {limit}):")
        print("=" * 120)
        print(f"{'ID (first 8 chars)':<20} {'Name':<25} {'Start Time':<20} {'Status':<12} {'Iterations':<10} {'Best Score':<12} {'Strategy':<15}")
        print("-" * 120)
        
        for session in sessions:
            session_id, name, start_time, end_time, status, iterations, best_score, strategy = session
            
            # Format values
            short_id = session_id[:8] + "..." if len(session_id) > 8 else session_id
            name_display = (name[:22] + "...") if name and len(name) > 25 else (name or "unnamed")
            start_display = start_time[:19] if start_time else "unknown"
            status_display = status or "unknown"
            iterations_display = str(iterations) if iterations else "0"
            score_display = f"{best_score:.5f}" if best_score else "0.00000"
            strategy_display = (strategy[:12] + "...") if strategy and len(strategy) > 15 else (strategy or "default")
            
            print(f"{short_id:<20} {name_display:<25} {start_display:<20} {status_display:<12} {iterations_display:<10} {score_display:<12} {strategy_display:<15}")
        
        print("-" * 120)
        print(f"\n💡 Usage:")
        print(f"   Continue last session:     python run_feature_discovery.py --resume")
        print(f"   Continue specific session: python run_feature_discovery.py --resume {sessions[0][0][:8]}")
        print(f"   Start new session:         python run_feature_discovery.py --new-session")
        print(f"   List more sessions:        python run_feature_discovery.py --list-sessions --session-limit 20")
        
    except Exception as e:
        print(f"❌ Error listing sessions: {e}")

def load_config_with_overrides(config_path: str) -> Dict[str, Any]:
    """
    Load main configuration and merge with overrides if it's not the base config.
    
    Args:
        config_path: Path to config file (may be override or base config)
        
    Returns:
        Merged configuration dictionary
    """
    base_config_path = 'config/mcts_config.yaml'
    
    # Load base configuration first
    try:
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded base configuration: {base_config_path}")
    except FileNotFoundError:
        print(f"❌ Base configuration not found: {base_config_path}")
        sys.exit(1)
    
    # If it's not the base config, load and merge overrides
    if config_path != base_config_path:
        try:
            with open(config_path, 'r') as f:
                overrides = yaml.safe_load(f)
            
            # Deep merge function
            def deep_merge(base, override):
                if isinstance(override, dict) and isinstance(base, dict):
                    for key, value in override.items():
                        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                            deep_merge(base[key], value)
                        else:
                            base[key] = value
                return base
            
            config = deep_merge(config, overrides)
            print(f"✓ Applied configuration overrides: {config_path}")
            
        except FileNotFoundError:
            print(f"❌ Override configuration not found: {config_path}")
            sys.exit(1)
    
    return config

def get_session_to_continue(config_path: str, session_id: str = None) -> str:
    """Get session ID to continue, either specified or most recent."""
    try:
        # Load config with overrides to get database path
        config = load_config_with_overrides(config_path)
        
        db_path = config['database']['path']
        
        if not os.path.exists(db_path):
            print("❌ No database found. Cannot continue any session.")
            return None
        
        # Connect to database
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if session_id:
            # Find session by ID (allow partial ID matching)
            query = '''
            SELECT session_id, session_name, start_time, status, total_iterations, best_score
            FROM sessions 
            WHERE session_id LIKE ? 
            ORDER BY start_time DESC 
            LIMIT 1
            '''
            cursor.execute(query, (f"{session_id}%",))
            session = cursor.fetchone()
            
            if not session:
                print(f"❌ Session not found with ID starting with: {session_id}")
                print("💡 Use --list-sessions to see available sessions")
                conn.close()
                return None
                
        else:
            # Get most recent session
            query = '''
            SELECT session_id, session_name, start_time, status, total_iterations, best_score
            FROM sessions 
            ORDER BY start_time DESC 
            LIMIT 1
            '''
            cursor.execute(query)
            session = cursor.fetchone()
            
            if not session:
                print("❌ No sessions found to continue.")
                conn.close()
                return None
        
        conn.close()
        
        # Display session info
        full_session_id, name, start_time, status, iterations, best_score = session
        short_id = full_session_id[:8]
        
        print(f"\n🔄 Continuing Session:")
        print(f"   ID: {short_id}... ({full_session_id})")
        print(f"   Name: {name or 'unnamed'}")
        print(f"   Started: {start_time[:19] if start_time else 'unknown'}")
        print(f"   Status: {status or 'unknown'}")
        print(f"   Previous iterations: {iterations or 0}")
        best_score_display = f"{best_score:.5f}" if best_score is not None else "0.00000"
        print(f"   Best score so far: {best_score_display}")
        print()
        
        return full_session_id
        
    except Exception as e:
        print(f"❌ Error getting session to continue: {e}")
        return None

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MCTS Feature Discovery System')
    parser.add_argument(
        '--config', 
        default='config/mcts_config.yaml',
        help='Configuration file path (default: mcts_config.yaml)'
    )
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration and exit'
    )
    parser.add_argument(
        '--resume',
        type=str,
        nargs='?',
        const='',
        metavar='SESSION_ID',
        help='Resume session: --resume (last session) or --resume SESSION_ID (specific session)'
    )
    parser.add_argument(
        '--new-session',
        action='store_true',
        help='Start new session (overrides config session mode)'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Use mock evaluator for fast testing'
    )
    parser.add_argument(
        '--real-autogluon',
        action='store_true',
        help='Use real AutoGluon with small dataset for testing'
    )
    parser.add_argument(
        '--list-sessions',
        action='store_true',
        help='List recent sessions and exit'
    )
    parser.add_argument(
        '--session-limit',
        type=int,
        default=10,
        help='Number of sessions to list (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Handle list sessions command
    if args.list_sessions:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}")
            sys.exit(1)
        list_sessions(args.config, args.session_limit)
        sys.exit(0)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load and validate configuration with overrides
    try:
        config = load_config_with_overrides(args.config)
        
        if args.validate_config:
            print("Configuration validation passed ✓")
            sys.exit(0)
        
        # Handle session management
        session_id_to_continue = None
        if args.resume is not None:
            # --resume was used (either with or without SESSION_ID)
            if args.resume:
                # Specific session ID provided: --resume SESSION_ID
                session_id_to_continue = get_session_to_continue(args.config, args.resume)
                if not session_id_to_continue:
                    sys.exit(1)
            else:
                # No session ID provided: --resume (continue last session)
                session_id_to_continue = get_session_to_continue(args.config)
                if not session_id_to_continue:
                    print("🆕 No sessions found to resume. Starting new session instead.")
                    config['session']['mode'] = 'new'
            
            if session_id_to_continue:
                config['session']['mode'] = 'continue'
                config['session']['resume_session_id'] = session_id_to_continue
        elif args.new_session:
            config['session']['mode'] = 'new'
        
        # Set test mode if specified
        if args.test_mode:
            config.setdefault('testing', {})['use_mock_evaluator'] = True
            config['session']['max_iterations'] = min(config['session']['max_iterations'], 5)  # Very short for testing
            print("TEST MODE: Using mock evaluator and limited iterations")
        
        # Add real autogluon test flag
        if args.real_autogluon:
            config.setdefault('testing', {})['use_mock_evaluator'] = False
            config['session']['max_iterations'] = min(config['session']['max_iterations'], 3)  # Very few iterations for real testing
            config['testing']['use_small_dataset'] = True
            print("REAL AUTOGLUON MODE: Using small dataset for real evaluation")
            
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Print startup banner
    logger.info("="*80)
    logger.info("MCTS-Driven Automated Feature Engineering System")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Session mode: {config['session']['mode']}")
    if session_id_to_continue:
        logger.info(f"Continuing session: {session_id_to_continue[:8]}...")
    logger.info(f"Max iterations: {config['session']['max_iterations']}")
    logger.info(f"Max runtime: {config['session']['max_runtime_hours']} hours")
    logger.info("="*80)
    
    # Prepare config overrides
    config_overrides = {}
    if args.test_mode:
        config_overrides['testing'] = {
            'use_mock_evaluator': True,
            'fast_test_mode': True,
            'use_small_dataset': True
        }
    
    # Run discovery process
    try:
        runner = FeatureDiscoveryRunner(args.config, config_overrides)
        results = runner.run_discovery()
        
        # Print final summary
        logger.info("="*80)
        logger.info("DISCOVERY COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        if 'search_results' in results:
            search_results = results['search_results']
            logger.info(f"Best Score: {search_results['best_score']:.5f}")
            logger.info(f"Total Iterations: {search_results['total_iterations']}")
            logger.info(f"Total Evaluations: {search_results['total_evaluations']}")
            logger.info(f"Runtime: {results['total_runtime']:.2f} seconds")
            
            if 'best_features_path' in results:
                logger.info(f"Best Feature Path: {' -> '.join(results['best_features_path'])}")
        
        logger.info("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Discovery process failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())