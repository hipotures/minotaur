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
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
from src.project_root import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    FeatureDiscoveryDB,
    MCTSEngine, 
    FeatureSpace,
    AutoGluonEvaluator,
    DatasetManager,
    initialize_timing,
    get_timing_collector,
    performance_monitor,
    AnalyticsGenerator,
    AUTOGLUON_AVAILABLE
)
from src.utils import (
    create_session_resolver, SessionResolutionError,
    add_legacy_resume_argument, add_universal_optional_session_args, 
    resolve_session_from_args
)
from src.logging_utils import setup_session_logging, set_session_context, clear_session_context

# Setup logging
def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config['logging']
    
    log_format = '%(asctime)s - [%(session_name)s] - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    from logging.handlers import RotatingFileHandler
    
    file_handler = RotatingFileHandler(
        log_config['log_file'],
        maxBytes=log_config['max_log_size_mb'] * 1024 * 1024,
        backupCount=log_config['backup_count']
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_format,
        handlers=[file_handler]
    )
    
    # Setup basic MCTS logger (session-specific handler will be added later)
    mcts_logger = logging.getLogger('mcts')
    mcts_logger.setLevel(logging.DEBUG)  # Always DEBUG for MCTS logger
    mcts_logger.propagate = True  # Always propagate to main log file
    
    # Note: Session-specific MCTS log file will be configured after session creation
    
    # Reduce verbosity of some libraries
    logging.getLogger('autogluon').setLevel(logging.INFO)  # Show AutoGluon logs
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Setup session-aware logging filters
    setup_session_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration initialized")


def setup_session_mcts_logging(session_name: str, config: Dict[str, Any]) -> None:
    """Setup session-specific MCTS logging to logs/mcts/session_name.log."""
    log_config = config['logging']
    main_log_level = log_config['level'].upper()
    
    # Only create session-specific MCTS log file when DEBUG level is enabled
    if main_log_level == 'DEBUG':
        from pathlib import Path
        from logging.handlers import RotatingFileHandler
        from src.logging_utils import SessionFilter
        
        # Ensure logs/mcts directory exists
        mcts_log_dir = Path('logs/mcts')
        mcts_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific log file
        session_log_file = mcts_log_dir / f"{session_name}.log"
        
        mcts_logger = logging.getLogger('mcts')
        
        # Check if handler already exists to avoid duplicates
        existing_handlers = [h for h in mcts_logger.handlers if hasattr(h, 'baseFilename') and session_log_file.as_posix() in h.baseFilename]
        if not existing_handlers:
            mcts_handler = RotatingFileHandler(
                session_log_file,
                maxBytes=log_config.get('mcts_log_size_mb', 50) * 1024 * 1024,
                backupCount=log_config.get('mcts_backup_count', 3)
            )
            mcts_handler.setFormatter(logging.Formatter(
                '%(asctime)s - [%(session_name)s] - MCTS - %(levelname)s - %(message)s'
            ))
            mcts_handler.addFilter(SessionFilter())
            mcts_logger.addHandler(mcts_handler)
            
            logger = logging.getLogger(__name__)
            logger.info(f"MCTS session logging enabled: {session_log_file} (DEBUG mode)")
    else:
        logger = logging.getLogger(__name__)
        logger.info(f"MCTS logging to main log only (level: {main_log_level})")

class FeatureDiscoveryRunner:
    """Main runner for MCTS feature discovery system."""
    
    def __init__(self, config_path: str, config_overrides: Dict[str, Any] = None, config: Dict[str, Any] = None):
        """Initialize the runner with configuration."""
        self.config_path = config_path
        if config is not None:
            self.config = config
        else:
            self.config = load_config_with_overrides(config_path)
        
        # Apply any config overrides
        if config_overrides:
            self._apply_config_overrides(config_overrides)
        
        # Initialize components
        self.db: Optional[FeatureDiscoveryDB] = None
        self.mcts_engine: Optional[MCTSEngine] = None
        self.feature_space: Optional[FeatureSpace] = None
        self.evaluator: Optional[AutoGluonEvaluator] = None
        self.dataset_manager: Optional[DatasetManager] = None
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
            
            # Initialize database first
            logger.info("Initializing database...")
            self.db = FeatureDiscoveryDB(self.config)
            
            # Initialize dataset manager with database connection
            logger.info("Initializing dataset manager...")
            self.dataset_manager = DatasetManager(self.config, self.db.db_service)
            
            # Validate dataset registration
            self._validate_dataset_configuration()
            
            # Set session context for logging after DB initialization
            if self.db.session_name:
                set_session_context(self.db.session_name)
                logger.info(f"Session context set to: {self.db.session_name}")
                
                # Setup session-specific MCTS logging
                setup_session_mcts_logging(self.db.session_name, self.config)
            
            # Initialize AutoGluon evaluator first (required for DuckDB manager)
            if not AUTOGLUON_AVAILABLE:
                logger.error("❌ CRITICAL: AutoGluon is not installed!")
                logger.error("📦 Please install AutoGluon: pip install autogluon")
                logger.error("🚫 Cannot proceed with real ML evaluation")
                raise ImportError("AutoGluon is required for feature evaluation")
            
            # Log AutoGluon version info
            try:
                import autogluon
                autogluon_version = autogluon.__version__
                logger.info(f"✅ AutoGluon v{autogluon_version} detected")
            except Exception:
                logger.info("✅ AutoGluon available (version unknown)")
                
            logger.info("Initializing AutoGluon evaluator...")
            self.evaluator = AutoGluonEvaluator(self.config, db_service=self.db.db_service)
            
            # Initialize feature space with DuckDB manager from db
            logger.info("Initializing feature space...")
            self.feature_space = FeatureSpace(self.config, duckdb_manager=self.db.db_service.db_manager)
            
            # Initialize MCTS engine
            logger.info("Initializing MCTS engine...")
            self.mcts_engine = MCTSEngine(self.config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def _validate_dataset_configuration(self) -> None:
        """Validate dataset configuration and registration."""
        logger = logging.getLogger(__name__)
        
        try:
            # Get dataset info from config
            dataset_info = self.dataset_manager.get_dataset_from_config()
            dataset_name = dataset_info['dataset_name']
            
            logger.info(f"🔍 Validating dataset: {dataset_name}")
            
            # Check if using registered dataset
            if dataset_info['is_registered']:
                # Validate registered dataset
                if not self.dataset_manager.validate_dataset_registration(dataset_name):
                    logger.error(f"❌ Dataset '{dataset_name}' validation failed")
                    logger.error("💡 Try re-registering: scripts/duckdb_manager.py datasets --register --help")
                    raise ValueError(f"Dataset '{dataset_name}' is not properly registered")
                
                # Update usage timestamp
                self.dataset_manager.update_dataset_usage(dataset_name)
                
                logger.info(f"✅ Using registered dataset: {dataset_name}")
                logger.info(f"   📊 Target column: {dataset_info['target_column']}")
                if dataset_info.get('train_records'):
                    logger.info(f"   📈 Train records: {dataset_info['train_records']:,}")
                if dataset_info.get('test_records'):
                    logger.info(f"   🧪 Test records: {dataset_info['test_records']:,}")
            else:
                # Legacy path-based system
                logger.warning("⚠️ Using legacy path-based dataset access")
                logger.warning("💡 Consider registering this dataset for better management:")
                logger.warning("   scripts/duckdb_manager.py datasets --register --dataset-name YOUR_DATASET --auto --dataset-path /path/to/data")
                
                # Basic validation for legacy system
                train_path = dataset_info.get('train_path')
                if not train_path or not Path(train_path).exists():
                    raise FileNotFoundError(f"Train file not found: {train_path}")
                
                logger.info(f"📊 Train file: {train_path}")
                if dataset_info.get('test_path'):
                    logger.info(f"🧪 Test file: {dataset_info['test_path']}")
            
        except Exception as e:
            # Print to stdout for immediate user feedback
            print(f"ℹ️ Dataset not available: {e}")
            
            # Show available datasets if validation fails
            try:
                available_datasets = self.dataset_manager.list_available_datasets()
                if available_datasets:
                    print("📋 Available registered datasets:")
                    for name, info in available_datasets.items():
                        status = "Active" if info['is_active'] else "Inactive"
                        print(f"   • {name} ({status}) - Target: {info['target_column']}")
                else:
                    print("📋 No datasets registered yet")
                    print("💡 Register a dataset: scripts/duckdb_manager.py datasets --register --help")
            except:
                pass
            
            # Exit gracefully instead of raising
            import sys
            print("🛑 Stopping due to dataset configuration issue")
            sys.exit(0)
    
    def _get_initial_features(self) -> set:
        """Get initial feature set for MCTS tree based on configured dataset."""
        dataset_name = self.config['autogluon'].get('dataset_name')
        
        if not dataset_name:
            raise ValueError("dataset_name not specified in autogluon config")
        
        # Load dataset configuration from database
        results = self.db.db_service.connection_manager.execute_query(
            "SELECT train_path, target_column, id_column FROM datasets WHERE name = :dataset_name",
            params={'dataset_name': dataset_name}
        )
        
        if not results:
            raise ValueError(f"Dataset '{dataset_name}' not found in database registry")
        
        dataset_info = results[0]
        train_path, target_column, id_column = dataset_info['train_path'], dataset_info['target_column'], dataset_info['id_column']
        
        # Get active train features from feature_catalog
        # This respects the feature configuration (is_active=True)
        try:
            # Use DuckDB connection to get active train features
            cache_dir = Path(self.config.get('project_root', '.')) / 'cache' / dataset_name
            cache_db_path = cache_dir / 'dataset.duckdb'
            
            # Always use CSV approach to avoid connection conflicts
            # The dataset-specific DuckDB file may already be open by AutoGluonEvaluator
            df = pd.read_csv(train_path, nrows=1)  # Read only header
            all_columns = set(col.lower() for col in df.columns)  # Convert to lowercase
            
            # Remove target and ID columns from features
            features = all_columns.copy()
            if target_column and target_column.lower() in features:
                features.remove(target_column.lower())
            if id_column and id_column.lower() in features:
                features.remove(id_column.lower())
            
            logger = logging.getLogger(__name__)
            logger.info(f"Loaded {len(features)} initial features from {dataset_name}: {sorted(features)}")
            
            return features
            
        except Exception as e:
            raise ValueError(f"Failed to load features from {train_path}: {e}")
    
    def run_discovery(self) -> Dict[str, Any]:
        """Run the complete feature discovery process."""
        logger = logging.getLogger(__name__)
        logger.info("Starting MCTS feature discovery process")
        
        try:
            # Initialize all components
            self._initialize_components()
            
            # Pre-build features at startup on 100% data
            self._prebuild_features()
            
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
            
            # Skip final evaluation - results from MCTS iterations are sufficient
            final_results = None
            
            # Get timing statistics
            timing_stats = None
            if self.timing_collector:
                timing_stats = self.timing_collector.get_stats()
                timing_export = self.timing_collector.export_timings()
                
                # Save timing data
                if hasattr(self.db, 'output_manager') and self.db.output_manager:
                    # Use session-based log directory
                    log_paths = self.db.output_manager.get_log_paths()
                    timing_file = log_paths['timing_data']
                else:
                    # Fallback to config-based directory
                    timing_dir = self.config.get('logging', {}).get('timing_output_dir', 'logs/timing')
                    os.makedirs(timing_dir, exist_ok=True)
                    timing_file = os.path.join(timing_dir, f"timing_data_{self.db.session_id[:8]}.json")
                
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
            
            # Create session summary
            if hasattr(self.db, 'output_manager') and self.db.output_manager:
                summary_file = self.db.output_manager.create_session_summary(results)
                logger.info(f"📄 Session summary created: {summary_file}")
            
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
                try:
                    # Try session-based export first
                    if hasattr(self.db, 'output_manager') and self.db.output_manager:
                        output_file = self.db.export_best_features_to_session()
                        logger.info(f"Exported best features code to session: {output_file}")
                    else:
                        # Fallback to config-based export
                        output_file = export_config['python_output']
                        self.db.export_best_features_code(output_file)
                        logger.info(f"Exported best features code to: {output_file}")
                except Exception as e:
                    logger.error(f"Failed to export best features code: {e}", exc_info=True)
            
            # Export session data
            if 'json' in export_config['formats']:
                try:
                    import json
                    # Try session-based export first
                    if hasattr(self.db, 'output_manager') and self.db.output_manager:
                        export_paths = self.db.output_manager.get_export_paths()
                        session_file = export_paths['discovery_results']
                    else:
                        # Fallback to current directory
                        session_file = f"discovery_session_{self.db.session_id[:8]}.json"
                    
                    with open(session_file, 'w') as f:
                        # Convert numpy types to native Python types for JSON serialization
                        json_results = self._serialize_for_json(results)
                        json.dump(json_results, f, indent=2)
                    logger.info(f"Exported session data to: {session_file}")
                except Exception as e:
                    logger.error(f"Failed to export session data: {e}", exc_info=True)
            
            # Generate HTML report if configured
            if 'html' in export_config['formats'] and export_config.get('include_plots', False):
                try:
                    self._generate_html_report(results)
                except Exception as e:
                    logger.error(f"Failed to generate HTML report: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Failed to export results: {e}", exc_info=True)
    
    def _serialize_for_json(self, obj):
        """Convert objects to JSON-serializable format."""
        import numpy as np
        import pandas as pd
        from dataclasses import is_dataclass, asdict
        
        if isinstance(obj, dict):
            # Handle dicts with non-string keys (like tuple keys in _ucb1_cache)
            result = {}
            for key, value in obj.items():
                # Convert non-string keys to strings
                if isinstance(key, tuple):
                    str_key = str(key)  # e.g., (1.4, 100) -> "(1.4, 100)"
                elif not isinstance(key, (str, int, float, bool, type(None))):
                    str_key = str(key)
                else:
                    str_key = key
                result[str_key] = self._serialize_for_json(value)
            return result
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'deque':
            # Handle collections.deque
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, set):
            try:
                return [self._serialize_for_json(item) for item in sorted(obj)]
            except TypeError:
                # If items can't be sorted (e.g., mixed types), just convert to list
                return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif is_dataclass(obj):
            # Handle dataclass objects like FeatureNode
            # Special handling for FeatureNode to avoid circular references
            if hasattr(obj, 'parent') and hasattr(obj, 'children'):
                # This is likely a FeatureNode - serialize without parent/children to avoid circular refs
                result = {}
                for field_name in obj.__dataclass_fields__:
                    if field_name not in ['parent', 'children']:
                        value = getattr(obj, field_name)
                        result[field_name] = self._serialize_for_json(value)
                # Add children count and parent info without circular reference
                result['children_count'] = len(obj.children) if obj.children else 0
                result['has_parent'] = obj.parent is not None
                return result
            else:
                return self._serialize_for_json(asdict(obj))
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
            
            # Initialize analytics generator with session output manager
            output_manager = self.db.output_manager if hasattr(self.db, 'output_manager') else None
            analytics = AnalyticsGenerator(self.config, output_manager=output_manager)
            
            # Generate report using actual database path (may be different for test mode)
            db_path = self.db.db_path if self.db else self.config['database']['path']
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
                    session_id_short = self.db.session_id[:8]
                    output_filename = f'partial_best_features_{session_id_short}.py'
                    self.db.export_best_features_code(output_filename, 10)
                    logger.info(f"Exported partial results to {output_filename}")
            
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
            
            # Clear session context from logging
            clear_session_context()
                
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def _prebuild_features(self) -> None:
        """Pre-build all possible features at startup on 100% dataset."""
        logger = logging.getLogger(__name__)
        logger.info("🏗️ Pre-building features on 100% dataset...")
        
        try:
            # Check if using DuckDB backend - skip feature cache pre-building
            data_config = self.config.get('data', {})
            backend = data_config.get('backend', 'duckdb').lower()
            
            if backend == 'duckdb':
                logger.info("✅ Using DuckDB backend - skipping feature cache pre-building")
                logger.info("📊 Features will be generated on-demand and cached in DuckDB")
                return
            
            # Fallback to feature cache manager for pandas backend
            from src.feature_cache import FeatureCacheManager
            from src.data_utils import prepare_training_data
            
            # Get dataset info via dataset manager
            dataset_info = self.dataset_manager.get_dataset_from_config()
            dataset_name = dataset_info['dataset_name']
            train_path = dataset_info['train_path']
            test_path = dataset_info.get('test_path')
            
            if not train_path:
                logger.warning("⚠️ No train path available - skipping feature pre-building")
                return
                
            # Initialize cache manager
            cache_manager = FeatureCacheManager(train_path)
            
            # Ensure base datasets are cached
            if test_path:
                cache_manager.ensure_base_datasets(train_path, test_path)
            
            # Load data via dataset manager (100% for pre-building)
            logger.info(f"📂 Loading full dataset: {dataset_name}")
            train_full_df, test_full_df = self.dataset_manager.load_dataset_files(dataset_name)
            
            logger.info(f"📊 Loaded full datasets: train={len(train_full_df)}, test={len(test_full_df)}")
            
            # Get all possible feature operations from feature space
            all_operations = list(self.feature_space.operations.keys())
            
            # Build feature definitions for cache manager
            feature_definitions = []
            
            # Skip pre-building for now - operations structure has changed
            # This would need to be reimplemented with the new modular feature system
            feature_definitions = []
            
            # Batch build all missing features
            cache_manager.batch_build_features(feature_definitions)
            
            # Log cache statistics
            cache_stats = cache_manager.get_cache_stats()
            logger.info(f"✅ Feature pre-building complete: {cache_stats['total_features']} features, {cache_stats['cache_size_mb']:.1f}MB")
            
        except Exception as e:
            logger.error(f"❌ Feature pre-building failed: {e}")
            logger.info("🔄 Continuing without pre-built features...")

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
        
        # Connect to database
        import duckdb
        conn = duckdb.connect(db_path)
        cursor = conn.cursor()
        
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
        print(f"   Continue last session:     python mcts.py --resume")
        print(f"   Continue specific session: python mcts.py --resume {sessions[0][0][:8]}")
        print(f"   Start new session:         python mcts.py --new-session")
        print(f"   List more sessions:        python mcts.py --list-sessions --session-limit 20")
        
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
    # Use PROJECT_ROOT to construct the base config path
    base_config_path = PROJECT_ROOT / 'config' / 'mcts_config.yaml'
    
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
        import duckdb
        conn = duckdb.connect(db_path)
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
        # Handle datetime object properly
        if start_time:
            start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(start_time, 'strftime') else str(start_time)[:19]
        else:
            start_time_str = 'unknown'
        print(f"   Started: {start_time_str}")
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
    # Add universal optional session argument (defaults to latest)
    add_universal_optional_session_args(parser, "MCTS")
    
    # Add legacy resume argument for backward compatibility
    add_legacy_resume_argument(parser)
    
    parser.add_argument(
        '--new-session',
        action='store_true',
        help='Start new session (overrides config session mode)'
    )
    parser.add_argument(
        '--force-resume',
        action='store_true',
        help='Force resume session despite configuration warnings'
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
    
    # If no arguments provided, show help and ask about continuing last session
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "="*60)
        
        # Check if there are any sessions to continue
        try:
            # Load config to check for sessions
            default_config = load_config_with_overrides("config/mcts_config.yaml")
            resolver = create_session_resolver(default_config)
            
            # Try to get the latest session
            try:
                latest_session = resolver.resolve_session("latest")
                print(f"\n📊 Ostatnia sesja:")
                print(f"   ID: {latest_session.session_id[:8]}...")
                print(f"   Nazwa: {latest_session.session_name}")
                print(f"   Status: {latest_session.status}")
                print(f"   Iteracje: {latest_session.total_iterations}")
                print(f"   Najlepszy wynik: {latest_session.best_score:.5f}" if latest_session.best_score else "   Najlepszy wynik: brak")
                
                # Ask user if they want to continue
                response = input("\n🔄 Czy chcesz kontynuować tę sesję? [t/N]: ").strip().lower()
                
                if response in ['t', 'tak', 'y', 'yes']:
                    # Set args to continue the session
                    sys.argv.extend(['--resume', 'latest'])
                    args = parser.parse_args()
                else:
                    print("\n💡 Aby rozpocząć nową sesję, użyj: ./mcts.py --config config/mcts_config.yaml --new")
                    print("💡 Aby kontynuować konkretną sesję: ./mcts.py --resume SESSION_ID")
                    sys.exit(0)
                    
            except (SessionResolutionError, Exception):
                print("\n📊 Brak poprzednich sesji.")
                print("\n💡 Aby rozpocząć nową sesję, użyj: ./mcts.py --config config/mcts_config.yaml")
                sys.exit(0)
                
        except Exception as e:
            # If anything fails, just show the help
            print(f"\n⚠️ Nie można sprawdzić sesji: {e}")
            print("\n💡 Aby rozpocząć nową sesję, użyj: ./mcts.py --config config/mcts_config.yaml")
            sys.exit(0)
    
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
            # Validate configuration using Pydantic schema
            from src.utils.config_validator import validate_configuration
            validation_result = validate_configuration(config)
            
            if validation_result.is_valid:
                print("✅ Configuration validation passed")
                if validation_result.has_warnings:
                    print("\n⚠️  Warnings:")
                    for warning in validation_result.warnings:
                        print(f"  • {warning}")
            else:
                print("❌ Configuration validation failed:")
                for error in validation_result.errors:
                    print(f"  • {error}")
                sys.exit(1)
            
            sys.exit(0)
        
        # Handle session management with universal resolver
        session_id_to_continue = None
        session_identifier = resolve_session_from_args(args)
        
        if session_identifier and not args.new_session:
            # Session resumption requested
            try:
                resolver = create_session_resolver(config)
                session_info = resolver.resolve_session(session_identifier)
                session_id_to_continue = session_info.session_id
                
                print(f"\n🔄 Resuming Session:")
                print(f"   ID: {session_info.session_id[:8]}... ({session_info.session_id})")
                print(f"   Name: {session_info.session_name}")
                print(f"   Started: {session_info.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Status: {session_info.status}")
                print(f"   Previous iterations: {session_info.total_iterations}")
                if session_info.best_score is not None:
                    print(f"   Best score so far: {session_info.best_score:.5f}")
                
                # Show planned iterations for this run
                max_iterations = config.get('session', {}).get('max_iterations', 100)
                print(f"\n📈 This run will execute: {max_iterations} iterations")
                print(f"   Total after completion: {session_info.total_iterations + max_iterations} iterations")
                print()
                
                config['session']['mode'] = 'continue'
                config['session']['resume_session_id'] = session_id_to_continue
                config['session']['force_resume'] = args.force_resume
                
                # Close resolver connection to avoid conflicts with main MCTS database connection
                resolver.connection_manager.close()
                
            except SessionResolutionError as e:
                print(f"❌ Session resolution failed: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Unexpected error resolving session: {e}")
                sys.exit(1)
                
        elif args.new_session:
            config['session']['mode'] = 'new'
        
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
    
    # Log test mode status
    test_mode = config.get('test_mode', False)
    if test_mode:
        logger.info("🧪 TEST MODE: Session will be marked as test (for cleanup)")
    else:
        logger.info("🚀 PRODUCTION MODE: Session will be marked as production")
    
    logger.info(f"Max iterations: {config['session']['max_iterations']}")
    logger.info(f"Max runtime: {config['session']['max_runtime_hours']} hours")
    logger.info("="*80)
    
    # Prepare config overrides
    config_overrides = {}
    
    # Run discovery process
    try:
        runner = FeatureDiscoveryRunner(args.config, config_overrides, config)
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
        
        # Output results as JSON for subprocess capture
        if runner.db and runner.db.session_id:
            import json
            output_data = {
                "session_id": runner.db.session_id,
                "session_name": getattr(runner.db, 'session_name', None),
                "iterations": 0,
                "score": None
            }
            
            # Get actual values from results
            if 'search_results' in results:
                search_results = results['search_results']
                output_data['iterations'] = search_results.get('total_iterations', 0)
                output_data['score'] = search_results.get('best_score', None)
            
            # Output JSON on a single line with clear marker
            print(f"MCTS_RESULT_JSON:{json.dumps(output_data)}")
            
            # Also log as info for debugging
            logger.info(f"Final session data: {json.dumps(output_data, indent=2)}")
        
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