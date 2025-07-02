"""
Database Service Manager for Minotaur MCTS System.

This module provides a high-level service layer that orchestrates
all database operations using the new repository-based architecture.
It serves as a bridge between the existing codebase and the new DB layer.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from .db import (
    DuckDBConnectionManager, MigrationRunner,
    SessionRepository, ExplorationRepository, FeatureRepository,
    FeatureImpactRepository, OperationPerformanceRepository, DatasetRepository,
    Session, SessionCreate, SessionUpdate, SessionStatus,
    ExplorationStep, ExplorationCreate,
    Feature, FeatureCreate, FeatureImpact,
    Dataset, DatasetCreate
)
from .session_output_manager import SessionOutputManager
from .utils.config_validator import (
    ConfigValidator, CompatibilityLevel, ValidationResult,
    validate_configuration, check_config_compatibility, calculate_configuration_hash
)


class DatabaseService:
    """
    High-level database service that orchestrates all database operations.
    
    This service provides a simplified interface for the MCTS system while
    using the new repository-based architecture underneath.
    """
    
    def __init__(self, config: Dict[str, Any], read_only: bool = False):
        """
        Initialize database service with configuration.
        
        Args:
            config: Main MCTS configuration dictionary
            read_only: If True, don't create new sessions (for read-only access)
        """
        self.config = config
        self.read_only = read_only
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection manager
        self.logger.info(f"🔧 Creating DuckDBConnectionManager with read_only={read_only}")
        self.connection_manager = DuckDBConnectionManager(config, read_only=read_only)
        
        # Initialize repositories
        self.session_repo = SessionRepository(self.connection_manager)
        self.exploration_repo = ExplorationRepository(self.connection_manager)
        self.feature_repo = FeatureRepository(self.connection_manager)
        self.feature_impact_repo = FeatureImpactRepository(self.connection_manager)
        self.operation_perf_repo = OperationPerformanceRepository(self.connection_manager)
        self.dataset_repo = DatasetRepository(self.connection_manager)
        
        # Configuration validator
        self.config_validator = ConfigValidator()
        
        # Run migrations to ensure schema is up to date (only for read-write connections)
        if not read_only:
            self._run_migrations()
        
        # Initialize current session
        self.current_session_id = None
        self.session_name = None
        self.output_manager: Optional[SessionOutputManager] = None
        
        if read_only:
            self.logger.debug("Database service initialized in read-only mode")
        else:
            self.logger.info("Database service initialized successfully")
    
    def _run_migrations(self) -> None:
        """Run database migrations to ensure schema is up to date."""
        try:
            self.logger.debug("🔄 Starting migration check...")
            migration_runner = MigrationRunner(self.connection_manager)
            status = migration_runner.get_migration_status()
            
            self.logger.debug(f"Migration status: {status}")
            
            if not status['is_up_to_date']:
                self.logger.info(f"Running {status['pending_migrations']} pending migrations...")
                self.logger.debug(f"Applied migrations: {status.get('applied_migrations', [])}")
                self.logger.debug(f"Pending migrations: {status.get('pending_migration_files', [])}")
                
                applied = migration_runner.run_migrations()
                self.logger.info(f"Applied {len(applied)} migrations successfully")
                self.logger.debug(f"Successfully applied migrations: {applied}")
            else:
                self.logger.debug("Database schema is up to date")
                
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            self.logger.debug(f"Migration error details: {e}", exc_info=True)
            raise
    
    def initialize_session(self, session_mode: str = 'new', 
                         resume_session_id: Optional[str] = None,
                         force_resume: bool = False) -> str:
        """
        Initialize a new session or resume existing one.
        
        Args:
            session_mode: Session mode ('new', 'continue', 'resume_best')
            resume_session_id: Specific session ID to resume
            force_resume: Force resume despite configuration warnings
            
        Returns:
            Session ID
        """
        if session_mode == 'new':
            return self._create_new_session()
        elif session_mode in ['continue', 'resume_best']:
            return self._resume_session(resume_session_id, force_resume)
        else:
            raise ValueError(f"Unknown session mode: {session_mode}")
    
    def _create_new_session(self) -> str:
        """Create a new session."""
        import uuid
        
        session_id = str(uuid.uuid4())
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Detect test mode
        is_test_mode = self.config.get('test_mode', False)
        self.logger.info(f"📊 Creating session with test_mode={is_test_mode}")
        
        # Validate configuration values
        config_dict = self.config.get_config() if hasattr(self.config, 'get_config') else self.config
        validation_result = self.config_validator.validate_values(config_dict)
        
        if not validation_result.is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(validation_result.errors)
            self.logger.error(error_msg)
            raise ValueError(f"Invalid configuration: {validation_result.errors[0]}")
        
        if validation_result.has_warnings:
            for warning in validation_result.warnings:
                self.logger.warning(f"Configuration warning: {warning}")
        
        # Calculate configuration hash and dataset hash
        config_hash = self.config_validator.calculate_config_hash(config_dict)
        dataset_hash = self._calculate_dataset_hash()
        
        # Prepare validation errors for storage (warnings only, since errors would block creation)
        validation_errors = None
        if validation_result.has_warnings:
            validation_errors = {
                'warnings': validation_result.warnings,
                'validation_timestamp': datetime.now().isoformat()
            }
        
        # Create session
        session_create = SessionCreate(
            session_id=session_id,
            session_name=session_name,
            config_snapshot=config_dict,
            config_hash=config_hash,
            is_test_mode=is_test_mode,
            dataset_hash=dataset_hash
        )
        
        session = self.session_repo.create_session(session_create)
        
        # Update session with validation information
        if validation_errors:
            self.session_repo.update_session_config_validation(
                session_id, config_hash, validation_errors
            )
        
        # Initialize output manager
        self.output_manager = SessionOutputManager(
            session_name=session_name,
            session_id=session_id,
            base_output_dir="outputs"
        )
        
        # Save session metadata
        environment_info = {
            'database_type': 'duckdb',
            'database_path': str(self.connection_manager.db_path),
            'is_test_mode': is_test_mode,
            'dataset_hash': dataset_hash
        }
        self.output_manager.save_metadata(self.config, environment_info)
        
        self.current_session_id = session_id
        self.session_name = session_name
        
        self.logger.info(f"Created new session: {session_name} (ID: {session_id[:8]}...)")
        return session_id
    
    def _resume_session(self, resume_session_id: Optional[str] = None, 
                      force_resume: bool = False) -> str:
        """Resume an existing session with configuration compatibility checking."""
        if resume_session_id:
            # Resume specific session
            session = self.session_repo.find_by_id(resume_session_id, 'session_id')
            if not session:
                raise ValueError(f"Session {resume_session_id} not found. Cannot resume non-existent session. "
                               f"Use --list-sessions to see available sessions.")
        else:
            # Resume most recent session
            recent_sessions = self.session_repo.get_recent_sessions(days=7, limit=1)
            if not recent_sessions:
                raise ValueError("No recent sessions found to resume. "
                               "Use --new-session to start a new session.")
            session = recent_sessions[0]
        
        # Get current configuration
        current_config = self.config.get_config() if hasattr(self.config, 'get_config') else self.config
        
        # Validate current configuration values first
        validation_result = self.config_validator.validate_values(current_config)
        if not validation_result.is_valid:
            error_msg = "Current configuration validation failed:\n" + "\n".join(validation_result.errors)
            self.logger.error(error_msg)
            raise ValueError(f"Invalid current configuration: {validation_result.errors[0]}")
        
        # Check configuration compatibility if session has stored config
        if session.config_snapshot:
            compatibility_result = self.config_validator.check_compatibility(
                session.config_snapshot, current_config
            )
            
            # Log compatibility check results
            for message in compatibility_result.messages:
                if compatibility_result.level == CompatibilityLevel.INCOMPATIBLE:
                    self.logger.error(message)
                elif compatibility_result.level == CompatibilityLevel.WARNING:
                    self.logger.warning(message)
                else:
                    self.logger.info(message)
            
            # Handle incompatible configurations
            if compatibility_result.level == CompatibilityLevel.INCOMPATIBLE:
                error_msg = f"Cannot resume session {session.session_id[:8]}... due to incompatible configuration changes:\n"
                error_msg += "\n".join(f"  • {change}" for change in compatibility_result.critical_changes)
                error_msg += "\n\n💡 Solution: Start a new session with --new-session"
                raise ValueError(error_msg)
            
            # Handle warning-level changes
            elif compatibility_result.level == CompatibilityLevel.WARNING and not force_resume:
                warning_msg = f"Session {session.session_id[:8]}... has configuration changes that may affect results:\n"
                warning_msg += "\n".join(f"  • {change}" for change in compatibility_result.warning_changes)
                warning_msg += "\n\n💡 Use --force-resume to continue anyway"
                raise ValueError(warning_msg)
            
            # Log successful compatibility check
            if compatibility_result.is_compatible:
                self.logger.info(f"✅ Configuration is compatible with session {session.session_id[:8]}...")
            elif force_resume:
                self.logger.warning(f"⚠️  Forcing resume of session {session.session_id[:8]}... despite configuration warnings")
        else:
            # Legacy session without stored config - log warning
            self.logger.warning(f"⚠️  Session {session.session_id[:8]}... has no stored configuration - cannot validate compatibility")
        
        # Load resume parameters from database view
        self._load_resume_parameters(session.session_id)
        
        self.current_session_id = session.session_id
        self.session_name = session.session_name
        
        self.logger.info(f"Resumed session: {session.session_name} (ID: {session.session_id[:8]}...)")
        return session.session_id
    
    def _load_resume_parameters(self, session_id: str) -> None:
        """Load MCTS resume parameters from session_resume_params view."""
        query = """
        SELECT next_iteration, total_evaluations, best_observed_score, 
               root_evaluation_score, has_exploration_history,
               total_evaluation_time, unique_nodes_count, last_iteration,
               session_total_iterations
        FROM session_resume_params 
        WHERE session_id = ?
        """
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute(query, [session_id]).fetchone()
            
            if result:
                self.resume_next_iteration = result[0]
                self.resume_total_evaluations = result[1] 
                self.resume_best_score = result[2]
                self.resume_root_score = result[3]
                self.resume_has_history = result[4]
                self.resume_total_eval_time = result[5]
                self.resume_unique_nodes = result[6]
                self.resume_last_iteration = result[7]
                self.resume_session_iterations = result[8]
                
                self.logger.info(f"Loaded resume parameters: next_iteration={self.resume_next_iteration}, "
                               f"total_evaluations={self.resume_total_evaluations}, "
                               f"best_score={self.resume_best_score:.5f}, "
                               f"root_score={self.resume_root_score:.5f}, "
                               f"eval_time={self.resume_total_eval_time:.1f}s, "
                               f"unique_nodes={self.resume_unique_nodes}")
            else:
                # Fallback for sessions without exploration history
                self.resume_next_iteration = 0
                self.resume_total_evaluations = 0
                self.resume_best_score = 0.0
                self.resume_root_score = None
                self.resume_has_history = False
                self.resume_total_eval_time = 0.0
                self.resume_unique_nodes = 0
                self.resume_last_iteration = -1
                self.resume_session_iterations = 0
                self.logger.warning(f"No resume parameters found for session {session_id[:8]}...")
    
    def get_resume_parameters(self) -> dict:
        """Get resume parameters for MCTS engine."""
        return {
            'next_iteration': getattr(self, 'resume_next_iteration', 0),
            'total_evaluations': getattr(self, 'resume_total_evaluations', 0),
            'best_score': getattr(self, 'resume_best_score', 0.0),
            'root_score': getattr(self, 'resume_root_score', None),
            'has_history': getattr(self, 'resume_has_history', False),
            'total_eval_time': getattr(self, 'resume_total_eval_time', 0.0),
            'unique_nodes': getattr(self, 'resume_unique_nodes', 0),
            'last_iteration': getattr(self, 'resume_last_iteration', -1),
            'session_iterations': getattr(self, 'resume_session_iterations', 0)
        }
    
    def _calculate_dataset_hash(self) -> str:
        """Calculate dataset hash from configuration."""
        import hashlib
        
        autogluon_config = self.config.get('autogluon', {})
        train_path = autogluon_config.get('train_path', '')
        test_path = autogluon_config.get('test_path', '')
        
        path_string = f"{train_path}|{test_path or ''}"
        return hashlib.md5(path_string.encode()).hexdigest()
    
    def log_exploration_step(self, 
                           iteration: int,
                           operation: str,
                           features_before: List[str],
                           features_after: List[str],
                           score: float,
                           eval_time: float,
                           autogluon_config: Dict,
                           ucb1_score: float = None,
                           parent_node_id: int = None,
                           memory_usage_mb: float = None,
                           mcts_node_id: int = None,
                           node_visits: int = 1) -> int:
        """
        Log a single MCTS exploration step.
        
        Args:
            iteration: Iteration number
            operation: Operation that was applied
            features_before: Features before operation
            features_after: Features after operation
            score: Evaluation score achieved
            eval_time: Time taken for evaluation
            autogluon_config: AutoGluon configuration used
            ucb1_score: UCB1 score for MCTS
            parent_node_id: Parent node ID in MCTS tree
            memory_usage_mb: Memory usage in MB
            mcts_node_id: MCTS internal node ID
            node_visits: Visit count after backpropagation
            
        Returns:
            Exploration step ID
        """
        if not self.current_session_id:
            raise RuntimeError("No active session. Call initialize_session() first.")
        
        # Extract target metric from config
        target_metric = autogluon_config.get('target_metric', 'unknown')
        
        step_data = ExplorationCreate(
            session_id=self.current_session_id,
            iteration=iteration,
            parent_node_id=parent_node_id,
            operation_applied=operation,
            features_before=features_before,
            features_after=features_after,
            evaluation_score=score,
            target_metric=target_metric,
            evaluation_time=eval_time,
            autogluon_config=autogluon_config,
            mcts_ucb1_score=ucb1_score,
            memory_usage_mb=memory_usage_mb,
            mcts_node_id=mcts_node_id,
            node_visits=node_visits
        )
        
        step = self.exploration_repo.log_exploration_step(step_data)
        return step.id
    
    def update_mcts_node_visits(self, node_id: int, visit_count: int, total_reward: float, 
                               average_reward: float) -> None:
        """
        Update MCTS node visit statistics during backpropagation.
        
        Args:
            node_id: MCTS internal node ID
            visit_count: Updated visit count
            total_reward: Updated total reward
            average_reward: Updated average reward
        """
        if not self.current_session_id:
            raise RuntimeError("No active session. Call initialize_session() first.")
        
        query = """
        UPDATE mcts_tree_nodes 
        SET visit_count = ?, 
            total_reward = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE session_id = ? AND node_id = ?
        """
        
        with self.connection_manager.get_connection() as conn:
            conn.execute(query, [visit_count, total_reward, self.current_session_id, node_id])
            
        self.logger.debug(f"Updated MCTS node {node_id}: visits={visit_count}, total_reward={total_reward:.5f}")
    
    def ensure_mcts_node_exists(self, node_id: int, parent_node_id: Optional[int], 
                               depth: int, operation_applied: Optional[str],
                               features_before: List[str], features_after: List[str],
                               base_features: List[str], applied_operations: List[str],
                               evaluation_score: Optional[float] = None,
                               evaluation_time: Optional[float] = None,
                               memory_usage_mb: Optional[float] = None) -> None:
        """
        Ensure MCTS node exists in mcts_tree_nodes table (upsert operation).
        
        Args:
            node_id: MCTS internal node ID
            parent_node_id: Parent's MCTS node ID (None for root)
            depth: Tree depth
            operation_applied: Operation that created this node (None for root)
            features_before: Features before operation
            features_after: Features after operation  
            base_features: Base features of the dataset
            applied_operations: List of operations applied to reach this node
            evaluation_score: Node evaluation score (if evaluated)
            evaluation_time: Time taken for evaluation
            memory_usage_mb: Memory usage in MB
        """
        if not self.current_session_id:
            raise RuntimeError("No active session. Call initialize_session() first.")
        
        # Check if node exists
        check_query = "SELECT COUNT(*) FROM mcts_tree_nodes WHERE session_id = ? AND node_id = ?"
        with self.connection_manager.get_connection() as conn:
            result = conn.execute(check_query, [self.current_session_id, node_id]).fetchone()
            exists = result[0] > 0 if result else False
        
        if not exists:
            # Insert new node
            insert_query = """
            INSERT INTO mcts_tree_nodes 
            (session_id, node_id, parent_node_id, depth, operation_applied,
             base_features, features_before, features_after, applied_operations,
             evaluation_score, evaluation_time, memory_usage_mb, visit_count, total_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0)
            """
            
            import json
            with self.connection_manager.get_connection() as conn:
                conn.execute(insert_query, [
                    self.current_session_id, node_id, parent_node_id, depth, operation_applied,
                    json.dumps(base_features), json.dumps(features_before), 
                    json.dumps(features_after), json.dumps(applied_operations),
                    evaluation_score, evaluation_time, memory_usage_mb
                ])
                
            self.logger.debug(f"Created MCTS node {node_id} in database")
        else:
            # Update existing node with evaluation results
            if evaluation_score is not None:
                update_query = """
                UPDATE mcts_tree_nodes 
                SET evaluation_score = ?, evaluation_time = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND node_id = ?
                """
                with self.connection_manager.get_connection() as conn:
                    conn.execute(update_query, [evaluation_score, evaluation_time, 
                                              self.current_session_id, node_id])
                    
                self.logger.debug(f"Updated MCTS node {node_id} evaluation in database")
    
    def register_feature(self,
                        name: str,
                        category: str,
                        python_code: str,
                        dependencies: List[str] = None,
                        description: str = "",
                        created_by: str = "mcts",
                        computational_cost: float = 1.0) -> int:
        """
        Register a new feature in the catalog.
        
        Args:
            name: Feature name
            category: Feature category
            python_code: Python code to generate feature
            dependencies: Required features/columns
            description: Feature description
            created_by: Who created the feature
            computational_cost: Relative computational cost
            
        Returns:
            Feature ID
        """
        from .db.models.feature import FeatureCategory, FeatureCreator
        
        # Convert string categories to enums
        try:
            category_enum = FeatureCategory(category)
        except ValueError:
            category_enum = FeatureCategory.FEATURE_TRANSFORMATIONS
        
        try:
            creator_enum = FeatureCreator(created_by)
        except ValueError:
            creator_enum = FeatureCreator.MCTS
        
        feature_data = FeatureCreate(
            feature_name=name,
            feature_category=category_enum,
            python_code=python_code,
            dependencies=dependencies or [],
            description=description,
            created_by=creator_enum,
            computational_cost=computational_cost
        )
        
        feature = self.feature_repo.register_feature(feature_data)
        return feature.id
    
    def update_feature_impact(self,
                            feature_name: str,
                            baseline_score: float,
                            with_feature_score: float,
                            context_features: List[str] = None) -> None:
        """
        Update impact analysis for a feature.
        
        Args:
            feature_name: Name of the feature
            baseline_score: Score without the feature
            with_feature_score: Score with the feature
            context_features: Other features in the evaluation set
        """
        if not self.current_session_id:
            raise RuntimeError("No active session. Call initialize_session() first.")
        
        self.feature_impact_repo.update_feature_impact(
            feature_name=feature_name,
            baseline_score=baseline_score,
            with_feature_score=with_feature_score,
            session_id=self.current_session_id,
            context_features=context_features or []
        )
    
    def update_operation_performance(self,
                                   operation_name: str,
                                   category: str,
                                   improvement: float,
                                   execution_time: float,
                                   success: bool = True) -> None:
        """
        Update performance statistics for an operation.
        
        Args:
            operation_name: Name of the operation
            category: Operation category
            improvement: Score improvement achieved
            execution_time: Time taken to execute
            success: Whether operation was successful
        """
        if not self.current_session_id:
            raise RuntimeError("No active session. Call initialize_session() first.")
        
        self.operation_perf_repo.update_operation_performance(
            operation_name=operation_name,
            category=category,
            improvement=improvement,
            execution_time=execution_time,
            session_id=self.current_session_id,
            success=success
        )
    
    def get_best_features(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top performing features by impact delta.
        
        Args:
            limit: Maximum number of features to return
            
        Returns:
            List of feature impact dictionaries
        """
        if not self.current_session_id:
            return []
        
        impacts = self.feature_impact_repo.get_top_performing_features(
            limit=limit,
            session_id=self.current_session_id
        )
        
        results = []
        
        # Get dataset name to access correct dataset database
        dataset_name = self.config.get('autogluon', {}).get('dataset_name')
        if not dataset_name:
            self.logger.warning("No dataset_name in config, returning impacts without feature details")
            for impact in impacts:
                results.append({
                    'feature_name': impact.feature_name,
                    'feature_category': 'unknown',
                    'impact_delta': impact.impact_delta,
                    'impact_percentage': impact.impact_percentage,
                    'with_feature_score': impact.with_feature_score,
                    'sample_size': impact.sample_size,
                    'python_code': f'# Feature: {impact.feature_name}',
                    'computational_cost': 1.0,
                    'session_id': impact.session_id
                })
            return results
        
        # Connect to dataset database for feature details
        import duckdb
        from pathlib import Path
        dataset_db_path = Path("cache") / dataset_name / "dataset.duckdb"
        
        if not dataset_db_path.exists():
            self.logger.warning(f"Dataset database not found: {dataset_db_path}")
            for impact in impacts:
                results.append({
                    'feature_name': impact.feature_name,
                    'feature_category': 'unknown',
                    'impact_delta': impact.impact_delta,
                    'impact_percentage': impact.impact_percentage,
                    'with_feature_score': impact.with_feature_score,
                    'sample_size': impact.sample_size,
                    'python_code': f'# Feature: {impact.feature_name}',
                    'computational_cost': 1.0,
                    'session_id': impact.session_id
                })
            return results
        
        try:
            with duckdb.connect(str(dataset_db_path)) as conn:
                for impact in impacts:
                    # Get feature details from dataset database
                    feature_result = conn.execute(
                        "SELECT feature_category, python_code, computational_cost FROM feature_catalog WHERE feature_name = ?", 
                        [impact.feature_name]
                    ).fetchone()
                    
                    if feature_result:
                        results.append({
                            'feature_name': impact.feature_name,
                            'feature_category': feature_result[0],
                            'impact_delta': impact.impact_delta,
                            'impact_percentage': impact.impact_percentage,
                            'with_feature_score': impact.with_feature_score,
                            'sample_size': impact.sample_size,
                            'python_code': feature_result[1],
                            'computational_cost': feature_result[2],
                            'session_id': impact.session_id
                        })
                    else:
                        # Feature not found in catalog
                        results.append({
                            'feature_name': impact.feature_name,
                            'feature_category': 'unknown',
                            'impact_delta': impact.impact_delta,
                            'impact_percentage': impact.impact_percentage,
                            'with_feature_score': impact.with_feature_score,
                            'sample_size': impact.sample_size,
                            'python_code': f'# Feature: {impact.feature_name}',
                            'computational_cost': 1.0,
                            'session_id': impact.session_id
                        })
        except Exception as e:
            self.logger.error(f"Error accessing dataset database: {e}")
            for impact in impacts:
                results.append({
                    'feature_name': impact.feature_name,
                    'feature_category': 'unknown',
                    'impact_delta': impact.impact_delta,
                    'impact_percentage': impact.impact_percentage,
                    'with_feature_score': impact.with_feature_score,
                    'sample_size': impact.sample_size,
                    'python_code': f'# Feature: {impact.feature_name}',
                    'computational_cost': 1.0,
                    'session_id': impact.session_id
                })
            return results
        
        return results
    
    def get_session_progress(self) -> Dict[str, Any]:
        """
        Get current session statistics.
        
        Returns:
            Dictionary with session progress information
        """
        if not self.current_session_id:
            return {}
        
        summary = self.session_repo.get_session_summary(self.current_session_id)
        if summary:
            return {
                'session_id': summary.session_id,
                'session_name': summary.session_name,
                'start_time': summary.start_time.isoformat(),
                'end_time': summary.end_time.isoformat() if summary.end_time else None,
                'total_iterations': summary.total_iterations,
                'min_score': summary.min_score,
                'max_score': summary.max_score,
                'improvement': summary.improvement,
                'avg_eval_time': summary.avg_eval_time,
                'total_eval_time': summary.total_eval_time,
                'status': summary.status.value,
                'target_metric': summary.target_metric
            }
        
        return {}
    
    def get_operation_rankings(self) -> List[Dict[str, Any]]:
        """
        Get operation effectiveness rankings.
        
        Returns:
            List of operation performance dictionaries
        """
        if not self.current_session_id:
            return []
        
        rankings = self.operation_perf_repo.get_operation_rankings(
            session_id=self.current_session_id
        )
        
        return [
            {
                'operation_name': op.operation_name,
                'effectiveness_score': op.effectiveness_score,
                'total_applications': op.total_applications,
                'success_rate': op.success_rate,
                'avg_improvement': op.avg_improvement,
                'session_id': op.session_id
            }
            for op in rankings
        ]
    
    def get_feature_code(self, feature_name: str) -> Optional[str]:
        """
        Get Python code for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Python code or None if not found
        """
        # Get dataset name to access correct dataset database
        dataset_name = self.config.get('autogluon', {}).get('dataset_name')
        if not dataset_name:
            return f'# Feature: {feature_name} (no dataset configured)'
        
        # Connect to dataset database for feature code
        import duckdb
        from pathlib import Path
        dataset_db_path = Path("cache") / dataset_name / "dataset.duckdb"
        
        if not dataset_db_path.exists():
            return f'# Feature: {feature_name} (dataset database not found)'
        
        try:
            with duckdb.connect(str(dataset_db_path)) as conn:
                result = conn.execute(
                    "SELECT python_code FROM feature_catalog WHERE feature_name = ?", 
                    [feature_name]
                ).fetchone()
                return result[0] if result else f'# Feature: {feature_name} (not found in catalog)'
        except Exception as e:
            self.logger.error(f"Error getting feature code: {e}")
            return f'# Feature: {feature_name} (error accessing catalog)'
    
    def close_session(self, status: str = 'completed') -> None:
        """
        Close the current session.
        
        Args:
            status: Final status for the session
        """
        if not self.current_session_id:
            return
        
        try:
            session_status = SessionStatus(status)
        except ValueError:
            session_status = SessionStatus.COMPLETED
        
        self.session_repo.close_session(self.current_session_id, session_status)
        
        self.logger.info(f"Session {self.current_session_id[:8]}... closed with status: {status}")
        
        self.current_session_id = None
        self.session_name = None
    
    def export_best_features_code(self, output_file: str, limit: int = 20) -> None:
        """
        Export Python code for the best discovered features.
        
        Args:
            output_file: Path to output file
            limit: Maximum number of features to export
        """
        best_features = self.get_best_features(limit)
        
        with open(output_file, 'w') as f:
            f.write("# Auto-generated best features from MCTS exploration\n")
            f.write(f"# Generated at: {datetime.now()}\n")
            f.write(f"# Session: {self.current_session_id}\n\n")
            
            f.write("import numpy as np\nimport pandas as pd\n\n")
            
            for i, feature in enumerate(best_features, 1):
                f.write(f"# {i}. {feature['feature_name']} - Impact: +{feature['impact_delta']:.5f}\n")
                f.write(f"# Category: {feature['feature_category']}\n")
                f.write(f"# Improvement: {feature['impact_percentage']:+.2f}%\n")
                f.write(feature['python_code'])
                f.write("\n\n")
        
        self.logger.info(f"Exported {len(best_features)} best features to {output_file}")
    
    def export_best_features_to_session(self, limit: int = 20) -> Optional[str]:
        """
        Export best features to session directory.
        
        Args:
            limit: Maximum number of features to export
            
        Returns:
            Path to exported file or None if no session
        """
        if not self.output_manager:
            return None
        
        export_paths = self.output_manager.get_export_paths()
        output_file = export_paths['best_features_code']
        
        self.export_best_features_code(str(output_file), limit)
        return str(output_file)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the database service.
        
        Returns:
            Dictionary with performance statistics
        """
        return self.connection_manager.get_performance_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on database service.
        
        Returns:
            Health check results
        """
        return self.connection_manager.health_check()
    
    def close(self) -> None:
        """Close database service and cleanup resources."""
        if self.current_session_id:
            self.close_session('interrupted')
        
        self.connection_manager.close()
        self.logger.info("Database service closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()