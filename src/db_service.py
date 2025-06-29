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


class DatabaseService:
    """
    High-level database service that orchestrates all database operations.
    
    This service provides a simplified interface for the MCTS system while
    using the new repository-based architecture underneath.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database service with configuration.
        
        Args:
            config: Main MCTS configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection manager
        self.connection_manager = DuckDBConnectionManager(config)
        
        # Initialize repositories
        self.session_repo = SessionRepository(self.connection_manager)
        self.exploration_repo = ExplorationRepository(self.connection_manager)
        self.feature_repo = FeatureRepository(self.connection_manager)
        self.feature_impact_repo = FeatureImpactRepository(self.connection_manager)
        self.operation_perf_repo = OperationPerformanceRepository(self.connection_manager)
        self.dataset_repo = DatasetRepository(self.connection_manager)
        
        # Run migrations to ensure schema is up to date
        self._run_migrations()
        
        # Initialize current session
        self.current_session_id = None
        self.session_name = None
        self.output_manager: Optional[SessionOutputManager] = None
        
        self.logger.info("Database service initialized successfully")
    
    def _run_migrations(self) -> None:
        """Run database migrations to ensure schema is up to date."""
        try:
            migration_runner = MigrationRunner(self.connection_manager)
            status = migration_runner.get_migration_status()
            
            if not status['is_up_to_date']:
                self.logger.info(f"Running {status['pending_migrations']} pending migrations...")
                applied = migration_runner.run_migrations()
                self.logger.info(f"Applied {len(applied)} migrations successfully")
            else:
                self.logger.debug("Database schema is up to date")
                
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise
    
    def initialize_session(self, session_mode: str = 'new', 
                         resume_session_id: Optional[str] = None) -> str:
        """
        Initialize a new session or resume existing one.
        
        Args:
            session_mode: Session mode ('new', 'continue', 'resume_best')
            resume_session_id: Specific session ID to resume
            
        Returns:
            Session ID
        """
        if session_mode == 'new':
            return self._create_new_session()
        elif session_mode in ['continue', 'resume_best']:
            return self._resume_session(resume_session_id)
        else:
            raise ValueError(f"Unknown session mode: {session_mode}")
    
    def _create_new_session(self) -> str:
        """Create a new session."""
        import uuid
        
        session_id = str(uuid.uuid4())
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Detect test mode
        testing_config = self.config.get('testing', {})
        is_test_mode = (
            self.config.get('session', {}).get('max_iterations', 0) <= 5 or
            testing_config.get('fast_test_mode', False) or
            testing_config.get('use_small_dataset', False)
        )
        
        # Calculate dataset hash
        dataset_hash = self._calculate_dataset_hash()
        
        # Create session
        session_create = SessionCreate(
            session_id=session_id,
            session_name=session_name,
            config_snapshot=self.config.get_config() if hasattr(self.config, 'get_config') else self.config,
            is_test_mode=is_test_mode,
            dataset_hash=dataset_hash
        )
        
        session = self.session_repo.create_session(session_create)
        
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
    
    def _resume_session(self, resume_session_id: Optional[str] = None) -> str:
        """Resume an existing session."""
        if resume_session_id:
            # Resume specific session
            session = self.session_repo.find_by_id(resume_session_id, 'session_id')
            if not session:
                self.logger.warning(f"Session {resume_session_id[:8]}... not found, creating new session")
                return self._create_new_session()
        else:
            # Resume most recent session
            recent_sessions = self.session_repo.get_recent_sessions(days=7, limit=1)
            if not recent_sessions:
                self.logger.info("No recent sessions found, creating new session")
                return self._create_new_session()
            session = recent_sessions[0]
        
        self.current_session_id = session.session_id
        self.session_name = session.session_name
        
        self.logger.info(f"Resumed session: {session.session_name} (ID: {session.session_id[:8]}...)")
        return session.session_id
    
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
                           memory_usage_mb: float = None) -> int:
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
            memory_usage_mb=memory_usage_mb
        )
        
        step = self.exploration_repo.log_exploration_step(step_data)
        return step.id
    
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
        for impact in impacts:
            # Get feature details
            feature = self.feature_repo.find_by_name(impact.feature_name)
            
            if feature:
                results.append({
                    'feature_name': impact.feature_name,
                    'feature_category': feature.feature_category.value,
                    'impact_delta': impact.impact_delta,
                    'impact_percentage': impact.impact_percentage,
                    'with_feature_score': impact.with_feature_score,
                    'sample_size': impact.sample_size,
                    'python_code': feature.python_code,
                    'computational_cost': feature.computational_cost,
                    'session_id': impact.session_id
                })
        
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
        return self.feature_repo.get_feature_code(feature_name)
    
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