"""
DuckDB Database Interface for MCTS Feature Discovery

Comprehensive logging and analytics for feature exploration sessions.
Supports session management, feature impact analysis, and performance tracking.
Uses DuckDB for improved JSON handling and analytical performance.

This is the refactored version that uses the new DatabaseService layer
for improved maintainability and performance.
"""

import json
import uuid
import os
import shutil
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from .session_output_manager import SessionOutputManager
from .db_service import DatabaseService

logger = logging.getLogger(__name__)

class FeatureDiscoveryDB:
    """
    Database interface for MCTS feature discovery logging and analytics.
    
    This class provides backward compatibility while using the new DatabaseService
    for improved performance and maintainability.
    """
    
    def __init__(self, config: Dict[str, Any], read_only: bool = False):
        """Initialize database with configuration parameters.
        
        Args:
            config: Configuration dictionary
            read_only: If True, don't create a new session (for read-only access)
        """
        self.config = config
        self.read_only = read_only
        
        # Initialize the new database service layer
        self.db_service = DatabaseService(config, read_only=read_only)
        
        # Extract session information from the service
        self.session_id = self.db_service.current_session_id
        self.session_name = self.db_service.session_name
        self.output_manager = self.db_service.output_manager
        
        # Expose db_service methods for backward compatibility
        self.update_session_progress = self.db_service.update_session_progress
        self.ensure_mcts_node_exists = self.db_service.ensure_mcts_node_exists
        
        # Backward compatibility properties
        self.db_config = config.get('database', {})
        self.db_path = self.db_service.connection_manager.db_path
        self.backup_path = self.db_config.get('backup_path', 'backups')
        self.db_type = self.db_config.get('type', 'duckdb')  # Support multiple database types
        self.schema = 'main'
        
        # Initialize session if not already done and not in read-only mode
        if not self.session_id and not read_only:
            session_mode = config.get('session', {}).get('mode', 'new')
            resume_session_id = config.get('session', {}).get('resume_session_id')
            force_resume = config.get('session', {}).get('force_resume', False)
            self.session_id = self.db_service.initialize_session(session_mode, resume_session_id, force_resume)
            self.session_name = self.db_service.session_name
            self.output_manager = self.db_service.output_manager
        
        if not read_only:
            logger.info(f"Initialized FeatureDiscoveryDB with session_id: {self.session_id[:8]}... (using DatabaseService)")
        else:
            logger.debug("Initialized FeatureDiscoveryDB in read-only mode (no session created)")
    
    def init_database(self):
        """Initialize database schema - delegated to DatabaseService."""
        # Database initialization is now handled by DatabaseService in constructor
        # This method is kept for backward compatibility
        logger.debug("Database initialization delegated to DatabaseService")
    
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
        """Log a single MCTS exploration step."""
        return self.db_service.log_exploration_step(
            iteration=iteration,
            operation=operation,
            features_before=features_before,
            features_after=features_after,
            score=score,
            eval_time=eval_time,
            autogluon_config=autogluon_config,
            ucb1_score=ucb1_score,
            parent_node_id=parent_node_id,
            memory_usage_mb=memory_usage_mb,
            mcts_node_id=mcts_node_id,
            node_visits=node_visits
        )
    
    def register_feature(self,
                        name: str,
                        category: str,
                        python_code: str,
                        dependencies: List[str] = None,
                        description: str = "",
                        created_by: str = "mcts",
                        computational_cost: float = 1.0) -> int:
        """Register a new feature in the catalog."""
        return self.db_service.register_feature(
            name=name,
            category=category,
            python_code=python_code,
            dependencies=dependencies,
            description=description,
            created_by=created_by,
            computational_cost=computational_cost
        )
    
    def update_feature_impact(self,
                            feature_name: str,
                            baseline_score: float,
                            with_feature_score: float,
                            context_features: List[str] = None) -> None:
        """Update impact analysis for a feature."""
        return self.db_service.update_feature_impact(
            feature_name=feature_name,
            baseline_score=baseline_score,
            with_feature_score=with_feature_score,
            context_features=context_features
        )
    
    def update_operation_performance(self,
                                   operation_name: str,
                                   category: str,
                                   improvement: float,
                                   execution_time: float,
                                   success: bool = True) -> None:
        """Update performance statistics for an operation."""
        return self.db_service.update_operation_performance(
            operation_name=operation_name,
            category=category,
            improvement=improvement,
            execution_time=execution_time,
            success=success
        )
    
    def get_best_features(self, limit: int = 10, session_id: str = None) -> List[Dict]:
        """Get top performing features by impact delta."""
        # If session_id is not provided, use current session
        # The DatabaseService will handle this automatically
        return self.db_service.get_best_features(limit=limit)
    
    def get_session_progress(self, session_id: str = None) -> Dict:
        """Get current session statistics."""
        return self.db_service.get_session_progress()
    
    def get_session_info(self) -> Dict:
        """Get information about the current session."""
        return {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'database_path': self.db_service.connection_manager.db_path,
            'output_directory': str(self.output_manager.session_dir)
        }
    
    def get_session_statistics(self) -> Dict:
        """Get statistics for the current session."""
        # Use session repository to get basic statistics
        session_repo = self.db_service.session_repo
        exploration_repo = self.db_service.exploration_repo
        
        # Get session details
        current_session = session_repo.get_by_id(self.session_id)
        if not current_session:
            return {'total_steps': 0, 'best_score': 0.0, 'status': 'unknown'}
        
        # Get exploration steps for this session
        steps = exploration_repo.find_all(
            where_clause="session_id = ?",
            params=(self.session_id,)
        )
        
        best_score = max([step.evaluation_score for step in steps]) if steps else 0.0
        
        return {
            'total_steps': len(steps),
            'best_score': best_score,
            'status': current_session.status,
            'session_id': self.session_id
        }
    
    def get_operation_rankings(self, session_id: str = None, limit: int = 10) -> List[Dict]:
        """Get operation effectiveness rankings."""
        return self.db_service.get_operation_rankings(limit=limit)
    
    def get_feature_code(self, feature_name: str) -> Optional[str]:
        """Get Python code for a specific feature."""
        return self.db_service.get_feature_code(feature_name)
    
    def export_best_features_code(self, output_file: str, limit: int = 20) -> None:
        """Export Python code for the best discovered features."""
        return self.db_service.export_best_features_code(output_file, limit)
    
    def export_best_features_to_session(self, limit: int = 20) -> str:
        """Export Python code for best features to session directory."""
        return self.db_service.export_best_features_to_session(limit)
    
    def close_session(self, status: str = 'completed') -> None:
        """Close the current session and mark it as completed."""
        return self.db_service.close_session(status)
    
    def update_mcts_node_visits(self, node_id: int, visit_count: int, total_reward: float, 
                               average_reward: float) -> None:
        """Update MCTS node visit statistics during backpropagation."""
        return self.db_service.update_mcts_node_visits(node_id, visit_count, total_reward, average_reward)
    
    def ensure_mcts_node_exists(self, node_id: int, parent_node_id: Optional[int], 
                               depth: int, operation_applied: Optional[str],
                               features_before: List[str], features_after: List[str],
                               base_features: List[str], applied_operations: List[str],
                               evaluation_score: Optional[float] = None,
                               evaluation_time: Optional[float] = None,
                               memory_usage_mb: Optional[float] = None) -> None:
        """Ensure MCTS node exists in mcts_tree_nodes table."""
        return self.db_service.ensure_mcts_node_exists(
            node_id, parent_node_id, depth, operation_applied,
            features_before, features_after, base_features, applied_operations,
            evaluation_score, evaluation_time, memory_usage_mb
        )
    
    def get_resume_parameters(self) -> dict:
        """Get resume parameters for MCTS engine - delegates to DatabaseService."""
        if self.db_service:
            return self.db_service.get_resume_parameters()
        return {
            'next_iteration': 0,
            'total_evaluations': 0,
            'best_score': 0.0,
            'root_score': None,
            'has_history': False
        }
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.close_session('interrupted')
        else:
            self.close_session('completed')