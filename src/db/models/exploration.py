"""
Exploration data models.

This module defines Pydantic models for MCTS exploration operations,
including exploration steps, nodes, and analysis results.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import json


class ExplorationStep(BaseModel):
    """
    MCTS exploration step model.
    
    Represents a single step in the MCTS exploration process,
    including the operation applied, features, and evaluation results.
    """
    
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    session_id: str = Field(..., description="Session this step belongs to")
    iteration: int = Field(..., ge=0, description="Iteration number in session (0=root, 1+=search)")
    timestamp: datetime = Field(default_factory=datetime.now, description="When step was executed")
    parent_node_id: Optional[int] = Field(None, description="Parent node ID in MCTS tree")
    operation_applied: str = Field(..., description="Feature operation that was applied")
    features_before: List[str] = Field(..., description="Feature list before operation")
    features_after: List[str] = Field(..., description="Feature list after operation")
    evaluation_score: float = Field(..., ge=0.0, le=1.0, description="Evaluation score (e.g., MAP@3)")
    target_metric: str = Field(..., description="Target metric used for evaluation")
    evaluation_time: float = Field(..., gt=0, description="Time taken for evaluation in seconds")
    autogluon_config: Dict[str, Any] = Field(..., description="AutoGluon configuration used")
    mcts_ucb1_score: Optional[float] = Field(None, description="UCB1 score for node selection")
    node_visits: int = Field(default=1, ge=1, description="Number of times node was visited")
    is_best_so_far: bool = Field(default=False, description="Whether this is best score so far")
    memory_usage_mb: Optional[float] = Field(None, ge=0, description="Memory usage in MB")
    notes: Optional[str] = Field(None, description="Additional notes about the step")
    mcts_node_id: Optional[int] = Field(None, description="MCTS internal node ID")
    
    @field_validator('features_before', 'features_after')
    @classmethod
    def validate_features(cls, v):
        """Validate feature lists."""
        if not isinstance(v, list):
            raise ValueError('Features must be a list')
        return v
    
    @field_validator('operation_applied')
    @classmethod
    def validate_operation(cls, v):
        """Validate operation name."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Operation applied cannot be empty')
        return v.strip()
    
    @field_validator('target_metric')
    @classmethod
    def validate_target_metric(cls, v):
        """Validate target metric name."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Target metric cannot be empty')
        return v.strip()
    
    @field_validator('autogluon_config')
    @classmethod
    def validate_autogluon_config(cls, v):
        """Validate AutoGluon configuration."""
        if not isinstance(v, dict):
            raise ValueError('AutoGluon config must be a dictionary')
        return v
    
    @property
    def feature_count_before(self) -> int:
        """Number of features before operation."""
        return len(self.features_before)
    
    @property
    def feature_count_after(self) -> int:
        """Number of features after operation."""
        return len(self.features_after)
    
    @property
    def features_added(self) -> List[str]:
        """Features that were added by the operation."""
        return [f for f in self.features_after if f not in self.features_before]
    
    @property
    def features_removed(self) -> List[str]:
        """Features that were removed by the operation."""
        return [f for f in self.features_before if f not in self.features_after]
    
    @property
    def net_feature_change(self) -> int:
        """Net change in feature count (positive = added, negative = removed)."""
        return self.feature_count_after - self.feature_count_before
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "session_id": "f09084df-01c1-446f-9996-4b59937691cb",
                "iteration": 1,
                "operation_applied": "npk_ratio_feature",
                "features_before": ["N", "P", "K", "pH"],
                "features_after": ["N", "P", "K", "pH", "NPK_ratio"],
                "evaluation_score": 0.7150,
                "target_metric": "MAP@3",
                "evaluation_time": 45.3,
                "autogluon_config": {"time_limit": 120, "presets": "medium_quality"},
                "is_best_so_far": True
            }
        }


class ExplorationNode(BaseModel):
    """
    MCTS tree node model.
    
    Represents a node in the MCTS exploration tree with visit statistics
    and relationship information.
    """
    
    node_id: int = Field(..., description="Unique node identifier")
    session_id: str = Field(..., description="Session this node belongs to")
    parent_node_id: Optional[int] = Field(None, description="Parent node ID")
    operation_applied: str = Field(..., description="Operation that created this node")
    features: List[str] = Field(..., description="Feature set at this node")
    visit_count: int = Field(default=0, ge=0, description="Number of times node was visited")
    total_reward: float = Field(default=0.0, description="Total reward accumulated")
    best_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Best score achieved at this node")
    ucb1_score: Optional[float] = Field(None, description="UCB1 score for selection")
    depth: int = Field(default=0, ge=0, description="Depth in the tree")
    is_expanded: bool = Field(default=False, description="Whether node has been expanded")
    is_terminal: bool = Field(default=False, description="Whether node is terminal")
    created_at: datetime = Field(default_factory=datetime.now, description="Node creation time")
    last_visited: Optional[datetime] = Field(None, description="Last visit time")
    
    @property
    def average_reward(self) -> float:
        """Calculate average reward per visit."""
        if self.visit_count > 0:
            return self.total_reward / self.visit_count
        return 0.0
    
    @property
    def feature_count(self) -> int:
        """Number of features at this node."""
        return len(self.features)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ExplorationPath(BaseModel):
    """
    Complete exploration path model.
    
    Represents a path from root to a specific node in the MCTS tree,
    showing the sequence of operations and feature evolution.
    """
    
    session_id: str = Field(..., description="Session this path belongs to")
    target_node_id: int = Field(..., description="Target node at end of path")
    path_nodes: List[ExplorationNode] = Field(..., description="Nodes in the path from root to target")
    total_depth: int = Field(..., ge=0, description="Total depth of the path")
    final_score: float = Field(..., ge=0.0, le=1.0, description="Final evaluation score")
    operations_sequence: List[str] = Field(..., description="Sequence of operations applied")
    
    @field_validator('path_nodes')
    @classmethod
    def validate_path_nodes(cls, v):
        """Validate that path nodes form a valid sequence."""
        if not v:
            raise ValueError('Path must contain at least one node')
        
        # Check that nodes form a valid parent-child sequence
        for i in range(1, len(v)):
            if v[i].parent_node_id != v[i-1].node_id:
                raise ValueError(f'Invalid path: node {i} parent mismatch')
        
        return v
    
    @field_validator('total_depth')
    @classmethod
    def validate_total_depth(cls, v, info):
        """Validate that total depth matches path length."""
        if info.data and 'path_nodes' in info.data:
            expected_depth = len(info.data['path_nodes']) - 1  # Root is depth 0
            if v != expected_depth:
                raise ValueError(f'Total depth {v} does not match path length {expected_depth}')
        return v
    
    @property
    def feature_evolution(self) -> List[List[str]]:
        """Feature sets at each step in the path."""
        return [node.features for node in self.path_nodes]
    
    @property
    def score_progression(self) -> List[float]:
        """Score progression along the path."""
        return [node.best_score for node in self.path_nodes]


class ExplorationCreate(BaseModel):
    """
    Model for creating new exploration steps.
    
    Used for logging new MCTS exploration steps with validation.
    """
    
    session_id: str = Field(..., description="Session this step belongs to")
    iteration: int = Field(..., ge=0, description="Iteration number in session (0=root, 1+=search)")
    parent_node_id: Optional[int] = Field(None, description="Parent node ID in MCTS tree")
    operation_applied: str = Field(..., description="Feature operation that was applied")
    features_before: List[str] = Field(..., description="Feature list before operation")
    features_after: List[str] = Field(..., description="Feature list after operation")
    evaluation_score: float = Field(..., ge=0.0, le=1.0, description="Evaluation score")
    target_metric: str = Field(..., description="Target metric used for evaluation")
    evaluation_time: float = Field(..., gt=0, description="Time taken for evaluation in seconds")
    autogluon_config: Dict[str, Any] = Field(..., description="AutoGluon configuration used")
    mcts_ucb1_score: Optional[float] = Field(None, description="UCB1 score for node selection")
    memory_usage_mb: Optional[float] = Field(None, ge=0, description="Memory usage in MB")
    notes: Optional[str] = Field(None, description="Additional notes about the step")
    mcts_node_id: Optional[int] = Field(None, description="MCTS internal node ID")
    node_visits: int = Field(default=1, ge=1, description="Visit count after backpropagation")


class ExplorationAnalysis(BaseModel):
    """
    Analysis results for exploration data.
    
    Provides statistical analysis and insights from exploration history.
    """
    
    session_id: str = Field(..., description="Session analyzed")
    total_steps: int = Field(..., ge=0, description="Total exploration steps")
    unique_operations: int = Field(..., ge=0, description="Number of unique operations")
    best_score: float = Field(..., ge=0.0, le=1.0, description="Best score achieved")
    score_improvement: float = Field(..., ge=0.0, description="Total score improvement")
    avg_evaluation_time: float = Field(..., ge=0.0, description="Average evaluation time")
    most_effective_operations: List[str] = Field(..., description="Most effective operations")
    feature_count_progression: List[int] = Field(..., description="Feature count over time")
    convergence_iteration: Optional[int] = Field(None, description="Iteration where convergence occurred")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "session_id": "f09084df-01c1-446f-9996-4b59937691cb",
                "total_steps": 15,
                "unique_operations": 8,
                "best_score": 0.7834,
                "score_improvement": 0.1284,
                "avg_evaluation_time": 42.5,
                "most_effective_operations": ["npk_ratio_feature", "soil_stress_indicator"],
                "feature_count_progression": [4, 5, 6, 7, 8, 9, 10],
                "convergence_iteration": 12
            }
        }