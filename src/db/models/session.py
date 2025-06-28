"""
Session data models.

This module defines Pydantic models for MCTS session management,
providing type safety and validation for session-related operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class SessionStrategy(str, Enum):
    """MCTS strategy enumeration."""
    DEFAULT = "default"
    UCB1 = "ucb1"
    UCT = "uct"
    THOMPSON_SAMPLING = "thompson_sampling"


class Session(BaseModel):
    """
    MCTS session model.
    
    Represents a complete MCTS feature discovery session with all
    metadata, configuration, and performance statistics.
    """
    
    session_id: str = Field(..., description="Unique session identifier")
    session_name: Optional[str] = Field(None, description="Human-readable session name")
    start_time: datetime = Field(default_factory=datetime.now, description="Session start time")
    end_time: Optional[datetime] = Field(None, description="Session end time")
    total_iterations: int = Field(default=0, ge=0, description="Total iterations completed")
    best_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Best evaluation score achieved")
    config_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Configuration used for session")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Current session status")
    strategy: SessionStrategy = Field(default=SessionStrategy.DEFAULT, description="MCTS strategy used")
    is_test_mode: bool = Field(default=False, description="Whether session is in test mode")
    notes: Optional[str] = Field(None, description="Additional session notes")
    dataset_hash: Optional[str] = Field(None, description="Hash of dataset used")
    
    @validator('session_id')
    def validate_session_id(cls, v):
        """Validate session ID format."""
        if not v or len(v) < 8:
            raise ValueError('Session ID must be at least 8 characters long')
        return v
    
    @validator('session_name')
    def validate_session_name(cls, v):
        """Validate session name if provided."""
        if v is not None and len(v.strip()) == 0:
            return None
        return v
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        """Validate that end_time is after start_time."""
        if v is not None and 'start_time' in values:
            start_time = values['start_time']
            if v < start_time:
                raise ValueError('End time must be after start time')
        return v
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate session duration in minutes."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() / 60.0
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if session is completed."""
        return self.status in [SessionStatus.COMPLETED, SessionStatus.INTERRUPTED, SessionStatus.FAILED]
    
    @property
    def iterations_per_minute(self) -> Optional[float]:
        """Calculate iterations per minute."""
        duration = self.duration_minutes
        if duration and duration > 0:
            return self.total_iterations / duration
        return None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "session_id": "f09084df-01c1-446f-9996-4b59937691cb",
                "session_name": "session_20250628_174253",
                "start_time": "2025-06-28T17:42:53",
                "end_time": "2025-06-28T17:51:23",
                "total_iterations": 3,
                "best_score": 0.7150,
                "status": "completed",
                "strategy": "ucb1",
                "is_test_mode": True,
                "dataset_hash": "abc123def456"
            }
        }


class SessionSummary(BaseModel):
    """
    Session summary model for analytics.
    
    Provides aggregated statistics and key metrics for a session
    without the full configuration details.
    """
    
    session_id: str
    session_name: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    total_iterations: int
    min_score: float = Field(ge=0.0, le=1.0)
    max_score: float = Field(ge=0.0, le=1.0)
    improvement: float = Field(ge=0.0, description="Score improvement (max - min)")
    avg_eval_time: float = Field(ge=0.0, description="Average evaluation time in seconds")
    total_eval_time: float = Field(ge=0.0, description="Total evaluation time in seconds")
    status: SessionStatus
    target_metric: Optional[str]
    
    @validator('improvement')
    def validate_improvement(cls, v, values):
        """Validate that improvement calculation is correct."""
        if 'min_score' in values and 'max_score' in values:
            expected = values['max_score'] - values['min_score']
            if abs(v - expected) > 0.0001:  # Allow small floating point differences
                raise ValueError('Improvement must equal max_score - min_score')
        return v
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as improvement over baseline."""
        if self.min_score > 0:
            return self.improvement / self.min_score
        return 0.0
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionCreate(BaseModel):
    """
    Model for creating new sessions.
    
    Used for session creation operations with only the required
    and optional fields that can be set at creation time.
    """
    
    session_id: str = Field(..., description="Unique session identifier")
    session_name: Optional[str] = Field(None, description="Human-readable session name")
    config_snapshot: Dict[str, Any] = Field(..., description="Configuration to use for session")
    strategy: SessionStrategy = Field(default=SessionStrategy.DEFAULT, description="MCTS strategy to use")
    is_test_mode: bool = Field(default=False, description="Whether session is in test mode")
    dataset_hash: Optional[str] = Field(None, description="Hash of dataset to use")
    notes: Optional[str] = Field(None, description="Initial session notes")
    
    @validator('config_snapshot')
    def validate_config_snapshot(cls, v):
        """Validate that config snapshot is not empty."""
        if not v:
            raise ValueError('Configuration snapshot cannot be empty')
        return v


class SessionUpdate(BaseModel):
    """
    Model for updating existing sessions.
    
    All fields are optional to allow partial updates.
    """
    
    session_name: Optional[str] = None
    end_time: Optional[datetime] = None
    total_iterations: Optional[int] = Field(None, ge=0)
    best_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    status: Optional[SessionStatus] = None
    notes: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }