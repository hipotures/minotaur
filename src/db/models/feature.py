"""
Feature data models.

This module defines Pydantic models for feature catalog operations,
including feature definitions, impact analysis, and performance tracking.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class FeatureCategory(str, Enum):
    """Feature category enumeration."""
    NPK_INTERACTIONS = "npk_interactions"
    ENVIRONMENTAL_STRESS = "environmental_stress"
    AGRICULTURAL_DOMAIN = "agricultural_domain"
    STATISTICAL_AGGREGATIONS = "statistical_aggregations"
    FEATURE_TRANSFORMATIONS = "feature_transformations"
    FEATURE_SELECTION = "feature_selection"


class FeatureCreator(str, Enum):
    """Feature creator enumeration."""
    MCTS = "mcts"
    LLM = "llm"
    MANUAL = "manual"
    AUTOMATED = "automated"


class Feature(BaseModel):
    """
    Feature catalog model.
    
    Represents a feature definition with its Python code,
    dependencies, and metadata.
    """
    
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    feature_name: str = Field(..., description="Unique feature name")
    feature_category: FeatureCategory = Field(..., description="Feature category")
    python_code: str = Field(..., description="Python code to generate the feature")
    dependencies: List[str] = Field(default_factory=list, description="Required features/columns")
    description: Optional[str] = Field(None, description="Feature description")
    created_by: FeatureCreator = Field(default=FeatureCreator.MCTS, description="Who/what created the feature")
    creation_timestamp: datetime = Field(default_factory=datetime.now, description="When feature was created")
    is_active: bool = Field(default=True, description="Whether feature is active")
    computational_cost: float = Field(default=1.0, ge=0.0, description="Relative computational cost")
    data_type: str = Field(default="float64", description="Expected data type of feature")
    
    @field_validator('feature_name')
    @classmethod
    def validate_feature_name(cls, v):
        """Validate feature name format."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Feature name cannot be empty')
        
        # Check for valid Python identifier (basic check)
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Feature name must be alphanumeric with underscores/hyphens')
        
        return v.strip()
    
    @field_validator('python_code')
    @classmethod
    def validate_python_code(cls, v):
        """Validate Python code is not empty."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Python code cannot be empty')
        return v
    
    @field_validator('dependencies')
    @classmethod
    def validate_dependencies(cls, v):
        """Validate dependencies list."""
        if not isinstance(v, list):
            raise ValueError('Dependencies must be a list')
        return v
    
    @field_validator('data_type')
    @classmethod
    def validate_data_type(cls, v):
        """Validate data type string."""
        valid_types = [
            'int8', 'int16', 'int32', 'int64',
            'uint8', 'uint16', 'uint32', 'uint64',
            'float16', 'float32', 'float64',
            'bool', 'object', 'string', 'category'
        ]
        if v not in valid_types:
            raise ValueError(f'Invalid data type. Must be one of: {valid_types}')
        return v
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "feature_name": "npk_ratio",
                "feature_category": "npk_interactions",
                "python_code": "df['npk_ratio'] = (df['N'] + df['P'] + df['K']) / 3",
                "dependencies": ["N", "P", "K"],
                "description": "Average NPK ratio for balanced fertilization",
                "created_by": "mcts",
                "computational_cost": 1.2,
                "data_type": "float64"
            }
        }


class FeatureImpact(BaseModel):
    """
    Feature impact analysis model.
    
    Tracks the performance impact of individual features
    on evaluation metrics.
    """
    
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    feature_name: str = Field(..., description="Feature name being analyzed")
    baseline_score: float = Field(..., ge=0.0, le=1.0, description="Baseline score without feature")
    with_feature_score: float = Field(..., ge=0.0, le=1.0, description="Score with feature included")
    impact_delta: float = Field(..., description="Score difference (with - without)")
    impact_percentage: float = Field(..., description="Percentage improvement")
    evaluation_context: List[str] = Field(default_factory=list, description="Other features in evaluation set")
    sample_size: int = Field(default=1, ge=1, description="Number of evaluations averaged")
    confidence_interval: Optional[List[float]] = Field(None, description="95% confidence interval [lower, upper]")
    statistical_significance: Optional[float] = Field(None, ge=0.0, le=1.0, description="P-value if available")
    first_discovered: datetime = Field(default_factory=datetime.now, description="When impact was first measured")
    last_evaluated: datetime = Field(default_factory=datetime.now, description="When last evaluated")
    session_id: str = Field(..., description="Session where impact was measured")
    
    # Removed validator - impact_delta can be averaged across samples
    # so it doesn't need to equal current baseline_score - with_feature_score
    
    @field_validator('impact_percentage')
    @classmethod
    def validate_impact_percentage(cls, v, info):
        """Validate impact percentage calculation."""
        if info.data and 'baseline_score' in info.data and 'impact_delta' in info.data:
            baseline = info.data['baseline_score']
            if baseline > 0:
                expected = (info.data['impact_delta'] / baseline) * 100
                if abs(v - expected) > 0.01:  # Allow small floating point differences
                    raise ValueError('Impact percentage calculation is incorrect')
        return v
    
    @field_validator('confidence_interval')
    @classmethod
    def validate_confidence_interval(cls, v):
        """Validate confidence interval format."""
        if v is not None:
            if not isinstance(v, list) or len(v) != 2:
                raise ValueError('Confidence interval must be a list of two values [lower, upper]')
            if v[0] > v[1]:
                raise ValueError('Confidence interval lower bound must be <= upper bound')
        return v
    
    @property
    def is_positive_impact(self) -> bool:
        """Check if feature has positive impact."""
        return self.impact_delta > 0
    
    @property
    def is_significant(self) -> bool:
        """Check if impact is statistically significant (p < 0.05)."""
        return self.statistical_significance is not None and self.statistical_significance < 0.05
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OperationPerformance(BaseModel):
    """
    Feature operation performance tracking model.
    
    Tracks the performance statistics for different types
    of feature operations.
    """
    
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    operation_name: str = Field(..., description="Name of the operation")
    operation_category: Optional[str] = Field(None, description="Category of operation")
    total_applications: int = Field(default=0, ge=0, description="Total times operation was applied")
    success_count: int = Field(default=0, ge=0, description="Number of successful applications")
    avg_improvement: float = Field(default=0.0, description="Average score improvement")
    best_improvement: float = Field(default=0.0, description="Best score improvement achieved")
    worst_result: float = Field(default=0.0, description="Worst score result")
    avg_execution_time: float = Field(default=0.0, ge=0.0, description="Average execution time in seconds")
    last_used: datetime = Field(default_factory=datetime.now, description="When operation was last used")
    effectiveness_score: float = Field(default=0.0, description="Overall effectiveness score")
    session_id: str = Field(..., description="Session where performance was tracked")
    
    @field_validator('success_count')
    @classmethod
    def validate_success_count(cls, v, info):
        """Validate that success count doesn't exceed total applications."""
        if info.data and 'total_applications' in info.data and v > info.data['total_applications']:
            raise ValueError('Success count cannot exceed total applications')
        return v
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_applications > 0:
            return (self.success_count / self.total_applications) * 100.0
        return 0.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        return 100.0 - self.success_rate
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FeatureCreate(BaseModel):
    """
    Model for creating new features.
    
    Used for feature registration operations with validation.
    """
    
    feature_name: str = Field(..., description="Unique feature name")
    feature_category: FeatureCategory = Field(..., description="Feature category")
    python_code: str = Field(..., description="Python code to generate the feature")
    dependencies: List[str] = Field(default_factory=list, description="Required features/columns")
    description: Optional[str] = Field(None, description="Feature description")
    created_by: FeatureCreator = Field(default=FeatureCreator.MCTS, description="Who/what created the feature")
    computational_cost: float = Field(default=1.0, ge=0.0, description="Relative computational cost")
    data_type: str = Field(default="float64", description="Expected data type of feature")


class FeatureUpdate(BaseModel):
    """
    Model for updating existing features.
    
    All fields are optional to allow partial updates.
    """
    
    feature_category: Optional[FeatureCategory] = None
    python_code: Optional[str] = None
    dependencies: Optional[List[str]] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    computational_cost: Optional[float] = Field(None, ge=0.0)
    data_type: Optional[str] = None


class FeatureAnalysis(BaseModel):
    """
    Feature analysis results model.
    
    Provides statistical analysis and insights about features
    in the catalog.
    """
    
    total_features: int = Field(..., ge=0, description="Total number of features")
    active_features: int = Field(..., ge=0, description="Number of active features")
    features_by_category: Dict[str, int] = Field(..., description="Feature count by category")
    features_by_creator: Dict[str, int] = Field(..., description="Feature count by creator")
    avg_computational_cost: float = Field(..., ge=0.0, description="Average computational cost")
    most_used_dependencies: List[str] = Field(..., description="Most frequently used dependencies")
    top_performing_features: List[str] = Field(..., description="Features with best impact")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "total_features": 45,
                "active_features": 42,
                "features_by_category": {
                    "npk_interactions": 12,
                    "environmental_stress": 8,
                    "agricultural_domain": 15
                },
                "features_by_creator": {
                    "mcts": 35,
                    "manual": 7,
                    "llm": 3
                },
                "avg_computational_cost": 1.3,
                "most_used_dependencies": ["N", "P", "K", "pH", "temperature"],
                "top_performing_features": ["npk_ratio", "soil_stress_indicator", "ph_balance"]
            }
        }