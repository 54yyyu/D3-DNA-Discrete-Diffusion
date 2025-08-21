"""
Pydantic models for D3-DNA API request/response schemas
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Request model for sequence generation"""
    
    # Core generation options
    num_samples: int = Field(default=100, ge=1, le=1000, description="Number of sequences to generate")
    steps: int = Field(default=249, ge=1, le=500, description="Number of sampling steps")
    dataset: str = Field(default="deepstarr", description="Dataset to use for generation")
    
    # DeepSTARR conditioning options
    dev_activity: Optional[float] = Field(default=None, description="Dev enhancer activity value")
    hk_activity: Optional[float] = Field(default=None, description="HK enhancer activity value")
    unconditional: bool = Field(default=False, description="Generate unconditionally (no conditioning)")
    
    # Evaluation options
    include_oracle: bool = Field(default=True, description="Include oracle model predictions")
    include_sp_mse: bool = Field(default=True, description="Compute SP-MSE against original data")
    specific_indices: Optional[str] = Field(default=None, description="Comma-separated indices (e.g. '11,12,40')")
    max_samples: Optional[int] = Field(default=None, ge=1, description="Maximum samples for evaluation")
    
    # Visualization options
    include_visualization: bool = Field(default=True, description="Include step-by-step visualization data")
    save_intermediate_steps: bool = Field(default=True, description="Save all intermediate diffusion steps")


class StepData(BaseModel):
    """Individual step data in the diffusion process"""
    step: int
    timestep: float
    noise_level: float
    sequences: List[List[int]]  # Token sequences
    score_matrices: List[List[List[float]]]  # Model score predictions
    prob_matrices: Optional[List[List[List[float]]]] = None  # Probability matrices
    oracle_mses: Optional[List[float]] = None  # Oracle MSE per sample
    oracle_predictions: Optional[List[List[float]]] = None  # Oracle predictions
    attribution_matrices: Optional[List[List[List[float]]]] = None  # Attribution matrices


class GenerationMetadata(BaseModel):
    """Metadata for generation results"""
    dataset: str
    num_samples: int
    sequence_length: int
    total_steps: int
    architecture: str
    split: Optional[str] = None
    save_oracle_mse: bool
    
    # Optional metadata arrays
    original_samples: Optional[List[List[int]]] = None
    ground_truth_labels: Optional[List[List[float]]] = None
    ground_truth_predictions: Optional[List[List[float]]] = None
    dataset_indices: Optional[List[int]] = None


class GenerationResponse(BaseModel):
    """Complete response for sequence generation with visualization data"""
    
    # Core results
    final_sequences: List[str]  # DNA strings
    generation_time: float
    
    # Complete visualization data (matching H5→JSON format)
    metadata: GenerationMetadata
    steps: List[StepData]
    
    # Additional evaluation metrics
    sp_mse: Optional[float] = None
    oracle_evaluation: Optional[str] = None


class DatasetInfo(BaseModel):
    """Information about available datasets"""
    name: str
    sequence_length: int
    signal_dim: int
    num_classes: int
    available_architectures: List[str]
    default_steps: int
    description: str


class ModelStatus(BaseModel):
    """Status of loaded models"""
    dataset: str
    architecture: str
    model_loaded: bool
    oracle_loaded: bool
    device: str
    loading_time: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    gpu_available: bool
    loaded_models: List[ModelStatus]
    memory_usage: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    status_code: int