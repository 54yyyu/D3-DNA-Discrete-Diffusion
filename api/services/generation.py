"""
Generation service for D3-DNA API

Wraps existing DeepSTARR evaluation and sampling components
"""

import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import OmegaConf

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model_zoo.deepstarr.evaluate import DeepSTARREvaluator
from model_zoo.deepstarr.sample import DeepSTARRSampler
from utils.visualization_logger import create_visualization_logger
from .model_loader import model_loader
from .visualization import VisualizationFormatter
from ..models import GenerationRequest, GenerationResponse


class GenerationService:
    """Service for generating DNA sequences with visualization data"""
    
    def __init__(self):
        self.token_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    async def generate_sequences(self, request: GenerationRequest, 
                               dataset_config: Dict[str, Any]) -> GenerationResponse:
        """Generate DNA sequences with complete visualization data"""
        
        start_time = time.time()
        
        try:
            # Get architecture from config
            architecture = dataset_config["architecture"]
            
            # Load model if not already cached
            model_key = f"{request.dataset}_{architecture}"
            if not model_loader.is_model_loaded(request.dataset, architecture):
                print(f"Loading model: {model_key}")
                model_loader.load_model(request.dataset, dataset_config)
            
            model_data = model_loader.get_model(request.dataset, architecture)
            
            # Determine mode based on request
            if request.mode == "evaluation":
                return await self._generate_with_evaluation(request, model_data, dataset_config, architecture, start_time)
            else:
                return await self._generate_pure_sampling(request, model_data, dataset_config, architecture, start_time)
                
        except Exception as e:
            print(f"Generation failed: {str(e)}")
            raise Exception(f"Generation failed: {str(e)}")
    
    async def _generate_with_evaluation(self, request: GenerationRequest, 
                                      model_data: Dict[str, Any], 
                                      dataset_config: Dict[str, Any],
                                      architecture: str,
                                      start_time: float) -> GenerationResponse:
        """Generate sequences using evaluation pipeline (with oracle, SP-MSE, etc.)"""
        
        if request.dataset.lower() != "deepstarr":
            raise ValueError(f"Evaluation not supported for dataset: {request.dataset}")
        
        # Create evaluator
        evaluator = DeepSTARREvaluator()
        
        # Create temporary checkpoint path (we already have loaded model)
        checkpoint_path = "dummy_path"  # Not actually used since we override model loading
        
        # Override the evaluator's load_model method to use our cached model
        def mock_load_model(cp_path, config, arch):
            return model_data["model"], model_data["graph"], model_data["noise"]
        evaluator.load_model = mock_load_model
        
        # Override oracle loading to use cached oracle
        def mock_load_oracle(oracle_path, data_path, use_evoaug=False):
            return model_data.get("oracle")
        evaluator.load_oracle_model = mock_load_oracle
        
        # Set configuration
        config = model_data["config"]
        
        # Determine split and sampling parameters
        split = "test"  # Default for evaluation
        steps = request.steps if request.steps != 249 else evaluator.get_sequence_length(config)
        
        # For evaluation mode, create a proper dataloader from the dataset
        # This will sample from the actual test set with proper indices
        dataloader = evaluator.create_dataloader(
            config=config, 
            split=split, 
            batch_size=request.num_samples,  # Use num_samples as batch size
            max_samples=request.num_samples,  # Limit to requested number of samples
            specific_indices=request.specific_indices
        )
        
        # Create visualization logger if requested (after dataloader to get proper sample count)
        viz_logger = None
        if request.include_visualization:
            sequence_length = evaluator.get_sequence_length(config)
            actual_samples = len(dataloader.dataset)
            
            viz_logger = create_visualization_logger(
                num_samples=actual_samples,
                sequence_length=sequence_length,
                num_steps=steps,
                dataset_name=request.dataset,
                architecture=architecture,
                split=split,
                save_oracle_mse=request.include_oracle,  # Re-enable oracle MSE since we have proper indices now
                device=model_data["device"],
                dataset_indices=getattr(evaluator, '_dataset_indices', None)  # Pass the dataset indices
            )
        
        # Sample sequences with evaluation
        sampled_sequences, target_labels = evaluator.sample_sequences_for_evaluation(
            checkpoint_path=checkpoint_path,
            config=config,
            dataloader=dataloader,
            num_steps=steps,
            architecture=architecture,
            show_progress=False,
            viz_logger=viz_logger,
            oracle_model=model_data.get("oracle"),
            data_path=str(dataset_config.get("data_file", ""))
        )
        
        # Compute SP-MSE if requested and oracle available
        sp_mse = None
        if request.include_sp_mse and model_data.get("oracle"):
            try:
                # Get original test data matching the same indices used in evaluation
                original_data = evaluator.get_original_test_data(str(dataset_config.get("data_file", "")))
                sp_mse = evaluator.compute_sp_mse(sampled_sequences, model_data["oracle"], original_data)
                print(f"SP-MSE computed: {sp_mse:.6f}")
            except Exception as e:
                print(f"SP-MSE computation failed: {e}")
                sp_mse = None
        
        generation_time = time.time() - start_time
        
        # Format response
        if viz_logger and request.include_visualization:
            response_data = VisualizationFormatter.format_complete_response(
                sampled_sequences, viz_logger, request.dataset, architecture,
                generation_time, sp_mse, split
            )
        else:
            # Basic response without visualization data
            sequences_str = VisualizationFormatter.sequences_to_strings(sampled_sequences)
            response_data = {
                "final_sequences": sequences_str,
                "generation_time": generation_time,
                "metadata": {
                    "dataset": request.dataset,
                    "num_samples": request.num_samples,
                    "sequence_length": sampled_sequences.shape[1],
                    "total_steps": steps,
                    "save_oracle_mse": request.include_oracle
                },
                "steps": []
            }
            
            if sp_mse is not None:
                response_data["sp_mse"] = sp_mse
                response_data["oracle_evaluation"] = "completed"
        
        return GenerationResponse(**response_data)
    
    async def _generate_pure_sampling(self, request: GenerationRequest,
                                    model_data: Dict[str, Any],
                                    dataset_config: Dict[str, Any],
                                    architecture: str,
                                    start_time: float) -> GenerationResponse:
        """Generate sequences using pure sampling pipeline"""
        
        if request.dataset.lower() != "deepstarr":
            raise ValueError(f"Sampling not supported for dataset: {request.dataset}")
        
        # Create sampler
        sampler = DeepSTARRSampler()
        
        # Override the sampler's load_model method to use our cached model
        def mock_load_model(cp_path, config, arch):
            return model_data["model"], model_data["graph"], model_data["noise"]
        sampler.load_model = mock_load_model
        
        config = model_data["config"]
        sequence_length = sampler.get_sequence_length(config)
        steps = request.steps if request.steps != 249 else sequence_length
        
        # Generate conditioning labels
        conditioning_labels = self._generate_conditioning_labels(request, model_data["device"])
        
        # Create visualization logger if requested
        viz_logger = None
        if request.include_visualization:
            viz_logger = create_visualization_logger(
                num_samples=request.num_samples,
                sequence_length=sequence_length,
                num_steps=steps,
                dataset_name=request.dataset,
                architecture=architecture,
                split=None,  # Pure sampling
                save_oracle_mse=False,  # No oracle for pure sampling
                device=model_data["device"]
            )
        
        # Sample sequences
        sampled_sequences = sampler.sample_sequences_with_pc_sampler(
            checkpoint_path="dummy_path",  # Not used
            config=config,
            num_samples=request.num_samples,
            steps=steps,
            architecture=architecture,
            conditioning_labels=conditioning_labels,
            viz_logger=viz_logger
        )
        
        generation_time = time.time() - start_time
        
        # Format response
        if viz_logger and request.include_visualization:
            response_data = VisualizationFormatter.format_complete_response(
                sampled_sequences, viz_logger, request.dataset, architecture,
                generation_time
            )
        else:
            # Basic response without visualization data
            sequences_str = VisualizationFormatter.sequences_to_strings(sampled_sequences)
            response_data = {
                "final_sequences": sequences_str,
                "generation_time": generation_time,
                "metadata": {
                    "dataset": request.dataset,
                    "num_samples": request.num_samples,
                    "sequence_length": sampled_sequences.shape[1],
                    "total_steps": steps,
                    "save_oracle_mse": False
                },
                "steps": []
            }
        
        return GenerationResponse(**response_data)
    
    def _generate_conditioning_labels(self, request: GenerationRequest, device: str) -> Optional[torch.Tensor]:
        """Generate conditioning labels based on request parameters"""
        
        if request.unconditional:
            return None
        
        if request.dataset.lower() == "deepstarr":
            # DeepSTARR has 2 activities: Dev and HK
            if request.dev_activity is not None and request.hk_activity is not None:
                # Use specified activities
                labels = torch.tensor([[request.dev_activity, request.hk_activity]], 
                                    device=device).expand(request.num_samples, -1)
            else:
                # Generate random activities
                labels = torch.randn(request.num_samples, 2, device=device)
            
            return labels
        
        # Default: random conditioning
        return torch.randn(request.num_samples, 2, device=device)
    


# Global generation service instance
generation_service = GenerationService()