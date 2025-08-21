"""
Visualization data formatter for D3-DNA API

Converts visualization logger data to JSON format matching the H5→JSON structure
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from ..models import GenerationMetadata, StepData


class VisualizationFormatter:
    """Converts visualization data to JSON format matching H5→JSON converter structure"""
    
    @staticmethod
    def format_step_data(viz_logger) -> List[StepData]:
        """Convert visualization logger step data to API format"""
        steps = []
        
        for i, step_info in enumerate(viz_logger.step_data):
            step_data = StepData(
                step=step_info.get('step', i),
                timestep=float(step_info.get('timestep', 0.0)),
                noise_level=float(step_info.get('noise_level', 0.0)),
                sequences=VisualizationFormatter._tensor_to_list(step_info.get('sequence')),
                score_matrices=VisualizationFormatter._tensor_to_list(step_info.get('score_matrix')),
            )
            
            # Add optional step data
            if 'prob_matrix' in step_info and step_info['prob_matrix'] is not None:
                step_data.prob_matrices = VisualizationFormatter._tensor_to_list(step_info['prob_matrix'])
            
            if 'oracle_mse' in step_info and step_info['oracle_mse'] is not None:
                step_data.oracle_mses = VisualizationFormatter._tensor_to_list(step_info['oracle_mse'])
            
            if 'oracle_predictions' in step_info and step_info['oracle_predictions'] is not None:
                step_data.oracle_predictions = VisualizationFormatter._tensor_to_list(step_info['oracle_predictions'])
                
            if 'attribution_matrix' in step_info and step_info['attribution_matrix'] is not None:
                step_data.attribution_matrices = VisualizationFormatter._tensor_to_list(step_info['attribution_matrix'])
            
            steps.append(step_data)
        
        return steps
    
    @staticmethod
    def format_metadata(viz_logger, dataset: str, architecture: str, split: Optional[str] = None) -> GenerationMetadata:
        """Convert visualization logger metadata to API format"""
        
        metadata = GenerationMetadata(
            dataset=dataset,
            num_samples=viz_logger.num_samples,
            sequence_length=viz_logger.sequence_length,
            total_steps=viz_logger.num_steps,
            architecture=architecture,
            split=split,
            save_oracle_mse=viz_logger.save_oracle_mse
        )
        
        # Add optional metadata arrays
        if hasattr(viz_logger, 'original_samples') and viz_logger.original_samples is not None:
            # Convert channel-first one-hot encoded original samples to integer tokens
            original_samples_tensor = viz_logger.original_samples
            
            if original_samples_tensor.dim() == 3 and original_samples_tensor.shape[1] == 4:
                # Channel-first one-hot encoded: [batch, 4, seq_len] -> [batch, seq_len]
                original_samples_tokens = torch.argmax(original_samples_tensor, dim=1)
                metadata.original_samples = VisualizationFormatter._tensor_to_list(original_samples_tokens)
            else:
                # Already integer tokens or different format
                metadata.original_samples = VisualizationFormatter._tensor_to_list(original_samples_tensor)
            
        if hasattr(viz_logger, 'ground_truth_labels') and viz_logger.ground_truth_labels is not None:
            metadata.ground_truth_labels = VisualizationFormatter._tensor_to_list(viz_logger.ground_truth_labels)
            
        if hasattr(viz_logger, 'ground_truth_predictions') and viz_logger.ground_truth_predictions is not None:
            metadata.ground_truth_predictions = VisualizationFormatter._tensor_to_list(viz_logger.ground_truth_predictions)
            
        if hasattr(viz_logger, 'dataset_indices') and viz_logger.dataset_indices is not None:
            metadata.dataset_indices = VisualizationFormatter._tensor_to_list(viz_logger.dataset_indices)
        
        return metadata
    
    @staticmethod
    def _tensor_to_list(tensor) -> List[Any]:
        """Convert tensor to nested Python list, handling various input types"""
        if tensor is None:
            return []
            
        if isinstance(tensor, torch.Tensor):
            # Move to CPU and convert to numpy, then to list
            return tensor.detach().cpu().numpy().tolist()
        elif isinstance(tensor, np.ndarray):
            return tensor.tolist()
        elif isinstance(tensor, (list, tuple)):
            return list(tensor)
        elif isinstance(tensor, (int, float, bool)):
            return [tensor]
        else:
            # Try to convert to list directly
            try:
                return list(tensor)
            except (TypeError, ValueError):
                return []
    
    @staticmethod
    def sequences_to_strings(sequences: torch.Tensor) -> List[str]:
        """Convert token sequences to DNA strings"""
        token_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        
        # Ensure sequences is 2D (batch_size, seq_length)
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        elif sequences.dim() > 2:
            # If it's one-hot encoded (batch_size, seq_length, 4), convert to indices
            if sequences.shape[-1] == 4:
                sequences = torch.argmax(sequences, dim=-1)
            else:
                sequences = sequences.view(-1, sequences.shape[-1])
        
        sequences_str = []
        for seq in sequences:
            # Handle each token in the sequence
            tokens = []
            for token in seq:
                if isinstance(token, torch.Tensor):
                    if token.numel() == 1:
                        tokens.append(token_to_nucleotide.get(token.item(), 'N'))
                    else:
                        # If token is still multi-dimensional, take the first element
                        tokens.append(token_to_nucleotide.get(token.flatten()[0].item(), 'N'))
                else:
                    tokens.append(token_to_nucleotide.get(int(token), 'N'))
            
            seq_str = ''.join(tokens)
            sequences_str.append(seq_str)
        
        return sequences_str
    
    @staticmethod
    def format_complete_response(final_sequences: torch.Tensor, viz_logger, 
                               dataset: str, architecture: str, generation_time: float,
                               sp_mse: Optional[float] = None, split: Optional[str] = None) -> Dict[str, Any]:
        """Format complete generation response with visualization data"""
        
        # Convert final sequences to DNA strings
        final_sequences_str = VisualizationFormatter.sequences_to_strings(final_sequences)
        
        # Format metadata and steps
        metadata = VisualizationFormatter.format_metadata(viz_logger, dataset, architecture, split)
        steps = VisualizationFormatter.format_step_data(viz_logger)
        
        response = {
            "final_sequences": final_sequences_str,
            "generation_time": generation_time,
            "metadata": metadata.model_dump(),
            "steps": [step.model_dump() for step in steps]
        }
        
        # Add evaluation metrics if available
        if sp_mse is not None:
            response["sp_mse"] = sp_mse
            response["oracle_evaluation"] = "completed"
        
        return response