"""
TREAD Token Routing Module for D3-DNA Discrete Diffusion

This module implements efficient token routing for discrete DNA sequence diffusion,
adapted from the TREAD architecture for use with discrete tokens rather than 
continuous latents.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class Router:
    """
    Token router for efficient discrete sequence diffusion.
    
    Handles token selection, routing, and restoration for DNA sequences,
    enabling more efficient training by processing only selected tokens
    through computationally expensive transformer layers.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the router.
        
        Args:
            seed: Random seed for reproducible token selection
        """
        self.seed = seed
        # We'll create device-specific generators on demand
        self._generators = {}
        
    def get_mask(self, x: torch.Tensor, selection_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Generate mask information for token selection.
        
        Args:
            x: Input token embeddings (batch_size, sequence_length, hidden_dim)
            selection_ratio: Fraction of tokens to select for routing (0.0 to 1.0)
            
        Returns:
            Dictionary containing mask information:
            - mask: Boolean mask indicating which tokens to route
            - ids_keep: Indices of tokens to keep in routing
            - ids_mask: Indices of tokens to skip in routing  
            - ids_shuffle: Random shuffle indices
            - ids_restore: Indices to restore original order
        """
        batch_size, num_tokens, _ = x.shape
        device = x.device
        
        # Calculate number of tokens to keep and mask
        num_keep = int(num_tokens * selection_ratio)
        num_mask = num_tokens - num_keep
        
        # Generate random noise for token selection
        # Get or create device-specific generator
        device_key = str(device)
        if device_key not in self._generators:
            self._generators[device_key] = torch.Generator(device=device).manual_seed(self.seed)
        
        generator = self._generators[device_key]
        
        # Use fork() if available, otherwise use the generator directly
        if hasattr(generator, 'fork'):
            noise_random = torch.rand(
                batch_size, num_tokens, 
                device=device, generator=generator.fork()
            )
        else:
            # Fallback: advance the generator state to maintain some randomness across calls
            # This isn't as robust as fork() but works for basic functionality
            _ = torch.rand(1, device=device, generator=generator)  # Advance state
            noise_random = torch.rand(
                batch_size, num_tokens, 
                device=device, generator=generator
            )
        
        # Get shuffle indices based on random noise
        ids_shuffle = torch.argsort(noise_random, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Select tokens to keep (route) and mask (skip)
        ids_keep = ids_shuffle[:, :num_keep]
        ids_mask = ids_shuffle[:, num_keep:]
        
        # Create boolean mask
        mask = torch.ones((batch_size, num_tokens), device=device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)  # False for tokens to keep/route
        
        return {
            'mask': mask,
            'ids_keep': ids_keep,
            'ids_mask': ids_mask, 
            'ids_shuffle': ids_shuffle,
            'ids_restore': ids_restore,
            'num_keep': num_keep,
            'num_mask': num_mask
        }
    
    def start_route(self, x: torch.Tensor, mask_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Begin token routing by selecting and shuffling tokens.
        
        Args:
            x: Input token embeddings (batch_size, sequence_length, hidden_dim)
            mask_info: Mask information from get_mask()
            
        Returns:
            Selected tokens for routing (batch_size, num_keep, hidden_dim)
        """
        ids_shuffle = mask_info['ids_shuffle']
        num_keep = mask_info['num_keep']
        
        # Shuffle tokens according to selection order
        x_shuffled = x.gather(
            1, ids_shuffle.unsqueeze(-1).expand(-1, -1, x.size(2))
        )
        
        # Keep only selected tokens for routing
        routed_tokens = x_shuffled[:, :num_keep, :]
        
        return routed_tokens
    
    def end_route(self, 
                  routed_tokens: torch.Tensor, 
                  mask_info: Dict[str, torch.Tensor], 
                  original_x: torch.Tensor) -> torch.Tensor:
        """
        Complete token routing by merging routed tokens back with original tokens.
        
        Args:
            routed_tokens: Processed tokens from routing (batch_size, num_keep, hidden_dim)  
            mask_info: Mask information from get_mask()
            original_x: Original token embeddings before routing
            
        Returns:
            Restored token embeddings (batch_size, sequence_length, hidden_dim)
        """
        batch_size, num_tokens = mask_info['mask'].shape
        num_keep = mask_info['num_keep']
        hidden_dim = routed_tokens.size(2)
        device = routed_tokens.device
        ids_restore = mask_info['ids_restore']
        ids_shuffle = mask_info['ids_shuffle']
        
        # Create output tensor
        x_unshuffled = torch.empty(
            (batch_size, num_tokens, hidden_dim), 
            device=device, dtype=routed_tokens.dtype
        )
        
        # Place routed tokens in first positions
        x_unshuffled[:, :num_keep, :] = routed_tokens
        
        # Get original tokens in shuffled order for unrouted positions
        x_shuffled = original_x.gather(
            1, ids_shuffle.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )
        
        # Place original unrouted tokens in remaining positions
        x_unshuffled[:, num_keep:, :] = x_shuffled[:, num_keep:, :]
        
        # Restore original token order
        x_restored = x_unshuffled.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )
        
        return x_restored
    
    def compute_routing_loss(self, 
                           routed_output: torch.Tensor,
                           original_output: torch.Tensor, 
                           mask_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute routing consistency loss to ensure routed and original outputs align.
        
        Args:
            routed_output: Output from routing path
            original_output: Output from original path  
            mask_info: Mask information from get_mask()
            
        Returns:
            Routing consistency loss
        """
        # Extract only the tokens that were routed for comparison
        ids_keep = mask_info['ids_keep']
        
        # Get routed tokens from original output for comparison
        original_routed = original_output.gather(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, original_output.size(2))
        )
        
        # Compute MSE loss between routed and original paths
        routing_loss = torch.nn.functional.mse_loss(
            routed_output, original_routed, reduction='mean'
        )
        
        return routing_loss


class RouterConfig:
    """Configuration container for routing parameters."""
    
    def __init__(self, 
                 selection_ratio: float = 0.5,
                 start_layer_idx: int = 2, 
                 end_layer_idx: int = 8,
                 routing_loss_weight: float = 0.0):
        """
        Args:
            selection_ratio: Fraction of tokens to route through expensive layers
            start_layer_idx: Layer index to start routing
            end_layer_idx: Layer index to end routing  
            routing_loss_weight: Weight for routing consistency loss
        """
        self.selection_ratio = selection_ratio
        self.start_layer_idx = start_layer_idx
        self.end_layer_idx = end_layer_idx
        self.routing_loss_weight = routing_loss_weight
        
        # Validation
        assert 0.0 <= selection_ratio <= 1.0, "selection_ratio must be in [0, 1]"
        assert start_layer_idx < end_layer_idx, "start_layer_idx must be < end_layer_idx"
        assert routing_loss_weight >= 0.0, "routing_loss_weight must be >= 0"