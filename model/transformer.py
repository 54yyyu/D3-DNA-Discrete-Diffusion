"""
Pure Transformer Architecture for D3-DNA Discrete Diffusion

This module contains the core transformer implementation that is completely
dataset-agnostic. All dataset-specific parameters are passed via configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from omegaconf import OmegaConf, DictConfig
import math

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

from . import rotary
from .layers import (
    LayerNorm, RMSNorm, TimestepEmbedder, LabelEmbedder, SwiGLUFFN, #EmbeddingLayer,
    modulate_fused, get_bias_dropout_scale
)


@torch.compile
def modulate(x, shift, scale):
    """Compiled modulation function for performance."""
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """
    Enhanced attention module with QK normalization support.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        
        # Choose normalization type
        norm_layer = RMSNorm if use_rmsnorm else LayerNorm
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor], 
                seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        
        # Apply QK normalization if enabled
        if not isinstance(self.q_norm, nn.Identity):
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q = self.q_norm(q)
            k = self.k_norm(k)
            qkv = torch.stack([q, k, v], dim=2)
        
        # Apply rotary position embedding
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        
        # Prepare sequence lengths for flash attention
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=x.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
            
        # Flash attention
        x = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0., causal=False)
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        
        # Output projection
        x = self.out_proj(x)
        x = self.dropout_layer(x)
        
        return x


class DDiTBlock(nn.Module):
    """
    Enhanced Diffusion Transformer Block with LightningDiT improvements.
    
    Supports:
    - QK normalization for better training stability
    - RMSNorm for efficiency
    - SwiGLU MLP for improved performance
    - No-shift AdaLN option
    """
    
    def __init__(self, dim: int, n_heads: int, cond_dim: int, 
                 mlp_ratio: int = 4, dropout: float = 0.1,
                 use_qknorm: bool = False, use_swiglu: bool = False, 
                 use_rmsnorm: bool = False, wo_shift: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.wo_shift = wo_shift

        # Choose normalization type
        norm_layer = RMSNorm if use_rmsnorm else LayerNorm

        # Attention layers
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=n_heads,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            dropout=dropout
        )

        # Feed-forward layers
        self.norm2 = norm_layer(dim)
        if use_swiglu:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_ratio * dim,
                out_features=dim
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_ratio * dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_ratio * dim, dim, bias=True)
            )

        # Adaptive layer normalization
        modulation_dim = 4 * dim if wo_shift else 6 * dim
        self.adaLN_modulation = nn.Linear(cond_dim, modulation_dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return get_bias_dropout_scale()(self.training)

    def forward(self, x: torch.Tensor, rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor], 
                c: torch.Tensor, seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_length, dim)
            rotary_cos_sin: Rotary position encoding (cos, sin)
            c: Conditioning tensor (batch_size, cond_dim)
            seqlens: Optional sequence lengths for variable-length attention
            
        Returns:
            Output tensor (batch_size, seq_length, dim)
        """
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # Adaptive layer normalization modulation
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c)[:, None].chunk(4, dim=2)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Multi-head self-attention
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        x = self.attn(x, rotary_cos_sin, seqlens)

        # Apply attention output with gating
        x = bias_dropout_scale_fn(x, None, gate_msa, x_skip, self.dropout)

        # Feed-forward network with gating
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), 
            None, gate_mlp, x, self.dropout
        )
        
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim, signal_dim=2):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        #remove if label embedding is used
        self.signal_embedding = nn.Linear(signal_dim, dim)
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x, y):
        vocab_embed = self.embedding[x] #return only this if label embedding is used
        if y is not None:
            signal_embed = self.signal_embedding(y.to(torch.float32))
            return torch.add(vocab_embed, signal_embed[:, None, :]) #[:, None, :] extra for deepstarr
        else:
            # For unconditional generation, return only vocab embedding
            return vocab_embed


class DDitFinalLayer(nn.Module):
    """Final output layer for the transformer."""
    
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int, use_rmsnorm: bool = False):
        super().__init__()
        norm_layer = RMSNorm if use_rmsnorm else LayerNorm
        self.norm_final = norm_layer(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    @torch.compile
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TransformerModel(nn.Module):
    """
    Pure Transformer SEDD Model.
    
    This implementation is completely dataset-agnostic. All dataset-specific
    parameters (num_classes, sequence_length) are passed via config.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        if isinstance(config, dict):
            config = OmegaConf.create(config)
            
        self.config = config
        
        # Extract dataset-agnostic parameters from config
        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)
        
        # These should be provided by dataset-specific config
        num_classes = config.dataset.num_classes
        class_dropout_prob = getattr(config.model, 'class_dropout_prob', 0.1)
        
        # LightningDiT enhancement options
        use_qknorm = getattr(config.model, 'use_qknorm', False)
        use_swiglu = getattr(config.model, 'use_swiglu', False)
        use_rmsnorm = getattr(config.model, 'use_rmsnorm', False)
        wo_shift = getattr(config.model, 'wo_shift', False)
        
        # Core components
        self.vocab_embed = EmbeddingLayer(
            dim=config.model.hidden_size, 
            vocab_dim=vocab_size,
            signal_dim=config.dataset.signal_dim,
        )

        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.label_embed = LabelEmbedder(num_classes, config.model.cond_dim, class_dropout_prob)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DDiTBlock(
                dim=config.model.hidden_size, 
                n_heads=config.model.n_heads, 
                cond_dim=config.model.cond_dim,
                dropout=config.model.dropout,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift
            ) 
            for _ in range(config.model.n_blocks)
        ])
        
        # Output layer
        self.output_layer = DDitFinalLayer(
            hidden_size=config.model.hidden_size, 
            out_channels=vocab_size, 
            cond_dim=config.model.cond_dim,
            use_rmsnorm=use_rmsnorm
        )
        
        # Model configuration
        self.scale_by_sigma = getattr(config.model, 'scale_by_sigma', False)

    def forward(self, indices: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                train: bool = True, sigma: Optional[torch.Tensor] = None, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            indices: Token indices (batch_size, seq_length)
            labels: Label/signal tensor (batch_size, signal_dim) or None for unconditional
            train: Training mode flag
            sigma: Noise level (batch_size,)
            layer_idx: Index of the layer to return the representation of
        Returns:
            Model output (batch_size, seq_length, vocab_size)
        """
        # Embedding
        x = self.vocab_embed(indices, labels)
        
        # Conditioning
        c = F.silu(self.sigma_map(sigma))
        # Note: label_embed could be added here if needed: + self.label_embed(labels, train)
        
        # Rotary position encoding
        rotary_cos_sin = self.rotary_emb(x)

        # Forward through transformer blocks
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
                if layer_idx is not None:
                    if i == layer_idx:
                        rep = x
                else:
                    rep = None
            x = self.output_layer(x, c)

        # Mask out the input tokens (standard diffusion technique)
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        
        return x, rep


def create_transformer_model(config: DictConfig) -> TransformerModel:
    """
    Factory function to create a transformer model.
    
    Args:
        config: Configuration containing model and dataset parameters
        
    Returns:
        Transformer model instance
    """
    return TransformerModel(config)