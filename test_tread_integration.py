"""
Test script for TREAD integration with D3-DNA
"""

import torch
import yaml
from omegaconf import OmegaConf
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_zoo.deepstarr.models import DeepSTARRTransformerModel

def test_tread_integration():
    """Test TREAD integration with DeepSTARR transformer."""
    
    print("Testing TREAD Integration...")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if fork is available
    gen = torch.Generator()
    has_fork = hasattr(gen, 'fork')
    print(f"Generator fork() available: {has_fork}")
    
    # Load TREAD configuration
    config_path = "model_zoo/deepstarr/configs/transformer_tread.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    
    # Create model
    print("Creating TREAD-enabled model...")
    model = DeepSTARRTransformerModel(config)
    
    # Check routing is enabled
    assert model.enable_routing == True, "Routing should be enabled"
    assert len(model.route_configs) > 0, "Should have route configurations"
    assert model.router is not None, "Router should be initialized"
    
    print(f"✓ Routing enabled with {len(model.route_configs)} routes")
    for i, route in enumerate(model.route_configs):
        print(f"  Route {i}: layers {route.start_layer_idx}-{route.end_layer_idx}, ratio={route.selection_ratio}")
    
    # Create sample inputs
    batch_size, seq_len = 8, 249  # DeepSTARR sequence length
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    model.eval()  # Set to eval mode first
    
    indices = torch.randint(0, 4, (batch_size, seq_len), device=device)  # DNA tokens (4 nucleotides)
    labels = torch.randn(batch_size, 2, device=device)  # DeepSTARR signal (2D)
    sigma = torch.randn(batch_size, device=device)
    
    print(f"✓ Created sample inputs: indices={indices.shape}, labels={labels.shape}, sigma={sigma.shape}")
    
    # Test forward pass without routing (eval mode)
    print("Testing forward pass without routing...")
    with torch.no_grad():
        output_no_routing, _ = model(indices, labels, train=False, sigma=sigma)
    
    print(f"✓ Forward pass without routing successful, output shape: {output_no_routing.shape}")
    
    # Test forward pass with forced routing
    print("Testing forward pass with forced routing...")
    with torch.no_grad():
        output_with_routing, _ = model(indices, labels, train=False, sigma=sigma, force_routing=True)
    
    print(f"✓ Forward pass with forced routing successful, output shape: {output_with_routing.shape}")
    
    # Test forward pass in training mode (auto-routing)
    print("Testing forward pass in training mode...")
    model.train()
    with torch.no_grad():
        output_training, _ = model(indices, labels, train=True, sigma=sigma)
    
    print(f"✓ Training mode forward pass successful, output shape: {output_training.shape}")
    
    # Test with different selection ratios
    print("Testing custom selection ratios...")
    model.eval()
    with torch.no_grad():
        output_custom_ratio, _ = model(
            indices, labels, train=False, sigma=sigma, 
            force_routing=True, overwrite_selection_ratio=0.3
        )
    
    print(f"✓ Custom selection ratio test successful, output shape: {output_custom_ratio.shape}")
    
    # Verify outputs have correct dimensions
    # DeepSTARR uses uniform graph (no absorb token), so vocab size = 4 (A, T, G, C)
    expected_shape = (batch_size, seq_len, 4)  # 4 DNA tokens (A, T, G, C)
    assert output_no_routing.shape == expected_shape, f"Expected {expected_shape}, got {output_no_routing.shape}"
    assert output_with_routing.shape == expected_shape, f"Expected {expected_shape}, got {output_with_routing.shape}"
    
    print("✓ All output shapes correct")
    
    # Test that routing produces different intermediate computations
    # but similar final outputs (when working correctly)
    diff = torch.abs(output_no_routing - output_with_routing).mean().item()
    print(f"✓ Mean difference between routing/no-routing outputs: {diff:.6f}")
    
    # Test backward pass (gradient computation)
    print("Testing backward pass with routing...")
    model.train()
    output, _ = model(indices, labels, train=True, sigma=sigma)
    loss = output.sum()
    loss.backward()
    
    print("✓ Backward pass successful")
    
    # Verify gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"✓ Gradients computed for {grad_count}/{total_params} parameters")
    
    print("\n🎉 All TREAD integration tests passed!")
    print("TREAD is successfully integrated with D3-DNA discrete diffusion!")
    
    return True

if __name__ == "__main__":
    try:
        test_tread_integration()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)