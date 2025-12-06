"""
Model Testing Script
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script provides unit tests and validation for the Vision Transformer model.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from vit_model import VisionTransformer, PatchEmbedding, MultiHeadAttention, TransformerBlock

# Author: RSK World
# Website: https://rskworld.in


def test_patch_embedding():
    """
    Test Patch Embedding layer
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("Testing Patch Embedding...")
    patch_embed = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
    x = torch.randn(2, 3, 224, 224)
    out = patch_embed(x)
    assert out.shape == (2, 196, 768), f"Expected (2, 196, 768), got {out.shape}"
    print("✓ Patch Embedding test passed")


def test_multi_head_attention():
    """
    Test Multi-Head Attention
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("Testing Multi-Head Attention...")
    attn = MultiHeadAttention(dim=768, heads=12, dropout=0.1)
    x = torch.randn(2, 197, 768)  # batch, seq_len, dim
    out = attn(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("✓ Multi-Head Attention test passed")


def test_transformer_block():
    """
    Test Transformer Block
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("Testing Transformer Block...")
    block = TransformerBlock(dim=768, heads=12, mlp_dim=3072, dropout=0.1)
    x = torch.randn(2, 197, 768)
    out = block(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("✓ Transformer Block test passed")


def test_vit_model():
    """
    Test complete ViT model
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("Testing Vision Transformer Model...")
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072
    )
    
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1000), f"Expected (2, 1000), got {out.shape}"
    print("✓ Vision Transformer test passed")


def test_model_gradients():
    """
    Test model gradient flow
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("Testing Model Gradients...")
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=1536
    )
    
    x = torch.randn(1, 3, 224, 224)
    y = torch.randint(0, 10, (1,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Forward pass
    output = model(x)
    loss = criterion(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check gradients
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients, "No gradients found"
    print("✓ Gradient flow test passed")


def test_different_configurations():
    """
    Test different model configurations
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("Testing Different Configurations...")
    
    configs = [
        {'image_size': 224, 'patch_size': 16, 'num_classes': 10, 'dim': 192, 'depth': 6, 'heads': 3, 'mlp_dim': 768},
        {'image_size': 224, 'patch_size': 32, 'num_classes': 100, 'dim': 384, 'depth': 12, 'heads': 6, 'mlp_dim': 1536},
        {'image_size': 384, 'patch_size': 16, 'num_classes': 1000, 'dim': 768, 'depth': 12, 'heads': 12, 'mlp_dim': 3072},
    ]
    
    for i, config in enumerate(configs):
        model = VisionTransformer(**config)
        x = torch.randn(1, 3, config['image_size'], config['image_size'])
        out = model(x)
        assert out.shape[1] == config['num_classes'], f"Config {i} failed"
        print(f"✓ Configuration {i+1} test passed")
    
    print("✓ All configurations test passed")


def test_model_parameters():
    """
    Test model parameter counting
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("Testing Model Parameters...")
    
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0, "Model has no parameters"
    assert trainable_params == total_params, "Some parameters are not trainable"
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print("✓ Parameter counting test passed")


def run_all_tests():
    """
    Run all tests
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("="*60)
    print("Running Vision Transformer Model Tests")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("="*60)
    
    tests = [
        test_patch_embedding,
        test_multi_head_attention,
        test_transformer_block,
        test_vit_model,
        test_model_gradients,
        test_different_configurations,
        test_model_parameters
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

