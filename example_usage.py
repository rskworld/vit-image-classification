"""
Example Usage Script for Vision Transformer (ViT)
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script demonstrates basic usage of the ViT model.
"""

import torch
from vit_model import VisionTransformer

# Author: RSK World
# Website: https://rskworld.in

def example_model_creation():
    """
    Example: Create a ViT model
    Author: RSK World
    Website: https://rskworld.in
    """
    print("Creating Vision Transformer model...")
    
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def example_forward_pass(model):
    """
    Example: Forward pass through the model
    Author: RSK World
    Website: https://rskworld.in
    """
    print("\nPerforming forward pass...")
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output represents logits for {output.shape[1]} classes")
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top5_probs, top5_indices = torch.topk(probabilities, 5)
    
    print("\nTop 5 Predictions:")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0]), 1):
        print(f"  {i}. Class {idx.item()}: {prob.item() * 100:.2f}%")
    
    return output


def example_different_configs():
    """
    Example: Create models with different configurations
    Author: RSK World
    Website: https://rskworld.in
    """
    print("\n" + "="*50)
    print("Creating models with different configurations...")
    
    configs = [
        {
            'name': 'ViT-Small',
            'params': {
                'image_size': 224,
                'patch_size': 16,
                'num_classes': 10,
                'dim': 384,
                'depth': 6,
                'heads': 6,
                'mlp_dim': 1536,
            }
        },
        {
            'name': 'ViT-Base',
            'params': {
                'image_size': 224,
                'patch_size': 16,
                'num_classes': 100,
                'dim': 768,
                'depth': 12,
                'heads': 12,
                'mlp_dim': 3072,
            }
        },
        {
            'name': 'ViT-Large',
            'params': {
                'image_size': 224,
                'patch_size': 16,
                'num_classes': 1000,
                'dim': 1024,
                'depth': 24,
                'heads': 16,
                'mlp_dim': 4096,
            }
        }
    ]
    
    for config in configs:
        model = VisionTransformer(**config['params'])
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n{config['name']}:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Config: {config['params']}")


if __name__ == '__main__':
    # Author: RSK World
    # Website: https://rskworld.in
    
    print("="*50)
    print("Vision Transformer (ViT) - Example Usage")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("="*50)
    
    # Example 1: Create model
    model = example_model_creation()
    
    # Example 2: Forward pass
    output = example_forward_pass(model)
    
    # Example 3: Different configurations
    example_different_configs()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("="*50)

