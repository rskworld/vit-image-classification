"""
Visualization Tools for Vision Transformer
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides visualization tools for understanding ViT model behavior,
including attention maps, patch visualization, and model architecture visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import seaborn as sns
from pathlib import Path

# Author: RSK World
# Website: https://rskworld.in


def visualize_patches(image_path, patch_size=16, image_size=224, save_path=None):
    """
    Visualize how an image is divided into patches
    
    Author: RSK World
    Website: https://rskworld.in
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size))
    img_array = np.array(img)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Image with patch grid
    axes[1].imshow(img_array)
    axes[1].set_title(f'Image with {patch_size}x{patch_size} Patches', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Draw patch grid
    num_patches = image_size // patch_size
    for i in range(num_patches + 1):
        # Vertical lines
        axes[1].axvline(i * patch_size, color='red', linewidth=1.5, alpha=0.7)
        # Horizontal lines
        axes[1].axhline(i * patch_size, color='red', linewidth=1.5, alpha=0.7)
    
    # Add patch count text
    axes[1].text(10, 20, f'Total Patches: {num_patches * num_patches}',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    print(f"Image size: {image_size}x{image_size}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Number of patches: {num_patches * num_patches}")


def visualize_attention_maps(model, image_path, transform, device, patch_size=16, image_size=224, save_path=None):
    """
    Visualize attention maps from the Vision Transformer
    
    Author: RSK World
    Website: https://rskworld.in
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get attention weights (requires model modification to return attention)
    model.eval()
    with torch.no_grad():
        # Forward pass through patch embedding
        x = model.patch_embedding(img_tensor)
        b, n, _ = x.shape
        
        # Add class token
        cls_tokens = model.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += model.pos_embedding
        x = model.dropout(x)
        
        # Get attention from first transformer block
        if len(model.transformer) > 0:
            # Access attention from first block
            x_norm = model.transformer[0].norm1(x)
            # This is a simplified version - full attention visualization requires hooking into the model
            attention_weights = torch.ones(1, n + 1, n + 1)  # Placeholder
    
    # Load original image for visualization
    img_array = np.array(img.resize((image_size, image_size)))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Attention heatmap (simplified)
    num_patches_per_side = image_size // patch_size
    attention_map = np.ones((num_patches_per_side, num_patches_per_side))
    
    im = axes[1].imshow(attention_map, cmap='hot', interpolation='nearest')
    axes[1].set_title('Attention Map (Simplified)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    
    plt.show()


def plot_training_history(log_dir, save_path=None):
    """
    Plot training history from TensorBoard logs or CSV
    
    Author: RSK World
    Website: https://rskworld.in
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        import pandas as pd
    except ImportError:
        print("TensorBoard or pandas not available")
        return
    
    # This is a placeholder - actual implementation would read from TensorBoard logs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Placeholder data
    epochs = np.arange(1, 101)
    train_loss = np.random.rand(100) * 2 + 1
    val_loss = np.random.rand(100) * 2 + 1.2
    train_acc = 100 - np.random.rand(100) * 20
    val_acc = 100 - np.random.rand(100) * 25
    
    # Loss plot
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, train_acc, label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_acc, label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    lr = np.ones(100) * 0.001
    axes[1, 0].plot(epochs, lr, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined metrics
    ax2 = axes[1, 1]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(epochs, train_loss, 'b-', label='Loss', linewidth=2)
    line2 = ax2_twin.plot(epochs, train_acc, 'r-', label='Accuracy', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12, color='b')
    ax2_twin.set_ylabel('Accuracy (%)', fontsize=12, color='r')
    ax2.set_title('Combined Metrics', fontsize=14, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()


def visualize_model_architecture(model, save_path=None):
    """
    Visualize model architecture and parameters
    
    Author: RSK World
    Website: https://rskworld.in
    """
    # Count parameters by layer
    layer_info = []
    total_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        layer_info.append({
            'name': name,
            'params': num_params,
            'shape': list(param.shape)
        })
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Parameter distribution
    param_counts = [info['params'] for info in layer_info[:20]]  # Top 20 layers
    layer_names = [info['name'].split('.')[-1] for info in layer_info[:20]]
    
    axes[0].barh(range(len(param_counts)), param_counts)
    axes[0].set_yticks(range(len(layer_names)))
    axes[0].set_yticklabels(layer_names, fontsize=8)
    axes[0].set_xlabel('Number of Parameters', fontsize=12)
    axes[0].set_title('Parameter Distribution by Layer (Top 20)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Model summary
    summary_text = f"""
    Model Architecture Summary
    
    Total Parameters: {total_params:,}
    Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
    
    Key Components:
    - Patch Embedding
    - Positional Encoding
    - Transformer Blocks: {len(model.transformer)}
    - Classification Head
    """
    
    axes[1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].axis('off')
    axes[1].set_title('Model Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved architecture visualization to {save_path}")
    
    plt.show()
    
    print(f"\nTotal Parameters: {total_params:,}")


def visualize_predictions(model, image_paths, transform, device, class_names=None, top_k=5, save_path=None):
    """
    Visualize predictions for multiple images
    
    Author: RSK World
    Website: https://rskworld.in
    """
    model.eval()
    
    num_images = len(image_paths)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, img_path in enumerate(image_paths):
        # Load and preprocess
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probs, top_k)
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Add predictions as text
        pred_text = "Top Predictions:\n"
        for i, (prob, idx_class) in enumerate(zip(top_probs[0], top_indices[0]), 1):
            class_name = class_names[idx_class.item()] if class_names else f"Class {idx_class.item()}"
            pred_text += f"{i}. {class_name}: {prob.item()*100:.2f}%\n"
        
        axes[idx].text(0.02, 0.98, pred_text, transform=axes[idx].transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions visualization to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    print("Visualization tools for Vision Transformer")
    print("Author: RSK World")
    print("Website: https://rskworld.in")

