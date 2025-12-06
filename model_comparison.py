"""
Model Comparison and Analysis Tools
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script provides tools for comparing different model configurations and analyzing performance.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm

from vit_model import VisionTransformer, create_vit_model
from utils import load_config, get_data_loaders, accuracy

# Author: RSK World
# Website: https://rskworld.in


def count_parameters(model):
    """
    Count model parameters
    
    Author: RSK World
    Website: https://rskworld.in
    """
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(model, input_shape, device, num_runs=100):
    """
    Measure inference time
    
    Author: RSK World
    Website: https://rskworld.in
    """
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time * 1000  # Return in milliseconds


def evaluate_model_performance(model, data_loader, device):
    """
    Evaluate model performance metrics
    
    Author: RSK World
    Website: https://rskworld.in
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def compare_models(model_configs, data_loader, device, input_shape=(1, 3, 224, 224)):
    """
    Compare multiple model configurations
    
    Author: RSK World
    Website: https://rskworld.in
    """
    results = []
    
    for config_name, model_config in model_configs.items():
        print(f"\nEvaluating {config_name}...")
        
        # Create model
        model = VisionTransformer(**model_config).to(device)
        
        # Count parameters
        num_params = count_parameters(model)
        
        # Measure inference time
        inference_time = measure_inference_time(model, input_shape, device)
        
        # Evaluate accuracy (if data available)
        if data_loader:
            acc = evaluate_model_performance(model, data_loader, device)
        else:
            acc = None
        
        results.append({
            'model': config_name,
            'parameters': num_params,
            'inference_time_ms': inference_time,
            'accuracy': acc
        })
        
        # Clean up
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return pd.DataFrame(results)


def plot_model_comparison(results_df, save_path=None):
    """
    Plot model comparison results
    
    Author: RSK World
    Website: https://rskworld.in
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Parameters comparison
    axes[0, 0].bar(results_df['model'], results_df['parameters'] / 1e6)
    axes[0, 0].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[0, 0].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Inference time comparison
    axes[0, 1].bar(results_df['model'], results_df['inference_time_ms'])
    axes[0, 1].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[0, 1].set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Accuracy comparison (if available)
    if results_df['accuracy'].notna().any():
        axes[1, 0].bar(results_df['model'], results_df['accuracy'])
        axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Parameters vs Inference Time
    scatter = axes[1, 1].scatter(results_df['parameters'] / 1e6, 
                                results_df['inference_time_ms'],
                                s=200, alpha=0.6)
    axes[1, 1].set_xlabel('Parameters (Millions)', fontsize=12)
    axes[1, 1].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[1, 1].set_title('Parameters vs Inference Time', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add labels
    for idx, row in results_df.iterrows():
        axes[1, 1].annotate(row['model'], 
                           (row['parameters'] / 1e6, row['inference_time_ms']),
                           fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


def analyze_model_complexity(model_configs):
    """
    Analyze model complexity metrics
    
    Author: RSK World
    Website: https://rskworld.in
    """
    analysis = {}
    
    for config_name, config in model_configs.items():
        # Calculate complexity metrics
        num_patches = (config['image_size'] // config['patch_size']) ** 2
        total_attention_heads = config['depth'] * config['heads']
        total_mlp_params = config['depth'] * config['mlp_dim'] * config['dim']
        
        analysis[config_name] = {
            'num_patches': num_patches,
            'total_attention_heads': total_attention_heads,
            'estimated_mlp_params': total_mlp_params,
            'depth': config['depth'],
            'embedding_dim': config['dim']
        }
    
    return pd.DataFrame(analysis).T


def main():
    """
    Main comparison function
    
    Author: RSK World
    Website: https://rskworld.in
    """
    # Define model configurations to compare
    model_configs = {
        'ViT-Tiny': {
            'image_size': 224,
            'patch_size': 16,
            'num_classes': 1000,
            'dim': 192,
            'depth': 6,
            'heads': 3,
            'mlp_dim': 768,
            'dropout': 0.1,
            'emb_dropout': 0.1
        },
        'ViT-Small': {
            'image_size': 224,
            'patch_size': 16,
            'num_classes': 1000,
            'dim': 384,
            'depth': 12,
            'heads': 6,
            'mlp_dim': 1536,
            'dropout': 0.1,
            'emb_dropout': 0.1
        },
        'ViT-Base': {
            'image_size': 224,
            'patch_size': 16,
            'num_classes': 1000,
            'dim': 768,
            'depth': 12,
            'heads': 12,
            'mlp_dim': 3072,
            'dropout': 0.1,
            'emb_dropout': 0.1
        },
        'ViT-Large': {
            'image_size': 224,
            'patch_size': 16,
            'num_classes': 1000,
            'dim': 1024,
            'depth': 24,
            'heads': 16,
            'mlp_dim': 4096,
            'dropout': 0.1,
            'emb_dropout': 0.1
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Compare models
    print("Comparing model configurations...")
    results = compare_models(model_configs, None, device)
    
    # Display results
    print("\n" + "="*60)
    print("Model Comparison Results")
    print("="*60)
    print(results.to_string(index=False))
    
    # Analyze complexity
    print("\n" + "="*60)
    print("Model Complexity Analysis")
    print("="*60)
    complexity = analyze_model_complexity(model_configs)
    print(complexity.to_string())
    
    # Plot comparison
    plot_model_comparison(results, save_path='model_comparison.png')
    
    # Save results
    results.to_csv('model_comparison_results.csv', index=False)
    complexity.to_csv('model_complexity_analysis.csv')
    print("\nResults saved to CSV files")


if __name__ == '__main__':
    main()

