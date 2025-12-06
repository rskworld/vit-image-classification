"""
Evaluation and Testing Script for Vision Transformer
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script provides comprehensive evaluation metrics and testing utilities.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path

from vit_model import create_vit_model
from utils import load_config, get_data_loaders, ImageDataset, get_data_transforms

# Author: RSK World
# Website: https://rskworld.in


def evaluate_model(model, data_loader, device, class_names=None):
    """
    Evaluate model on a dataset
    
    Author: RSK World
    Website: https://rskworld.in
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'class_names': class_names
    }


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix
    
    Author: RSK World
    Website: https://rskworld.in
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else range(len(np.unique(y_true))),
                yticklabels=class_names if class_names else range(len(np.unique(y_true))),
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def calculate_metrics(y_true, y_pred, y_probs, class_names=None):
    """
    Calculate comprehensive evaluation metrics
    
    Author: RSK World
    Website: https://rskworld.in
    """
    # Classification report
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   output_dict=True)
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    metrics = {
        'overall_accuracy': accuracy_score(y_true, y_pred),
        'per_class_accuracy': dict(zip(
            class_names if class_names else range(len(per_class_accuracy)),
            per_class_accuracy
        )),
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    return metrics


def plot_class_accuracy(metrics, class_names=None, save_path=None):
    """
    Plot per-class accuracy
    
    Author: RSK World
    Website: https://rskworld.in
    """
    per_class_acc = metrics['per_class_accuracy']
    classes = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), accuracies, color='steelblue', alpha=0.7)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class accuracy plot to {save_path}")
    
    plt.show()


def evaluate_top_k_accuracy(model, data_loader, device, k_values=[1, 3, 5]):
    """
    Calculate top-k accuracy
    
    Author: RSK World
    Website: https://rskworld.in
    """
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Calculating top-k accuracy"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    top_k_accuracies = {}
    for k in k_values:
        _, top_k_preds = torch.topk(all_outputs, k, dim=1)
        correct = top_k_preds.eq(all_labels.view(-1, 1).expand_as(top_k_preds))
        top_k_acc = correct.any(dim=1).float().mean().item()
        top_k_accuracies[f'top_{k}'] = top_k_acc * 100
    
    return top_k_accuracies


def main():
    """
    Main evaluation function
    
    Author: RSK World
    Website: https://rskworld.in
    """
    parser = argparse.ArgumentParser(description='Evaluate Vision Transformer Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Path to test/validation data')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = create_vit_model(config).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Get data loader
    if args.data_dir:
        data_dir = Path(args.data_dir) / args.split
        transform = get_data_transforms(config, is_train=False)
        dataset = ImageDataset(data_dir, transform=transform)
        data_loader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers']
        )
    else:
        _, data_loader = get_data_loaders(config)
        if args.split == 'train':
            data_loader, _ = get_data_loaders(config)
        elif args.split == 'val':
            _, data_loader = get_data_loaders(config)
    
    # Get class names (if available)
    class_names = None
    if args.data_dir:
        class_dirs = sorted([d.name for d in Path(args.data_dir).iterdir() if d.is_dir()])
        class_names = class_dirs if class_dirs else None
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, data_loader, device, class_names)
    
    # Calculate metrics
    metrics = calculate_metrics(
        results['labels'],
        results['predictions'],
        results['probabilities'],
        class_names
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"\nPer-Class Accuracy:")
    for class_name, acc in metrics['per_class_accuracy'].items():
        print(f"  {class_name}: {acc*100:.2f}%")
    
    # Top-k accuracy
    print("\nCalculating top-k accuracy...")
    top_k_results = evaluate_top_k_accuracy(model, data_loader, device)
    print("Top-K Accuracy:")
    for k, acc in top_k_results.items():
        print(f"  {k}: {acc:.2f}%")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to file
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'overall_accuracy': metrics['overall_accuracy'],
            'per_class_accuracy': metrics['per_class_accuracy'],
            'top_k_accuracy': top_k_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            class_names,
            save_path=output_dir / 'confusion_matrix.png'
        )
        plot_class_accuracy(
            metrics,
            class_names,
            save_path=output_dir / 'class_accuracy.png'
        )
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()

