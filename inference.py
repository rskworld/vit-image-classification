"""
Inference Script for Vision Transformer (ViT)
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script performs inference on images using a trained ViT model.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import numpy as np
from pathlib import Path

from vit_model import create_vit_model
from utils import load_config, get_data_transforms


def load_model(model_path, config, device):
    """
    Load trained model
    
    Author: RSK World
    Website: https://rskworld.in
    """
    model = create_vit_model(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_image(model, image_path, transform, device, top_k=5):
    """
    Predict class for a single image
    
    Author: RSK World
    Website: https://rskworld.in
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
    
    return top_probs.cpu().numpy()[0], top_indices.cpu().numpy()[0]


def main():
    """
    Main inference function
    
    Author: RSK World
    Website: https://rskworld.in
    """
    parser = argparse.ArgumentParser(description='ViT Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to show')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, config, device)
    print("Model loaded successfully!")
    
    # Get transform
    transform = get_data_transforms(config, is_train=False)
    
    # Predict
    print(f"Predicting on image: {args.image_path}")
    probs, indices = predict_image(model, args.image_path, transform, device, args.top_k)
    
    # Display results
    print("\nTop Predictions:")
    print("-" * 50)
    for i, (prob, idx) in enumerate(zip(probs, indices), 1):
        print(f"{i}. Class {idx}: {prob * 100:.2f}%")
    print("-" * 50)


if __name__ == '__main__':
    main()

