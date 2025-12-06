"""
Setup Script for Vision Transformer Project
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script helps set up the project environment and dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

# Author: RSK World
# Website: https://rskworld.in


def check_python_version():
    """
    Check if Python version is compatible
    
    Author: RSK World
    Website: https://rskworld.in
    """
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    print(f"Python version: {sys.version}")
    return True


def install_requirements():
    """
    Install project requirements
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Error installing requirements")
        return False


def create_directories():
    """
    Create necessary directories
    
    Author: RSK World
    Website: https://rskworld.in
    """
    directories = [
        'data/train',
        'data/val',
        'data/test',
        'models',
        'logs',
        'evaluation_results',
        'outputs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def check_cuda():
    """
    Check CUDA availability
    
    Author: RSK World
    Website: https://rskworld.in
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available!")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("CUDA is not available. Using CPU.")
            return False
    except ImportError:
        print("PyTorch not installed yet")
        return False


def verify_installation():
    """
    Verify installation
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("\nVerifying installation...")
    
    try:
        import torch
        import torchvision
        import numpy
        import matplotlib
        import PIL
        import sklearn
        import einops
        import yaml
        
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False


def main():
    """
    Main setup function
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print("="*60)
    print("Vision Transformer (ViT) Project Setup")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Install requirements
    print("\nInstalling requirements...")
    install_requirements()
    
    # Check CUDA
    print("\nChecking CUDA...")
    check_cuda()
    
    # Verify installation
    verify_installation()
    
    print("\n" + "="*60)
    print("Setup completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your data in data/train and data/val directories")
    print("2. Update config.yaml with your settings")
    print("3. Run: python train.py --config config.yaml")
    print("="*60)


if __name__ == '__main__':
    main()

