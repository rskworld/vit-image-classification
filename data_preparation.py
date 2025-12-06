"""
Data Preparation and Download Utilities
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script helps prepare and organize data for training the Vision Transformer model.
"""

import os
import shutil
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
import random

# Author: RSK World
# Website: https://rskworld.in


def create_directory_structure(base_dir='data'):
    """
    Create standard directory structure for image classification
    
    Author: RSK World
    Website: https://rskworld.in
    """
    base_path = Path(base_dir)
    train_path = base_path / 'train'
    val_path = base_path / 'val'
    test_path = base_path / 'test'
    
    for path in [train_path, val_path, test_path]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    
    return train_path, val_path, test_path


def split_data(source_dir, train_dir, val_dir, test_dir=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train, validation, and test sets
    
    Author: RSK World
    Website: https://rskworld.in
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist!")
        return
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(class_dirs)} classes")
    
    for class_dir in tqdm(class_dirs, desc="Splitting classes"):
        class_name = class_dir.name
        
        # Get all image files
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.png')) + \
                     list(class_dir.glob('*.jpeg'))
        
        if len(image_files) == 0:
            print(f"No images found in {class_dir}")
            continue
        
        # Shuffle
        random.shuffle(image_files)
        
        # Split
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:] if test_dir else []
        
        # Create class directories
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        if test_dir:
            (test_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for file in train_files:
            shutil.copy2(file, train_dir / class_name / file.name)
        
        for file in val_files:
            shutil.copy2(file, val_dir / class_name / file.name)
        
        if test_dir:
            for file in test_files:
                shutil.copy2(file, test_dir / class_name / file.name)
        
        print(f"{class_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")


def download_file(url, destination, chunk_size=8192):
    """
    Download a file with progress bar
    
    Author: RSK World
    Website: https://rskworld.in
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    """
    Extract zip file
    
    Author: RSK World
    Website: https://rskworld.in
    """
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def organize_data_by_class(source_dir, target_dir):
    """
    Organize images into class folders based on filename patterns or subdirectories
    
    Author: RSK World
    Website: https://rskworld.in
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # If source already has class folders, just copy structure
    if any(d.is_dir() for d in source_path.iterdir()):
        print("Source directory already has class structure, copying...")
        for class_dir in source_path.iterdir():
            if class_dir.is_dir():
                shutil.copytree(class_dir, target_path / class_dir.name, dirs_exist_ok=True)
    else:
        # Organize by filename pattern (e.g., class_image.jpg)
        print("Organizing images by filename pattern...")
        image_files = list(source_path.glob('*.jpg')) + \
                     list(source_path.glob('*.png')) + \
                     list(source_path.glob('*.jpeg'))
        
        for img_file in tqdm(image_files, desc="Organizing"):
            # Extract class name from filename (modify pattern as needed)
            # Example: class1_image001.jpg -> class1
            parts = img_file.stem.split('_')
            if len(parts) > 1:
                class_name = parts[0]
            else:
                class_name = 'unknown'
            
            class_dir = target_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, class_dir / img_file.name)


def get_data_statistics(data_dir):
    """
    Get statistics about the dataset
    
    Author: RSK World
    Website: https://rskworld.in
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist!")
        return
    
    stats = {
        'total_classes': 0,
        'total_images': 0,
        'class_distribution': {}
    }
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            stats['total_classes'] += 1
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpeg'))
            num_images = len(image_files)
            stats['total_images'] += num_images
            stats['class_distribution'][class_dir.name] = num_images
    
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    print(f"Total Classes: {stats['total_classes']}")
    print(f"Total Images: {stats['total_images']}")
    print(f"\nClass Distribution:")
    for class_name, count in sorted(stats['class_distribution'].items()):
        print(f"  {class_name}: {count} images")
    print("="*50)
    
    return stats


def main():
    """
    Main function for data preparation
    
    Author: RSK World
    Website: https://rskworld.in
    """
    parser = argparse.ArgumentParser(description='Data Preparation Utilities')
    parser.add_argument('--action', type=str, required=True,
                       choices=['create_structure', 'split', 'organize', 'stats'],
                       help='Action to perform')
    parser.add_argument('--source_dir', type=str, help='Source directory')
    parser.add_argument('--target_dir', type=str, default='data', help='Target directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    
    args = parser.parse_args()
    
    if args.action == 'create_structure':
        create_directory_structure(args.target_dir)
    
    elif args.action == 'split':
        if not args.source_dir:
            print("Error: --source_dir is required for split action")
            return
        
        train_dir, val_dir, test_dir = create_directory_structure(args.target_dir)
        split_data(
            args.source_dir,
            train_dir,
            val_dir,
            test_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
    
    elif args.action == 'organize':
        if not args.source_dir:
            print("Error: --source_dir is required for organize action")
            return
        organize_data_by_class(args.source_dir, args.target_dir)
    
    elif args.action == 'stats':
        get_data_statistics(args.target_dir if args.target_dir else args.source_dir)


if __name__ == '__main__':
    main()

