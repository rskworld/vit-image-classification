"""
Utility Functions for Vision Transformer Training
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module contains utility functions for data loading, augmentation, and training helpers.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import yaml
import numpy as np
from pathlib import Path


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    
    Author: RSK World
    Website: https://rskworld.in
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_data_transforms(config, is_train=True):
    """
    Get data augmentation transforms
    
    Author: RSK World
    Website: https://rskworld.in
    """
    aug_config = config['augmentation']
    normalize = aug_config['normalize']
    
    if is_train:
        transform_list = [
            transforms.Resize((config['model']['image_size'], config['model']['image_size'])),
        ]
        
        if aug_config.get('random_crop', False):
            transform_list.append(transforms.RandomCrop(config['model']['image_size'], padding=4))
        
        if aug_config.get('random_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if aug_config.get('color_jitter', False):
            transform_list.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
        ])
    else:
        transform_list = [
            transforms.Resize((config['model']['image_size'], config['model']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
        ]
    
    return transforms.Compose(transform_list)


class ImageDataset(Dataset):
    """
    Custom Dataset for Image Classification
    
    Author: RSK World
    Website: https://rskworld.in
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images and labels
        if self.data_dir.exists():
            for class_idx, class_name in enumerate(sorted(os.listdir(self.data_dir))):
                class_dir = self.data_dir / class_name
                if class_dir.is_dir():
                    # Get all image files (jpg, png, jpeg)
                    img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
                    for img_file in img_files:
                        self.images.append(str(img_file))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(config):
    """
    Create data loaders for training and validation
    
    Author: RSK World
    Website: https://rskworld.in
    """
    train_transform = get_data_transforms(config, is_train=True)
    val_transform = get_data_transforms(config, is_train=False)
    
    train_dataset = ImageDataset(config['data']['train_dir'], transform=train_transform)
    val_dataset = ImageDataset(config['data']['val_dir'], transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint
    
    Author: RSK World
    Website: https://rskworld.in
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint
    
    Author: RSK World
    Website: https://rskworld.in
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {filepath}")
    return epoch, loss


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy
    
    Author: RSK World
    Website: https://rskworld.in
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing
    
    Author: RSK World
    Website: https://rskworld.in
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

