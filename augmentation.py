"""
Advanced Data Augmentation Utilities
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides advanced data augmentation techniques for Vision Transformer training.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import random
import math

# Author: RSK World
# Website: https://rskworld.in


class RandomErasing:
    """
    Random Erasing augmentation
    
    Author: RSK World
    Website: https://rskworld.in
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, img):
        if random.random() < self.p:
            return F.erase(img, *self._get_params(img))
        return img
    
    def _get_params(self, img):
        img_c, img_h, img_w = F.get_image_num_channels(img), F.get_image_size(img)[1], F.get_image_size(img)[0]
        area = img_h * img_w
        
        for _ in range(10):
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            
            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                return i, j, h, w, self.value
        
        return 0, 0, img_h, img_w, self.value


class MixUp:
    """
    MixUp augmentation
    
    Author: RSK World
    Website: https://rskworld.in
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index, :]
        y_a, y_b = batch_labels, batch_labels[index]
        
        return mixed_images, y_a, y_b, lam


class CutMix:
    """
    CutMix augmentation
    
    Author: RSK World
    Website: https://rskworld.in
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch_images.size(), lam)
        batch_images[:, :, bbx1:bbx2, bby1:bby2] = batch_images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_images.size()[-1] * batch_images.size()[-2]))
        
        y_a, y_b = batch_labels, batch_labels[index]
        return batch_images, y_a, y_b, lam
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


def get_advanced_augmentation(image_size=224, augmentation_type='standard'):
    """
    Get advanced augmentation pipeline
    
    Author: RSK World
    Website: https://rskworld.in
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augmentation_type == 'standard':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_type == 'aggressive':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            RandomErasing(p=0.5),
            normalize
        ])
    
    elif augmentation_type == 'light':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_type == 'auto_augment':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            normalize
        ])
    
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def visualize_augmentations(image_path, augmentation_type='standard', num_samples=8, image_size=224):
    """
    Visualize different augmentation effects
    
    Author: RSK World
    Website: https://rskworld.in
    """
    import matplotlib.pyplot as plt
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Get augmentation
    aug = get_advanced_augmentation(image_size, augmentation_type)
    
    # Apply augmentations
    augmented_images = []
    for _ in range(num_samples):
        aug_img = aug(img)
        # Denormalize for visualization
        aug_img = aug_img.permute(1, 2, 0).numpy()
        aug_img = aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        aug_img = np.clip(aug_img, 0, 1)
        augmented_images.append(aug_img)
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, aug_img in enumerate(augmented_images):
        axes[idx].imshow(aug_img)
        axes[idx].set_title(f'Augmentation {idx+1}', fontsize=10)
        axes[idx].axis('off')
    
    plt.suptitle(f'Augmentation Type: {augmentation_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Advanced Data Augmentation Utilities")
    print("Author: RSK World")
    print("Website: https://rskworld.in")

