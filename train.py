"""
Training Script for Vision Transformer (ViT)
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script trains the Vision Transformer model on image classification tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from vit_model import create_vit_model
from utils import (
    load_config, get_data_loaders, save_checkpoint,
    load_checkpoint, accuracy, WarmupCosineScheduler
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Author: RSK World
    Website: https://rskworld.in
    """
    model.train()
    running_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1_acc += acc1.item()
        top5_acc += acc5.item()
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc1': f'{acc1.item():.2f}%',
            'acc5': f'{acc5.item():.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc1 = top1_acc / len(train_loader)
    epoch_acc5 = top5_acc / len(train_loader)
    
    return epoch_loss, epoch_acc1, epoch_acc5


def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model
    
    Author: RSK World
    Website: https://rskworld.in
    """
    model.eval()
    running_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc += acc1.item()
            top5_acc += acc5.item()
            running_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc1': f'{acc1.item():.2f}%',
                'acc5': f'{acc5.item():.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc1 = top1_acc / len(val_loader)
    epoch_acc5 = top5_acc / len(val_loader)
    
    return epoch_loss, epoch_acc1, epoch_acc5


def main():
    """
    Main training function
    
    Author: RSK World
    Website: https://rskworld.in
    """
    parser = argparse.ArgumentParser(description='Train Vision Transformer')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_vit_model(config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        config['training']['warmup_epochs'],
        config['training']['num_epochs'],
        config['training']['learning_rate']
    )
    
    # Data loaders
    train_loader, val_loader = get_data_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # TensorBoard writer
    writer = SummaryWriter(config['training']['log_dir'])
    
    # Create save directory
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Update learning rate
        lr = scheduler.step(epoch)
        writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Top1_Train', train_acc1, epoch)
        writer.add_scalar('Accuracy/Top1_Val', val_acc1, epoch)
        writer.add_scalar('Accuracy/Top5_Train', train_acc5, epoch)
        writer.add_scalar('Accuracy/Top5_Val', val_acc5, epoch)
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            config['training']['save_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save best model
        if val_acc1 > best_acc:
            best_acc = val_acc1
            best_model_path = os.path.join(
                config['training']['save_dir'],
                'best_model.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc1: {train_acc1:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc1: {val_acc1:.2f}%")
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()

