"""
Training module for image classification models.
Includes training loop, validation, early stopping, and learning rate scheduling.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

import sys
sys.path.append('..')
from config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE,
    EARLY_STOPPING_PATIENCE, MODEL_SAVE_DIR
)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs to wait for improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
    
    def restore_best_model(self, model):
        if self.restore_best_weights and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print("Restored best model weights")


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, 
                epochs=EPOCHS, 
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                device=DEVICE,
                use_scheduler=True,
                use_early_stopping=True,
                model_name='model'):
    """
    Full training pipeline with validation, LR scheduling, and early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs (int): Number of training epochs
        learning_rate (float): Initial learning rate
        weight_decay (float): L2 regularization strength
        device: Device to use for training
        use_scheduler (bool): Use learning rate scheduler
        use_early_stopping (bool): Use early stopping
        model_name (str): Name for saving the model
    
    Returns:
        history: Dictionary with training history
    """
    # Move model to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                          weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            verbose=True
        )
    
    # Early stopping
    early_stopping = None
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    print(f"\nTraining {model_name} on {device}")
    print(f"Epochs: {epochs}, LR: {learning_rate}, Weight Decay: {weight_decay}")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), 
                      os.path.join(MODEL_SAVE_DIR, f'{model_name}_best.pth'))
        
        # Early stopping check
        if early_stopping is not None:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                early_stopping.restore_best_model(model)
                break
    
    total_time = time.time() - start_time
    
    print("=" * 60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(MODEL_SAVE_DIR, f'{model_name}_final.pth'))
    
    return history


if __name__ == "__main__":
    # Quick test of training pipeline
    from src.data_loader import get_data_loaders
    from src.models.custom_cnn import CustomCNN
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=128)
    
    # Create model
    model = CustomCNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train for 2 epochs (quick test)
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=2,
        model_name='custom_cnn_test'
    )
    
    print("\nTraining test completed!")
