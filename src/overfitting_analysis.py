"""
Overfitting analysis and regularization demonstration.
Compares models with and without regularization techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

import sys
sys.path.append('..')
from config import DEVICE, RESULTS_DIR, set_seed
from src.data_loader import get_data_loaders
from src.models.custom_cnn import CustomCNN, CustomCNNNoRegularization
from src.train import train_one_epoch, validate


def train_for_analysis(model, train_loader, val_loader, epochs=30, 
                       device=DEVICE, learning_rate=0.001):
    """
    Train model and record detailed history for overfitting analysis.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        device: Device to use
        learning_rate: Learning rate
    
    Returns:
        history: Training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    return history


def plot_overfitting_comparison(history_no_reg, history_with_reg, save_path=None):
    """
    Plot comparison of models with and without regularization.
    
    Args:
        history_no_reg: History from model without regularization
        history_with_reg: History from model with regularization
        save_path: Path to save the figure
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss - No Regularization
    axes[0, 0].plot(history_no_reg['train_loss'], label='Train Loss', 
                    color='#2196F3', linewidth=2)
    axes[0, 0].plot(history_no_reg['val_loss'], label='Val Loss', 
                    color='#FF5722', linewidth=2)
    axes[0, 0].set_title('Without Regularization - Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss - With Regularization
    axes[0, 1].plot(history_with_reg['train_loss'], label='Train Loss', 
                    color='#2196F3', linewidth=2)
    axes[0, 1].plot(history_with_reg['val_loss'], label='Val Loss', 
                    color='#FF5722', linewidth=2)
    axes[0, 1].set_title('With Regularization - Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy - No Regularization
    axes[1, 0].plot(history_no_reg['train_acc'], label='Train Acc', 
                    color='#4CAF50', linewidth=2)
    axes[1, 0].plot(history_no_reg['val_acc'], label='Val Acc', 
                    color='#9C27B0', linewidth=2)
    axes[1, 0].set_title('Without Regularization - Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy - With Regularization
    axes[1, 1].plot(history_with_reg['train_acc'], label='Train Acc', 
                    color='#4CAF50', linewidth=2)
    axes[1, 1].plot(history_with_reg['val_acc'], label='Val Acc', 
                    color='#9C27B0', linewidth=2)
    axes[1, 1].set_title('With Regularization - Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Overfitting Analysis: Effect of Regularization', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved overfitting comparison to {save_path}")
    
    plt.show()
    return fig


def plot_generalization_gap(history_no_reg, history_with_reg, save_path=None):
    """
    Plot generalization gap (train acc - val acc) over epochs.
    
    Args:
        history_no_reg: History from model without regularization
        history_with_reg: History from model with regularization
        save_path: Path to save the figure
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    gap_no_reg = [t - v for t, v in zip(history_no_reg['train_acc'], 
                                         history_no_reg['val_acc'])]
    gap_with_reg = [t - v for t, v in zip(history_with_reg['train_acc'], 
                                          history_with_reg['val_acc'])]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(gap_no_reg) + 1)
    ax.plot(epochs, gap_no_reg, label='Without Regularization', 
            color='#FF5722', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, gap_with_reg, label='With Regularization', 
            color='#4CAF50', linewidth=2, marker='s', markersize=4)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(epochs, gap_no_reg, alpha=0.3, color='#FF5722')
    ax.fill_between(epochs, gap_with_reg, alpha=0.3, color='#4CAF50')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Generalization Gap (Train Acc - Val Acc)', fontsize=12)
    ax.set_title('Generalization Gap Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved generalization gap plot to {save_path}")
    
    plt.show()
    return fig


def demonstrate_regularization_techniques():
    """
    Demonstrate different regularization techniques:
    1. Dropout
    2. Batch Normalization
    3. Data Augmentation
    4. Weight Decay (L2 Regularization)
    """
    print("=" * 60)
    print("REGULARIZATION TECHNIQUES DEMONSTRATION")
    print("=" * 60)
    
    techniques = {
        'Dropout': {
            'description': 'Randomly zeroes elements during training',
            'effect': 'Prevents co-adaptation of neurons',
            'typical_rate': '0.2 - 0.5'
        },
        'Batch Normalization': {
            'description': 'Normalizes layer inputs to have mean=0, var=1',
            'effect': 'Stabilizes training, acts as regularizer',
            'typical_rate': 'Applied after conv/linear layers'
        },
        'Data Augmentation': {
            'description': 'Applies random transforms to training data',
            'effect': 'Increases effective dataset size',
            'typical_rate': 'RandomCrop, RandomFlip, ColorJitter'
        },
        'Weight Decay (L2)': {
            'description': 'Adds penalty for large weights to loss',
            'effect': 'Encourages smaller, more generalizable weights',
            'typical_rate': '1e-4 to 1e-5'
        },
        'Early Stopping': {
            'description': 'Stops training when val loss stops improving',
            'effect': 'Prevents overfitting to training data',
            'typical_rate': 'Patience: 5-10 epochs'
        }
    }
    
    for name, info in techniques.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Effect: {info['effect']}")
        print(f"  Typical Value: {info['typical_rate']}")
    
    print("\n" + "=" * 60)
    return techniques


def run_overfitting_analysis(epochs=20, use_augmentation=True):
    """
    Run complete overfitting analysis.
    
    Args:
        epochs: Number of epochs for training
        use_augmentation: Whether to use data augmentation
    """
    set_seed(42)
    
    print("=" * 60)
    print("OVERFITTING ANALYSIS")
    print("=" * 60)
    
    # Get data loaders
    print("\nLoading data...")
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=128, 
        use_augmentation=use_augmentation
    )
    
    train_loader_no_aug, val_loader_no_aug, _ = get_data_loaders(
        batch_size=128, 
        use_augmentation=False
    )
    
    # Train model WITHOUT regularization
    print("\n" + "-" * 40)
    print("Training model WITHOUT regularization...")
    print("-" * 40)
    model_no_reg = CustomCNNNoRegularization()
    history_no_reg = train_for_analysis(
        model_no_reg, train_loader_no_aug, val_loader_no_aug, epochs=epochs
    )
    
    # Train model WITH regularization
    print("\n" + "-" * 40)
    print("Training model WITH regularization...")
    print("-" * 40)
    model_with_reg = CustomCNN(dropout_rate=0.5)
    history_with_reg = train_for_analysis(
        model_with_reg, train_loader, val_loader, epochs=epochs
    )
    
    # Plot comparisons
    plot_overfitting_comparison(
        history_no_reg, history_with_reg,
        save_path=os.path.join(RESULTS_DIR, 'overfitting_comparison.png')
    )
    
    plot_generalization_gap(
        history_no_reg, history_with_reg,
        save_path=os.path.join(RESULTS_DIR, 'generalization_gap.png')
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    final_gap_no_reg = history_no_reg['train_acc'][-1] - history_no_reg['val_acc'][-1]
    final_gap_with_reg = history_with_reg['train_acc'][-1] - history_with_reg['val_acc'][-1]
    
    print(f"\nWithout Regularization:")
    print(f"  Final Train Acc: {history_no_reg['train_acc'][-1]:.2f}%")
    print(f"  Final Val Acc: {history_no_reg['val_acc'][-1]:.2f}%")
    print(f"  Generalization Gap: {final_gap_no_reg:.2f}%")
    
    print(f"\nWith Regularization:")
    print(f"  Final Train Acc: {history_with_reg['train_acc'][-1]:.2f}%")
    print(f"  Final Val Acc: {history_with_reg['val_acc'][-1]:.2f}%")
    print(f"  Generalization Gap: {final_gap_with_reg:.2f}%")
    
    print(f"\nImprovement: {final_gap_no_reg - final_gap_with_reg:.2f}% reduction in gap")
    
    # Demonstrate techniques
    demonstrate_regularization_techniques()
    
    return {
        'history_no_reg': history_no_reg,
        'history_with_reg': history_with_reg
    }


if __name__ == "__main__":
    # Run analysis
    results = run_overfitting_analysis(epochs=15)
