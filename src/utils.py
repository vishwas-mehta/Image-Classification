"""
Utility functions for training, evaluation, and visualization.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import sys
sys.path.append('..')
from config import MODEL_SAVE_DIR, RESULTS_DIR, CLASSES


def save_model(model, optimizer, epoch, loss, accuracy, filename):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Validation loss
        accuracy: Validation accuracy
        filename: Filename for the checkpoint
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_SAVE_DIR, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        filepath: Path to checkpoint
        optimizer: Optional optimizer to restore
    
    Returns:
        epoch, loss, accuracy from checkpoint
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.2f}%")
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


def get_model_summary(model, input_size=(1, 3, 32, 32)):
    """
    Print model summary with layer shapes and parameters.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    """
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    print(f"\nModel size: {total_size / 1024**2:.2f} MB")
    print("=" * 60)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': total_size / 1024**2
    }


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save the figure
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='#2196F3', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', color='#FF5722', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', color='#4CAF50', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', color='#9C27B0', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()
    return fig


def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix (numpy array)
        save_path: Optional path to save the figure
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return fig


def plot_sample_predictions(images, labels, predictions, num_samples=10, save_path=None):
    """
    Display sample predictions with images.
    
    Args:
        images: Image tensors
        labels: True labels
        predictions: Predicted labels
        num_samples: Number of samples to display
        save_path: Optional path to save the figure
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    from src.augmentation import denormalize
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        
        img = denormalize(images[i].cpu())
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        true_label = CLASSES[labels[i]]
        pred_label = CLASSES[predictions[i]]
        correct = labels[i] == predictions[i]
        
        color = 'green' if correct else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', 
                     color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")
    
    plt.show()
    return fig


def plot_model_comparison(results_dict, save_path=None):
    """
    Compare multiple models using bar charts.
    
    Args:
        results_dict: Dictionary with model names as keys and accuracy as values
        save_path: Optional path to save the figure
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    models = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    bars = ax.bar(models, accuracies, color=colors[:len(models)], edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Comparison on CIFAR-10', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.show()
    return fig


def plot_class_accuracy(class_accuracies, save_path=None):
    """
    Plot per-class accuracy.
    
    Args:
        class_accuracies: List of accuracies for each class
        save_path: Optional path to save the figure
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(CLASSES))
    colors = plt.cm.viridis(np.linspace(0, 1, len(CLASSES)))
    
    bars = ax.bar(x, class_accuracies, color=colors, edgecolor='black')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class accuracy saved to {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Create dummy history
    history = {
        'train_loss': [2.0, 1.5, 1.2, 1.0, 0.8],
        'val_loss': [2.1, 1.6, 1.4, 1.2, 1.1],
        'train_acc': [30, 45, 55, 65, 72],
        'val_acc': [28, 42, 50, 58, 65]
    }
    
    # Test plotting
    plot_training_history(history, save_path='results/test_training_curves.png')
    
    # Test model comparison
    results = {'CustomCNN': 75.5, 'ResNet18': 82.3}
    plot_model_comparison(results, save_path='results/test_model_comparison.png')
    
    print("All utility tests completed!")
