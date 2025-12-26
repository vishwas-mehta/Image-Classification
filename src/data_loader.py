"""
Data loading and preprocessing module for CIFAR-10 dataset.
Handles train/validation/test splits and DataLoader creation.
"""

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import numpy as np

import sys
sys.path.append('..')
from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, VALIDATION_SPLIT,
    CIFAR10_MEAN, CIFAR10_STD, SEED
)
from src.augmentation import get_train_transforms, get_test_transforms


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, 
                     validation_split=VALIDATION_SPLIT, use_augmentation=True):
    """
    Create train, validation, and test data loaders for CIFAR-10.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        validation_split (float): Fraction of training data to use for validation
        use_augmentation (bool): Whether to use data augmentation for training
    
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
    """
    # Get transforms
    train_transform = get_train_transforms() if use_augmentation else get_test_transforms()
    test_transform = get_test_transforms()
    
    # Download and load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=test_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create train/validation split
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))
    
    # Shuffle indices
    np.random.seed(SEED)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_class_names():
    """Return CIFAR-10 class names."""
    return ('airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')


def show_sample_images(data_loader, num_images=8):
    """
    Display sample images from a data loader.
    
    Args:
        data_loader: PyTorch DataLoader
        num_images (int): Number of images to display
    """
    import matplotlib.pyplot as plt
    
    # Get a batch of images
    images, labels = next(iter(data_loader))
    
    # Denormalize images
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    images = images * std + mean
    
    # Plot images
    fig, axes = plt.subplots(1, num_images, figsize=(2*num_images, 2))
    class_names = get_class_names()
    
    for i in range(num_images):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(class_names[labels[i]])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Test the data loader
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Print batch info
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
