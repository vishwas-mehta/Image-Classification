"""
Data augmentation utilities for training.
Includes various transforms for data augmentation and preprocessing.
"""

from torchvision import transforms
import sys
sys.path.append('..')
from config import CIFAR10_MEAN, CIFAR10_STD


def get_train_transforms():
    """
    Get training transforms with data augmentation.
    
    Augmentation techniques used:
    - Random horizontal flip
    - Random crop with padding
    - Random rotation
    - Color jitter
    - Normalization
    
    Returns:
        torchvision.transforms.Compose: Training transforms
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])


def get_test_transforms():
    """
    Get test/validation transforms (no augmentation).
    
    Only applies:
    - ToTensor conversion
    - Normalization
    
    Returns:
        torchvision.transforms.Compose: Test transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])


def get_minimal_augmentation():
    """
    Get minimal augmentation (only horizontal flip).
    Useful for comparison studies.
    
    Returns:
        torchvision.transforms.Compose: Minimal augmentation transforms
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])


def get_heavy_augmentation():
    """
    Get heavy augmentation for regularization studies.
    
    Includes all basic augmentations plus:
    - Cutout/RandomErasing
    - AutoAugment-style transforms
    
    Returns:
        torchvision.transforms.Compose: Heavy augmentation transforms
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
    ])


def denormalize(tensor):
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        Denormalized tensor
    """
    import torch
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    
    if tensor.device.type != 'cpu':
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    return tensor * std + mean


if __name__ == "__main__":
    # Test augmentation
    import matplotlib.pyplot as plt
    from torchvision import datasets
    import torch
    import numpy as np
    
    # Load a sample image
    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    sample_img, label = dataset[0]
    
    # Apply different augmentations
    transforms_dict = {
        'Original': transforms.ToTensor(),
        'Train Aug': get_train_transforms(),
        'Heavy Aug': get_heavy_augmentation(),
        'Minimal Aug': get_minimal_augmentation()
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for row in range(2):
        for col, (name, transform) in enumerate(transforms_dict.items()):
            ax = axes[row, col]
            img_tensor = transform(sample_img)
            
            # Denormalize if normalized
            if 'Aug' in name:
                img_tensor = denormalize(img_tensor)
            
            img = img_tensor.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f'{name}' if row == 0 else '')
            ax.axis('off')
    
    plt.suptitle('Data Augmentation Examples')
    plt.tight_layout()
    plt.savefig('results/augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
