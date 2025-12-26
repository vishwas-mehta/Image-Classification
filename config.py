"""
Configuration file for Image Classification project.
Contains all hyperparameters and settings.
"""

import torch

# ==================== Device Configuration ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Data Configuration ====================
DATA_DIR = './data'
BATCH_SIZE = 128
NUM_WORKERS = 4
VALIDATION_SPLIT = 0.1  # 10% of training data for validation

# CIFAR-10 Statistics (for normalization)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-10 Classes
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
NUM_CLASSES = 10

# ==================== Training Configuration ====================
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Learning Rate Scheduler
LR_SCHEDULER_FACTOR = 0.1
LR_SCHEDULER_PATIENCE = 5

# Early Stopping
EARLY_STOPPING_PATIENCE = 10

# ==================== Model Configuration ====================
# Custom CNN
CNN_DROPOUT_RATE = 0.5

# ==================== Paths ====================
MODEL_SAVE_DIR = './models'
RESULTS_DIR = './results'

# ==================== Random Seed ====================
SEED = 42


def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get the device to use for training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device
