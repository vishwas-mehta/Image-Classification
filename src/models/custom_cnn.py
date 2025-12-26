"""
Custom CNN architecture for CIFAR-10 image classification.
A deep convolutional neural network built from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A convolutional block with Conv2D -> BatchNorm -> ReLU -> MaxPool.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 padding=1, pool=True):
        super(ConvBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for CIFAR-10 classification.
    
    Architecture:
    - 3 Convolutional blocks with increasing filters (32 -> 64 -> 128)
    - Each block: Conv2D + BatchNorm + ReLU + MaxPool
    - 2 Fully connected layers with dropout
    - Output: 10 classes
    
    Args:
        num_classes (int): Number of output classes (default: 10)
        dropout_rate (float): Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        # Input: 3x32x32
        self.conv1 = ConvBlock(3, 32)      # -> 32x16x16
        self.conv2 = ConvBlock(32, 64)     # -> 64x8x8
        self.conv3 = ConvBlock(64, 128)    # -> 128x4x4
        
        # Additional conv without pooling for more features
        self.conv4 = ConvBlock(128, 256, pool=False)  # -> 256x4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CustomCNNLight(nn.Module):
    """
    A lighter version of CustomCNN for comparison.
    Demonstrates overfitting on smaller models.
    """
    
    def __init__(self, num_classes=10):
        super(CustomCNNLight, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CustomCNNNoRegularization(nn.Module):
    """
    CustomCNN without any regularization (no dropout, no batch norm).
    Used for demonstrating overfitting behavior.
    """
    
    def __init__(self, num_classes=10):
        super(CustomCNNNoRegularization, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test the model
    model = CustomCNN()
    print(f"CustomCNN Architecture:")
    print(model)
    print(f"\nTotal parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
