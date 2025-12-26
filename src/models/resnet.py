"""
ResNet18 implementation for CIFAR-10 classification.
Uses torchvision's ResNet with modifications for 32x32 images.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    """
    ResNet18 adapted for CIFAR-10 (32x32 images).
    
    Modifications from standard ResNet18:
    - Smaller initial conv kernel (3x3 instead of 7x7)
    - No initial max pooling
    - Output layer modified for 10 classes
    
    Args:
        num_classes (int): Number of output classes (default: 10)
        pretrained (bool): Use pretrained weights (default: False)
    """
    
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.resnet = models.resnet18(weights=weights)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Modify for CIFAR-10 (32x32 images)
        # Replace first conv layer (smaller kernel, no stride)
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Remove max pooling (keep spatial resolution for small images)
        self.resnet.maxpool = nn.Identity()
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Initialize modified layers
        self._initialize_modified_layers()
    
    def _initialize_modified_layers(self):
        """Initialize the modified layers."""
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.kaiming_normal_(self.resnet.fc.weight, mode='fan_out')
        nn.init.constant_(self.resnet.fc.bias, 0)
    
    def forward(self, x):
        return self.resnet(x)
    
    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_resnet18(num_classes=10, pretrained=False):
    """
    Factory function to create ResNet18 model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
    
    Returns:
        ResNet18 model
    """
    return ResNet18(num_classes=num_classes, pretrained=pretrained)


class ResNet18Scratch(nn.Module):
    """
    ResNet18 implemented from scratch for educational purposes.
    """
    
    def __init__(self, num_classes=10):
        super(ResNet18Scratch, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """Create a residual layer."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet18/34."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        
        return out


if __name__ == "__main__":
    # Test the models
    print("=" * 50)
    print("ResNet18 (torchvision-based):")
    model1 = ResNet18()
    print(f"Total parameters: {model1.get_num_parameters():,}")
    
    x = torch.randn(1, 3, 32, 32)
    output = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n" + "=" * 50)
    print("ResNet18 (from scratch):")
    model2 = ResNet18Scratch()
    params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"Total parameters: {params:,}")
    
    output = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
