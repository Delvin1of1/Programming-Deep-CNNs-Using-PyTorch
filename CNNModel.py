import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    Simple CNN model for CIFAR-10 classification.
    Architecture:
    - Conv2d(3, 6, 5) -> MaxPool -> Conv2d(6, 16, 5) -> MaxPool
    - Flatten -> FC(400, 128) -> FC(128, 64) -> FC(64, 10)
    """
    
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # Input: 3 channels (RGB), Output: 6 feature maps, Kernel: 5x5
        self.pool = nn.MaxPool2d(2, 2)   # Pooling layer (reused)
        self.conv2 = nn.Conv2d(6, 16, 5)  # Input: 6 channels, Output: 16 feature maps, Kernel: 5x5
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 128)  # 16 feature maps * 5 * 5 spatial dimensions
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes for CIFAR-10
        
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x):
        # First conv block: Conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv block: Conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer with Softmax
        x = self.sm(self.fc3(x))
        
        return x