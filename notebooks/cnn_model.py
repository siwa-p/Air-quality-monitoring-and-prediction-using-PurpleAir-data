import torch.nn as nn
import torch

class CNN(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))
        
    def __init__(self, channels_l0=64, n_blocks=4, stride=2):
        super().__init__()
        self.cnn_layers = nn.ModuleList()
        
        # First convolutional layer with stride to reduce spatial dimensions
        self.cnn_layers.append(nn.Conv3d(1, channels_l0, kernel_size=5, stride=stride, padding=2))
        self.cnn_layers.append(nn.ReLU())
        
        c1 = channels_l0
        for _ in range(n_blocks):
            c2 = c1 * 2
            block = self.Block(c1, c2, stride=1)  # Keep channels consistent in each block
            self.cnn_layers.append(block)
            c1 = c2  # Update the number of input channels for the next block
        
        # Final convolutional layer before upsampling
        self.cnn_layers.append(nn.Conv3d(c1, c1, kernel_size=3, padding=1))
        self.cnn_layers.append(nn.BatchNorm3d(c1))
        
        # upsampling
        self.cnn_layers.append(nn.ConvTranspose3d(c1, 1, kernel_size=4, stride=stride, padding=1, output_padding=0))
        self.cnn_layers.append(nn.BatchNorm3d(1))
        
        self.cnn = nn.Sequential(*self.cnn_layers)
        
        # A 2D convolutional layer to finalize the output
        self.Conv2d = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        residual = x 
        x = self.cnn(x)
        x = x + residual  # Skip connection
        x = x[:, :, -1, :, :]  # Selecting the last time step
        return self.Conv2d(x)
