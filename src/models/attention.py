"""
Attention mechanism modules for MS lesion segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Spatial attention module for focusing on lesion locations.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Size of the convolution kernel
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W, D)
            
        Returns:
            Attention-weighted tensor
        """
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel attention module for modality importance weighting.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction factor for hidden dimension
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W, D)
            
        Returns:
            Attention-weighted tensor
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention for detecting small and diffuse lesions.
    """
    
    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input channels
        """
        super(MultiScaleAttention, self).__init__()
        
        # Multi-scale convolutions
        self.conv1x1 = nn.Conv3d(channels, channels // 4, 1)
        self.conv3x3 = nn.Conv3d(channels, channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv3d(channels, channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv3d(channels, channels // 4, 7, padding=3)
        
        self.conv_out = nn.Conv3d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W, D)
            
        Returns:
            Multi-scale attention-weighted tensor
        """
        # Extract multi-scale features
        feat1 = self.relu(self.conv1x1(x))
        feat3 = self.relu(self.conv3x3(x))
        feat5 = self.relu(self.conv5x5(x))
        feat7 = self.relu(self.conv7x7(x))
        
        # Concatenate multi-scale features
        concat = torch.cat([feat1, feat3, feat5, feat7], dim=1)
        
        # Generate attention map
        attention = self.sigmoid(self.conv_out(concat))
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (combines channel and spatial attention).
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction factor for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W, D)
            
        Returns:
            Attention-weighted tensor
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

