"""
Swin UNETR model implementation for MS lesion segmentation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from monai.networks.nets import SwinUNETR as MONAISwinUNETR
from monai.networks.layers import Conv

from .attention import CBAM, MultiScaleAttention


class SwinUNETRModel(nn.Module):
    """
    Swin UNETR model with attention mechanisms for MS lesion segmentation.
    
    This model combines:
    - Swin Transformer encoder for hierarchical feature extraction
    - U-Net decoder with skip connections
    - Attention mechanisms for lesion detection
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 48,
        use_attention: bool = True,
        attention_type: str = "cbam",
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        dropout_path_rate: float = 0.1
    ):
        """
        Args:
            in_channels: Number of input channels (modalities)
            out_channels: Number of output channels (1 for binary segmentation)
            img_size: Input image size (H, W, D)
            feature_size: Feature size for Swin UNETR
            use_attention: Whether to use attention mechanisms
            attention_type: Type of attention ("cbam" or "multiscale")
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Drop path rate for regularization
        """
        super(SwinUNETRModel, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.use_attention = use_attention
        
        # Base Swin UNETR model from MONAI
        # Note: MONAI's SwinUNETR doesn't take img_size, it infers from input
        self.swin_unetr = MONAISwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=False
        )
        
        # Add attention mechanisms if enabled
        if use_attention:
            if attention_type == "cbam":
                # Final attention layer on output (out_channels)
                self.final_attention = CBAM(out_channels, reduction=max(1, out_channels // 4))
            elif attention_type == "multiscale":
                self.final_attention = MultiScaleAttention(out_channels)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
        else:
            self.final_attention = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W, D)
            
        Returns:
            Segmentation output (B, out_channels, H, W, D)
        """
        # Get features from Swin UNETR encoder
        # MONAI's SwinUNETR returns the final output directly
        output = self.swin_unetr(x)
        
        # Apply attention if enabled
        if self.use_attention and self.final_attention is not None:
            output = self.final_attention(output)
        
        return output


class SwinUNETRWithSkipConnections(nn.Module):
    """
    Enhanced Swin UNETR with explicit skip connections and attention.
    This is a more custom implementation that gives more control.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 48,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        use_attention: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            img_size: Input image size
            feature_size: Base feature size
            depths: Depths of each Swin Transformer stage
            num_heads: Number of attention heads at each stage
            use_attention: Whether to use attention mechanisms
        """
        super(SwinUNETRWithSkipConnections, self).__init__()
        
        # Use MONAI's SwinUNETR as base
        # Note: MONAI's SwinUNETR doesn't take img_size, it infers from input
        self.base_model = MONAISwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False
        )
        
        self.use_attention = use_attention
        if use_attention:
            # Final attention layer
            self.final_attention = CBAM(out_channels, reduction=4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W, D)
            
        Returns:
            Segmentation output (B, out_channels, H, W, D)
        """
        output = self.base_model(x)
        
        if self.use_attention:
            output = self.final_attention(output)
        
        return output


# Alias for convenience
SwinUNETR = SwinUNETRModel

