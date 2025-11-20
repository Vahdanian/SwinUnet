"""
Loss functions for MS lesion segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for imbalanced segmentation.
    """
    
    def __init__(self, smooth: float = 1e-5):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = float(smooth)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, C, H, W, D)
            target: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Dice loss value
        """
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        # Return Dice loss (1 - Dice)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice loss and Binary Cross-Entropy loss.
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-5):
        """
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            smooth: Smoothing factor for Dice loss
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice_loss = DiceLoss(smooth=float(smooth))
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W, D)
            target: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Combined loss value
        """
        # BCE loss (expects logits)
        bce = self.bce_loss(pred, target)
        
        # Dice loss (expects probabilities)
        pred_sigmoid = torch.sigmoid(pred)
        dice = self.dice_loss(pred_sigmoid, target)
        
        # Combined loss
        total_loss = self.dice_weight * dice + self.bce_weight * bce
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W, D)
            target: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Focal loss value
        """
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Compute p_t
        p_t = torch.exp(-bce)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce
        
        return focal_loss.mean()


class DiceBCEFocalLoss(nn.Module):
    """
    Combined Dice, BCE, and Focal loss.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.4,
        bce_weight: float = 0.3,
        focal_weight: float = 0.3,
        smooth: float = 1e-5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            focal_weight: Weight for Focal loss
            smooth: Smoothing factor for Dice loss
            focal_alpha: Alpha for Focal loss
            focal_gamma: Gamma for Focal loss
        """
        super(DiceBCEFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W, D)
            target: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Combined loss value
        """
        # BCE loss
        bce = self.bce_loss(pred, target)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        dice = self.dice_loss(pred_sigmoid, target)
        
        # Focal loss
        focal = self.focal_loss(pred, target)
        
        # Combined
        total_loss = (
            self.dice_weight * dice +
            self.bce_weight * bce +
            self.focal_weight * focal
        )
        
        return total_loss

