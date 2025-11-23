"""
Loss functions for MS lesion segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice loss for imbalanced segmentation.
    Improved version that computes per-sample Dice for better gradients.
    """
    
    def __init__(self, smooth: float = 1e-5, per_sample: bool = True):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            per_sample: If True, compute Dice per sample and average (better gradients)
        """
        super(DiceLoss, self).__init__()
        self.smooth = float(smooth)
        self.per_sample = per_sample
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, C, H, W, D)
            target: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Dice loss value
        """
        if self.per_sample:
            # Compute Dice per sample for better gradients
            batch_size = pred.shape[0]
            dice_scores = []
            
            for i in range(batch_size):
                pred_flat = pred[i].view(-1)
                target_flat = target[i].view(-1)
                
                intersection = (pred_flat * target_flat).sum()
                dice = (2.0 * intersection + self.smooth) / (
                    pred_flat.sum() + target_flat.sum() + self.smooth
                )
                dice_scores.append(dice)
            
            # Average Dice across batch
            dice = torch.stack(dice_scores).mean()
        else:
            # Original global computation
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            
            intersection = (pred_flat * target_flat).sum()
            dice = (2.0 * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
            )
        
        # Return Dice loss (1 - Dice)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice loss and Binary Cross-Entropy loss.
    Improved with per-sample Dice computation.
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-5, pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            smooth: Smoothing factor for Dice loss
            pos_weight: Weight for positive class in BCE (for class imbalance)
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice_loss = DiceLoss(smooth=float(smooth), per_sample=True)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
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


class TverskyLoss(nn.Module):
    """
    Tversky loss - generalization of Dice loss with alpha/beta parameters.
    Better for handling class imbalance by penalizing false positives/negatives differently.
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-5):
        """
        Args:
            alpha: Weight for false positives (higher = more penalty on FP)
            beta: Weight for false negatives (higher = more penalty on FN)
            smooth: Smoothing factor
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, C, H, W, D)
            target: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Tversky loss value
        """
        batch_size = pred.shape[0]
        tversky_scores = []
        
        for i in range(batch_size):
            pred_flat = pred[i].view(-1)
            target_flat = target[i].view(-1)
            
            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()
            
            tversky = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            tversky_scores.append(tversky)
        
        tversky = torch.stack(tversky_scores).mean()
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss - combines Tversky loss with focal term.
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-5):
        """
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            gamma: Focal parameter (higher = more focus on hard examples)
            smooth: Smoothing factor
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, C, H, W, D)
            target: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Focal Tversky loss value
        """
        batch_size = pred.shape[0]
        focal_tversky_scores = []
        
        for i in range(batch_size):
            pred_flat = pred[i].view(-1)
            target_flat = target[i].view(-1)
            
            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()
            
            tversky = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            
            # Apply focal term
            focal_tversky = torch.pow(1.0 - tversky, self.gamma)
            focal_tversky_scores.append(focal_tversky)
        
        return torch.stack(focal_tversky_scores).mean()


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
        
        self.dice_loss = DiceLoss(smooth=smooth, per_sample=True)
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


class EnhancedCombinedLoss(nn.Module):
    """
    Enhanced combined loss with Tversky and better weighting.
    Optimized for medical image segmentation with class imbalance.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.3,
        tversky_weight: float = 0.3,
        bce_weight: float = 0.2,
        focal_weight: float = 0.2,
        smooth: float = 1e-5,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None
    ):
        """
        Args:
            dice_weight: Weight for Dice loss
            tversky_weight: Weight for Tversky loss
            bce_weight: Weight for BCE loss
            focal_weight: Weight for Focal loss
            smooth: Smoothing factor
            tversky_alpha: Alpha for Tversky loss
            tversky_beta: Beta for Tversky loss
            focal_alpha: Alpha for Focal loss
            focal_gamma: Gamma for Focal loss
            pos_weight: Weight for positive class in BCE (for class imbalance)
        """
        super(EnhancedCombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(smooth=smooth, per_sample=True)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W, D)
            target: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Combined loss value
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # All losses
        dice = self.dice_loss(pred_sigmoid, target)
        tversky = self.tversky_loss(pred_sigmoid, target)
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        # Combined
        total_loss = (
            self.dice_weight * dice +
            self.tversky_weight * tversky +
            self.bce_weight * bce +
            self.focal_weight * focal
        )
        
        return total_loss

