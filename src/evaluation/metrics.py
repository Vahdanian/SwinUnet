"""
Evaluation metrics for MS lesion segmentation
"""

import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from typing import Union


def dice_score(pred: Union[torch.Tensor, np.ndarray], 
               target: Union[torch.Tensor, np.ndarray],
               smooth: float = 1e-5) -> Union[torch.Tensor, float]:
    """
    Compute Dice Similarity Coefficient.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor
        
    Returns:
        Dice score
    """
    if isinstance(pred, torch.Tensor):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
        return dice
    else:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = np.sum(pred_flat * target_flat)
        dice = (2.0 * intersection + smooth) / (
            np.sum(pred_flat) + np.sum(target_flat) + smooth
        )
        return float(dice)


def sensitivity(pred: Union[torch.Tensor, np.ndarray],
                target: Union[torch.Tensor, np.ndarray],
                smooth: float = 1e-5) -> Union[torch.Tensor, float]:
    """
    Compute sensitivity (true positive rate / recall).
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor
        
    Returns:
        Sensitivity value
    """
    if isinstance(pred, torch.Tensor):
        tp = ((pred == 1) & (target == 1)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()
        sensitivity = (tp + smooth) / (tp + fn + smooth)
        return sensitivity
    else:
        tp = np.sum((pred == 1) & (target == 1))
        fn = np.sum((pred == 0) & (target == 1))
        sensitivity = (tp + smooth) / (tp + fn + smooth)
        return float(sensitivity)


def specificity(pred: Union[torch.Tensor, np.ndarray],
                target: Union[torch.Tensor, np.ndarray],
                smooth: float = 1e-5) -> Union[torch.Tensor, float]:
    """
    Compute specificity (true negative rate).
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor
        
    Returns:
        Specificity value
    """
    if isinstance(pred, torch.Tensor):
        tn = ((pred == 0) & (target == 0)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        specificity = (tn + smooth) / (tn + fp + smooth)
        return specificity
    else:
        tn = np.sum((pred == 0) & (target == 0))
        fp = np.sum((pred == 1) & (target == 0))
        specificity = (tn + smooth) / (tn + fp + smooth)
        return float(specificity)


def hausdorff_distance(pred: Union[torch.Tensor, np.ndarray],
                       target: Union[torch.Tensor, np.ndarray],
                       percentile: float = 95.0) -> float:
    """
    Compute Hausdorff distance (95th percentile) between boundaries.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        percentile: Percentile for Hausdorff distance (default: 95)
        
    Returns:
        Hausdorff distance in voxels
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Get boundary points
    pred_coords = np.argwhere(pred > 0.5)
    target_coords = np.argwhere(target > 0.5)
    
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return float('inf')
    
    # Compute directed Hausdorff distances
    d1 = directed_hausdorff(pred_coords, target_coords)[0]
    d2 = directed_hausdorff(target_coords, pred_coords)[0]
    
    # Symmetric Hausdorff distance
    hausdorff = max(d1, d2)
    
    # For percentile version, compute distances from each point
    if percentile < 100:
        distances = []
        for p in pred_coords:
            dists = np.sqrt(np.sum((target_coords - p) ** 2, axis=1))
            distances.append(np.min(dists))
        for t in target_coords:
            dists = np.sqrt(np.sum((pred_coords - t) ** 2, axis=1))
            distances.append(np.min(dists))
        
        if len(distances) > 0:
            hausdorff = np.percentile(distances, percentile)
    
    return float(hausdorff)


def compute_all_metrics(pred: Union[torch.Tensor, np.ndarray],
                        target: Union[torch.Tensor, np.ndarray]) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        Dictionary with all metrics
    """
    # Convert to binary if needed
    if isinstance(pred, torch.Tensor):
        if pred.max() <= 1.0 and pred.min() >= 0.0:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
    else:
        if pred.max() <= 1.0 and pred.min() >= 0.0:
            pred_binary = (pred > 0.5).astype(np.float32)
        else:
            pred_binary = (1 / (1 + np.exp(-pred)) > 0.5).astype(np.float32)
    
    metrics = {
        'dice': dice_score(pred_binary, target),
        'sensitivity': sensitivity(pred_binary, target),
        'specificity': specificity(pred_binary, target),
        'hausdorff_distance': hausdorff_distance(pred_binary, target)
    }
    
    return metrics

