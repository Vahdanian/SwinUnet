"""
Visualization utilities for MS lesion segmentation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import torch

from ..utils.io_utils import save_nifti


def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    modality: str = "FLAIR"
):
    """
    Visualize prediction overlaid on MRI slice.
    
    Args:
        image: Input MRI image (H, W, D) or (H, W) for single slice
        prediction: Predicted mask (H, W, D) or (H, W)
        ground_truth: Ground truth mask (optional)
        slice_idx: Slice index to visualize (if 3D, uses middle slice if None)
        save_path: Path to save figure (optional)
        modality: Modality name for title
    """
    # Handle 3D volumes
    if len(image.shape) == 3:
        if slice_idx is None:
            slice_idx = image.shape[2] // 2
        image_slice = image[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx]
        if ground_truth is not None:
            gt_slice = ground_truth[:, :, slice_idx]
    else:
        image_slice = image
        pred_slice = prediction
        if ground_truth is not None:
            gt_slice = ground_truth
    
    # Normalize image for display
    image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
    
    # Create figure
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title(f'{modality} Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(image_slice, cmap='gray')
        axes[1].imshow(gt_slice, alpha=0.5, cmap='Reds')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(image_slice, cmap='gray')
        axes[2].imshow(pred_slice, alpha=0.5, cmap='Blues')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title(f'{modality} Image')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(image_slice, cmap='gray')
        axes[1].imshow(pred_slice, alpha=0.5, cmap='Blues')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics(history: dict, save_path: Optional[str] = None):
    """
    Plot training/validation metrics over epochs.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save figure (optional)
    """
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history.get('train_loss', []), label='Train Loss', marker='o')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice
    axes[0, 1].plot(epochs, history.get('train_dice', []), label='Train Dice', marker='o')
    if 'val_dice' in history and len(history['val_dice']) > 0:
        axes[0, 1].plot(epochs, history['val_dice'], label='Val Dice', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Training and Validation Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    if 'learning_rate' in history:
        axes[1, 0].plot(epochs, history['learning_rate'], marker='o', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
    
    # Combined view
    ax2 = axes[1, 1]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(epochs, history.get('train_loss', []), 'b-', label='Train Loss', marker='o')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        line2 = ax2.plot(epochs, history['val_loss'], 'b--', label='Val Loss', marker='s')
    
    line3 = ax2_twin.plot(epochs, history.get('train_dice', []), 'r-', label='Train Dice', marker='o')
    if 'val_dice' in history and len(history['val_dice']) > 0:
        line4 = ax2_twin.plot(epochs, history['val_dice'], 'r--', label='Val Dice', marker='s')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss', color='b')
    ax2_twin.set_ylabel('Dice Score', color='r')
    ax2.set_title('Loss and Dice Over Time')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_results(
    predictions: np.ndarray,
    output_dir: str,
    patient_id: str,
    timepoint: str,
    affine: Optional[np.ndarray] = None
):
    """
    Save segmentation results as NIfTI files.
    
    Args:
        predictions: Predicted masks (H, W, D) or (B, H, W, D)
        output_dir: Output directory
        patient_id: Patient identifier
        timepoint: Timepoint identifier
        affine: Affine transformation matrix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle batch dimension
    if len(predictions.shape) == 4 and predictions.shape[0] > 1:
        # Batch of predictions
        for i, pred in enumerate(predictions):
            output_path = os.path.join(
                output_dir, f"{patient_id}_{timepoint}_pred_{i}.nii.gz"
            )
            save_nifti(pred, output_path, affine=affine, dtype=np.uint8)
    else:
        # Single prediction
        if len(predictions.shape) == 4:
            predictions = predictions[0]
        
        output_path = os.path.join(
            output_dir, f"{patient_id}_{timepoint}_pred.nii.gz"
        )
        save_nifti(predictions, output_path, affine=affine, dtype=np.uint8)

