"""
Preprocessing functions for MRI scans
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
import nibabel as nib


def normalize_intensity(data: np.ndarray, method: str = "zscore", 
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize intensity values of MRI scan.
    
    Args:
        data: Input image data
        method: Normalization method ("zscore" or "minmax")
        mask: Optional mask to compute statistics only from masked region
        
    Returns:
        Normalized image data
    """
    if mask is not None:
        masked_data = data[mask > 0]
        if len(masked_data) == 0:
            mask = None
    
    if method == "zscore":
        if mask is not None:
            mean = np.mean(data[mask > 0])
            std = np.std(data[mask > 0])
        else:
            mean = np.mean(data)
            std = np.std(data)
        
        if std == 0:
            return data - mean
        
        normalized = (data - mean) / std
        return normalized.astype(np.float32)
    
    elif method == "minmax":
        if mask is not None:
            min_val = np.min(data[mask > 0])
            max_val = np.max(data[mask > 0])
        else:
            min_val = np.min(data)
            max_val = np.max(data)
        
        if max_val == min_val:
            return np.zeros_like(data, dtype=np.float32)
        
        normalized = (data - min_val) / (max_val - min_val)
        return normalized.astype(np.float32)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resample_volume(data: np.ndarray, original_spacing: Tuple[float, float, float],
                   target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                   order: int = 3) -> np.ndarray:
    """
    Resample volume to target voxel spacing.
    
    Args:
        data: Input volume data
        original_spacing: Original voxel spacing (x, y, z)
        target_spacing: Target voxel spacing (x, y, z)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        Resampled volume
    """
    zoom_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
    resampled = ndimage.zoom(data, zoom_factors, order=order, mode='nearest')
    return resampled.astype(np.float32)


def co_register_modalities(reference: np.ndarray, moving: np.ndarray,
                          method: str = "affine") -> np.ndarray:
    """
    Co-register moving image to reference image.
    
    Note: This is a simplified version. For production use, consider using
    SimpleITK or ANTs for more robust registration.
    
    Args:
        reference: Reference image (typically T1 or FLAIR)
        moving: Moving image to be registered
        method: Registration method ("affine" or "rigid")
        
    Returns:
        Registered moving image
    """
    # Simplified implementation - assumes images are already roughly aligned
    # For proper registration, use SimpleITK or ANTs
    # This is a placeholder that returns the moving image as-is
    # In practice, you would use:
    # - SimpleITK for rigid/affine registration
    # - ANTs for more advanced deformable registration
    
    return moving.astype(np.float32)


def crop_padding(data: np.ndarray, target_size: Tuple[int, int, int],
                mode: str = "center") -> Tuple[np.ndarray, Tuple[slice, ...]]:
    """
    Crop or pad volume to target size.
    
    Args:
        data: Input volume
        target_size: Target size (x, y, z)
        mode: Cropping mode ("center" or "random")
        
    Returns:
        Cropped/padded volume and slice indices
    """
    current_size = data.shape
    slices = []
    
    for i, (curr, target) in enumerate(zip(current_size, target_size)):
        if curr > target:
            # Crop
            if mode == "center":
                start = (curr - target) // 2
                end = start + target
            else:  # random
                start = np.random.randint(0, curr - target + 1)
                end = start + target
            slices.append(slice(start, end))
        else:
            # No cropping needed
            slices.append(slice(0, curr))
    
    cropped = data[tuple(slices)]
    
    # Pad if necessary
    padding = []
    for i, (curr, target) in enumerate(zip(cropped.shape, target_size)):
        if curr < target:
            pad_before = (target - curr) // 2
            pad_after = target - curr - pad_before
            padding.append((pad_before, pad_after))
        else:
            padding.append((0, 0))
    
    if any(p[0] > 0 or p[1] > 0 for p in padding):
        padded = np.pad(cropped, padding, mode='constant', constant_values=0)
        return padded.astype(np.float32), tuple(slices)
    
    return cropped.astype(np.float32), tuple(slices)


def create_brain_mask(data: np.ndarray, threshold_percentile: float = 5.0) -> np.ndarray:
    """
    Create a simple brain mask by thresholding.
    
    Args:
        data: Input image data
        threshold_percentile: Percentile to use as threshold
        
    Returns:
        Binary brain mask
    """
    threshold = np.percentile(data, threshold_percentile)
    mask = data > threshold
    
    # Remove small connected components
    from scipy.ndimage import label, find_objects
    labeled, num_features = label(mask)
    
    if num_features > 0:
        # Keep largest component
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        largest_component = np.argmax(sizes) + 1
        mask = labeled == largest_component
    
    return mask.astype(np.uint8)

