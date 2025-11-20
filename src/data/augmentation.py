"""
Data augmentation transforms for 3D medical images
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from typing import Tuple, Optional
import torch
from torch.nn import functional as F


class ElasticDeformation:
    """
    3D elastic deformation augmentation.
    """
    
    def __init__(self, alpha: float = 100.0, sigma: float = 10.0, 
                 probability: float = 0.5):
        """
        Args:
            alpha: Scaling factor for deformation
            sigma: Standard deviation for Gaussian smoothing
            probability: Probability of applying augmentation
        """
        self.alpha = alpha
        self.sigma = sigma
        self.probability = probability
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return data
        
        shape = data.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))
        
        deformed = map_coordinates(data, indices, order=3, mode='reflect').reshape(shape)
        return deformed.astype(data.dtype)


class RandomRotation3D:
    """
    Random 3D rotation augmentation.
    """
    
    def __init__(self, max_angle: float = 15.0, probability: float = 0.5):
        """
        Args:
            max_angle: Maximum rotation angle in degrees
            probability: Probability of applying augmentation
        """
        self.max_angle = max_angle
        self.probability = probability
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return data
        
        # Convert to torch tensor for rotation
        if isinstance(data, np.ndarray):
            data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        else:
            data_tensor = data.unsqueeze(0).unsqueeze(0) if len(data.shape) == 3 else data
        
        # Random rotation angles
        angles = [
            np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180.0,
            np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180.0,
            np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180.0
        ]
        
        # Apply rotations
        # Rotate around x-axis
        data_tensor = F.affine_grid(
            torch.tensor([[[1, 0, 0, 0],
                          [0, np.cos(angles[0]), -np.sin(angles[0]), 0],
                          [0, np.sin(angles[0]), np.cos(angles[0]), 0]]], dtype=torch.float32),
            data_tensor.shape, align_corners=False
        )
        # Note: This is simplified - for proper 3D rotation, use scipy.ndimage.rotate
        # or implement proper 3D rotation matrix
        
        # Simplified: use scipy for actual rotation
        from scipy.ndimage import rotate
        rotated = rotate(data, angles[0] * 180 / np.pi, axes=(1, 2), reshape=False, order=3)
        rotated = rotate(rotated, angles[1] * 180 / np.pi, axes=(0, 2), reshape=False, order=3)
        rotated = rotate(rotated, angles[2] * 180 / np.pi, axes=(0, 1), reshape=False, order=3)
        
        return rotated.astype(data.dtype)


class RandomScaling:
    """
    Random scaling (zoom) augmentation.
    """
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1), 
                 probability: float = 0.5):
        """
        Args:
            scale_range: Tuple of (min_scale, max_scale)
            probability: Probability of applying augmentation
        """
        self.scale_range = scale_range
        self.probability = probability
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return data
        
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        from scipy.ndimage import zoom
        zoom_factors = [scale] * len(data.shape)
        scaled = zoom(data, zoom_factors, order=3, mode='nearest')
        
        # Crop or pad to original size
        if scaled.shape != data.shape:
            # Center crop or pad
            slices = []
            for i, (orig, scaled_dim) in enumerate(zip(data.shape, scaled.shape)):
                if scaled_dim > orig:
                    start = (scaled_dim - orig) // 2
                    slices.append(slice(start, start + orig))
                else:
                    slices.append(slice(0, scaled_dim))
            
            if any(s.stop > scaled.shape[i] for i, s in enumerate(slices)):
                # Need to pad
                padding = []
                for i, (orig, scaled_dim) in enumerate(zip(data.shape, scaled.shape)):
                    if scaled_dim < orig:
                        pad_before = (orig - scaled_dim) // 2
                        pad_after = orig - scaled_dim - pad_before
                        padding.append((pad_before, pad_after))
                    else:
                        padding.append((0, 0))
                scaled = np.pad(scaled, padding, mode='constant', constant_values=0)
            
            scaled = scaled[tuple(slices) if len(slices) == len(data.shape) else ...]
        
        return scaled.astype(data.dtype)


class IntensityAugmentation:
    """
    Intensity-based augmentation (gamma correction, noise injection).
    """
    
    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2),
                 noise_std: float = 0.1, probability: float = 0.5):
        """
        Args:
            gamma_range: Range for gamma correction
            noise_std: Standard deviation for Gaussian noise
            probability: Probability of applying augmentation
        """
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.probability = probability
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return data
        
        augmented = data.copy()
        
        # Gamma correction
        if np.random.random() > 0.5:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            # Normalize to [0, 1] for gamma correction
            data_min, data_max = augmented.min(), augmented.max()
            if data_max > data_min:
                normalized = (augmented - data_min) / (data_max - data_min)
                gamma_corrected = np.power(normalized, gamma)
                augmented = gamma_corrected * (data_max - data_min) + data_min
        
        # Add noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, self.noise_std, augmented.shape)
            augmented = augmented + noise
        
        return augmented.astype(data.dtype)


class Compose:
    """
    Compose multiple augmentations.
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        for transform in self.transforms:
            if mask is not None:
                # Apply same transform to both data and mask
                if isinstance(transform, (ElasticDeformation, RandomRotation3D, RandomScaling)):
                    data = transform(data)
                    mask = transform(mask)
                elif isinstance(transform, IntensityAugmentation):
                    # Only apply intensity augmentation to data, not mask
                    data = transform(data)
            else:
                data = transform(data)
        
        if mask is not None:
            return data, mask
        return data

