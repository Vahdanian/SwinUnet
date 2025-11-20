"""
Dataset class for loading MS lesion segmentation data
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict

from ..utils.io_utils import (
    load_nifti, get_patient_list, get_timepoints_for_patient,
    get_modality_path, get_mask_path
)
from .preprocessing import normalize_intensity, crop_padding
from .augmentation import Compose, ElasticDeformation, RandomRotation3D, RandomScaling, IntensityAugmentation


class MSLesionDataset(Dataset):
    """
    Dataset class for MS lesion segmentation.
    Loads multi-modal MRI scans (FLAIR, T1, PD, T2) and corresponding masks.
    """
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: Optional[List[str]] = None,
        use_preprocessed: bool = True,
        normalize: bool = True,
        augmentation: bool = False,
        target_size: Optional[Tuple[int, int, int]] = None,
        modalities: List[str] = None
    ):
        """
        Args:
            data_dir: Root directory containing patient folders
            patient_ids: List of patient IDs to include (None = all)
            use_preprocessed: Use preprocessed scans if available
            normalize: Apply intensity normalization
            augmentation: Enable data augmentation
            target_size: Target volume size for cropping/padding (H, W, D)
            modalities: List of modalities to load (default: ["flair", "mprage", "pd", "t2"])
        """
        self.data_dir = data_dir
        self.use_preprocessed = use_preprocessed
        self.normalize = normalize
        self.augmentation = augmentation
        self.target_size = target_size
        
        if modalities is None:
            modalities = ["flair", "mprage", "pd", "t2"]
        self.modalities = modalities
        
        # Get patient list
        if patient_ids is None:
            patient_ids = get_patient_list(data_dir, pattern="training*")
        self.patient_ids = patient_ids
        
        # Build sample list: (patient_id, timepoint)
        self.samples = []
        for patient_id in self.patient_ids:
            timepoints = get_timepoints_for_patient(data_dir, patient_id)
            for timepoint in timepoints:
                self.samples.append((patient_id, timepoint))
        
        # Setup augmentation
        if augmentation:
            self.augment = Compose([
                RandomRotation3D(max_angle=15.0, probability=0.5),
                RandomScaling(scale_range=(0.9, 1.1), probability=0.5),
                ElasticDeformation(alpha=100.0, sigma=10.0, probability=0.5),
                IntensityAugmentation(gamma_range=(0.8, 1.2), noise_std=0.1, probability=0.5)
            ])
        else:
            self.augment = None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with:
                - 'image': Multi-modal image tensor (C, H, W, D)
                - 'mask': Ground truth mask tensor (1, H, W, D) or None for test data
                - 'patient_id': Patient identifier
                - 'timepoint': Timepoint identifier
        """
        patient_id, timepoint = self.samples[idx]
        
        # Load all modalities
        image_channels = []
        for modality in self.modalities:
            file_path = get_modality_path(
                self.data_dir, patient_id, timepoint, modality, self.use_preprocessed
            )
            
            if file_path is None:
                raise FileNotFoundError(
                    f"Could not find {modality} for {patient_id} timepoint {timepoint}"
                )
            
            # Load image
            data = load_nifti(file_path)
            
            # Normalize
            if self.normalize:
                data = normalize_intensity(data, method="zscore")
            
            image_channels.append(data)
        
        # Stack modalities
        image = np.stack(image_channels, axis=0)  # (C, H, W, D)
        
        # Load mask if available
        mask = None
        mask_path = get_mask_path(self.data_dir, patient_id, timepoint, mask_number=1)
        if mask_path and os.path.exists(mask_path):
            mask = load_nifti(mask_path)
            # Binarize mask (some datasets have multiple classes)
            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)  # (1, H, W, D)
        
        # Crop/pad to target size
        if self.target_size is not None:
            if mask is not None:
                # Crop/pad both image and mask together
                image_cropped, _ = crop_padding(image[0], self.target_size)
                mask_cropped, _ = crop_padding(mask[0], self.target_size)
                
                # Reconstruct
                image_cropped_all = []
                for c in range(image.shape[0]):
                    img_c, _ = crop_padding(image[c], self.target_size)
                    image_cropped_all.append(img_c)
                image = np.stack(image_cropped_all, axis=0)
                mask = np.expand_dims(mask_cropped, axis=0)
            else:
                image_cropped_all = []
                for c in range(image.shape[0]):
                    img_c, _ = crop_padding(image[c], self.target_size)
                    image_cropped_all.append(img_c)
                image = np.stack(image_cropped_all, axis=0)
        
        # Apply augmentation
        if self.augmentation and self.augment is not None and mask is not None:
            # Apply same augmentation to image and mask
            # Note: Augmentation expects (H, W, D) format
            augmented_channels = []
            for c in range(image.shape[0]):
                aug_img, aug_mask = self.augment(image[c], mask[0])
                augmented_channels.append(aug_img)
            image = np.stack(augmented_channels, axis=0)
            mask = np.expand_dims(aug_mask, axis=0)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        if mask is not None:
            mask = torch.from_numpy(mask).float()
        
        return {
            'image': image,
            'mask': mask,
            'patient_id': patient_id,
            'timepoint': timepoint
        }


class TestDataset(MSLesionDataset):
    """
    Dataset class for test data (no ground truth masks).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override patient list to use test data
        if 'patient_ids' not in kwargs or kwargs['patient_ids'] is None:
            self.patient_ids = get_patient_list(self.data_dir, pattern="test*")
        
        # Rebuild samples list
        self.samples = []
        for patient_id in self.patient_ids:
            timepoints = get_timepoints_for_patient(self.data_dir, patient_id)
            for timepoint in timepoints:
                self.samples.append((patient_id, timepoint))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get test sample (no mask)."""
        sample = super().__getitem__(idx)
        # Remove mask from test samples
        sample['mask'] = None
        return sample

