"""
Utility functions for file I/O operations
"""

import os
import glob
import nibabel as nib
import numpy as np
from typing import List, Tuple, Optional


def load_nifti(file_path: str, return_affine: bool = False) -> np.ndarray:
    """
    Load a NIfTI file and return the image data as a numpy array.
    
    Args:
        file_path: Path to the NIfTI file (.nii or .nii.gz)
        return_affine: If True, also return the affine matrix
        
    Returns:
        Image data as numpy array, optionally with affine matrix
    """
    nii_img = nib.load(file_path)
    data = nii_img.get_fdata()
    
    if return_affine:
        return data, nii_img.affine
    return data


def save_nifti(data: np.ndarray, file_path: str, affine: Optional[np.ndarray] = None, 
               dtype: Optional[np.dtype] = None) -> None:
    """
    Save a numpy array as a NIfTI file.
    
    Args:
        data: Image data as numpy array
        file_path: Output file path
        affine: Affine transformation matrix (default: identity)
        dtype: Data type for saving (default: same as input)
    """
    if dtype is None:
        dtype = data.dtype
    
    if affine is None:
        affine = np.eye(4)
    
    data = data.astype(dtype)
    nii_img = nib.Nifti1Image(data, affine)
    nib.save(nii_img, file_path)


def get_patient_list(data_dir: str, pattern: str = "training*") -> List[str]:
    """
    Extract patient IDs from directory structure.
    
    Args:
        data_dir: Root directory containing patient folders
        pattern: Glob pattern to match patient folders (default: "training*")
        
    Returns:
        List of patient IDs (folder names)
    """
    patient_dirs = glob.glob(os.path.join(data_dir, pattern))
    patient_list = [os.path.basename(d) for d in patient_dirs]
    patient_list.sort()
    return patient_list


def get_timepoints_for_patient(data_dir: str, patient_id: str, 
                               modality: str = "flair") -> List[str]:
    """
    Get list of timepoints for a specific patient.
    
    Args:
        data_dir: Root directory containing patient folders
        patient_id: Patient identifier (e.g., "training01")
        modality: Modality to check (default: "flair")
        
    Returns:
        List of timepoint strings (e.g., ["01", "02", "03"])
    """
    patient_dir = os.path.join(data_dir, patient_id)
    orig_dir = os.path.join(patient_dir, "orig")
    
    if not os.path.exists(orig_dir):
        orig_dir = os.path.join(patient_dir, "preprocessed")
    
    pattern = f"{patient_id}_*_{modality}*.nii*"
    files = glob.glob(os.path.join(orig_dir, pattern))
    
    timepoints = []
    for file in files:
        filename = os.path.basename(file)
        # Extract timepoint from filename (e.g., "training01_02_flair.nii.gz" -> "02")
        parts = filename.split("_")
        if len(parts) >= 2:
            timepoint = parts[1]
            if timepoint not in timepoints:
                timepoints.append(timepoint)
    
    timepoints.sort()
    return timepoints


def get_modality_path(data_dir: str, patient_id: str, timepoint: str, 
                     modality: str, use_preprocessed: bool = True) -> Optional[str]:
    """
    Get the file path for a specific patient, timepoint, and modality.
    
    Args:
        data_dir: Root directory containing patient folders
        patient_id: Patient identifier
        timepoint: Timepoint identifier (e.g., "01")
        modality: Modality name (e.g., "flair", "mprage", "pd", "t2")
        use_preprocessed: If True, use preprocessed folder, else use orig folder
        
    Returns:
        File path if found, None otherwise
    """
    patient_dir = os.path.join(data_dir, patient_id)
    
    if use_preprocessed:
        scan_dir = os.path.join(patient_dir, "preprocessed")
        pattern = f"{patient_id}_{timepoint}_{modality}_pp.nii"
    else:
        scan_dir = os.path.join(patient_dir, "orig")
        pattern = f"{patient_id}_{timepoint}_{modality}.nii.gz"
    
    file_path = os.path.join(scan_dir, pattern)
    
    if os.path.exists(file_path):
        return file_path
    
    # Try alternative naming
    if use_preprocessed:
        pattern_alt = f"{patient_id}_{timepoint}_{modality}_pp.nii"
    else:
        pattern_alt = f"{patient_id}_{timepoint}_{modality}.nii.gz"
    
    file_path_alt = os.path.join(scan_dir, pattern_alt)
    if os.path.exists(file_path_alt):
        return file_path_alt
    
    return None


def get_mask_path(data_dir: str, patient_id: str, timepoint: str, 
                 mask_number: int = 1) -> Optional[str]:
    """
    Get the file path for a ground truth mask.
    
    Args:
        data_dir: Root directory containing patient folders
        patient_id: Patient identifier
        timepoint: Timepoint identifier
        mask_number: Mask number (1 or 2, default: 1)
        
    Returns:
        File path if found, None otherwise
    """
    patient_dir = os.path.join(data_dir, patient_id)
    masks_dir = os.path.join(patient_dir, "masks")
    
    pattern = f"{patient_id}_{timepoint}_mask{mask_number}.nii"
    file_path = os.path.join(masks_dir, pattern)
    
    if os.path.exists(file_path):
        return file_path
    
    return None

