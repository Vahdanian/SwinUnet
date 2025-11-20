"""
Preprocessing script for MRI scans
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.io_utils import load_nifti, save_nifti, get_patient_list, get_timepoints_for_patient, get_modality_path
from src.data.preprocessing import normalize_intensity, resample_volume, create_brain_mask


def preprocess_patient(data_dir: str, patient_id: str, output_dir: str,
                      modalities: list, normalize: bool = True,
                      target_spacing: tuple = None):
    """Preprocess all scans for a single patient."""
    patient_output_dir = os.path.join(output_dir, patient_id, "preprocessed")
    os.makedirs(patient_output_dir, exist_ok=True)
    
    timepoints = get_timepoints_for_patient(data_dir, patient_id)
    
    for timepoint in timepoints:
        for modality in modalities:
            # Load original scan
            file_path = get_modality_path(data_dir, patient_id, timepoint, modality, use_preprocessed=False)
            
            if file_path is None:
                print(f"Warning: Could not find {modality} for {patient_id} timepoint {timepoint}")
                continue
            
            # Load image
            data, affine = load_nifti(file_path, return_affine=True)
            
            # Normalize intensity
            if normalize:
                # Create brain mask for normalization
                brain_mask = create_brain_mask(data)
                data = normalize_intensity(data, method="zscore", mask=brain_mask)
            
            # Resample if target spacing specified
            if target_spacing is not None:
                # Get original spacing from affine (simplified)
                original_spacing = (1.0, 1.0, 1.0)  # Default, should be extracted from affine
                data = resample_volume(data, original_spacing, target_spacing)
            
            # Save preprocessed image
            output_filename = f"{patient_id}_{timepoint}_{modality}_pp.nii"
            output_path = os.path.join(patient_output_dir, output_filename)
            save_nifti(data, output_path, affine=affine, dtype=np.float32)
            
            print(f"Processed: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess MRI scans')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing original scans')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for preprocessed scans')
    parser.add_argument('--normalize', action='store_true',
                       help='Apply intensity normalization')
    parser.add_argument('--resample', action='store_true',
                       help='Resample to target spacing')
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help='Target voxel spacing (x y z)')
    parser.add_argument('--modalities', type=str, nargs='+',
                       default=['flair', 'mprage', 'pd', 't2'],
                       help='Modalities to process')
    parser.add_argument('--patients', type=str, nargs='+', default=None,
                       help='Specific patients to process (default: all)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MRI Preprocessing")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Normalize: {args.normalize}")
    print(f"Resample: {args.resample}")
    if args.resample:
        print(f"Target spacing: {args.target_spacing}")
    print(f"Modalities: {args.modalities}")
    print("=" * 60)
    
    # Get patient list
    if args.patients:
        patient_ids = args.patients
    else:
        patient_ids = get_patient_list(args.input_dir, pattern="training*")
        # Also check for test patients
        test_patients = get_patient_list(args.input_dir, pattern="test*")
        patient_ids.extend(test_patients)
    
    print(f"\nProcessing {len(patient_ids)} patients...")
    
    # Process each patient
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        try:
            preprocess_patient(
                args.input_dir,
                patient_id,
                args.output_dir,
                args.modalities,
                normalize=args.normalize,
                target_spacing=tuple(args.target_spacing) if args.resample else None
            )
        except Exception as e:
            print(f"\nError processing {patient_id}: {e}")
            continue
    
    print("\nPreprocessing completed!")


if __name__ == '__main__':
    main()

