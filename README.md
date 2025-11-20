# MS Lesion Segmentation using Swin UNETR

## Project Overview

This project implements a deep learning pipeline for Multiple Sclerosis (MS) lesion segmentation in longitudinal MRI scans using a Swin UNETR (Swin Transformer-based U-Net) architecture. The method leverages self-supervised pretraining on unlabeled MRI data and multi-modal MRI inputs (T1-weighted, T2-weighted, PD-weighted, and FLAIR) to segment MS lesions with attention mechanisms optimized for detecting small and diffuse lesions.

## Dataset: ISBI 2015 MS Longitudinal Challenge

### Data Source

The dataset used in this project is from the **ISBI 2015 MS Longitudinal Challenge** (also known as the 2017-NI-Carass-Longitudinal-multiple-sclerosis-lesion-segmentation-Resource-and-challenge).

**Citation:**
```
A. Carass, S. Roy, A. Jog, et al., "Longitudinal Multiple Sclerosis Lesion Segmentation: 
Resource and Challenge", NeuroImage, 148(C):77-102, 2017.
```

**Download Information:**
- Official website: http://www.iacl.ece.jhu.edu/MSChallenge
- The dataset is provided by the Image Analysis and Communications Lab (IACL) at Johns Hopkins University
- Version: 1.0 (2015-02-11)

### Training Dataset Structure

**Location:** `ISBI_2015/training/`

**Statistics:**
- **Number of patients:** 5 (training01 through training05)
- **Modalities per scan:** 4 (FLAIR, MPRAGE/T1, PD, T2)
- **Time points:** 4 timepoints for most patients (training01, training02, training04, training05), 5 timepoints for training03
- **Ground truth:** Available as manual segmentation masks (typically 2 masks per timepoint for inter-rater agreement)

**Folder Structure:**
```
ISBI_2015/training/
├── training01/
│   ├── orig/              # Original compressed NIfTI files (.nii.gz)
│   │   ├── training01_01_flair.nii.gz
│   │   ├── training01_01_mprage.nii.gz
│   │   ├── training01_01_pd.nii.gz
│   │   ├── training01_01_t2.nii.gz
│   │   └── ... (additional timepoints)
│   ├── preprocessed/      # Preprocessed NIfTI files (.nii)
│   │   └── ... (preprocessed versions)
│   └── masks/             # Ground truth segmentation masks
│       ├── training01_01_mask1.nii
│       ├── training01_01_mask2.nii
│       └── ... (multiple masks per timepoint)
├── training02/
│   └── ... (same structure)
└── ...
```

**File Naming Convention:**
- Original files: `{patient_id}_{timepoint}_{modality}.nii.gz`
- Preprocessed files: `{patient_id}_{timepoint}_{modality}_pp.nii`
- Mask files: `{patient_id}_{timepoint}_mask{number}.nii`

**Modalities:**
- **FLAIR** (Fluid Attenuated Inversion Recovery): Best for detecting MS lesions
- **MPRAGE** (Magnetization Prepared Rapid Gradient Echo): T1-weighted high-resolution structural images
- **PD** (Proton Density): Provides good contrast for white matter
- **T2** (T2-weighted): Highlights pathological changes

**Time Points:**
- Each patient has multiple longitudinal scans (typically 4-5 timepoints)
- Timepoints are numbered sequentially (01, 02, 03, etc.)
- This enables tracking lesion evolution over time

**Ground Truth Masks:**
- Multiple manual segmentations per timepoint (mask1, mask2, etc.)
- Binary masks indicating lesion locations
- Created by expert radiologists

### Test Dataset Structure

**Location:** `ISBI_2015/testdata_website/`

**Statistics:**
- **Number of patients:** 14 (test01 through test14)
- **Modalities per scan:** 4 (same as training: FLAIR, MPRAGE, PD, T2)
- **Time points:** 4 timepoints per patient (consistent across all test patients)
- **Ground truth:** Not publicly available (for challenge evaluation)

**Folder Structure:**
```
ISBI_2015/testdata_website/
├── test01/
│   ├── orig/              # Original compressed NIfTI files (.nii.gz)
│   │   └── ... (same format as training)
│   └── preprocessed/      # Preprocessed NIfTI files (.nii)
│       └── ... (preprocessed versions)
├── test02/
│   └── ... (same structure)
└── ...
```

**Differences from Training Data:**
1. **No masks folder:** Ground truth segmentations are not provided for test data
2. **More patients:** 14 test patients vs. 5 training patients
3. **Consistent timepoints:** All test patients have exactly 4 timepoints (training patients have 4-5 timepoints)

**Preprocessing Expectations:**
- Test data includes preprocessed versions in the `preprocessed/` folder
- Preprocessing likely includes intensity normalization and spatial alignment
- The model should be compatible with the same preprocessing pipeline used for training

## Method Description

### Architecture: Swin UNETR

**Swin UNETR** (Swin Transformer-based U-Net for Medical Image Segmentation) combines:
- **Swin Transformer:** Hierarchical vision transformer with shifted windows for efficient self-attention
- **U-Net Architecture:** Encoder-decoder structure with skip connections for precise localization
- **3D Medical Image Processing:** Optimized for volumetric medical imaging

**Key Components:**

1. **Encoder (Swin Transformer Backbone):**
   - Multi-scale feature extraction using shifted window attention
   - Hierarchical representation learning at different resolutions
   - Efficient computation through window-based self-attention

2. **Decoder (U-Net Style):**
   - Progressive upsampling with skip connections
   - Feature fusion from encoder at multiple scales
   - Precise boundary localization for small lesions

3. **Multi-Modal Fusion:**
   - Input: 4-channel volume (FLAIR, T1, PD, T2)
   - Early fusion: Concatenate modalities as input channels
   - Attention mechanisms to weight important modalities

4. **Self-Supervised Pretraining:**
   - Leverage unlabeled MRI data for representation learning
   - Contrastive learning or masked image modeling
   - Transfer learned features to segmentation task

5. **Attention Mechanisms:**
   - Spatial attention for lesion localization
   - Channel attention for modality importance
   - Multi-scale attention for small and diffuse lesions

### Training Pipeline

**Preprocessing Steps:**
1. **Intensity Normalization:**
   - Z-score normalization per modality
   - Robust to intensity variations across scanners

2. **Spatial Alignment/Co-registration:**
   - Align all modalities to a common space
   - Resample to uniform voxel spacing
   - Handle different acquisition parameters

3. **Data Augmentation:**
   - Elastic deformation (simulate anatomical variations)
   - Rotation (random 3D rotations)
   - Scaling (random zoom in/out)
   - Intensity augmentation (gamma correction, noise injection)
   - Applied on-the-fly during training

**Loss Function:**
- **Dice Loss:** Primary loss for imbalanced segmentation
  - Handles class imbalance (lesions vs. background)
  - Directly optimizes Dice Similarity Coefficient
- **Combined Loss:** Dice Loss + Binary Cross-Entropy
  - BCE provides additional gradient signal
  - Weighted combination for optimal performance

**Training Strategy:**
- **Self-Supervised Pretraining:**
  - Train on unlabeled MRI data
  - Learn robust feature representations
  - Transfer to supervised segmentation

- **Supervised Fine-tuning:**
  - Train on labeled training data
  - Multi-modal input (4 channels)
  - Early stopping based on validation Dice score

- **Cross-Validation:**
  - K-fold cross-validation on training set
  - Monitor performance to prevent overfitting
  - Select best model based on validation metrics

**Optimization:**
- Optimizer: AdamW with weight decay
- Learning rate: Cosine annealing schedule
- Batch size: Adjusted based on GPU memory
- Early stopping: Stop if validation Dice doesn't improve for N epochs

### Evaluation Framework

**Metrics:**
1. **Dice Similarity Coefficient (DSC):**
   - Primary metric for segmentation accuracy
   - Measures overlap between prediction and ground truth
   - Range: 0 (no overlap) to 1 (perfect overlap)

2. **Sensitivity (True Positive Rate):**
   - Ability to detect lesions
   - Important for clinical applications

3. **Specificity (True Negative Rate):**
   - Ability to correctly identify non-lesion regions
   - Reduces false positives

4. **Additional Metrics:**
   - Hausdorff Distance (boundary accuracy)
   - Volume Difference (lesion volume estimation)

**Evaluation Protocol:**
- Evaluate on held-out validation set during training
- Final evaluation on test set (if ground truth available)
- Per-patient and aggregate statistics
- Longitudinal analysis (tracking lesions over time)

## Project Structure

```
SwinUnet/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/                   # Configuration files
│   ├── model_config.yaml     # Model architecture parameters
│   └── training_config.yaml  # Training hyperparameters
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/                 # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset class for loading MRI scans
│   │   ├── preprocessing.py  # Preprocessing functions
│   │   └── augmentation.py   # Data augmentation transforms
│   ├── models/               # Model architectures
│   │   ├── __init__.py
│   │   ├── swin_unetr.py     # Swin UNETR model implementation
│   │   └── attention.py      # Attention mechanism modules
│   ├── training/             # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py        # Main training loop
│   │   ├── losses.py         # Loss function implementations
│   │   └── optimizer.py      # Optimizer and scheduler setup
│   ├── evaluation/           # Evaluation modules
│   │   ├── __init__.py
│   │   ├── metrics.py        # Segmentation metrics (Dice, etc.)
│   │   └── visualization.py  # Result visualization
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── io_utils.py       # File I/O helpers
├── scripts/                  # Executable scripts
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── preprocess.py        # Preprocessing script
├── notebooks/                # Jupyter notebooks for analysis
│   └── data_exploration.ipynb
└── ISBI_2015/               # Dataset directory
    ├── training/            # Training dataset
    └── testdata_website/    # Test dataset
```

## Package/Module Descriptions

### `src/data/`

**Purpose:** Data loading, preprocessing, and augmentation for MRI scans.

**Modules:**
- **`dataset.py`**: 
  - `MSLesionDataset`: PyTorch Dataset class for loading multi-modal MRI scans and masks
  - Handles patient-wise and timepoint-wise data organization
  - Supports both training (with masks) and test (without masks) modes

- **`preprocessing.py`**:
  - `normalize_intensity()`: Z-score normalization per modality
  - `co_register_modalities()`: Spatial alignment of multi-modal scans
  - `resample_volume()`: Resampling to uniform voxel spacing
  - `crop_padding()`: Handle variable image sizes

- **`augmentation.py`**:
  - `ElasticDeformation`: 3D elastic deformation transform
  - `RandomRotation3D`: Random 3D rotations
  - `RandomScaling`: Random zoom operations
  - `IntensityAugmentation`: Gamma correction, noise injection

### `src/models/`

**Purpose:** Neural network model architectures.

**Modules:**
- **`swin_unetr.py`**:
  - `SwinUNETR`: Main model class implementing Swin UNETR architecture
  - Encoder: Swin Transformer with hierarchical feature extraction
  - Decoder: U-Net style decoder with skip connections
  - Multi-modal input handling (4-channel input)

- **`attention.py`**:
  - `SpatialAttention`: Attention mechanism for spatial feature weighting
  - `ChannelAttention`: Attention for modality/channel importance
  - `MultiScaleAttention`: Multi-scale attention for small lesions

### `src/training/`

**Purpose:** Training pipeline, loss functions, and optimization.

**Modules:**
- **`trainer.py`**:
  - `Trainer`: Main training class
  - Handles training loop, validation, checkpointing
  - Implements early stopping and learning rate scheduling

- **`losses.py`**:
  - `DiceLoss`: Dice loss for imbalanced segmentation
  - `CombinedLoss`: Dice + BCE combined loss
  - `FocalLoss`: Optional focal loss variant

- **`optimizer.py`**:
  - `get_optimizer()`: AdamW optimizer setup
  - `get_scheduler()`: Cosine annealing learning rate scheduler

### `src/evaluation/`

**Purpose:** Model evaluation and metrics computation.

**Modules:**
- **`metrics.py`**:
  - `dice_score()`: Dice Similarity Coefficient computation
  - `sensitivity()`: Sensitivity (recall) calculation
  - `specificity()`: Specificity calculation
  - `hausdorff_distance()`: Boundary accuracy metric

- **`visualization.py`**:
  - `visualize_prediction()`: Overlay predictions on MRI slices
  - `plot_metrics()`: Training/validation curves
  - `save_results()`: Save segmentation results as NIfTI files

### `src/utils/`

**Purpose:** Utility functions for file I/O and common operations.

**Modules:**
- **`io_utils.py`**:
  - `load_nifti()`: Load NIfTI files using nibabel
  - `save_nifti()`: Save arrays as NIfTI files
  - `get_patient_list()`: Extract patient IDs from directory structure

## Dependencies

### Core Dependencies

- **Python 3.8+**
- **PyTorch 1.12+**: Deep learning framework
- **torchvision**: Image transforms and utilities
- **nibabel**: NIfTI file I/O
- **numpy**: Numerical computations
- **scipy**: Scientific computing (for preprocessing)
- **scikit-image**: Image processing utilities

### Optional Dependencies

- **monai**: Medical imaging deep learning framework (useful for Swin UNETR implementation)
- **SimpleITK**: Advanced medical image processing
- **matplotlib**: Visualization
- **tensorboard**: Training visualization
- **tqdm**: Progress bars

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage Instructions

### 1. Data Preprocessing

Before training, preprocess the raw MRI scans:

```bash
python scripts/preprocess.py \
    --input_dir ISBI_2015/training \
    --output_dir ISBI_2015/training_preprocessed \
    --normalize \
    --co_register
```

**Options:**
- `--input_dir`: Directory containing original scans
- `--output_dir`: Output directory for preprocessed data
- `--normalize`: Apply intensity normalization
- `--co_register`: Perform spatial co-registration

### 2. Training

Train the Swin UNETR model:

```bash
python scripts/train.py \
    --data_dir ISBI_2015/training \
    --config config/training_config.yaml \
    --output_dir outputs/experiment_01 \
    --pretrain \
    --cross_validate
```

**Options:**
- `--data_dir`: Directory containing training data
- `--config`: Path to training configuration file
- `--output_dir`: Directory to save checkpoints and logs
- `--pretrain`: Enable self-supervised pretraining
- `--cross_validate`: Use k-fold cross-validation

### 3. Evaluation

Evaluate trained model on test data:

```bash
python scripts/evaluate.py \
    --model_path outputs/experiment_01/best_model.pth \
    --test_dir ISBI_2015/testdata_website \
    --output_dir results/test_predictions \
    --save_predictions
```

**Options:**
- `--model_path`: Path to trained model checkpoint
- `--test_dir`: Directory containing test data
- `--output_dir`: Directory to save predictions
- `--save_predictions`: Save segmentation masks as NIfTI files

### 4. Test Mode (Code Verification)

Before running full training, you can verify that the code runs correctly using test mode. Test mode uses CPU and a small subset of data for quick verification.

#### Quick Test Script

Run a comprehensive test that verifies all components:

```bash
python scripts/test_run.py
```

This script will:
- Test data loading
- Test model creation and forward pass
- Test a single training step
- Run a mini training loop (2 epochs, 2 batches)

#### Test Mode Training

Run training in test mode using the test configuration:

```bash
python scripts/train.py --test
```

Or explicitly specify the test config:

```bash
python scripts/train.py --config config/test_config.yaml --device cpu
```

**Test Mode Features:**
- **Device**: Forces CPU usage (no GPU required)
- **Data**: Limited to 4 samples for quick testing
- **Model**: Smaller model size (64×64×64 input, feature_size=24)
- **Training**: Only 2 epochs
- **Augmentation**: Disabled for faster processing
- **Batch size**: 1 (suitable for CPU)
- **No checkpoints**: Doesn't save models to save time

**Test Configuration:**
- Uses `config/test_config.yaml` for training settings
- Uses `config/test_model_config.yaml` for model architecture
- Output directory: `outputs/test_run/`

**Expected Runtime:**
- Test script: ~2-5 minutes on CPU
- Test mode training: ~10-20 minutes on CPU (depending on hardware)

**Verification Checklist:**
After running test mode, verify:
- ✓ Data loads without errors
- ✓ Model creates and runs forward pass
- ✓ Training loop completes without errors
- ✓ Loss decreases (even slightly)
- ✓ Metrics (Dice score) are computed

If all tests pass, the code is ready for full training on GPU with the complete dataset.

### 4. Configuration Files

Edit `config/training_config.yaml` to adjust hyperparameters:

```yaml
model:
  name: swin_unetr
  in_channels: 4
  out_channels: 1
  img_size: [96, 96, 96]
  feature_size: 48

training:
  batch_size: 2
  num_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 10

data:
  augmentation:
    elastic_deformation: true
    rotation: true
    scaling: true
```

## Notes

- **Dataset Discovery:** The dataset structure was independently explored and documented based on the actual file organization in `ISBI_2015/`.

- **Implementation Status:** This README documents the planned implementation. The actual code implementation will follow this structure and methodology.

- **Focus:** The project prioritizes creating a working end-to-end pipeline quickly rather than exhaustive hyperparameter tuning or achieving state-of-the-art performance.

- **Documentation:** All major components and steps are documented in this README. Additional inline code documentation will be provided in the implementation.

## License

The dataset is subject to the ISBI 2015 MS Challenge license terms (see `ISBI_2015/training/license.txt`). The code implementation in this repository follows standard open-source practices.

## Acknowledgments

- ISBI 2015 MS Longitudinal Challenge organizers
- Johns Hopkins University IACL for providing the dataset
- Original paper: Carass et al., "Longitudinal Multiple Sclerosis Lesion Segmentation: Resource and Challenge", NeuroImage, 2017

