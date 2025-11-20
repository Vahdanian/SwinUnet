# Test Mode Guide

## Overview

Test mode is designed to quickly verify that the code is runnable without requiring GPU resources or processing the full dataset. It uses CPU and a small subset of data to ensure all components work correctly.

## Quick Start

### Option 1: Quick Test Script (Recommended)

Run the comprehensive test script that verifies all components:

```bash
python scripts/test_run.py
```

This will:
1. Test data loading from the dataset
2. Test model creation and forward pass
3. Test a single training step
4. Run a mini training loop (2 epochs, 2 batches)

**Expected output:**
```
================================================================
MS Lesion Segmentation - Code Runnability Test
================================================================
This script verifies that the code can run on CPU
with a small subset of data.
================================================================

================================================================
Test 1: Data Loading
================================================================
✓ Dataset created successfully
  Number of samples: 2
✓ Sample loaded successfully
  Image shape: torch.Size([4, 64, 64, 64])
  Mask shape: torch.Size([1, 64, 64, 64])
  Patient ID: training01
  Timepoint: 01

================================================================
Test 2: Model Creation
================================================================
✓ Model created successfully
  Total parameters: 1,234,567
✓ Forward pass successful
  Input shape: torch.Size([1, 4, 64, 64, 64])
  Output shape: torch.Size([1, 1, 64, 64, 64])

================================================================
Test 3: Training Step
================================================================
✓ Batch loaded
✓ Forward pass completed
  Loss: 0.7234
✓ Backward pass completed
  Training step successful
✓ Metrics computed
  Dice score: 0.1234

================================================================
Test 4: Mini Training Loop
================================================================
Running 2 training steps...
  Epoch 1, Batch 1: Loss=0.7234, Dice=0.1234
  Epoch 2, Batch 1: Loss=0.7123, Dice=0.1456
✓ Mini training loop completed successfully

================================================================
Test Summary
================================================================
✓ PASS: Data Loading
✓ PASS: Model Creation
✓ PASS: Training Step
✓ PASS: Mini Training Loop

================================================================
✓ ALL TESTS PASSED!
================================================================
The code is runnable. You can now run full training with:
  python scripts/train.py --test
================================================================
```

### Option 2: Test Mode Training

Run a short training session in test mode:

```bash
python scripts/train.py --test
```

This will:
- Use CPU (no GPU required)
- Train on only 4 samples
- Use a smaller model (64×64×64 input)
- Run for 2 epochs only
- Save outputs to `outputs/test_run/`

## Test Configuration Details

### Data Settings
- **Max samples**: 4 (for quick testing)
- **Image size**: 64×64×64 (smaller than full training)
- **Batch size**: 1 (suitable for CPU)
- **Augmentation**: Disabled (faster processing)
- **Workers**: 0 (no multiprocessing)

### Model Settings
- **Input size**: [64, 64, 64]
- **Feature size**: 24 (reduced from 48)
- **Attention heads**: [2, 4, 8, 16] (reduced)
- **Window size**: [4, 4, 4] (smaller)

### Training Settings
- **Epochs**: 2 (just for verification)
- **Device**: CPU (forced)
- **Checkpoints**: Disabled
- **TensorBoard**: Disabled

## Verification Checklist

After running test mode, verify:

- [ ] Data loads without errors
- [ ] Model creates successfully
- [ ] Forward pass completes
- [ ] Training loop runs without errors
- [ ] Loss is computed and decreases (even slightly)
- [ ] Metrics (Dice score) are computed
- [ ] No memory errors or crashes

## Troubleshooting

### Issue: "Data directory not found"
**Solution**: Ensure the ISBI_2015 dataset is in the project root:
```
SwinUnet/
├── ISBI_2015/
│   ├── training/
│   └── testdata_website/
```

### Issue: "CUDA out of memory" (even in test mode)
**Solution**: Test mode should use CPU. Check that `device: "cpu"` is set in `config/test_config.yaml`

### Issue: "Module not found" errors
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Slow performance in test mode
**Solution**: This is expected on CPU. Test mode is designed for verification, not speed. Full training should use GPU.

## Next Steps

Once test mode passes successfully:

1. **Full Training**: Run training with full configuration:
   ```bash
   python scripts/train.py --config config/training_config.yaml
   ```

2. **GPU Training**: Ensure you have a GPU available and CUDA installed for faster training.

3. **Custom Configuration**: Modify `config/training_config.yaml` for your specific needs.

## Files Created in Test Mode

- `outputs/test_run/`: Output directory (if training is run)
- No model checkpoints are saved in test mode (to save time)

## Expected Runtime

- **test_run.py**: 2-5 minutes on CPU
- **train.py --test**: 10-20 minutes on CPU (depending on hardware)

These times are for verification only. Full training will take significantly longer and should be done on GPU.

