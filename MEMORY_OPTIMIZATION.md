# Memory Optimization Guide

## Issues Fixed

### 1. Deprecation Warning ✅
- **Fixed**: Changed `torch.cuda.amp.autocast()` to `torch.amp.autocast(device_type='cuda', dtype=torch.float16)`
- **Impact**: No more deprecation warnings, future-proof code

### 2. CUDA Out of Memory Error ✅
- **Added**: Gradient accumulation support
- **Added**: Automatic cache clearing between batches
- **Added**: Non-blocking data transfer

## Memory Optimization Features

### Gradient Accumulation
- **What it does**: Accumulates gradients over multiple batches before updating weights
- **Benefits**: 
  - Reduces memory usage (effective batch size = batch_size × gradient_accumulation_steps)
  - Allows training with larger effective batch sizes on limited GPU memory
- **Configuration**: Set `gradient_accumulation_steps: 2` (or higher) in training config
- **Example**: With batch_size=2 and gradient_accumulation_steps=2, effective batch size = 4

### Automatic Cache Clearing
- **What it does**: Clears CUDA cache between batches to reduce memory fragmentation
- **Benefits**: Helps prevent OOM errors from memory fragmentation
- **Configuration**: Set `empty_cache: true` in training config

### Non-blocking Data Transfer
- **What it does**: Uses `non_blocking=True` for data transfers
- **Benefits**: Overlaps data transfer with computation, slightly faster

## Recommended Settings for Different GPU Sizes

### For 16GB GPU (like yours)
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 2  # Effective batch size = 4
  use_amp: true
  empty_cache: true
```

### For 8GB GPU
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 4  # Effective batch size = 4
  use_amp: true
  empty_cache: true
```

### For 24GB+ GPU
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 1  # No accumulation needed
  use_amp: true
  empty_cache: false  # Not needed with enough memory
```

## Additional Memory Saving Tips

1. **Reduce Batch Size**: If still getting OOM, reduce `batch_size` to 1
2. **Increase Gradient Accumulation**: Increase `gradient_accumulation_steps` to maintain effective batch size
3. **Reduce Image Size**: Reduce `target_size` in data config (e.g., from [96,96,96] to [64,64,64])
4. **Reduce Feature Size**: Reduce `feature_size` in model config (e.g., from 48 to 32)
5. **Disable Mixed Precision**: If AMP causes issues, set `use_amp: false` (but this uses more memory)
6. **Use Gradient Checkpointing**: Can be added to model for more memory savings (trades memory for compute)

## Current Configuration

The training config has been updated with:
- `gradient_accumulation_steps: 2` - Accumulate over 2 batches
- `empty_cache: true` - Clear cache between batches

This should significantly reduce memory usage while maintaining training effectiveness.

## Monitoring Memory Usage

You can monitor GPU memory usage during training:
```python
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## If Still Getting OOM Errors

1. **Reduce batch_size to 1** in `config/training_config.yaml`
2. **Increase gradient_accumulation_steps to 4 or 8**
3. **Reduce target_size** to [64, 64, 64] or [48, 48, 48]
4. **Reduce feature_size** to 32 in model config
5. **Close other applications** using GPU memory
6. **Restart Python kernel** to clear any cached memory

