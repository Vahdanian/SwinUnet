"""
Quick test script to verify code runnability
This script runs a minimal training loop on CPU with a small dataset
"""

import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import SwinUNETR
from src.data import MSLesionDataset
from src.training import Trainer, CombinedLoss, get_optimizer, get_scheduler
from src.evaluation.metrics import dice_score


def test_data_loading():
    """Test if data can be loaded."""
    print("=" * 60)
    print("Test 1: Data Loading")
    print("=" * 60)
    
    try:
        dataset = MSLesionDataset(
            data_dir="ISBI_2015/training",
            use_preprocessed=True,
            normalize=True,
            augmentation=False,
            target_size=(64, 64, 64),
            modalities=["flair", "mprage", "pd", "t2"]
        )
        
        # Limit to 2 samples for testing
        if len(dataset) > 2:
            dataset, _ = random_split(dataset, [2, len(dataset) - 2],
                                    generator=torch.Generator().manual_seed(42))
        
        print(f"[OK] Dataset created successfully")
        print(f"  Number of samples: {len(dataset)}")
        
        # Try loading one sample
        sample = dataset[0]
        print(f"[OK] Sample loaded successfully")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape if sample['mask'] is not None else 'None'}")
        print(f"  Patient ID: {sample['patient_id']}")
        print(f"  Timepoint: {sample['timepoint']}")
        
        return True, dataset
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_creation():
    """Test if model can be created."""
    print("\n" + "=" * 60)
    print("Test 2: Model Creation")
    print("=" * 60)
    
    try:
        model = SwinUNETR(
            in_channels=4,
            out_channels=1,
            img_size=(64, 64, 64),
            feature_size=24,  # Smaller for testing
            use_attention=True,
            attention_type="cbam"
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 4, 64, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"[OK] Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True, model
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_step(model, dataset):
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("Test 3: Training Step")
    print("=" * 60)
    
    try:
        device = "cpu"
        model = model.to(device)
        model.train()
        
        # Create a small data loader
        train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Create optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        
        # Run one batch
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        print(f"[OK] Batch loaded")
        print(f"  Image shape: {images.shape}")
        print(f"  Mask shape: {masks.shape}")
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        print(f"[OK] Forward pass completed")
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"[OK] Backward pass completed")
        print(f"  Training step successful")
        
        # Compute metric
        with torch.no_grad():
            pred_binary = (torch.sigmoid(outputs) > 0.5).float()
            dice = dice_score(pred_binary, masks)
            print(f"[OK] Metrics computed")
            print(f"  Dice score: {dice.item():.4f}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training(model, dataset):
    """Test a mini training loop (2 epochs, 2 batches)."""
    print("\n" + "=" * 60)
    print("Test 4: Mini Training Loop")
    print("=" * 60)
    
    try:
        device = "cpu"
        model = model.to(device)
        
        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Create trainer components
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        
        print("Running 2 training steps...")
        for epoch in range(1, 3):
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 2:  # Limit to 2 batches
                    break
                
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    pred_binary = (torch.sigmoid(outputs) > 0.5).float()
                    dice = dice_score(pred_binary, masks)
                
                print(f"  Epoch {epoch}, Batch {batch_idx + 1}: Loss={loss.item():.4f}, Dice={dice.item():.4f}")
        
        print(f"[OK] Mini training loop completed successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Mini training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MS Lesion Segmentation - Code Runnability Test")
    print("=" * 60)
    print("This script verifies that the code can run on CPU")
    print("with a small subset of data.")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists("ISBI_2015/training"):
        print("\n[FAIL] ERROR: Data directory 'ISBI_2015/training' not found!")
        print("  Please ensure the ISBI_2015 dataset is in the project root.")
        return False
    
    results = []
    
    # Test 1: Data loading
    success, dataset = test_data_loading()
    results.append(("Data Loading", success))
    if not success:
        print("\n[FAIL] Cannot proceed without data loading. Exiting.")
        return False
    
    # Test 2: Model creation
    success, model = test_model_creation()
    results.append(("Model Creation", success))
    if not success:
        print("\n[FAIL] Cannot proceed without model. Exiting.")
        return False
    
    # Test 3: Training step
    success = test_training_step(model, dataset)
    results.append(("Training Step", success))
    
    # Test 4: Mini training loop
    success = test_mini_training(model, dataset)
    results.append(("Mini Training Loop", success))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 60)
        print("The code is runnable. You can now run full training with:")
        print("  python scripts/train.py --test")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[FAIL] SOME TESTS FAILED")
        print("=" * 60)
        print("Please check the error messages above.")
        print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

