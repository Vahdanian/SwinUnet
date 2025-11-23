"""
Training script for MS lesion segmentation
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import SwinUNETR
from src.data import MSLesionDataset
from src.training import Trainer, CombinedLoss, DiceLoss, EnhancedCombinedLoss, get_optimizer, get_scheduler


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: dict):
    """Create training and validation data loaders."""
    # Create dataset
    dataset = MSLesionDataset(
        data_dir=config['data']['data_dir'],
        use_preprocessed=config['data']['use_preprocessed'],
        normalize=config['data']['normalize'],
        augmentation=config['data']['augmentation'],
        target_size=tuple(config['data']['target_size']) if config['data'].get('target_size') else None,
        modalities=config['data']['modalities']
    )
    
    # Limit dataset size for testing if specified
    max_samples = config['data'].get('max_samples', None)
    if max_samples is not None and max_samples < len(dataset):
        print(f"Limiting dataset to {max_samples} samples for testing")
        dataset, _ = random_split(
            dataset, [max_samples, len(dataset) - max_samples],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Split into train and validation
    train_split = config['data']['train_split']
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    # Disable pin_memory for CPU testing
    pin_memory = config.get('device', 'cuda') == 'cuda'
    
    # Cap num_workers to avoid warnings (max 2 workers recommended on some systems)
    num_workers = config['data']['num_workers']
    if num_workers > 0:
        import os
        # Use min of configured workers and system CPU count, but cap at reasonable max
        max_recommended = min(2, os.cpu_count() or 1)
        if num_workers > max_recommended:
            print(f"Warning: Reducing num_workers from {num_workers} to {max_recommended} to avoid potential issues")
            num_workers = max_recommended
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def create_model(config: dict) -> torch.nn.Module:
    """Create model from configuration."""
    model_config_path = config['model']['config_path']
    model_config = load_config(model_config_path)
    
    model_params = model_config['model']
    swin_params = model_config.get('swin', {})
    
    model = SwinUNETR(
        in_channels=model_params['in_channels'],
        out_channels=model_params['out_channels'],
        img_size=tuple(model_params['img_size']),
        feature_size=model_params['feature_size'],
        use_attention=model_params['use_attention'],
        attention_type=model_params.get('attention_type', 'cbam'),
        drop_rate=swin_params.get('drop_rate', 0.1),
        attn_drop_rate=swin_params.get('attn_drop_rate', 0.1),
        dropout_path_rate=swin_params.get('dropout_path_rate', 0.1)
    )
    
    return model


def compute_class_weights(dataset, device='cpu'):
    """
    Compute class weights for imbalanced data.
    Returns pos_weight for BCE loss.
    """
    total_pixels = 0
    positive_pixels = 0
    
    print("Computing class weights from dataset...")
    for i in range(min(len(dataset), 100)):  # Sample up to 100 samples
        sample = dataset[i]
        if sample['mask'] is not None:
            mask = sample['mask']
            total_pixels += mask.numel()
            positive_pixels += mask.sum().item()
    
    if positive_pixels > 0 and total_pixels > positive_pixels:
        neg_pixels = total_pixels - positive_pixels
        pos_weight = torch.tensor(neg_pixels / positive_pixels, device=device)
        print(f"Class weights computed: pos_weight={pos_weight.item():.4f}")
        return pos_weight
    else:
        print("Using default class weights")
        return None


def create_loss_function(config: dict, pos_weight: torch.Tensor = None) -> torch.nn.Module:
    """Create loss function from configuration."""
    loss_config = config['loss']
    loss_type = loss_config['type']
    
    if loss_type == 'dice':
        smooth = float(loss_config.get('smooth', 1e-5))
        return DiceLoss(smooth=smooth)
    elif loss_type == 'combined':
        return CombinedLoss(
            dice_weight=float(loss_config.get('dice_weight', 0.5)),
            bce_weight=float(loss_config.get('bce_weight', 0.5)),
            smooth=float(loss_config.get('smooth', 1e-5)),
            pos_weight=pos_weight
        )
    elif loss_type == 'enhanced':
        return EnhancedCombinedLoss(
            dice_weight=float(loss_config.get('dice_weight', 0.3)),
            tversky_weight=float(loss_config.get('tversky_weight', 0.3)),
            bce_weight=float(loss_config.get('bce_weight', 0.2)),
            focal_weight=float(loss_config.get('focal_weight', 0.2)),
            smooth=float(loss_config.get('smooth', 1e-5)),
            tversky_alpha=float(loss_config.get('tversky_alpha', 0.7)),
            tversky_beta=float(loss_config.get('tversky_beta', 0.3)),
            focal_alpha=float(loss_config.get('focal_alpha', 0.25)),
            focal_gamma=float(loss_config.get('focal_gamma', 2.0)),
            pos_weight=pos_weight
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def main():
    parser = argparse.ArgumentParser(description='Train MS Lesion Segmentation Model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory from config')
    parser.add_argument('--device', type=str, default=None,
                       help='Override device from config')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (uses test_config.yaml)')
    
    args = parser.parse_args()
    
    # Load configuration - use test config if test mode
    if args.test:
        config_path = 'config/test_config.yaml'
        print("=" * 60)
        print("RUNNING IN TEST MODE")
        print("=" * 60)
        print("This will use CPU and a small subset of data for quick verification")
        print("=" * 60)
    else:
        config_path = args.config
    
    config = load_config(config_path)
    
    # Override config with command line arguments (but respect test mode)
    if not args.test:
        if args.data_dir:
            config['data']['data_dir'] = args.data_dir
        if args.output_dir:
            config['output']['output_dir'] = args.output_dir
    if args.device:
        config['device'] = args.device
    elif args.test:
        # Force CPU in test mode
        config['device'] = 'cpu'
    
    # Set device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    if not args.test:
        print("=" * 60)
        print("MS Lesion Segmentation Training")
        print("=" * 60)
        print(f"Configuration: {config_path}")
        print(f"Data directory: {config['data']['data_dir']}")
        print(f"Output directory: {config['output']['output_dir']}")
        print(f"Device: {device}")
        print("=" * 60)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compute class weights for imbalanced data
    pos_weight = None
    if config['loss'].get('compute_class_weights', False):
        # Create a temporary dataset to compute weights
        temp_dataset = MSLesionDataset(
            data_dir=config['data']['data_dir'],
            use_preprocessed=config['data']['use_preprocessed'],
            normalize=config['data']['normalize'],
            augmentation=False,  # No augmentation for weight computation
            target_size=tuple(config['data']['target_size']) if config['data'].get('target_size') else None,
            modalities=config['data']['modalities']
        )
        pos_weight = compute_class_weights(temp_dataset, device=device)
    
    # Create loss function
    criterion = create_loss_function(config, pos_weight=pos_weight)
    
    # Create optimizer
    optimizer = get_optimizer(
        model,
        optimizer_type=config['training']['optimizer'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler_params = config['training'].get('scheduler_params', {})
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=config['training']['scheduler'],
        num_epochs=config['training']['num_epochs'],
        warmup_epochs=warmup_epochs,
        **scheduler_params
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        output_dir=config['output']['output_dir'],
        save_best=config['output']['save_best'],
        save_checkpoints=config['output'].get('save_checkpoints', True),
        early_stopping_patience=config['training']['early_stopping_patience'],
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        use_amp=config['training'].get('use_amp', False),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        empty_cache=config['training'].get('empty_cache', False)
    )
    
    # Train
    history = trainer.train(config['training']['num_epochs'])
    
    print("\nTraining completed!")
    print(f"Best validation Dice: {trainer.best_val_dice:.4f}")


if __name__ == '__main__':
    main()

