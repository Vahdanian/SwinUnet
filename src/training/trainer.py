"""
Training pipeline for MS lesion segmentation
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from tqdm import tqdm
import json

from ..evaluation.metrics import dice_score
from .losses import CombinedLoss, DiceLoss
from .optimizer import get_optimizer, get_scheduler


class Trainer:
    """
    Main training class for MS lesion segmentation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        output_dir: str = "outputs",
        save_best: bool = True,
        save_checkpoints: bool = True,
        early_stopping_patience: int = 10,
        gradient_clip_val: float = 1.0,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        empty_cache: bool = False
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (if None, will be created)
            scheduler: Learning rate scheduler (optional)
            criterion: Loss function (if None, uses CombinedLoss)
            device: Device to train on ("cuda" or "cpu")
            output_dir: Directory to save checkpoints and logs
            save_best: Whether to save best model based on validation
            early_stopping_patience: Number of epochs to wait before early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.save_best = save_best
        self.save_checkpoints = save_checkpoints
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_val = gradient_clip_val
        self.use_amp = use_amp and device == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.empty_cache = empty_cache
        
        # Mixed precision scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = get_optimizer(model, learning_rate=1e-4)
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Setup loss function
        if criterion is None:
            self.criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        else:
            self.criterion = criterion
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'val_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
        
        # Early stopping
        self.best_val_dice = 0.0
        self.epochs_without_improvement = 0
        self.best_model_state = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        self.optimizer.zero_grad()  # Zero gradients at the start
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Clear cache if requested
                if self.empty_cache and self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Compute metrics (use unscaled loss for display)
            with torch.no_grad():
                pred_binary = (torch.sigmoid(outputs) > 0.5).float()
                dice = dice_score(pred_binary, masks)
            
            # Accumulate loss (multiply by accumulation steps to get true loss)
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_dice += dice.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'dice': f'{dice.item():.4f}'
            })
        
        # Handle remaining gradients if batch count is not divisible by accumulation steps
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Clear cache if requested
            if self.empty_cache and self.device == "cuda":
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision if enabled
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Compute metrics
                pred_binary = (torch.sigmoid(outputs) > 0.5).float()
                dice = dice_score(pred_binary, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice.item():.4f}'
                })
                
                # Clear cache if requested
                if self.empty_cache and self.device == "cuda":
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def train(self, num_epochs: int) -> Dict[str, list]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            
            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics.get('loss', 0.0))
                self.history['val_dice'].append(val_metrics.get('dice', 0.0))
            
            # Learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Dice: {train_metrics['dice']:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics.get('loss', 0.0):.4f}, Val Dice: {val_metrics.get('dice', 0.0):.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            if self.save_checkpoints:
                self.save_checkpoint(epoch, val_metrics.get('dice', train_metrics['dice']))
            
            # Early stopping
            if self.val_loader is not None:
                val_dice = val_metrics.get('dice', 0.0)
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self.epochs_without_improvement = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"Best validation Dice: {self.best_val_dice:.4f}")
                    break
        
        # Save best model
        if self.save_best and self.best_model_state is not None:
            best_model_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.best_model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_dice': self.best_val_dice,
                'history': self.history
            }, best_model_path)
            print(f"\nBest model saved to {best_model_path}")
        
        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, metric: float):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
            'history': self.history
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']

