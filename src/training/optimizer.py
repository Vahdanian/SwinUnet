"""
Optimizer and learning rate scheduler setup
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, LambdaLR, SequentialLR
from typing import Dict, Any


def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional optimizer parameters
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int, base_lr: float):
    """
    Create a warmup scheduler that linearly increases LR from 0 to base_lr.
    
    Args:
        optimizer: Optimizer instance
        warmup_epochs: Number of warmup epochs
        base_lr: Base learning rate to warmup to
        
    Returns:
        Warmup scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    warmup_epochs: int = 0,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with optional warmup.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ("cosine", "plateau", "step", "cosine_warmup")
        num_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs (0 = no warmup)
        **kwargs: Additional scheduler parameters
        
    Returns:
        Scheduler instance
    """
    base_lr = optimizer.param_groups[0]['lr']
    
    if scheduler_type.lower() == "cosine":
        T_max = kwargs.get("T_max", num_epochs)
        eta_min = kwargs.get("eta_min", 0)
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        
        if warmup_epochs > 0:
            warmup_scheduler = get_warmup_scheduler(optimizer, warmup_epochs, base_lr)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = main_scheduler
            
    elif scheduler_type.lower() == "cosine_warmup":
        # Cosine annealing with warmup
        T_max = kwargs.get("T_max", num_epochs - warmup_epochs)
        eta_min = kwargs.get("eta_min", 0)
        
        if warmup_epochs > 0:
            warmup_scheduler = get_warmup_scheduler(optimizer, warmup_epochs, base_lr)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=eta_min
            )
            
    elif scheduler_type.lower() == "plateau":
        mode = kwargs.get("mode", "min")
        factor = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 10)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )
    elif scheduler_type.lower() == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

