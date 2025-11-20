"""
Optimizer and learning rate scheduler setup
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
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


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ("cosine", "plateau", "step")
        num_epochs: Total number of training epochs
        **kwargs: Additional scheduler parameters
        
    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == "cosine":
        T_max = kwargs.get("T_max", num_epochs)
        eta_min = kwargs.get("eta_min", 0)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
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
            patience=patience,
            verbose=True
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

