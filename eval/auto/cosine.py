"""
Learning rate scheduler:
Warmup + Cosine decay (per iteration).

Features
✓ linear warmup
✓ cosine annealing
✓ supports multiple param groups (requires each group has a "name")
✓ step-based (not epoch-based)
"""

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, num_epochs, iter_per_epoch):
        """
        Args:
            optimizer: torch optimizer
            warmup_epochs: number of warmup epochs
            num_epochs: total training epochs
            iter_per_epoch: iterations per epoch
        """

        self.optimizer = optimizer

        # Save initial learning rates (by group name)
        self.base_lrs = {
            group["name"]: group["lr"] for group in optimizer.param_groups
        }

        # Convert epochs → iterations
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch

        self.iter = 0
        self.current_lr = 0.0

        # Initialize lr at step 0
        self.init_lr()

    # -------------------------------------------------------------------------
    # LR computation
    # -------------------------------------------------------------------------
    def get_lr(self, base_lr):
        """Compute learning rate at current iteration."""

        # Linear warmup
        if self.iter < self.warmup_iter:
            if self.warmup_iter == 0:
                return base_lr
            return base_lr * self.iter / self.warmup_iter

        # Cosine decay
        decay_iter = self.total_iter - self.warmup_iter
        if decay_iter <= 0:
            return base_lr

        progress = (self.iter - self.warmup_iter) / decay_iter

        return 0.5 * base_lr * (1 + np.cos(np.pi * progress))

    # -------------------------------------------------------------------------
    # Update optimizer
    # -------------------------------------------------------------------------
    def update_param_groups(self):
        """Apply computed LR to each param group."""
        for group in self.optimizer.param_groups:
            new_lr = self.get_lr(self.base_lrs[group["name"]])
            group["lr"] = new_lr
            self.current_lr = new_lr

    # -------------------------------------------------------------------------
    # Interface
    # -------------------------------------------------------------------------
    def step(self):
        """Advance one iteration."""
        self.update_param_groups()
        self.iter += 1

    def init_lr(self):
        """Set LR before the first training step."""
        self.update_param_groups()
