"""
Image Logging Callback for PyTorch Lightning.

This callback periodically:
    • calls `pl_module.log_images(...)`
    • saves images locally
    • optionally logs them to Weights & Biases

Supports:
    - tensor images in [-1, 1]
    - heatmaps
    - automatic mixed precision
    - train / validation splits

Designed for research reproducibility and long experiments.
"""

import os
from typing import Dict, Optional, Union

import numpy as np
import torch
import torchvision
import wandb
from einops import rearrange
from matplotlib import pyplot as plt
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from bert.util import exists, isheatmap


# ==============================================================
# Image Logger
# ==============================================================


class ImageLoggingCallback(Callback):
    """
    Periodically log images produced by the LightningModule.

    The module must implement:
        log_images(batch, split="train"/"val", **kwargs) -> Dict[str, Tensor]
    """

    def __init__(
        self,
        batch_frequency: int,
        max_images: int,
        clamp: bool = True,
        increase_log_steps: bool = True,
        rescale_to_uint8: bool = True,
        disabled: bool = False,
        log_on_batch_idx: bool = False,
        log_first_step: bool = False,
        log_images_kwargs: Optional[dict] = None,
        log_before_first_step: bool = False,
        enable_autocast: bool = True,
    ):
        super().__init__()

        # logging behaviour
        self.batch_frequency = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

        # precision / formatting
        self.enable_autocast = enable_autocast
        self.rescale_to_uint8 = rescale_to_uint8

        # extra kwargs passed to pl_module.log_images
        self.log_images_kwargs = log_images_kwargs or {}

        # early exponential logging (useful at start of training)
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_frequency)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_frequency]

    # ==========================================================
    # Utilities
    # ==========================================================

    def _should_log(self, index: int) -> bool:
        """Check whether current step should trigger logging."""
        if index:
            index -= 1

        should = (
            (index % self.batch_frequency == 0 or index in self.log_steps)
            and (index > 0 or self.log_first_step)
        )

        if should:
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass

        return should

    # ==========================================================
    # Local Saving
    # ==========================================================

    @rank_zero_only
    def _save_images_locally(
        self,
        save_dir: str,
        split: str,
        images: Dict[str, torch.Tensor],
        global_step: int,
        epoch: int,
        batch_idx: int,
        pl_module: Optional[pl.LightningModule] = None,
    ) -> None:
        """Save images to disk and optionally send to W&B."""

        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)

        for name, value in images.items():
            filename = f"{name}_gs-{global_step:06}_e-{epoch:06}_b-{batch_idx:06}.png"
            path = os.path.join(root, filename)

            # --------------------------------------------------
            # Heatmap logging
            # --------------------------------------------------
            if isheatmap(value):
                fig, ax = plt.subplots()
                ax = ax.matshow(value.cpu().numpy(), cmap="hot", interpolation="lanczos")
                plt.colorbar(ax)
                plt.axis("off")
                plt.savefig(path)
                plt.close()
                continue

            # --------------------------------------------------
            # Standard image tensor logging
            # --------------------------------------------------
            grid = torchvision.utils.make_grid(value.squeeze(2), nrow=4)

            if self.rescale_to_uint8:
                grid = (grid + 1.0) / 2.0  # [-1,1] → [0,1]

            grid = rearrange(grid.squeeze(1), "c h w -> h w c").numpy()
            grid = (grid * 255).astype(np.uint8)

            img = Image.fromarray(grid)
            img.save(path)

            # optional W&B
            if exists(pl_module):
                assert isinstance(pl_module.logger, WandbLogger), (
                    "Image logging currently supports WandbLogger only."
                )
                pl_module.logger.log_image(
                    key=f"{split}/{name}",
                    images=[img],
                    step=pl_module.global_step,
                )

    # ==========================================================
    # Main Image Collection
    # ==========================================================

    @rank_zero_only
    def _collect_and_log(
        self,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
        split: str,
    ) -> None:
        """Call model, process tensors, and log them."""

        check_index = batch_idx if self.log_on_batch_idx else pl_module.global_step

        if not (
            self._should_log(check_index)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            return

        is_training = pl_module.training
        if is_training:
            pl_module.eval()

        # AMP inference
        autocast_kwargs = {
            "enabled": self.enable_autocast,
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }

        with torch.no_grad(), torch.cuda.amp.autocast(**autocast_kwargs):
            images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

        # limit + cpu transfer
        for k in images:
            N = min(images[k].shape[0], self.max_images)

            if not isheatmap(images[k]):
                images[k] = images[k][:N]

            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()

                if self.clamp and not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self._save_images_locally(
            pl_module.logger.save_dir,
            split,
            images,
            pl_module.global_step,
            pl_module.current_epoch,
            batch_idx,
            pl_module if isinstance(pl_module.logger, WandbLogger) else None,
        )

        if is_training:
            pl_module.train()

    # ==========================================================
    # Lightning Hooks
    # ==========================================================

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self._collect_and_log(pl_module, batch, batch_idx, "train")

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self._collect_and_log(pl_module, batch, batch_idx, "train")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not self.disabled and pl_module.global_step > 0:
            self._collect_and_log(pl_module, batch, batch_idx, "val")


# ==============================================================
# W&B Initializer
# ==============================================================


@rank_zero_only
def init_wandb(save_dir, opt, config, group_name, run_name):
    """
    Initialize Weights & Biases safely in distributed setups.
    """

    print(f"Setting WANDB_DIR to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = save_dir

    if opt.debug:
        wandb.init(project=opt.projectname, mode="offline", group=group_name)
    else:
        wandb.init(
            project=opt.projectname,
            config=config,
            settings=wandb.Settings(code_dir="./bert"),
            group=group_name,
            name=run_name,
        )
