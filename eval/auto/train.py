"""
Training script for Audio / Visual / Audiovisual ASR using PyTorch Lightning.

Features:
- Deterministic seeds for reproducibility
- Checkpointing with top-K saving
- Learning rate monitoring
- Optional multi-GPU distributed training (DDP)
- Supports checkpoint averaging after training
"""

import os
import hydra
import logging
import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from avg_ckpts import ensemble
from datamodule.data_module import DataModule


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    # -------------------------
    # Reproducibility
    # -------------------------
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    # -------------------------
    # Callbacks
    # -------------------------
    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        filename="{epoch}",
        save_last=True,
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # -------------------------
    # Load model
    # -------------------------
    if cfg.data.modality in ["audio", "visual"]:
        from lightning import ModelModule
    elif cfg.data.modality == "audiovisual":
        from lightning_av import ModelModule
    else:
        raise ValueError(f"Unsupported modality: {cfg.data.modality}")

    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)

    # -------------------------
    # Trainer setup
    # -------------------------
    trainer = Trainer(
        **cfg.trainer,
        # logger=WandbLogger(name=cfg.exp_name, project="auto_avsr"),  # optional
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=False),  # distributed multi-GPU
    )

    # -------------------------
    # Training
    # -------------------------
    trainer.fit(model=modelmodule, datamodule=datamodule)

    # -------------------------
    # Optional: Average last 10 checkpoints
    # -------------------------
    ensemble(cfg)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
