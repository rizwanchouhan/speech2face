"""
Testing entry point.

✓ build model
✓ build datamodule
✓ load pretrained weights
✓ run evaluation
"""

import hydra
import torch
from pytorch_lightning import Trainer

from datamodule.data_module import DataModule


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def build_model(cfg):
    """Select correct LightningModule depending on modality."""
    if cfg.data.modality in ["audio", "visual"]:
        from lightning import ModelModule
    elif cfg.data.modality == "audiovisual":
        from lightning_av import ModelModule
    else:
        raise ValueError(f"Unsupported modality: {cfg.data.modality}")

    return ModelModule(cfg)


def load_pretrained_weights(modelmodule, path):
    """Load checkpoint into inner model."""
    if not path:
        raise ValueError("pretrained_model_path must be provided for testing.")

    state = torch.load(path, map_location=lambda storage, loc: storage)
    modelmodule.model.load_state_dict(state)


# -----------------------------------------------------------------------------
# Hydra main
# -----------------------------------------------------------------------------
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    # -------------------------------------------------------------------------
    # Build components
    # -------------------------------------------------------------------------
    modelmodule = build_model(cfg)
    datamodule = DataModule(cfg)

    # -------------------------------------------------------------------------
    # Load weights
    # -------------------------------------------------------------------------
    load_pretrained_weights(modelmodule, cfg.pretrained_model_path)

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    trainer = Trainer(
        num_nodes=1,
        gpus=1,
    )

    # -------------------------------------------------------------------------
    # Run test
    # -------------------------------------------------------------------------
    trainer.test(model=modelmodule, datamodule=datamodule)


# -----------------------------------------------------------------------------
# Start
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
