"""
Setup callback utilities.

This callback is responsible for:
- Creating logging / checkpoint / config directories
- Saving project & lightning configs at the start of training
- Auto-saving a checkpoint if an exception occurs
- Handling special multi-node edge cases
"""

import os
import time

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


# -----------------------------------------------------------------------------
# Global flags
# -----------------------------------------------------------------------------
# Some environments require small synchronization delays to avoid race
# conditions when multiple nodes try to access the filesystem.
MULTINODE_HACKS = True


# -----------------------------------------------------------------------------
# Setup Callback
# -----------------------------------------------------------------------------
class SetupCallback(Callback):
    """
    A PyTorch Lightning callback that prepares the training environment.

    Responsibilities
    ----------------
    ✓ Create required directories
    ✓ Save configs for reproducibility
    ✓ Save emergency checkpoints on crash
    ✓ Handle multinode directory quirks
    """

    def __init__(
        self,
        resume,
        now,
        logdir,
        ckptdir,
        cfgdir,
        config,
        lightning_config,
        debug,
        ckpt_name=None,
    ):
        super().__init__()

        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.ckpt_name = ckpt_name

    # -------------------------------------------------------------------------
    # Exception Handling
    # -------------------------------------------------------------------------
    @rank_zero_only
    def on_exception(self, trainer: pl.Trainer, pl_module, exception):
        """
        If training crashes, save a checkpoint so progress is not lost.
        """
        print(f"Exception occurred: {exception}")

        if not self.debug and trainer.global_rank == 0:
            print("Summoning checkpoint.")

            ckpt_filename = "last.ckpt" if self.ckpt_name is None else self.ckpt_name
            ckpt_path = os.path.join(self.ckptdir, ckpt_filename)

            trainer.save_checkpoint(ckpt_path)

    # -------------------------------------------------------------------------
    # Training Start
    # -------------------------------------------------------------------------
    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        """
        Runs once before training actually begins.

        Creates folders and stores:
        - project configuration
        - lightning configuration
        """
        if trainer.global_rank == 0:
            # -----------------------------------------------------------------
            # Create directories
            # -----------------------------------------------------------------
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # Additional folder for step-based checkpoints
            if (
                "callbacks" in self.lightning_config
                and "metrics_over_trainsteps_checkpoint"
                in self.lightning_config["callbacks"]
            ):
                os.makedirs(
                    os.path.join(self.ckptdir, "trainstep_checkpoints"),
                    exist_ok=True,
                )

            # -----------------------------------------------------------------
            # Print & save configs
            # -----------------------------------------------------------------
            print("Project config")
            print(OmegaConf.to_yaml(self.config))

            # Small wait for distributed setups to avoid FS conflicts
            if MULTINODE_HACKS:
                time.sleep(5)

            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, f"{self.now}-project.yaml"),
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))

            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"),
            )

        else:
            # -----------------------------------------------------------------
            # Non-zero ranks
            # -----------------------------------------------------------------
            # In some setups ModelCheckpoint may create a logdir for every
            # process. We move those into a child_runs folder to keep things tidy.
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst_root, name = os.path.split(self.logdir)
                dst = os.path.join(dst_root, "child_runs", name)

                os.makedirs(os.path.split(dst)[0], exist_ok=True)

                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass
