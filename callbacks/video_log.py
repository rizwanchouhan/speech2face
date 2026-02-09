"""
Video logging callback.

Features
--------
✓ Logs image grids or videos locally
✓ Optionally attaches audio
✓ Supports WandB uploads
✓ Works for both train & validation
✓ Mixed precision safe
"""

import os
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import torchvision
import wandb
from PIL import Image
from einops import rearrange
from moviepy import editor as mpy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from bert.util import exists, suppress_output, default


# -----------------------------------------------------------------------------
# Utility: Save video (+ optional audio)
# -----------------------------------------------------------------------------
@suppress_output
def save_audio_video(
    video,
    audio=None,
    frame_rate=25,
    sample_rate=16000,
    save_path="temp.mp4",
    keep_intermediate=False,
):
    """
    Save a video tensor and optional audio into a single mp4 file.

    Parameters
    ----------
    video : Tensor (t, c, h, w)
    audio : Tensor (channels, t)
    """
    save_path = str(save_path)

    try:
        # Write temporary silent video
        torchvision.io.write_video(
            "temp_video.mp4",
            rearrange(video.detach().cpu(), "t c h w -> t h w c").to(torch.uint8),
            frame_rate,
        )

        video_clip = mpy.VideoFileClip("temp_video.mp4")

        # Attach audio if provided
        if audio is not None:
            torchaudio.save("temp_audio.wav", audio.detach().cpu(), sample_rate)
            audio_clip = mpy.AudioFileClip("temp_audio.wav")
            video_clip = video_clip.set_audio(audio_clip)

        video_clip.write_videofile(
            save_path, fps=frame_rate, codec="libx264", audio_codec="aac", verbose=False
        )

        if not keep_intermediate:
            os.remove("temp_video.mp4")
            if audio is not None:
                os.remove("temp_audio.wav")

        return 1

    except Exception as e:
        print(e)
        print("Saving video to file failed")
        return 0


# -----------------------------------------------------------------------------
# Video Logger Callback
# -----------------------------------------------------------------------------
class VideoLogger(Callback):
    """
    Logs videos produced by `pl_module.log_videos()`.

    Supports:
    - progressive logging frequency
    - wandb upload
    - train & validation schedules
    """

    def __init__(
        self,
        batch_frequency,
        max_videos,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_videos_kwargs=None,
        log_before_first_step=False,
        enable_autocast=True,
        batch_frequency_val=None,
    ):
        super().__init__()

        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_videos = max_videos

        # progressive schedule (1,2,4,8,...)
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]

        self.batch_freq_val = default(batch_frequency_val, self.batch_freq)
        self.log_steps_val = [2**n for n in range(int(np.log2(self.batch_freq_val)) + 1)]
        if not increase_log_steps:
            self.log_steps_val = [self.batch_freq_val]

        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_videos_kwargs = log_videos_kwargs or {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

    # -------------------------------------------------------------------------
    # Local + WandB logging
    # -------------------------------------------------------------------------
    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        log_elements,
        raw_audio,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "videos", split)

        for name, element in log_elements.items():
            # -----------------------------------------------------------------
            # IMAGE GRID (B,C,H,W)
            # -----------------------------------------------------------------
            if len(element.shape) == 4:
                grid = torchvision.utils.make_grid(element, nrow=4)

                if self.rescale:
                    grid = (grid + 1.0) / 2.0

                grid = grid.permute(1, 2, 0).numpy()
                grid = (grid * 255).astype(np.uint8)

                filename = f"{name}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
                path = os.path.join(root, filename)
                os.makedirs(os.path.dirname(path), exist_ok=True)

                img = Image.fromarray(grid)
                img.save(path)

                if exists(pl_module):
                    assert isinstance(pl_module.logger, WandbLogger)
                    pl_module.logger.log_image(
                        key=f"{split}/{name}",
                        images=[img],
                        step=pl_module.global_step,
                    )

            # -----------------------------------------------------------------
            # VIDEO (B,T,C,H,W)
            # -----------------------------------------------------------------
            elif len(element.shape) == 5:
                video = element

                if self.rescale:
                    video = (video + 1.0) / 2.0

                video = (video * 255.0).permute(0, 2, 1, 3, 4).cpu().to(torch.uint8)

                for i in range(video.shape[0]):
                    filename = f"{name}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}_{i}.mp4"
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.dirname(path), exist_ok=True)

                    log_audio = raw_audio[i] if raw_audio is not None else None

                    success = save_audio_video(
                        video[i],
                        audio=log_audio.unsqueeze(0) if log_audio is not None else None,
                        frame_rate=25,
                        sample_rate=16000,
                        save_path=path,
                        keep_intermediate=False,
                    )

                    if exists(pl_module):
                        assert isinstance(pl_module.logger, WandbLogger)
                        pl_module.logger.experiment.log(
                            {
                                f"{split}/{name}": wandb.Video(
                                    path if success else video,
                                    fps=25,
                                    format="mp4",
                                )
                            }
                        )

    # -------------------------------------------------------------------------
    # Trigger logging
    # -------------------------------------------------------------------------
    @rank_zero_only
    def log_video(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        if (
            self.check_frequency(check_idx, split=split)
            and hasattr(pl_module, "log_videos")
            and callable(pl_module.log_videos)
            and self.max_videos > 0
        ):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }

            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                videos = pl_module.log_videos(batch, split=split, **self.log_videos_kwargs)

            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N].detach().float().cpu()

                if self.clamp:
                    videos[k] = torch.clamp(videos[k], -1.0, 1.0)

            raw_audio = batch.get("raw_audio", None)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                videos,
                raw_audio,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module if isinstance(pl_module.logger, WandbLogger) else None,
            )

            if is_train:
                pl_module.train()

    # -------------------------------------------------------------------------
    # Frequency rules
    # -------------------------------------------------------------------------
    def check_frequency(self, check_idx, split="train"):
        if check_idx:
            check_idx -= 1

        if split == "val":
            freq = self.batch_freq_val
            steps = self.log_steps_val
        else:
            freq = self.batch_freq
            steps = self.log_steps

        if ((check_idx % freq) == 0 or (check_idx in steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                steps.pop(0)
            except IndexError:
                pass
            return True

        return False

    # -------------------------------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------------------------------
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_video(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self.log_video(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_video(pl_module, batch, batch_idx, split="val")

        if hasattr(pl_module, "calibrate_grad_norm"):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
