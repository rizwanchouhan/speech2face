"""
Lightning DataModule for Audio-Visual Speech Recognition (AVSR)
Handles dataset loading, sampling, padding, and batching for training, validation, and testing.
"""

import os

import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset
from .samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from .transforms import AudioTransform, VideoTransform


# -------------------------
# Collate and padding utils
# -------------------------
def pad(samples, pad_val=0.0):
    """
    Pads or truncates variable-length sequences in a batch to the maximum length.

    Args:
        samples (List[Tensor]): List of tensors of shape [T, ...]
        pad_val (float): Value to pad shorter sequences with

    Returns:
        collated_batch (Tensor): Padded tensor of shape [B, T, ...]
        lengths (List[int]): Original sequence lengths
    """
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)

    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )

    # Ensure proper shape for targets
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # [B, T, 1] for targets
    return collated_batch, lengths


def collate_pad(batch):
    """
    Collate function to handle batches of audio/video/targets with variable lengths.

    Args:
        batch (List[Dict]): List of dataset items

    Returns:
        Dict: Dictionary of padded tensors and corresponding lengths
    """
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


# -------------------------
# DataModule class
# -------------------------
class DataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for AVSR.
    Handles training, validation, and test data loaders with appropriate sampling and batching.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.cfg.gpus = torch.cuda.device_count()
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes

    def _dataloader(self, ds, sampler, collate_fn):
        return torch.utils.data.DataLoader(
            ds,
            num_workers=12,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    # -------------------------
    # Training dataloader
    # -------------------------
    def train_dataloader(self):
        ds_args = self.cfg.data.dataset
        train_ds = AVDataset(
            root_dir=ds_args.root_dir,
            label_path=os.path.join(ds_args.root_dir, ds_args.label_dir, ds_args.train_file),
            subset="train",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train"),
        )

        sampler = ByFrameCountSampler(train_ds, self.cfg.data.max_frames)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = RandomSamplerWrapper(sampler)

        return self._dataloader(train_ds, sampler, collate_pad)

    # -------------------------
    # Validation dataloader
    # -------------------------
    def val_dataloader(self):
        ds_args = self.cfg.data.dataset
        val_ds = AVDataset(
            root_dir=ds_args.root_dir,
            label_path=os.path.join(ds_args.root_dir, ds_args.label_dir, ds_args.val_file),
            subset="val",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val"),
        )

        sampler = ByFrameCountSampler(val_ds, self.cfg.data.max_frames_val, shuffle=False)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)

        return self._dataloader(val_ds, sampler, collate_pad)

    # -------------------------
    # Test dataloader
    # -------------------------
    def test_dataloader(self):
        ds_args = self.cfg.data.dataset
        test_ds = AVDataset(
            root_dir=ds_args.root_dir,
            label_path=os.path.join(ds_args.root_dir, ds_args.label_dir, ds_args.test_file),
            subset="test",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("test", snr_target=self.cfg.decode.snr_target),
            video_transform=VideoTransform("test"),
        )

        return torch.utils.data.DataLoader(test_ds, batch_size=None)
