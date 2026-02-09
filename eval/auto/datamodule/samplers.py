"""
Custom Samplers for Audio-Visual Speech Recognition (AVSR)

Includes:
- ByFrameCountSampler: Batching sequences based on frame counts.
- DatasetFromSampler: Wraps any sampler as a dataset.
- DistributedSamplerWrapper: Enables distributed training with any sampler.
- RandomSamplerWrapper: Wraps a sampler with random shuffling.
"""

from operator import itemgetter
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, RandomSampler, Sampler
from fairseq.data import data_utils


# -------------------------
# Sampler: Batch by frame count
# -------------------------
class ByFrameCountSampler(Sampler):
    """
    Sampler that creates batches where the total number of frames does not exceed max_frames_per_gpu.
    Useful for variable-length sequences in AVSR datasets.
    """

    def __init__(self, dataset, max_frames_per_gpu, shuffle=True, seed=0):
        self.dataset = dataset
        self.max_frames_per_gpu = max_frames_per_gpu
        self.sizes = [item[2] for item in self.dataset.list]  # input_length of each sample
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        batch_indices = data_utils.batch_by_size(
            self._get_indices(),
            lambda i: self.sizes[i],
            max_tokens=max_frames_per_gpu,
        )
        self.num_batches = len(batch_indices)

    def _get_indices(self):
        """
        Returns indices of dataset samples, optionally shuffled within equal-length groups.
        """
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = [torch.randperm(len(self.dataset), generator=g).tolist()]
        else:
            order = [list(range(len(self.dataset)))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]  # Sort by length descending

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        batch_indices = data_utils.batch_by_size(
            self._get_indices(),
            lambda i: self.sizes[i],
            max_tokens=self.max_frames_per_gpu,
        )
        return iter(batch_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


# -------------------------
# Wrap a Sampler as a Dataset
# -------------------------
class DatasetFromSampler(Dataset):
    """
    Turns any Sampler into a Dataset.
    Useful for distributed training wrappers.
    """

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


# -------------------------
# Distributed training wrapper
# -------------------------
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper to use any sampler in a distributed training setup.
    Each process gets a subset of the original sampler.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        super().__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        return iter(itemgetter(*indexes_of_indexes)(self.dataset))

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.sampler.set_epoch(epoch)


# -------------------------
# Random sampler wrapper
# -------------------------
class RandomSamplerWrapper(RandomSampler):
    """
    Wraps a sampler to provide random sampling of its indices.
    """

    def __init__(self, sampler):
        super().__init__(DatasetFromSampler(sampler))
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        return iter(itemgetter(*indexes_of_indexes)(self.dataset))
