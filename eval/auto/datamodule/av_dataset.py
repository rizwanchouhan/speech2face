"""
Audio-Visual Dataset and Utility Functions
Handles loading, preprocessing, and padding/trimming of audio and video data
for training or inference in AVSR (Audio-Visual Speech Recognition) pipelines.
"""

import os

import torch
import torchaudio
import torchvision
import torch.nn.functional as F


# -------------------------
# Utility functions
# -------------------------
def cut_or_pad(data: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    """
    Pads or trims the data along a given dimension to the target size.
    
    Args:
        data (torch.Tensor): Input tensor
        size (int): Target size along the specified dimension
        dim (int): Dimension along which to cut/pad (default=0)
        
    Returns:
        torch.Tensor: Tensor of the specified size along `dim`
    """
    current_size = data.size(dim)
    if current_size < size:
        # Pad with zeros
        pad_size = size - current_size
        pad_shape = [0] * (2 * data.dim())
        pad_shape[-2 * (dim + 1) + 1] = pad_size  # pad end of the target dimension
        data = F.pad(data, pad_shape, "constant", 0)
    elif current_size > size:
        # Trim
        data = data.narrow(dim, 0, size)
    assert data.size(dim) == size, f"Data size {data.size(dim)} != target {size}"
    return data


def load_video(path: str) -> torch.Tensor:
    """
    Loads a video from file and returns a tensor (T, C, H, W)
    
    Args:
        path (str): Path to the video file
    
    Returns:
        torch.Tensor: Video tensor (T, C, H, W)
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute(0, 3, 1, 2)  # Convert THWC -> TCHW
    return vid


def load_audio(path: str) -> torch.Tensor:
    """
    Loads audio waveform from a .wav file corresponding to the video file
    
    Args:
        path (str): Path to the video file (audio file assumed to be .wav with same name)
    
    Returns:
        torch.Tensor: Audio tensor (T, 1)
    """
    waveform, _ = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)  # Convert to (T, 1)


# -------------------------
# Dataset class
# -------------------------
class AVDataset(torch.utils.data.Dataset):
    """
    Audio-Visual Dataset for loading video, audio, and targets.
    
    Supports modalities:
    - 'video': only video frames
    - 'audio': only audio
    - 'audiovisual': both video and audio
    """

    def __init__(
        self,
        root_dir: str,
        label_path: str,
        subset: str,
        modality: str,
        audio_transform=None,
        video_transform=None,
        rate_ratio: int = 640,
    ):
        self.root_dir = root_dir
        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)
        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path: str):
        """
        Load the dataset file list with paths, input lengths, and token IDs
        
        Returns:
            List[Tuple]: (dataset_name, rel_path, input_length, token_id_tensor)
        """
        paths_counts_labels = []
        for line in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = line.split(",")
            token_tensor = torch.tensor([int(tok) for tok in token_id.split()])
            paths_counts_labels.append((dataset_name, rel_path, int(input_length), token_tensor))
        return paths_counts_labels

    def __getitem__(self, idx: int):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)

        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}

        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}

        elif self.modality == "audiovisual":
            video = load_video(path)
            audio = load_audio(path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            return {"video": video, "audio": audio, "target": token_id}

    def __len__(self):
        return len(self.list)
