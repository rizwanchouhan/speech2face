import math
import os
import sys
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from torchvision.io import read_video, read_image
import torchaudio
from safetensors.torch import load_file as load_safetensors

# -------------------------
# Project Path Setup
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from bert.util import (  # noqa
    default,
    instantiate_from_config,
    trim_pad_audio,
    get_raw_audio,
    save_audio_video,
)

# -------------------------
# Helper Functions
# -------------------------

def merge_overlapping_segments(segments: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Merge overlapping segments by averaging frames in the overlap region.

    Args:
        segments: Tensor of shape (b, t, ...) [b: segments, t: frames, ...: extra dims]
        overlap: Number of frames overlapping between consecutive segments.

    Returns:
        Merged video tensor of shape (num_frames, ...)
    """
    b, t, *other_dims = segments.shape
    num_frames = (b - 1) * (t - overlap) + t

    # Initialize output and count tensors
    output_shape = [num_frames] + other_dims
    output = torch.zeros(output_shape, dtype=segments.dtype, device=segments.device)
    count = torch.zeros(output_shape, dtype=torch.float32, device=segments.device)

    current_index = 0
    for i in range(b):
        end_index = current_index + t
        output[current_index:end_index] += rearrange(segments[i], "... -> ...")
        count[current_index:end_index] += 1
        current_index += t - overlap

    count[count == 0] = 1
    output /= count
    return output


def create_emotion_list(
    emotion_states: List[str],
    total_frames: int,
    accentuate: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create valence and arousal values for each frame based on emotion states.

    Args:
        emotion_states: List of emotion names (e.g., ["happy", "sad"])
        total_frames: Total number of frames to generate values for
        accentuate: Whether to increase emotion intensity

    Returns:
        Tuple of (valence_tensor, arousal_tensor)
    """
    # Base emotion mappings
    emotion_values = {
        "happy": (0.85, 0.75),
        "angry": (-0.443, 0.908),
        "surprised": (0.0, 0.85),
        "sad": (-0.85, -0.35),
        "neutral": (0.0, 0.0),
        "fear": (0.181, 0.949),
        "disgusted": (-0.8, 0.5),
        "contempt": (0.307, 0.535),
        "calm": (0.65, -0.5),
        "excited": (0.9, 0.9),
        "bored": (-0.6, -0.9),
        "confused": (-0.3, 0.4),
        "anxious": (-0.85, 0.9),
        "confident": (0.7, 0.4),
        "frustrated": (-0.8, 0.6),
        "amused": (0.7, 0.5),
        "proud": (0.8, 0.4),
        "ashamed": (-0.8, -0.3),
        "grateful": (0.7, 0.2),
        "jealous": (-0.7, 0.5),
        "hopeful": (0.7, 0.3),
        "disappointed": (-0.7, -0.3),
        "curious": (0.5, 0.5),
        "overwhelmed": (-0.6, 0.8),
    }

    if accentuate:
        emotion_values = {
            k: (max(min(v[0] * 1.5, 1), -1), max(min(v[1] * 1.5, 1), -1))
            for k, v in emotion_values.items()
        }
        emotion_values["neutral"] = (0.0, 0.0)

    # Single emotion repeated
    if len(emotion_states) == 1:
        v, a = emotion_values[emotion_states[0]]
        valence = torch.full((total_frames,), v)
        arousal = torch.full((total_frames,), a)
    else:
        # Multi-emotion interpolation
        frames_per_transition = total_frames // (len(emotion_states) - 1)
        valence, arousal = [], []
        for i in range(len(emotion_states) - 1):
            start_v, start_a = emotion_values[emotion_states[i]]
            end_v, end_a = emotion_values[emotion_states[i + 1]]
            valence.extend(np.linspace(start_v, end_v, frames_per_transition))
            arousal.extend(np.linspace(start_a, end_a, frames_per_transition))
        valence = torch.tensor(valence[:total_frames])
        arousal = torch.tensor(arousal[:total_frames])

    return valence, arousal


def get_audio_indexes(main_index: int, n_audio_frames: int, max_len: int) -> List[int]:
    """
    Return audio frame indexes centered around main_index, with padding if necessary.
    """
    audio_ids = [0] * max(n_audio_frames - main_index, 0)
    audio_ids += list(range(max(main_index - n_audio_frames, 0),
                            min(main_index + n_audio_frames + 1, max_len)))
    audio_ids += [max_len - 1] * max(main_index + n_audio_frames - max_len + 1, 0)
    return audio_ids


# -------------------------
# Pipeline Input Preparation
# -------------------------

def create_pipeline_inputs(
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_interpolation: torch.Tensor,
    num_frames: int,
    video_emb: Optional[torch.Tensor] = None,
    emotions: Optional[List[str]] = None,
    overlap: int = 1,
    add_zero_flag: bool = False,
    is_image_model: bool = False,
    accentuate: bool = False,
) -> Tuple:
    """
    Prepares chunks, keyframe inputs, audio embeddings, and emotion tensors
    for the video generation pipeline.
    """
    audio_interpolation_chunks = []
    audio_image_preds = []
    emotions_chunks = []

    # Generate emotion tensors
    if emotions is not None:
        valence, arousal = create_emotion_list(emotions, audio.shape[0], accentuate)
    else:
        valence, arousal = create_emotion_list(["neutral"], audio.shape[0], accentuate)

    step = max(num_frames - overlap, 1)  # Ensure step is at least 1

    audio_image_preds_idx = []
    audio_interp_preds_idx = []

    for i in range(0, audio.shape[0] - num_frames + 1, step):
        segment_end = i + num_frames

        # Process first and last frames for each segment
        for idx in [i, segment_end - 1]:
            if idx not in audio_image_preds_idx:
                if is_image_model:
                    audio_image_preds.append(audio[get_audio_indexes(idx, 2, len(audio))])
                else:
                    audio_image_preds.append(audio[idx])
                audio_image_preds_idx.append(idx)
                emotions_chunks.append((valence[idx], arousal[idx]))

        audio_interpolation_chunks.append(audio_interpolation[i:segment_end])
        audio_interp_preds_idx.append([i, segment_end - 1])

    # Handle last incomplete chunk
    remaining_frames = len(audio_interpolation) - segment_end
    if remaining_frames > 0:
        last_chunk = audio_interpolation[segment_end:]
        if len(last_chunk) < num_frames:
            last_chunk = torch.cat([last_chunk, last_chunk[-1].repeat(num_frames - len(last_chunk), 1, 1)])
        audio_interpolation_chunks.append(last_chunk)
        audio_interp_preds_idx.append([segment_end, segment_end + num_frames - 1])

    # Add zero flag every num_frames if requested
    if add_zero_flag:
        first_element = audio_image_preds[0]
        first_element_emotions = (valence[0], arousal[0])
        for i in range(0, len(audio_image_preds), num_frames):
            audio_image_preds.insert(i, first_element)
            audio_image_preds_idx.insert(i, None)
            emotions_chunks.insert(i, first_element_emotions)

    # Prepare lists for interpolation
    to_remove = [idx is None for idx in audio_image_preds_idx]
    audio_image_preds_idx_clone = list(audio_image_preds_idx)

    interpolation_cond_list = []
    for i in range(0, len(audio_image_preds_idx) - 1, overlap if overlap > 0 else 2):
        interpolation_cond_list.append([audio_image_preds_idx[i], audio_image_preds_idx[i + 1]])

    # Pad audio_image_preds to multiple of num_frames
    frames_needed = (num_frames - (len(audio_image_preds) % num_frames)) % num_frames
    audio_image_preds += [audio_image_preds[-1]] * frames_needed
    emotions_chunks += [emotions_chunks[-1]] * frames_needed
    to_remove += [True] * frames_needed
    audio_image_preds_idx_clone += [audio_image_preds_idx_clone[-1]] * frames_needed

    return (
        audio_interpolation_chunks,
        audio_image_preds,
        video_emb[0] if video_emb is not None else None,
        video[0],
        emotions_chunks,
        0,  # random_cond_idx
        to_remove,
        audio_interp_preds_idx,
        audio_image_preds_idx_clone,
    )
