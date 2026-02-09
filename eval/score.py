"""
LipScore Metric for evaluating lip reading similarity using the auto_avsr model.

Features:
- Computes cosine similarity between predicted and ground-truth video features
- Uses a pretrained Lip Reading model (auto_avsr)
- Supports Hydra configuration
- Compatible with PyTorch Lightning metrics API
"""

import os
from typing import List

import torch
import torch.nn.functional as F
from torchmetrics import Metric
import hydra
import numpy as np

from auto_avsr.lightning import ModelModule
from auto_avsr.datamodule.transforms import VideoTransform
from auto_avsr.preparation.detectors.retinaface.detector import LandmarksDetector
from auto_avsr.preparation.detectors.retinaface.video_process import VideoProcess


class LipScore(Metric):
    """
    LipScore metric for evaluating lip reading model similarity.

    Args:
        modality (str): Data modality, e.g., 'video'
        pretrained_model_path (str): Path to pretrained checkpoint
        config_dir (str): Path to Hydra config folder
        device (torch.device): Device for model inference
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        modality: str = "video",
        pretrained_model_path: str = "PATH_TO_CHECKPOINTS/vsr_trlrs2lrs3vox2avsp_base_updated.pth",
        config_dir: str = "PATH_TO_EVALUATION_FOLDER/auto_avsr/configs",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()

        # -------------------------
        # Check files
        # -------------------------
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Model file not found at {pretrained_model_path}")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config directory not found at {config_dir}")

        # -------------------------
        # Load configuration using Hydra
        # -------------------------
        with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    f"data.modality={modality}",
                    f"pretrained_model_path={pretrained_model_path}",
                ],
            )
        self.cfg = cfg
        self.modality = cfg.data.modality
        self.device = device

        # -------------------------
        # Initialize model components
        # -------------------------
        self.video_transform = VideoTransform(subset="test")
        self.landmarks_detector = LandmarksDetector(device=str(device))
        self.video_process = VideoProcess(convert_gray=False)

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(
            torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
        )
        self.modelmodule.to(device)
        self.modelmodule.eval()

        # -------------------------
        # Metric states
        # -------------------------
        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_words", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("cosine_similarities", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.counter = 0

    # -------------------------
    # Update metric with new batch
    # -------------------------
    def update(self, videos_pred: List[torch.Tensor], videos_gt: List[torch.Tensor]):
        """
        Update the metric with predicted and ground-truth video tensors.

        Args:
            videos_pred (List[torch.Tensor]): Predicted video batch
            videos_gt (List[torch.Tensor]): Ground-truth video batch
        """
        assert len(videos_pred) == len(videos_gt), "Mismatch in number of videos"

        for pred, gt in zip(videos_pred, videos_gt):
            # Convert to HWC and scale
            pred = pred.permute(0, 2, 3, 1).numpy() * 255
            gt = gt.permute(0, 2, 3, 1).numpy() * 255

            # Extract features and landmarks
            feature_pred, _ = self.get_feature(pred)
            feature_gt, _ = self.get_feature(gt)

            # Cosine similarity
            similarity = F.cosine_similarity(feature_pred, feature_gt, dim=1)
            self.cosine_similarities += similarity.mean().cpu().item()
            self.counter += 1

    # -------------------------
    # Compute final metric
    # -------------------------
    def compute(self):
        return {"cosine_sim": self.cosine_similarities.item() / self.counter}

    # -------------------------
    # Feature extraction helper
    # -------------------------
    def get_feature(self, video: torch.Tensor):
        """
        Extract model features from video after landmarks detection and transformation.

        Args:
            video (torch.Tensor): Video tensor (T, H, W, C)
        Returns:
            feature (torch.Tensor): Feature tensor
            landmarks (List[np.ndarray]): Detected landmarks
        """
        landmarks = self.landmarks_detector(video)
        video_processed = self.video_process(video, landmarks)
        video_tensor = torch.tensor(video_processed).permute(0, 3, 1, 2)  # T, C, H, W
        video_tensor = self.video_transform(video_tensor).to(self.device)

        with torch.no_grad():
            feature = self.modelmodule.forward_no_beam(video_tensor)

        return feature, landmarks


# -----------------------------------------------------------------------------
# Demo / standalone test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import torchvision

    video_path = "PATH_TO_VIDEO"
    video = torchvision.io.read_video(video_path, pts_unit="sec")[0]
    video = video.permute(0, 3, 1, 2) / 255  # T, C, H, W

    lipscore = LipScore()
    lipscore.update([video], [video])
    print(lipscore.compute())
