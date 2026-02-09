"""
Inference pipeline for audio / video / audiovisual speech recognition.

Responsibilities
✓ load media
✓ detect landmarks (video)
✓ preprocess inputs
✓ apply transforms
✓ run model
"""

import os
import hydra
import torch
import torchaudio
import torchvision

from datamodule.transforms import AudioTransform, VideoTransform
from datamodule.av_dataset import cut_or_pad


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super().__init__()

        self.cfg = cfg
        self.modality = cfg.data.modality
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------------------------------------------------------------
        # Transforms
        # ---------------------------------------------------------------------
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")

        if self.modality in ["video", "audiovisual"]:
            self._init_video_pipeline(detector)
            self.video_transform = VideoTransform(subset="test")

        # ---------------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------------
        if self.modality in ["audio", "video"]:
            from lightning import ModelModule
        else:
            from lightning_av import ModelModule

        self.modelmodule = ModelModule(cfg)

        state = torch.load(
            cfg.pretrained_model_path,
            map_location=lambda storage, loc: storage,
        )
        self.modelmodule.model.load_state_dict(state)

        self.modelmodule.to(self.device)
        self.modelmodule.eval()

    # -------------------------------------------------------------------------
    # Video pipeline initialization
    # -------------------------------------------------------------------------
    def _init_video_pipeline(self, detector):
        if detector == "mediapipe":
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)

        elif detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector(device="cuda:0")
            self.video_process = VideoProcess(convert_gray=False)

        else:
            raise ValueError(f"Unknown detector: {detector}")

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)

        if not os.path.isfile(data_filename):
            raise FileNotFoundError(f"{data_filename} does not exist.")

        # ---------------------------------------------------------------------
        # Audio branch
        # ---------------------------------------------------------------------
        if self.modality in ["audio", "audiovisual"]:
            audio, sr = self.load_audio(data_filename)
            audio = self.audio_process(audio, sr)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio).to(self.device)

        # ---------------------------------------------------------------------
        # Video branch
        # ---------------------------------------------------------------------
        if self.modality in ["video", "audiovisual"]:
            video = self.load_video(data_filename)
            landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)

            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video).to(self.device)

        # ---------------------------------------------------------------------
        # Inference
        # ---------------------------------------------------------------------
        with torch.no_grad():
            if self.modality == "video":
                transcript = self.modelmodule(video)

            elif self.modality == "audio":
                transcript = self.modelmodule(audio)

            else:  # audiovisual
                self._check_av_sync(audio, video)
                transcript = self.modelmodule(video, audio)

        return transcript

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path, normalize=True)
        return waveform, sample_rate

    def load_video(self, path):
        return torchvision.io.read_video(path, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sr=16000):
        """Resample + convert to mono."""
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sr
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _check_av_sync(self, audio, video):
        """
        Ensure audio/video length match expected fps.
        Ideal ratio = 640 samples per frame (~25 fps).
        """
        ratio = len(audio) // len(video)

        if not (530 < ratio < 670):
            raise ValueError(
                "Video FPS should be between ~24 and 30."
            )

        if ratio != 640:
            print(
                f"Expected ~25fps but got {len(video) * 16000 / len(audio):.1f}. "
                "Audio will be padded/cut."
            )
            audio = cut_or_pad(audio, len(video) * 640)

        return audio


# -----------------------------------------------------------------------------
# Hydra entry point
# -----------------------------------------------------------------------------
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.file_path)
    print(f"transcript: {transcript}")


if __name__ == "__main__":
    main()
