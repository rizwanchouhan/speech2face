"""
Lightning module for Audio-Visual Speech Recognition.

This module:
✓ builds ESPNet conformer AV model
✓ supports pretrained loading / transfer
✓ runs beam-search decoding
✓ computes WER during testing
✓ logs losses & accuracy in Lightning style
"""

import torch
import torchaudio

from pytorch_lightning import LightningModule

from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def compute_word_level_distance(reference: str, hypothesis: str) -> int:
    """Compute WER edit distance on word level."""
    return torchaudio.functional.edit_distance(
        reference.lower().split(),
        hypothesis.lower().split(),
    )


# -----------------------------------------------------------------------------
# Main Lightning Module
# -----------------------------------------------------------------------------
class ModelModule(LightningModule):
    """
    PyTorch Lightning wrapper around ESPNet AV-ASR model.
    """

    def __init__(self, cfg):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Backbone configuration
        self.backbone_args = self.cfg.model.audiovisual_backbone

        # Text processing utilities
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        # Core ESPNet model
        self.model = E2E(len(self.token_list), self.backbone_args)

        # Load pretrained weights if provided
        self._load_pretrained_model()

    # -------------------------------------------------------------------------
    # Pretrained / transfer loading
    # -------------------------------------------------------------------------
    def _load_pretrained_model(self):
        if not self.cfg.pretrained_model_path:
            return

        ckpt = torch.load(
            self.cfg.pretrained_model_path,
            map_location=lambda storage, loc: storage,
        )

        print(f"Loading pretrained model from {self.cfg.pretrained_model_path}")

        if self.cfg.transfer_frontend:
            print("→ Transferring frontend weights")
            tmp_ckpt = {
                k: v
                for k, v in ckpt["model_state_dict"].items()
                if k.startswith("trunk.") or k.startswith("frontend3D.")
            }
            self.model.encoder.frontend.load_state_dict(tmp_ckpt)

        elif self.cfg.transfer_encoder:
            print("→ Transferring encoder weights")
            tmp_ckpt = {
                k.replace("encoder.", ""): v
                for k, v in ckpt.items()
                if k.startswith("encoder.")
            }
            self.model.encoder.load_state_dict(tmp_ckpt, strict=True)

        else:
            print("→ Loading full model weights")
            self.model.load_state_dict(ckpt)

    # -------------------------------------------------------------------------
    # Optimizers & schedulers
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "name": "model",
                    "params": self.model.parameters(),
                    "lr": self.cfg.optimizer.lr,
                }
            ],
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=(0.9, 0.98),
        )

        scheduler = WarmupCosineScheduler(
            optimizer,
            self.cfg.optimizer.warmup_epochs,
            self.cfg.trainer.max_epochs,
            len(self.trainer.datamodule.train_dataloader()),
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    # -------------------------------------------------------------------------
    # Inference (single sample)
    # -------------------------------------------------------------------------
    def forward(self, video, audio):
        """
        Decode one video/audio pair into text.
        """
        beam_search = get_beam_search_decoder(self.model, self.token_list)

        video_feat, _ = self.model.encoder(video.unsqueeze(0).to(self.device), None)
        audio_feat, _ = self.model.aux_encoder(audio.unsqueeze(0).to(self.device), None)

        fused = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))
        fused = fused.squeeze(0)

        nbest_hyps = beam_search(fused)
        nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]

        pred_ids = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        prediction = self.text_transform.post_process(pred_ids).replace("<eos>", "")

        return prediction

    # -------------------------------------------------------------------------
    # Train / Val steps
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, step_type="val")

    def _shared_step(self, batch, batch_idx, step_type):
        """
        Forward + loss computation used by train & val.
        """
        loss, loss_ctc, loss_att, acc = self.model(
            batch["videos"],
            batch["audios"],
            batch["video_lengths"],
            batch["audio_lengths"],
            batch["targets"],
        )

        batch_size = len(batch["videos"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_epoch=True, batch_size=batch_size)
            self.log("loss_att", loss_att, on_epoch=True, batch_size=batch_size)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)

            self.log(
                "monitoring_step",
                torch.tensor(self.global_step, dtype=torch.float32),
            )
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        return loss

    # -------------------------------------------------------------------------
    # Test (WER computation)
    # -------------------------------------------------------------------------
    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def test_step(self, sample, sample_idx):
        video_feat, _ = self.model.encoder(sample["video"].unsqueeze(0).to(self.device), None)
        audio_feat, _ = self.model.aux_encoder(sample["audio"].unsqueeze(0).to(self.device), None)

        fused = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))
        fused = fused.squeeze(0)

        nbest_hyps = self.beam_search(fused)
        nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]

        pred_ids = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(pred_ids).replace("<eos>", "")

        target_ids = sample["target"]
        actual = self.text_transform.post_process(target_ids)

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())

    def on_test_epoch_end(self):
        wer = self.total_edit_distance / self.total_length
        self.log("wer", wer)

    # -------------------------------------------------------------------------
    # Dataloader epoch sync
    # -------------------------------------------------------------------------
    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()


# -----------------------------------------------------------------------------
# Beam search factory
# -----------------------------------------------------------------------------
def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    """
    Create ESPNet batch beam search decoder.
    """

    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None,
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
