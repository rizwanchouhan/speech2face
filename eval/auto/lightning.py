"""
Audio / Video Speech Recognition LightningModule.

- Supports beam search decoding.
- Tracks word-level edit distance for WER evaluation.
- Flexible pretrained model loading.
"""

import torch
import torchaudio
from pytorch_lightning import LightningModule

# local imports
from .cosine import WarmupCosineScheduler
from .datamodule.transforms import TextTransform
from .espnet.nets.batch_beam_search import BatchBeamSearch
from .espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from .espnet.nets.scorers.length_bonus import LengthBonus
from .espnet.nets.scorers.ctc import CTCPrefixScorer


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def compute_word_level_distance(seq1: str, seq2: str) -> int:
    """Compute word-level edit distance."""
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    """Construct a batch beam search decoder for ASR output."""
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None,  # no external LM
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


# -----------------------------------------------------------------------------
# Lightning Module
# -----------------------------------------------------------------------------
class ModelModule(LightningModule):
    """LightningModule for audio/video ASR."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Choose backbone depending on modality
        if cfg.data.modality == "audio":
            self.backbone_args = cfg.model.audio_backbone
        elif cfg.data.modality == "video":
            self.backbone_args = cfg.model.visual_backbone
        else:
            raise ValueError(f"Unsupported modality: {cfg.data.modality}")

        # Text processing
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        # Model
        self.model = E2E(len(self.token_list), self.backbone_args)

        # Pretrained weights
        if cfg.pretrained_model_path:
            self._load_pretrained(cfg)

    # -------------------------------------------------------------------------
    # Pretrained loading
    # -------------------------------------------------------------------------
    def _load_pretrained(self, cfg):
        """Load pretrained checkpoint with optional frontend/encoder transfer."""
        ckpt = torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
        if cfg.transfer_frontend:
            tmp_ckpt = {
                k: v for k, v in ckpt["model_state_dict"].items()
                if k.startswith("trunk.") or k.startswith("frontend3D.")
            }
            self.model.encoder.frontend.load_state_dict(tmp_ckpt)
        elif cfg.transfer_encoder:
            tmp_ckpt = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
            self.model.encoder.load_state_dict(tmp_ckpt, strict=True)
        else:
            self.model.load_state_dict(ckpt)

    # -------------------------------------------------------------------------
    # Optimizers
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}],
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=(0.9, 0.98),
        )

        scheduler = WarmupCosineScheduler(
            optimizer,
            self.cfg.optimizer.warmup_epochs,
            self.cfg.trainer.max_epochs,
            len(self.trainer.datamodule.train_dataloader()),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, sample):
        """Beam search decoding."""
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = [h.asdict() for h in self.beam_search(enc_feat)[:1]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def forward_no_beam(self, sample):
        """Return encoder features without beam search."""
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        return enc_feat.squeeze(0)

    # -------------------------------------------------------------------------
    # Lightning steps
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        """Compute word-level distance for WER calculation."""
        enc_feat, _ = self.model.encoder(sample["input"].unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = [h.asdict() for h in self.beam_search(enc_feat)[:1]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        actual = self.text_transform.post_process(sample["target"])
        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())

    # -------------------------------------------------------------------------
    # Shared step logic
    # -------------------------------------------------------------------------
    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc = self.model(
            batch["inputs"], batch["input_lengths"], batch["targets"]
        )
        batch_size = len(batch["inputs"])

        log_prefix = "" if step_type == "train" else "_val"
        self.log(f"loss{log_prefix}", loss, on_step=(step_type=="train"), on_epoch=True, batch_size=batch_size)
        self.log(f"loss_ctc{log_prefix}", loss_ctc, batch_size=batch_size)
        self.log(f"loss_att{log_prefix}", loss_att, batch_size=batch_size)
        self.log(f"decoder_acc{log_prefix}", acc, on_step=(step_type=="train"), on_epoch=True, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    # -------------------------------------------------------------------------
    # Epoch hooks
    # -------------------------------------------------------------------------
    def on_train_epoch_start(self):
        """Set epoch for distributed sampler if applicable."""
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_test_epoch_start(self):
        """Initialize WER tracking and beam search decoder."""
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_test_epoch_end(self):
        """Compute final WER."""
        self.log("wer", self.total_edit_distance / self.total_length)
