"""
Speech2Face Training Launcher
=============================

Entry point for training / testing using PyTorch Lightning.

Features
--------
- Config driven initialization (OmegaConf)
- Resume from checkpoints
- Automatic log directory handling
- Optional Weights & Biases logging
- Multi-GPU & multi-node ready
- Signal-based emergency checkpointing

This script is intentionally verbose and explicit to make
research workflows easier to debug and extend.
"""

import argparse
import datetime
import glob
import inspect
import os
import signal
import sys
from inspect import Parameter

import pytorch_lightning as pl
import torch
import wandb
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from bert.util import instantiate_from_config
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import BertTokenizer, BertModel
import numpy as np

# ==================================================
# 1. Dataset Class
# ==================================================
class Speech2FaceDataset(Dataset):
    def __init__(self, data_root, dataset_list, num_mfcc=13, max_frames=200):
        """
        data_root: Root folder of dataset
        dataset_list: text file listing audio and corresponding facial motion files
        num_mfcc: MFCC feature dimension
        """
        self.data_root = data_root
        self.samples = []
        self.num_mfcc = num_mfcc
        self.max_frames = max_frames
        
        with open(dataset_list, "r") as f:
            for line in f:
                audio_file, motion_file = line.strip().split()
                self.samples.append((audio_file, motion_file))
        
        # Load tokenizer and EmotionBERT
        self.tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
        self.emotionbert = BertModel.from_pretrained("monologg/bert-base-cased-goemotions-original")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_file, motion_file = self.samples[idx]
        audio_path = os.path.join(self.data_root, audio_file)
        motion_path = os.path.join(self.data_root, motion_file)
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(0)  # mono
        
        # Extract MFCC features
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=self.num_mfcc,
        )(waveform)  # (num_mfcc, time)
        mfcc = mfcc.transpose(0, 1)  # (time, num_mfcc)
        T_audio = mfcc.shape[0]
        
        # Load motion targets
        motion = torch.load(motion_path)  # (time, facial_dim)
        T_motion = motion.shape[0]
        
        # Pad/trim to max_frames
        if T_audio > self.max_frames:
            mfcc = mfcc[:self.max_frames]
        else:
            pad = self.max_frames - T_audio
            mfcc = torch.cat([mfcc, torch.zeros(pad, self.num_mfcc)], dim=0)
        
        if T_motion > self.max_frames:
            motion = motion[:self.max_frames]
        else:
            pad = self.max_frames - T_motion
            motion = torch.cat([motion, torch.zeros(pad, motion.shape[1])], dim=0)
        
        # Generate transcript using filename (or ASR system placeholder)
        transcript = os.path.basename(audio_file).split(".")[0].replace("_", " ")
        tokenized = self.tokenizer(transcript, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        with torch.no_grad():
            emo_emb = self.emotionbert(**tokenized).last_hidden_state.squeeze(0)  # (seq_len, emb_dim)
        
        # Interpolate EmotionBERT embeddings to match audio frames
        emo_emb_interp = nn.functional.interpolate(
            emo_emb.unsqueeze(0).transpose(1,2), size=self.max_frames, mode='linear', align_corners=False
        ).transpose(1,2).squeeze(0)  # (time, emb_dim)
        
        return mfcc, emo_emb_interp, motion

# ==================================================
# 2. Multimodal Fusion + Diffusion Model
# ==================================================
class MultimodalFusion(nn.Module):
    def __init__(self, mfcc_dim=13, emo_dim=768, hidden_dim=512):
        super().__init__()
        self.mfcc_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=mfcc_dim, nhead=1), num_layers=2
        )
        self.emo_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emo_dim, nhead=4), num_layers=2
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.proj = nn.Linear(mfcc_dim + emo_dim, hidden_dim)
    
    def forward(self, mfcc, emo):
        # mfcc: (batch, time, mfcc_dim)
        # emo: (batch, time, emo_dim)
        batch, T, _ = mfcc.shape
        mfcc_enc = self.mfcc_encoder(mfcc.transpose(0,1)).transpose(0,1)  # (batch, time, mfcc_dim)
        emo_enc = self.emo_encoder(emo.transpose(0,1)).transpose(0,1)      # (batch, time, emo_dim)
        fused = torch.cat([mfcc_enc, emo_enc], dim=-1)
        fused = self.proj(fused)
        
        # Cross-attention (self-attention variant)
        attn_out, _ = self.cross_attention(fused.transpose(0,1), fused.transpose(0,1), fused.transpose(0,1))
        return attn_out.transpose(0,1)  # (batch, time, hidden_dim)

class FacialAnimationDecoder(nn.Module):
    def __init__(self, input_dim=512, facial_dim=3*68):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, facial_dim)
        )
    
    def forward(self, fused):
        return self.fc(fused)

# ==================================================
# 3. Training Loop
# ==================================================
def train():
    dataset = Speech2FaceDataset(data_root="data/", dataset_list="data/filelist.txt")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    fusion_model = MultimodalFusion()
    decoder = FacialAnimationDecoder()
    
    fusion_model = fusion_model.cuda()
    decoder = decoder.cuda()
    
    optimizer = torch.optim.Adam(list(fusion_model.parameters()) + list(decoder.parameters()), lr=3e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(10):
        for mfcc, emo_emb, motion in loader:
            mfcc, emo_emb, motion = mfcc.cuda(), emo_emb.cuda(), motion.cuda()
            
            fused = fusion_model(mfcc, emo_emb)
            pred_motion = decoder(fused)
            
            loss = criterion(pred_motion, motion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()


def fetch_trainer_default_args():
    """Collect default Trainer arguments from Lightning."""
    signature = dict(inspect.signature(Trainer.__init__).parameters)
    signature.pop("self")

    return {
        k: signature[k].default
        for k in signature
        if signature[k] != Parameter.empty
    }


def build_cli_parser(**kwargs):
    """Create command line interface."""
    parser = argparse.ArgumentParser(**kwargs)

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    # ------------------------------------------------------------------
    # basic experiment control
    # ------------------------------------------------------------------
    parser.add_argument("-n", "--run_name", type=str, default="", nargs="?")
    parser.add_argument("--no_date", type=str2bool, default=False, nargs="?")
    parser.add_argument("-r", "--resume", type=str, default="", nargs="?")
    parser.add_argument("-b", "--base", nargs="*", default=list())
    parser.add_argument("-t", "--train", type=str2bool, default=True, nargs="?")
    parser.add_argument("--no_test", type=str2bool, default=False, nargs="?")
    parser.add_argument("-s", "--seed", type=int, default=23)
    parser.add_argument("-f", "--postfix", type=str, default="")
    parser.add_argument("-l", "--logdir", type=str, default="logs")

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    parser.add_argument("--wandb", type=str2bool, default=False, nargs="?")
    parser.add_argument("--projectname", type=str, default="speech2face")

    # ------------------------------------------------------------------
    # hardware / precision
    # ------------------------------------------------------------------
    parser.add_argument("--enable_tf32", type=str2bool, default=False, nargs="?")

    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # add lightning trainer args
    defaults = fetch_trainer_default_args()
    for key in defaults:
        parser.add_argument("--" + key, default=defaults[key])

    return parser


def find_latest_checkpoint(logdir):
    """Return latest checkpoint from log directory."""
    pattern = os.path.join(logdir, "checkpoints", "last**.ckpt")
    candidates = natsorted(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError("No checkpoint found.")

    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda x: os.path.getmtime(x))

    ckpt = candidates[-1]
    print(f"Resuming from: {ckpt}")
    return ckpt


@rank_zero_only
def setup_wandb(save_dir, run_name, project, offline=False):
    """Initialize Weights & Biases."""
    os.makedirs(save_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = save_dir

    wandb.init(
        project=project,
        name=run_name,
        mode="offline" if offline else "online",
        settings=wandb.Settings(code_dir="./"),
    )


# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    sys.path.append(os.getcwd())

    # ------------------------------------------------------------------
    # parse arguments
    # ------------------------------------------------------------------
    parser = build_cli_parser()
    opt, unknown = parser.parse_known_args()

    if opt.run_name and opt.resume:
        raise ValueError("Cannot set both --run_name and --resume.")

    # ------------------------------------------------------------------
    # log directory naming
    # ------------------------------------------------------------------
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    if opt.resume:
        if os.path.isfile(opt.resume):
            ckpt_path = opt.resume
            logdir = "/".join(ckpt_path.split("/")[:-2])
        else:
            logdir = opt.resume.rstrip("/")
            ckpt_path = find_latest_checkpoint(logdir)

        opt.resume_from_checkpoint = ckpt_path
        base_cfgs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_cfgs + opt.base
        run_name = os.path.basename(logdir)

    else:
        name = "_" + opt.run_name if opt.run_name else ""
        run_name = (now + name + opt.postfix) if not opt.no_date else name
        run_name = run_name.lstrip("_")
        logdir = os.path.join(opt.logdir, run_name)
        print("LOGDIR:", logdir)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # ------------------------------------------------------------------
    # reproducibility
    # ------------------------------------------------------------------
    seed_everything(opt.seed, workers=True)

    # ------------------------------------------------------------------
    # TF32 control
    # ------------------------------------------------------------------
    if opt.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled")

    # ------------------------------------------------------------------
    # load configs
    # ------------------------------------------------------------------
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli_conf = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli_conf)

    lightning_conf = config.pop("lightning", OmegaConf.create())
    trainer_conf = lightning_conf.get("trainer", OmegaConf.create())

    # default GPU
    trainer_conf["accelerator"] = "gpu"

    # override with CLI
    defaults = fetch_trainer_default_args()
    for k in defaults:
        if getattr(opt, k) != defaults[k]:
            trainer_conf[k] = getattr(opt, k)

    trainer_opt = argparse.Namespace(**trainer_conf)
    lightning_conf.trainer = trainer_conf

    # ------------------------------------------------------------------
    # model & data
    # ------------------------------------------------------------------
    model = instantiate_from_config(config.model)
    data = instantiate_from_config(config.data)
    data.prepare_data()

    # ------------------------------------------------------------------
    # logger
    # ------------------------------------------------------------------
    if opt.wandb:
        setup_wandb(os.path.join(os.getcwd(), logdir), run_name, opt.projectname)
        logger = pl.loggers.WandbLogger(
            name=run_name, project=opt.projectname, save_dir=logdir
        )
        logger.log_hyperparams(config)
    else:
        logger = pl.loggers.CSVLogger(save_dir=logdir, name="csv")

    # ------------------------------------------------------------------
    # checkpoint callback
    # ------------------------------------------------------------------
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=ckptdir,
        filename="{epoch:06}",
        save_last=True,
        save_top_k=3 if hasattr(model, "monitor") else -1,
        monitor=getattr(model, "monitor", None),
    )

    # ------------------------------------------------------------------
    # trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        **vars(trainer_opt),
        logger=logger,
        callbacks=[
            checkpoint_cb,
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelSummary(max_depth=1),
        ],
    )

    trainer.logdir = logdir

    # ------------------------------------------------------------------
    # learning rate scaling
    # ------------------------------------------------------------------
    if "batch_size" in config.data.params:
        bs = config.data.params.batch_size
    else:
        bs = config.data.params.train.loader.batch_size

    base_lr = config.model.base_learning_rate

    if opt.scale_lr:
        ngpu = torch.cuda.device_count()
        accumulate = trainer_conf.get("accumulate_grad_batches", 1)
        model.learning_rate = accumulate * ngpu * bs * base_lr
        print("Scaled LR:", model.learning_rate)
    else:
        model.learning_rate = base_lr
        print("Base LR:", model.learning_rate)

    # ------------------------------------------------------------------
    # signal handlers (emergency save / debug)
    # ------------------------------------------------------------------
    def emergency_checkpoint(*args, **kwargs):
        if trainer.global_rank == 0:
            path = os.path.join(ckptdir, "last.ckpt")
            print("Saving checkpoint to", path)
            trainer.save_checkpoint(path)

    def debug_break(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb

            pudb.set_trace()

    signal.signal(signal.SIGUSR1, emergency_checkpoint)
    signal.signal(signal.SIGUSR2, debug_break)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    if opt.train:
        trainer.fit(model, data, ckpt_path=opt.resume_from_checkpoint)

    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
