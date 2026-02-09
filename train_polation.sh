#!/usr/bin/env bash
# ==============================================================
# Interpolation Training Script
# ==============================================================
#
# Launches training for the audio-driven interpolation model.
#
# Usage:
#   bash train_interpolation.sh <filelist> [num_workers] [batch_size] [num_gpus]
#
# Example:
#   bash train_interpolation.sh filelists/train.txt 8 2 4
#
# ==============================================================


# --------------------------------------------------------------
# Required argument
# --------------------------------------------------------------
FILELIST_PATH=$1

if [ -z "$FILELIST_PATH" ]; then
  echo "❌ Please provide a filelist."
  echo "Usage: bash train_interpolation.sh <filelist> [num_workers] [batch_size] [num_gpus]"
  exit 1
fi


# --------------------------------------------------------------
# Optional arguments with defaults
# --------------------------------------------------------------
NUM_WORKERS=${2:-6}
BATCH_SIZE=${3:-1}
NUM_DEVICES=${4:-1}


# --------------------------------------------------------------
# Static dataset locations
# --------------------------------------------------------------
VIDEO_DIR="videos"
AUDIO_DIR="audios"
AUDIO_EMB_DIR="audios_emb"
VIDEO_LATENT_DIR="videos_emb"


# --------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------
CONFIG_PATH="configs/example_training/interpolation/interpolation_base.yaml"
BASE_LR="1.e-5"

# Distributed / Lightning
NUM_NODES=1
STRATEGY="deepspeed_stage_1"
PRECISION=32
GRAD_ACCUM=1


# --------------------------------------------------------------
# Model behaviour
# --------------------------------------------------------------
AUDIO_CONDITION_METHOD="both_keyframes"

FREEZE_KEYS='["time_"]'
UNFREEZE_KEYS='["time_embed"]'


# --------------------------------------------------------------
# Loss configuration
# --------------------------------------------------------------
LAMBDA_LOWER=2


# --------------------------------------------------------------
# Audio embedding options
# --------------------------------------------------------------
AUDIO_IN_VIDEO=False
ADD_EXTRA_AUDIO_EMB=True


# --------------------------------------------------------------
# Launch training
# --------------------------------------------------------------
python main.py \
  --base ${CONFIG_PATH} \
  --wandb True \
  lightning.trainer.num_nodes=${NUM_NODES} \
  lightning.strategy=${STRATEGY} \
  lightning.trainer.precision=${PRECISION} \
  lightning.trainer.devices=${NUM_DEVICES} \
  lightning.trainer.accumulate_grad_batches=${GRAD_ACCUM} \
  model.base_learning_rate=${BASE_LR} \
  \
  data.params.train.datapipeline.filelist=${FILELIST_PATH} \
  data.params.train.datapipeline.video_folder=${VIDEO_DIR} \
  data.params.train.datapipeline.audio_folder=${AUDIO_DIR} \
  data.params.train.datapipeline.audio_emb_folder=${AUDIO_EMB_DIR} \
  data.params.train.datapipeline.latent_folder=${VIDEO_LATENT_DIR} \
  data.params.train.datapipeline.audio_in_video=${AUDIO_IN_VIDEO} \
  data.params.train.datapipeline.load_all_possible_indexes=False \
  data.params.train.datapipeline.add_extra_audio_emb=${ADD_EXTRA_AUDIO_EMB} \
  \
  data.params.train.loader.num_workers=${NUM_WORKERS} \
  data.params.train.loader.batch_size=${BATCH_SIZE} \
  \
  model.params.network_config.params.audio_cond_method=${AUDIO_CONDITION_METHOD} \
  model.params.loss_fn_config.params.lambda_lower=${LAMBDA_LOWER} \
  "model.params.to_freeze=${FREEZE_KEYS}" \
  "model.params.to_unfreeze=${UNFREEZE_KEYS}"
