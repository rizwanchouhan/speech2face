#!/bin/bash
# SPDX-License-Identifier: MIT
# Full inference pipeline:
# 1. Compute video embeddings
# 2. Compute audio embeddings
# 3. Create filelists
# 4. Run inference

# -----------------------------
# Parse command-line arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --video_dir)
            video_dir="${2}"
            shift 2
            ;;
        --audio_dir)
            audio_dir="${2}"
            shift 2
            ;;
        --output_folder)
            output_folder="${2}"
            shift 2
            ;;
        --keyframes_ckpt)
            keyframes_ckpt="${2}"
            shift 2
            ;;
        --interpolation_ckpt)
            interpolation_ckpt="${2}"
            shift 2
            ;;
        --compute_until)
            compute_until="${2}"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# -----------------------------
# Set defaults
# -----------------------------
video_dir=${video_dir:-"data/videos"}
audio_dir=${audio_dir:-"data/audios"}
output_folder=${output_folder:-"outputs"}
keyframes_ckpt=${keyframes_ckpt:-None}
interpolation_ckpt=${interpolation_ckpt:-None}
compute_until=${compute_until:-45}

echo "Using settings:"
echo "  Video directory       : $video_dir"
echo "  Audio directory       : $audio_dir"
echo "  Output folder         : $output_folder"
echo "  Keyframes checkpoint  : $keyframes_ckpt"
echo "  Interpolation checkpoint: $interpolation_ckpt"
echo "  Compute until         : $compute_until"

# -----------------------------
# Define directories & files
# -----------------------------
script_dir="scripts"
util_dir="$script_dir/util"
video_latent_dir="video_crop_emb"
audio_emb_dir="audio_emb"
filelist="filelist_inference.txt"
filelist_audio="filelist_inference_audio.txt"

mkdir -p "$output_folder"

# -----------------------------
# Step 1: Compute video embeddings
# -----------------------------
echo "Step 1: Computing video embeddings..."
python "$util_dir/video_to_latent.py" \
    --filelist "$video_dir"

# -----------------------------
# Step 2: Compute audio embeddings
# -----------------------------
echo "Step 2: Computing audio embeddings..."
for model_type in beats wavlm wav2vec2; do
    python "$util_dir/get_audio_embeddings.py" \
        --audio_path "$audio_dir/*.wav" \
        --model_type "$model_type" \
        --skip_video
done

# -----------------------------
# Step 3: Create filelists
# -----------------------------
echo "Step 3: Creating filelists for inference..."
python "$util_dir/create_filelist.py" \
    --root_dir "$video_dir" \
    --dest_file "$filelist" \
    --ext ".mp4" ".png" ".jpg"

python "$util_dir/create_filelist.py" \
    --root_dir "$audio_dir" \
    --dest_file "$filelist_audio" \
    --ext ".wav"

# -----------------------------
# Step 4: Run inference
# -----------------------------
echo "Step 4: Running inference..."
"$script_dir/inference.sh" \
    "$output_folder" \
    "$filelist" \
    "$keyframes_ckpt" \
    "$interpolation_ckpt" \
    "$compute_until" \
    "$filelist_audio"

echo "Inference pipeline completed successfully!"
