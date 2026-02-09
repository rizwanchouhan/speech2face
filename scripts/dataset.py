#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Create WebDataset tar archives from video/audio files

import os
import tarfile
import argparse
from tqdm import tqdm


def create_webdataset(root_path, tar_output_path, filelist_path, batch_size=1, debug=False):
    """
    Create WebDataset tar files from videos, audio, and audio embeddings.

    Args:
        root_path (str): root directory containing 'video' and 'audio' folders
        tar_output_path (str): directory where tar files will be saved
        filelist_path (str): text file with list of video files
        batch_size (int): number of samples per tar file
        debug (bool): print debug info
    """
    audio_path = os.path.join(root_path, "audio")
    os.makedirs(tar_output_path, exist_ok=True)

    batch_count = 0
    tar_file = None
    tar_file_path = ""

    # Read all video file paths
    with open(filelist_path, "r") as f:
        video_files = [line.strip() for line in f.readlines()]

    max_files = max(1, len(video_files) // batch_size)
    n_number_length = len(str(max_files))

    for line in tqdm(video_files, desc="Creating tar files", total=len(video_files)):
        video_file_path = line
        speaker_id, video_file = os.path.split(video_file_path)
        speaker_id = os.path.basename(speaker_id)
        base_filename, _ = os.path.splitext(video_file)

        audio_file = base_filename + ".wav"
        audio_emb_file = base_filename + "_emb.pt"

        audio_file_path = os.path.join(audio_path, speaker_id, audio_file)
        audio_emb_file_path = os.path.join(audio_path, speaker_id, audio_emb_file)

        if not os.path.exists(audio_file_path):
            if debug:
                print(f"[DEBUG] Missing audio file: {audio_file_path}")
            continue

        if batch_size and batch_count % batch_size == 0:
            if tar_file is not None:
                tar_file.close()
                print(f"Created tar file: {tar_file_path}")
            file_number = str(batch_count // batch_size).zfill(n_number_length)
            tar_file_path = os.path.join(tar_output_path, f"batch_{file_number}.tar")
            tar_file = tarfile.open(tar_file_path, "w")
            if debug:
                print(f"[DEBUG] Opening new tar: {tar_file_path}")

        # Add video, audio, and embedding to tar
        tar_file.add(video_file_path, arcname=os.path.join("video", speaker_id, video_file))
        tar_file.add(audio_file_path, arcname=os.path.join("audio", speaker_id, audio_file))
        tar_file.add(audio_emb_file_path, arcname=os.path.join("audio", speaker_id, audio_emb_file))

        batch_count += 1

    if tar_file is not None:
        tar_file.close()
        print(f"Created final tar file: {tar_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WebDataset tar files from videos and audios")
    parser.add_argument("root_path", help="Root directory containing 'audio' and 'video' folders")
    parser.add_argument("tar_output_path", help="Directory to save the tar files")
    parser.add_argument("filelist_path", help="Text file listing video files")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples per tar file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    create_webdataset(args.root_path, args.tar_output_path, args.filelist_path, args.batch_size, args.debug)
