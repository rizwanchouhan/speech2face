#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Create a WebDataset from video and audio files

import os
import argparse
from itertools import islice
import webdataset as wds


def create_webdataset(filelist_path, audio_model=""):
    """
    Generator that yields samples for WebDataset.

    Args:
        filelist_path (str): Path to the text file listing video files.
        audio_model (str): Optional audio model suffix for embedding files.
    """
    with open(filelist_path, "r") as filelist:
        filelist = filelist.readlines()

    for line in filelist:
        video_file_path = line.strip()
        speaker_dir, video_file = os.path.split(video_file_path)
        audio_path = speaker_dir.replace("video", "audio")
        speaker_id = os.path.basename(speaker_dir)
        base_filename, _ = os.path.splitext(video_file)
        audio_file = base_filename + ".wav"
        audio_embedding_file = base_filename + f"_{audio_model}_emb.pt" if audio_model else base_filename + "_emb.pt"

        audio_file_path = os.path.join(audio_path, audio_file)
        audio_emb_file_path = os.path.join(audio_path, audio_embedding_file)

        if not os.path.exists(audio_file_path):
            print(f"[Warning] No audio file for video: {video_file_path}")
            continue
        if not os.path.exists(audio_emb_file_path):
            print(f"[Warning] No audio embedding file for video: {video_file_path}")
            continue

        with open(video_file_path, "rb") as f:
            video_bytes = f.read()
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
        with open(audio_emb_file_path, "rb") as f:
            audio_emb_bytes = f.read()

        yield {
            "__key__": f"{speaker_id}/{base_filename}",
            "mp4": video_bytes,
            "wav": audio_bytes,
            "pt": audio_emb_bytes,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a WebDataset from video and audio files.")
    parser.add_argument("tar_output_path", help="Output path for the tar files.")
    parser.add_argument("filelist_path", help="Path to the text file containing the list of video files.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples per tar file.")
    parser.add_argument("--audio_model", type=str, default="", help="Audio model to use for embedding files.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process.")

    args = parser.parse_args()

    os.makedirs(args.tar_output_path, exist_ok=True)

    with wds.ShardWriter(f"{args.tar_output_path}/out-%06d.tar", maxcount=args.batch_size) as sink:
        samples = create_webdataset(args.filelist_path, args.audio_model)
        if args.max_samples:
            samples = islice(samples, args.max_samples)
        for sample in samples:
            sink.write(sample)

    print(f"WebDataset creation completed. Tar files saved in {args.tar_output_path}")
