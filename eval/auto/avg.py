"""
Checkpoint averaging utilities.

This file provides helpers to:
✓ average the last N checkpoints
✓ export a merged model file for evaluation or inference
"""

import os
from typing import List

import torch


# -----------------------------------------------------------------------------
# Core averaging
# -----------------------------------------------------------------------------
def average_checkpoints(checkpoint_paths: List[str]):
    """
    Average model weights from multiple Lightning checkpoints.

    Each checkpoint is expected to contain:
        {"state_dict": ...}

    Only parameters starting with "model." are kept.
    """

    averaged_state = None
    num_ckpts = len(checkpoint_paths)

    for ckpt_path in checkpoint_paths:
        print(f"Loading checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        state_dict = ckpt["state_dict"]

        # Remove lightning prefix -> keep pure model weights
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}

        if averaged_state is None:
            averaged_state = state_dict
        else:
            for key in averaged_state.keys():
                averaged_state[key] += state_dict[key]

    # -------------------------------------------------------------------------
    # Compute mean
    # -------------------------------------------------------------------------
    print(f"Averaging {num_ckpts} checkpoints...")

    for key in averaged_state.keys():
        if averaged_state[key] is None:
            continue

        if averaged_state[key].is_floating_point():
            averaged_state[key] /= num_ckpts
        else:
            averaged_state[key] //= num_ckpts

    return averaged_state


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def ensemble(args):
    """
    Average the last 10 epochs and save result to disk.

    Output:
        exp_dir/exp_name/model_avg_10.pth
    """

    # Collect last N checkpoints
    start_epoch = args.trainer.max_epochs - 10
    end_epoch = args.trainer.max_epochs

    checkpoint_paths = [
        os.path.join(args.exp_dir, args.exp_name, f"epoch={epoch}.ckpt")
        for epoch in range(start_epoch, end_epoch)
    ]

    output_path = os.path.join(
        args.exp_dir,
        args.exp_name,
        "model_avg_10.pth",
    )

    print("Preparing checkpoint averaging")
    print(f"→ from epoch {start_epoch} to {end_epoch - 1}")
    print(f"→ saving to {output_path}")

    averaged_state = average_checkpoints(checkpoint_paths)

    torch.save(averaged_state, output_path)

    print("Done ✅")

    return output_path
