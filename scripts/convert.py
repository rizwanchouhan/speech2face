#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Converts ZeRO stage 1/2/3 checkpoints to a single FP32 PyTorch state_dict.

import argparse
import torch
import os
import glob
import math
import re
from collections import OrderedDict
from dataclasses import dataclass

from deepspeed.utils import logger
from deepspeed.checkpoint.constants import (
    DS_VERSION,
    OPTIMIZER_STATE_DICT,
    SINGLE_PARTITION_OF_FP32_GROUPS,
    FP32_FLAT_GROUPS,
    ZERO_STAGE,
    PARTITION_COUNT,
    PARAM_SHAPES,
    BUFFER_NAMES,
    FROZEN_PARAM_SHAPES,
    FROZEN_PARAM_FRAGMENTS,
)

debug = 0
device = torch.device("cpu")


@dataclass
class ZeroModelState:
    buffers: dict
    param_shapes: dict
    shared_params: list
    ds_version: int
    frozen_param_shapes: dict
    frozen_param_fragments: dict


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_checkpoint_files(checkpoint_dir, pattern):
    files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)), key=natural_keys)
    if len(files) == 0:
        raise FileNotFoundError(f"No '{pattern}' files in '{checkpoint_dir}'")
    return files


def parse_model_states(files):
    states = []
    for f in files:
        sd = torch.load(f, map_location=device)
        buffers = {k: v.float() for k, v in sd["module"].items() if k in sd[BUFFER_NAMES]}
        param_shapes = sd[PARAM_SHAPES]
        frozen_param_shapes = sd.get(FROZEN_PARAM_SHAPES)
        frozen_param_fragments = sd.get(FROZEN_PARAM_FRAGMENTS)
        shared_params = [[k, v] for k, v in sd["shared_params"].items()]
        states.append(
            ZeroModelState(
                buffers=buffers,
                param_shapes=param_shapes,
                shared_params=shared_params,
                ds_version=sd.get(DS_VERSION),
                frozen_param_shapes=frozen_param_shapes,
                frozen_param_fragments=frozen_param_fragments,
            )
        )
    return states


def parse_optim_states(files):
    state_dicts = [torch.load(f, map_location=device) for f in files]
    zero_stage = state_dicts[0][OPTIMIZER_STATE_DICT][ZERO_STAGE]
    world_size = state_dicts[0][OPTIMIZER_STATE_DICT][PARTITION_COUNT]
    if isinstance(world_size, list):
        world_size = max(world_size)
    fp32_key = SINGLE_PARTITION_OF_FP32_GROUPS if zero_stage <= 2 else FP32_FLAT_GROUPS
    fp32_groups = []
    if zero_stage <= 2:
        fp32_groups = [sd[OPTIMIZER_STATE_DICT][fp32_key] for sd in state_dicts]
    else:
        fp32_groups = [torch.cat(sd[OPTIMIZER_STATE_DICT][fp32_key], 0) for sd in state_dicts]
    return zero_stage, world_size, fp32_groups


def zero3_partition_info(numel, world_size):
    remainder = numel % world_size
    padding = (world_size - remainder) if remainder else 0
    part_numel = math.ceil(numel / world_size)
    return part_numel, padding


def merge_zero2_params(state_dict, world_size, fp32_groups, model_states, exclude_frozen=False):
    state_dict.update(model_states[0].buffers)
    if not exclude_frozen and model_states[0].frozen_param_shapes:
        for name, shape in model_states[0].frozen_param_shapes.items():
            state_dict[name] = model_states[0].frozen_param_fragments[name]

    # Merge trainable
    for shapes, fp32_vector in zip(model_states[0].param_shapes, [torch.cat([g[i] for g in fp32_groups], 0) for i in range(len(fp32_groups[0]))]):
        offset = 0
        for name, shape in shapes.items():
            numel = shape.numel()
            state_dict[name] = fp32_vector[offset: offset + numel].view(shape)
            offset += numel

    # Shared params
    for k, v in model_states[0].shared_params:
        if v in state_dict:
            state_dict[k] = state_dict[v]
    return state_dict


def merge_zero3_params(state_dict, world_size, fp32_groups, model_states, exclude_frozen=False):
    state_dict.update(model_states[0].buffers)
    if not exclude_frozen and model_states[0].frozen_param_shapes:
        for name, shape in model_states[0].frozen_param_shapes.items():
            fragments = [ms.frozen_param_fragments[name] for ms in model_states]
            state_dict[name] = torch.cat(fragments, 0)[:shape.numel()].view(shape)

    # Trainable
    param_shapes = {k: v for d in model_states[0].param_shapes for k, v in d.items()}
    offset = 0
    for name, shape in param_shapes.items():
        part_numel, _ = zero3_partition_info(shape.numel(), world_size)
        state_dict[name] = torch.cat([g[offset:offset + part_numel] for g in fp32_groups], 0)[:shape.numel()].view(shape)
        offset += part_numel

    # Shared params
    for k, v in model_states[0].shared_params:
        if v in state_dict:
            state_dict[k] = state_dict[v]
    return state_dict


def get_fp32_state_dict(checkpoint_dir, exclude_frozen=False):
    optim_files = get_checkpoint_files(checkpoint_dir, "*_optim_states.pt")
    zero_stage, world_size, fp32_groups = parse_optim_states(optim_files)
    model_files = get_checkpoint_files(checkpoint_dir, "*_model_states.pt")
    model_states = parse_model_states(model_files)
    if zero_stage <= 2:
        return merge_zero2_params(OrderedDict(), world_size, fp32_groups, model_states, exclude_frozen)
    return merge_zero3_params(OrderedDict(), world_size, fp32_groups, model_states, exclude_frozen)


def convert_checkpoint(checkpoint_dir, output_file, exclude_frozen=False):
    state_dict = get_fp32_state_dict(checkpoint_dir, exclude_frozen)
    print(f"Saving FP32 state dict to '{output_file}'")
    torch.save(state_dict, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str, help="DeepSpeed checkpoint folder")
    parser.add_argument("output_file", type=str, help="Output PyTorch FP32 state dict file")
    parser.add_argument("--exclude_frozen", action="store_true", help="Exclude frozen parameters")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug")
    args = parser.parse_args()

    debug = args.debug
    convert_checkpoint(args.checkpoint_dir, args.output_file, args.exclude_frozen)
