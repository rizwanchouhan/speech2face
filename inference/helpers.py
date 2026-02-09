"""
Stable Diffusion Sampling Utilities

Includes:
- WatermarkEmbedder: Adds a fixed watermark to images
- Img2ImgDiscretizationWrapper: Wraps a discretizer with strength scaling
- do_sample: Text-to-image sampling
- do_img2img: Image-to-image sampling
- get_batch: Prepares conditioning batch
- get_input_image_tensor: Converts PIL image to model tensor
- perform_save_locally: Saves watermarked images locally
"""

import math
import os
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from imwatermark import WatermarkEncoder
from omegaconf import ListConfig
from PIL import Image
from torch import autocast

from bert.util import append_dims


# -------------------------
# Watermarking
# -------------------------
# Fixed 48-bit watermark
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]


class WatermarkEmbedder:
    """Adds a fixed watermark to a batch of images."""

    def __init__(self, watermark: List[int]):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor of shape ([N,] B, RGB, H, W) in [0,1]

        Returns:
            Tensor of same shape, watermarked
        """
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]

        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]

        # Apply watermark to each image
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")

        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(image.device)
        image = torch.clamp(image / 255, min=0.0, max=1.0)

        if squeeze:
            image = image[0]
        return image


embed_watermark = WatermarkEmbedder(WATERMARK_BITS)


def perform_save_locally(save_path: str, samples: torch.Tensor):
    """Saves watermarked images locally."""
    os.makedirs(save_path, exist_ok=True)
    base_count = len(os.listdir(save_path))
    samples = embed_watermark(samples)

    for sample in samples:
        sample_np = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample_np.astype(np.uint8)).save(os.path.join(save_path, f"{base_count:09}.png"))
        base_count += 1


# -------------------------
# Image-to-Image Discretization
# -------------------------
class Img2ImgDiscretizationWrapper:
    """Wraps a discretizer and prunes the sigmas according to img2img strength."""

    def __init__(self, discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        sigmas = self.discretization(*args, **kwargs)
        sigmas = torch.flip(sigmas, (0,))
        prune_len = max(int(self.strength * len(sigmas)), 1)
        sigmas = sigmas[:prune_len]
        sigmas = torch.flip(sigmas, (0,))
        return sigmas


# -------------------------
# Batch & Conditioning Helpers
# -------------------------
def get_unique_embedder_keys_from_conditioner(conditioner):
    return list({x.input_key for x in conditioner.embedders})


def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], math.prod(N)).reshape(N).tolist()
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = torch.tensor([value_dict["orig_height"], value_dict["orig_width"]]).to(device).repeat(*N, 1)
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = torch.tensor([value_dict["crop_coords_top"], value_dict["crop_coords_left"]]).to(device).repeat(*N, 1)
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            batch_uc["aesthetic_score"] = torch.tensor([value_dict["negative_aesthetic_score"]]).to(device).repeat(*N, 1)
        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = torch.tensor([value_dict["target_height"], value_dict["target_width"]]).to(device).repeat(*N, 1)
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_input_image_tensor(image: Image.Image, device="cuda"):
    """Converts PIL Image to tensor for img2img processing."""
    w, h = image.size
    width, height = w - w % 64, h - h % 64
    image = image.resize((width, height))
    arr = np.array(image.convert("RGB"))[None].transpose(0, 3, 1, 2)
    return torch.from_numpy(arr).to(dtype=torch.float32).div(127.5).sub_(1.0).to(device)


# -------------------------
# Sampling Functions
# -------------------------
def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    batch2model_input: Optional[List] = None,
    return_latents=False,
    filter=None,
    device="cuda",
):
    force_uc_zero_embeddings = force_uc_zero_embeddings or []
    batch2model_input = batch2model_input or []

    with torch.no_grad(), autocast(device):
        with model.ema_scope():
            batch, batch_uc = get_batch(get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, [num_samples])
            c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc, force_uc_zero_embeddings)

            for k in c:
                if k != "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][:math.prod([num_samples])].to(device), (c, uc))

            additional_inputs = {k: batch[k] for k in batch2model_input}
            shape = (math.prod([num_samples]), C, H // F, W // F)
            randn = torch.randn(shape).to(device)

            def denoiser(x, sigma, c_):
                return model.denoiser(model.model, x, sigma, c_, **additional_inputs)

            samples_z = sampler(denoiser, randn, cond=c, uc=uc)
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, 0.0, 1.0)

            if filter:
                samples = filter(samples)
            return (samples, samples_z) if return_latents else samples


def do_img2img(
    img,
    model,
    sampler,
    value_dict,
    num_samples,
    force_uc_zero_embeddings=[],
    additional_kwargs={},
    offset_noise_level: float = 0.0,
    return_latents=False,
    skip_encode=False,
    filter=None,
    device="cuda",
):
    with torch.no_grad(), autocast(device):
        with model.ema_scope():
            batch, batch_uc = get_batch(get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, [num_samples])
            c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc, force_uc_zero_embeddings)

            for k in c:
                c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))
            for k in additional_kwargs:
                c[k] = uc[k] = additional_kwargs[k]

            z = img if skip_encode else model.encode_first_stage(img)
            noise = torch.randn_like(z)
            sigmas = sampler.discretization(sampler.num_steps)
            sigma = sigmas[0].to(z.device)

            if offset_noise_level > 0.0:
                noise = noise + offset_noise_level * append_dims(torch.randn(z.shape[0], device=z.device), z.ndim)
            noised_z = z + noise * append_dims(sigma, z.ndim)
            noised_z = noised_z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

            def denoiser(x, sigma_, c_):
                return model.denoiser(model.model, x, sigma_, c_)

            samples_z = sampler(denoiser, noised_z, cond=c, uc=uc)
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, 0.0, 1.0)

            if filter:
                samples = filter(samples)
            return (samples, samples_z) if return_latents else samples
