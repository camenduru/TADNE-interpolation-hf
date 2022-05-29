#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import subprocess
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

if os.environ.get('SYSTEM') == 'spaces':
    subprocess.call('git apply ../patch'.split(), cwd='stylegan2-pytorch')

sys.path.insert(0, 'stylegan2-pytorch')

from model import Generator

TITLE = 'TADNE (This Anime Does Not Exist) Interpolation'
DESCRIPTION = '''The original TADNE site is https://thisanimedoesnotexist.ai/.

Expected execution time on Hugging Face Spaces: 4s for one image

Related Apps:
- [TADNE](https://huggingface.co/spaces/hysts/TADNE)
- [TADNE Image Viewer](https://huggingface.co/spaces/hysts/TADNE-image-viewer)
- [TADNE Image Selector](https://huggingface.co/spaces/hysts/TADNE-image-selector)
- [TADNE Image Search with DeepDanbooru](https://huggingface.co/spaces/hysts/TADNE-image-search-with-DeepDanbooru)
'''
ARTICLE = '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.tadne-interpolation" alt="visitor badge"/></center>'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_model(device: torch.device) -> nn.Module:
    model = Generator(512, 1024, 4, channel_multiplier=2)
    path = hf_hub_download('hysts/TADNE',
                           'models/aydao-anime-danbooru2019s-512-5268480.pt')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['g_ema'])
    model.eval()
    model.to(device)
    model.latent_avg = checkpoint['latent_avg'].to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.style_dim)).to(device)
        model([z], truncation=0.7, truncation_latent=model.latent_avg)
    return model


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, z_dim)).to(device).float()


@torch.inference_mode()
def generate_image(model: nn.Module, z: torch.Tensor, truncation_psi: float,
                   randomize_noise: bool) -> np.ndarray:
    out, _ = model([z],
                   truncation=truncation_psi,
                   truncation_latent=model.latent_avg,
                   randomize_noise=randomize_noise)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


@torch.inference_mode()
def generate_interpolated_images(seed0: int, seed1: int, num_intermediate: int,
                                 psi0: float, psi1: float,
                                 randomize_noise: bool, model: nn.Module,
                                 device: torch.device) -> list[np.ndarray]:
    seed0 = int(np.clip(seed0, 0, np.iinfo(np.uint32).max))
    seed1 = int(np.clip(seed1, 0, np.iinfo(np.uint32).max))

    z0 = generate_z(model.style_dim, seed0, device)
    if num_intermediate == -1:
        out = generate_image(model, z0, psi0, randomize_noise)
        return [out], None

    z1 = generate_z(model.style_dim, seed1, device)
    vec = z1 - z0
    dvec = vec / (num_intermediate + 1)
    zs = [z0 + dvec * i for i in range(num_intermediate + 2)]
    dpsi = (psi1 - psi0) / (num_intermediate + 1)
    psis = [psi0 + dpsi * i for i in range(num_intermediate + 2)]
    res = []
    for z, psi in zip(zs, psis):
        out = generate_image(model, z, psi, randomize_noise)
        res.append(out)
    return res


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(device)

    func = functools.partial(generate_interpolated_images,
                             model=model,
                             device=device)
    func = functools.update_wrapper(func, generate_interpolated_images)

    examples = [
        [29703, 55376, 3, 0.7, 0.7, False],
        [34141, 36864, 5, 0.7, 0.7, False],
        [74650, 88322, 7, 0.7, 0.7, False],
        [84314, 70317410, 9, 0.7, 0.7, False],
        [55376, 55376, 5, 0.3, 1.3, False],
    ]

    gr.Interface(
        func,
        [
            gr.inputs.Number(default=29703, label='Seed 1'),
            gr.inputs.Number(default=55376, label='Seed 2'),
            gr.inputs.Slider(-1,
                             21,
                             step=1,
                             default=3,
                             label='Number of Intermediate Frames'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi 1'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi 2'),
            gr.inputs.Checkbox(default=False, label='Randomize Noise'),
        ],
        gr.Gallery(type='numpy', label='Output Images'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
        cache_examples=False,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
