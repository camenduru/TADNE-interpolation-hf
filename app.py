#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import random
import shlex
import subprocess
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

if os.environ.get('SYSTEM') == 'spaces':
    with open('patch') as f:
        subprocess.run(shlex.split('patch -p1'),
                       cwd='stylegan2-pytorch',
                       stdin=f)
    if not torch.cuda.is_available():
        with open('patch-cpu') as f:
            subprocess.run(shlex.split('patch -p1'),
                           cwd='stylegan2-pytorch',
                           stdin=f)

sys.path.insert(0, 'stylegan2-pytorch')

from model import Generator

DESCRIPTION = '''# [TADNE](https://thisanimedoesnotexist.ai/) (This Anime Does Not Exist) interpolation

Related Apps:
- [TADNE](https://huggingface.co/spaces/hysts/TADNE)
- [TADNE Image Viewer](https://huggingface.co/spaces/hysts/TADNE-image-viewer)
- [TADNE Image Selector](https://huggingface.co/spaces/hysts/TADNE-image-selector)
- [TADNE Image Search with DeepDanbooru](https://huggingface.co/spaces/hysts/TADNE-image-search-with-DeepDanbooru)
'''

MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def load_model(device: torch.device) -> nn.Module:
    model = Generator(512, 1024, 4, channel_multiplier=2)
    path = hf_hub_download('public-data/TADNE',
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
    seed0 = int(np.clip(seed0, 0, MAX_SEED))
    seed1 = int(np.clip(seed1, 0, MAX_SEED))

    z0 = generate_z(model.style_dim, seed0, device)
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(device)
fn = functools.partial(generate_interpolated_images,
                       model=model,
                       device=device)

examples = [
    [29703, 55376, 3, 0.7, 0.7, False],
    [34141, 36864, 5, 0.7, 0.7, False],
    [74650, 88322, 7, 0.7, 0.7, False],
    [84314, 70317410, 9, 0.7, 0.7, False],
    [55376, 55376, 5, 0.3, 1.3, False],
]

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            seed_1 = gr.Slider(label='Seed 1',
                               minimum=0,
                               maximum=MAX_SEED,
                               step=1,
                               value=29703)
            seed_2 = gr.Slider(label='Seed 2',
                               minimum=0,
                               maximum=MAX_SEED,
                               step=1,
                               value=55376)
            num_intermediate_frames = gr.Slider(
                label='Number of Intermediate Frames',
                minimum=1,
                maximum=21,
                step=1,
                value=3,
            )
            psi_1 = gr.Slider(label='Truncation psi 1',
                              minimum=0,
                              maximum=2,
                              step=0.05,
                              value=0.7)
            psi_2 = gr.Slider(label='Truncation psi 2',
                              minimum=0,
                              maximum=2,
                              step=0.05,
                              value=0.7)
            randomize_noise = gr.Checkbox(label='Randomize Noise', value=False)
            run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Gallery(label='Output')

    inputs = [
        seed_1,
        seed_2,
        num_intermediate_frames,
        psi_1,
        psi_2,
        randomize_noise,
    ]
    gr.Examples(
        examples=examples,
        inputs=inputs,
        outputs=result,
        fn=fn,
        cache_examples=os.getenv('CACHE_EXAMPLES') == '1',
    )
    run_button.click(
        fn=fn,
        inputs=inputs,
        outputs=result,
        api_name='run',
    )
demo.queue(max_size=10).launch()
