#!/usr/bin/env python
"""
src/training/architectures.py
=============================
One interface over the denoising architectures, so each can be swept and scored
under an identical protocol.

Why this exists
---------------
Only the U-Net has ever been hyperparameter-searched on line-emission data. The
autoencoder and VAE numbers still quoted in the report are Week-2 results on
64x64 *continuum patches*, and the DDPM was hand-tuned once. Comparing a swept
U-Net against untuned models and concluding the U-Net architecture wins is not a
defensible claim -- the tuning budget is confounded with the architecture.

Two further problems made the comparison unfair in the opposite direction, and
both are fixed here rather than papered over:

  * the autoencoder and VAE ended in a sigmoid. Under the shared dirty-scale
    normalisation the clean target can exceed 1, so a sigmoid *cannot represent
    the target*. Removing it from the U-Net was half of the 5ed8fc6 fix that took
    Moment-0 from -6402% to positive. Both are built here with `linear_head=True`.
  * neither was parameterised, so neither could be swept at all. Both now take
    `base_channels`, matching the U-Net's width knob -- the parameter the sweep
    re-analysis suggests matters most once training duration is controlled for.

The VAE is the one architecture whose training objective genuinely differs: it
carries a KL term, so `kl_weight` is part of its search space. Everything else
about the protocol -- fixed PSNR/SSIM/MSE scoring, early stopping, the cube-level
split -- is identical across architectures by construction.
"""

import math
from typing import Callable, Dict, Sequence

import torch
import torch.nn as nn

from src.models.autoencoder import DenoisingAutoencoder
from src.models.unet import UNet
from src.models.vae import DenoisingVAE


def _build_unet(base_channels=32, channel_multipliers=(1, 2, 4), use_beam=False,
                n_neighbors=0, out_channels=1, **_):
    # n_neighbors > 0 is 2.5D spectral context: the input is the channel plus k neighbours
    # each side along velocity, so in_channels is 2k+1. Must match the dataset's own
    # n_neighbors or the first batch will fail on the channel dimension.
    # out_channels > 1 is for a velocity-aware objective: a moment-1 penalty needs the model
    # to predict a line profile, not one slice. Must equal in_channels in that case.
    return UNet(in_channels=2 * int(n_neighbors) + 1, out_channels=int(out_channels),
                base_channels=base_channels,
                channel_multipliers=list(channel_multipliers), time_emb_dim=128,
                num_res_blocks=2, groups=math.gcd(8, base_channels),
                beam_dim=4 if use_beam else 0)


def _build_ae(base_channels=32, **_):
    return DenoisingAutoencoder(base_channels=base_channels, linear_head=True)


def _build_vae(base_channels=32, latent_dim=128, **_):
    return DenoisingVAE(latent_dim=latent_dim, base_channels=base_channels,
                        linear_head=True)


def _fwd_unet(model, dirty, beam=None):
    """U-Net takes a timestep argument; denoising uses a constant t=0."""
    t = torch.zeros(dirty.size(0), dtype=torch.long, device=dirty.device)
    out = model(dirty, t, beam) if beam is not None else model(dirty, t)
    return out, {}


def _fwd_plain(model, dirty, beam=None):
    return model(dirty), {}


def _fwd_vae(model, dirty, beam=None):
    recon, mu, log_var = model(dirty)
    return recon, {"mu": mu, "log_var": log_var}


def _kl_term(aux) -> torch.Tensor:
    """
    KL(q(z|x) || N(0,I)) per sample, averaged over the batch.

    The latent here is a spatial map, not a vector, so the sum runs over channels
    and both spatial axes before averaging over the batch -- summing over
    everything including the batch would make the term scale with batch size and
    silently change the effective kl_weight when the OOM guard halves the batch.
    """
    mu, log_var = aux["mu"], aux["log_var"]
    per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),
                                  dim=tuple(range(1, mu.dim())))
    return per_sample.mean()


# name -> (builder, forward adapter, extra-loss term or None, sweep space)
ARCHITECTURES: Dict[str, dict] = {
    "unet": {
        "build": _build_unet,
        "forward": _fwd_unet,
        "extra_loss": None,
        "space": {
            "base_channels": [16, 32, 48, 64],
            "channel_multipliers": [(1, 2, 4), (1, 2, 2, 4), (1, 2, 4, 8)],
            "lr": (1e-4, 3e-3),
            "alpha": (0.5, 0.95),
            "sched_patience": [3, 5, 8],
        },
    },
    "autoencoder": {
        "build": _build_ae,
        "forward": _fwd_plain,
        "extra_loss": None,
        "space": {
            "base_channels": [16, 32, 48, 64],
            "lr": (1e-4, 3e-3),
            "alpha": (0.5, 0.95),
            "sched_patience": [3, 5, 8],
        },
    },
    "vae": {
        "build": _build_vae,
        "forward": _fwd_vae,
        "extra_loss": _kl_term,
        "space": {
            "base_channels": [16, 32, 48],
            "latent_dim": [32, 64, 128],
            "lr": (1e-4, 3e-3),
            "alpha": (0.5, 0.95),
            "sched_patience": [3, 5, 8],
            # log-uniform: the useful range spans orders of magnitude, and too
            # large a weight collapses the latent (posterior collapse) while too
            # small makes the VAE an autoencoder with extra steps.
            "kl_weight": (1e-6, 1e-2),
        },
    },
}


def build_model(arch: str, **cfg) -> nn.Module:
    """Instantiate `arch` from a config dict; unknown keys are ignored."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}; "
                         f"expected one of {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch]["build"](**cfg)


def forward_fn(arch: str) -> Callable:
    """Return the adapter that maps (model, dirty[, beam]) -> (prediction, aux)."""
    return ARCHITECTURES[arch]["forward"]


def extra_loss_fn(arch: str):
    """Return the architecture's extra loss term (VAE's KL), or None."""
    return ARCHITECTURES[arch]["extra_loss"]


def search_space(arch: str) -> dict:
    """Return the sweep space for `arch`."""
    return dict(ARCHITECTURES[arch]["space"])


def param_count(arch: str, **cfg) -> int:
    """Parameter count for a config, for reporting capacity alongside scores."""
    return sum(p.numel() for p in build_model(arch, **cfg).parameters())


def supported() -> Sequence[str]:
    return tuple(ARCHITECTURES)
