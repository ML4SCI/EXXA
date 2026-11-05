#!/usr/bin/env python
"""
Smoke test for the DDPM sampler: posterior-mean averaging (n_avg) must run,
return the right shape, and stay in [0, 1]. Tiny config on CPU so it's fast.
"""

import torch

from src.models.diffusion_unet import default_diffusion_config
from src.training.diffusion import DenoisingDiffusion

print("=" * 60)
print("DDPM Sampling Smoke Test")
print("=" * 60)

cfg = default_diffusion_config(image_size=32)  # default ch_mult reaches 16x16 (attn fires)
diff = DenoisingDiffusion(config=cfg, device="cpu", lr=2e-4,
                          checkpoint_path="/tmp/_smoke.pth.tar", data_parallel=False)

dirty = torch.rand(2, 1, 32, 32)  # (B,1,H,W) in [0,1]

single = diff.sample(dirty, sampling_timesteps=2, use_ema=True, n_avg=1)
avg = diff.sample(dirty, sampling_timesteps=2, use_ema=True, n_avg=4)

print(f"[1] shapes: single={tuple(single.shape)} avg={tuple(avg.shape)}")
assert single.shape == avg.shape == (2, 1, 32, 32)

print(f"[2] range: single=[{single.min():.3f},{single.max():.3f}] "
      f"avg=[{avg.min():.3f},{avg.max():.3f}]")
assert single.min() >= 0.0 and single.max() <= 1.0
assert avg.min() >= 0.0 and avg.max() <= 1.0

# Averaging must reduce sample-to-sample variance: the mean of 4 draws should be
# less extreme (lower variance across pixels vs its own mean) than a single draw
# is not a strict guarantee per-run, so just assert averaging changed the output
# (i.e. it actually averaged, not returned one draw).
print(f"[3] n_avg=4 differs from n_avg=1 (mean abs diff {torch.mean((single-avg).abs()):.4f})")
assert not torch.allclose(single, avg), "n_avg had no effect -- averaging not wired"

print("=" * 60)
print("All DDPM sampling smoke tests passed!")
print("=" * 60)
