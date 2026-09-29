#!/usr/bin/env python
"""
Smoke tests for beam conditioning + the sweep harness:
  1. beam_features_of produces the 4-vector from a FITS-like header.
  2. UNet(beam_dim=4) forward accepts a beam vector; beam actually changes output.
  3. train_unet runs end-to-end with early stopping on a tiny synthetic dataset.
  4. run_sweep completes runs and writes the CSV.
"""

import os
import tempfile

import numpy as np
import torch

from src.data.fits_cube_dataset import beam_features_of
from src.models.unet import UNet
from src.training.sweep import run_sweep, train_unet

print("=" * 60)
print("Beam + Sweep Smoke Test")
print("=" * 60)

# [1] beam features from an example header (BPA deg, BMAJ/BMIN deg)
hdr = {"BPA": 16.933288583074418, "BMAJ": 4.26294365362412e-05, "BMIN": 3.29226140071572e-05}
feat = beam_features_of(hdr)
bpa = np.deg2rad(hdr["BPA"])
assert np.allclose(feat, [np.sin(2 * bpa), np.cos(2 * bpa),
                          hdr["BMAJ"] * 3600, hdr["BMIN"] * 3600], atol=1e-6)
assert beam_features_of({}).tolist() == [0, 0, 0, 0]  # missing keys -> zeros
print(f"[1] beam_features_of OK: {np.round(feat, 4).tolist()}")

# [2] beam-conditioned forward; beam must influence the output
torch.manual_seed(0)
net = UNet(in_channels=1, out_channels=1, base_channels=16,
           channel_multipliers=[1, 2], groups=8, beam_dim=4)
x = torch.randn(2, 1, 32, 32)
t = torch.zeros(2, dtype=torch.long)
b1 = torch.tensor([[0.56, 0.83, 0.153, 0.119]] * 2)
b2 = torch.tensor([[-0.9, 0.1, 0.5, 0.4]] * 2)
with torch.no_grad():
    o1, o2 = net(x, t, b1), net(x, t, b2)
assert o1.shape == (2, 1, 32, 32)
assert not torch.allclose(o1, o2), "beam vector had no effect on output"
print(f"[2] beam-conditioned forward OK (beam changes output, "
      f"mean abs diff {torch.mean((o1-o2).abs()):.4f})")

# [3] train_unet end-to-end on synthetic (dirty, clean, beam) items
rng = np.random.default_rng(0)
items = [(torch.rand(1, 32, 32), torch.rand(1, 32, 32),
          torch.tensor([0.5, 0.8, 0.15, 0.12])) for _ in range(16)]
res = train_unet(items[:12], items[12:], "cpu", base_channels=16,
                 channel_multipliers=(1, 2), lr=1e-3, alpha=0.8, batch_size=4,
                 use_beam=True, min_epochs=1, max_epochs=2, patience=1, verbose=False)
assert res["epochs_run"] <= 2 and res["best_epoch"] >= 1
assert all(k in res for k in ("psnr", "ssim", "mse", "best_val_loss"))
print(f"[3] train_unet OK: {res['epochs_run']} epochs, "
      f"PSNR {res['psnr']:.2f} dB, best ep {res['best_epoch']}")

# [4] run_sweep writes CSV rows
with tempfile.TemporaryDirectory() as td:
    csv_path = os.path.join(td, "sweep.csv")
    tiny_space = {"base_channels": [16], "channel_multipliers": [(1, 2)],
                  "lr": (1e-3, 1e-3), "alpha": (0.8, 0.8),
                  "sched_patience": [3], "use_beam": [True, False]}
    out = run_sweep(items[:12], items[12:], "cpu", n_runs=2, out_csv=csv_path,
                    space=tiny_space, batch_size=4, min_epochs=1, max_epochs=1,
                    patience=1, verbose=False)
    assert len(out) == 2
    with open(csv_path) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
print(f"[4] run_sweep OK: 2 runs, CSV rows written")

print("=" * 60)
print("All beam + sweep smoke tests passed!")
print("=" * 60)
