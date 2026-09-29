#!/usr/bin/env python
"""
Tests for the multi-architecture training path.

The point of this module is fairness: only the U-Net was ever swept on
line-emission data, so any architecture claim is currently confounded with tuning
budget. Two of these cases guard defects that would make a comparison invalid in
opposite directions:

  [3] a sigmoid head CANNOT represent a target above 1, and under the shared
      dirty-scale normalisation the clean target routinely exceeds 1. An AE/VAE
      left with its sigmoid would lose for a reason unrelated to architecture.
  [7] the same metric function must score every architecture. If each had its own
      measurement path the numbers would not be comparable at all.

  1. every advertised architecture builds and forwards at 256x256.
  2. the defaults reproduce the original fixed architectures (back-compat).
  3. linear_head removes the output squashing: outputs can exceed 1.
  4. base_channels actually scales capacity.
  5. train_unet trains each architecture end to end and returns finite metrics.
  6. the VAE's kl_weight changes the objective (and 0 disables it).
  7. val_metrics dispatches per architecture.
  8. checkpoints record the architecture, and the U-Net's legacy label is kept.
  9. the KL term is batch-size invariant.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoencoder import DenoisingAutoencoder
from src.models.vae import DenoisingVAE
from src.training.architectures import (ARCHITECTURES, _kl_term, build_model,
                                        extra_loss_fn, forward_fn, param_count,
                                        search_space, supported)
from src.training.sweep import train_unet, val_metrics

print("=" * 68)
print("Multi-architecture training path test")
print("=" * 68)

device = torch.device("cpu")

# [1] build + forward at the real working resolution
x = torch.randn(2, 1, 256, 256)
for arch in supported():
    m = build_model(arch, base_channels=8, channel_multipliers=(1, 2), latent_dim=8)
    pred, aux = forward_fn(arch)(m, x)
    assert pred.shape == (2, 1, 256, 256), (arch, pred.shape)
    assert torch.isfinite(pred).all(), arch
    print(f"[1] {arch:<12} forward OK at 256x256 -> {tuple(pred.shape)}"
          f"{'  (+mu/log_var)' if aux else ''}")

# [2] defaults must reproduce the original fixed architectures
ae_default = DenoisingAutoencoder()
assert isinstance(ae_default.sigmoid, torch.nn.Sigmoid), "AE default lost its sigmoid"
assert ae_default.enc1.block[0].out_channels == 32, "AE default width changed"
vae_default = DenoisingVAE()
assert isinstance(vae_default.sigmoid, torch.nn.Sigmoid), "VAE default lost its sigmoid"
assert vae_default.enc1.block[0].out_channels == 32, "VAE default width changed"
assert vae_default.conv_mu.out_channels == 128, "VAE default latent_dim changed"
print("[2] defaults unchanged: 32-wide, sigmoid retained (continuum results safe)")

# [3] THE FAIRNESS CASE: linear_head must allow outputs above 1.
# A sigmoid model is mathematically incapable of it, so a target >1 is unreachable.
big = torch.randn(4, 1, 64, 64) * 6.0
with torch.no_grad():
    sig_out = DenoisingAutoencoder(base_channels=8, linear_head=False)(big)
    lin_out = DenoisingAutoencoder(base_channels=8, linear_head=True)(big)
assert float(sig_out.max()) <= 1.0, f"sigmoid head exceeded 1: {float(sig_out.max())}"
assert not isinstance(DenoisingAutoencoder(base_channels=8, linear_head=True).sigmoid,
                      torch.nn.Sigmoid), "linear_head did not remove the sigmoid"
# the built models used for sweeps must all be linear-headed
for arch in ("autoencoder", "vae"):
    built = build_model(arch, base_channels=8, latent_dim=8)
    assert not isinstance(built.sigmoid, torch.nn.Sigmoid), \
        f"{arch} was built with a sigmoid: it cannot represent targets >1"
print(f"[3] sigmoid caps at {float(sig_out.max()):.3f}; linear head is unbounded "
      f"(range {float(lin_out.min()):.2f}..{float(lin_out.max()):.2f}), "
      "sweep-built AE/VAE are linear")

# [4] base_channels scales capacity
for arch in ("unet", "autoencoder", "vae"):
    small = param_count(arch, base_channels=8, channel_multipliers=(1, 2), latent_dim=8)
    big_n = param_count(arch, base_channels=32, channel_multipliers=(1, 2), latent_dim=8)
    assert big_n > small * 2, (arch, small, big_n)
    print(f"[4] {arch:<12} base_channels 8 -> 32 grows {small:,} -> {big_n:,} params")


class TinyDS(torch.utils.data.Dataset):
    """(dirty, clean) pairs where clean deliberately exceeds 1, as the real data does."""

    def __init__(self, n=6, size=32):
        g = torch.Generator().manual_seed(0)
        self.clean = torch.rand(n, 1, size, size, generator=g) * 1.3
        self.dirty = (self.clean + 0.1 * torch.randn(n, 1, size, size, generator=g))

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        return self.dirty[i], self.clean[i]


ds = TinyDS()
BASE = dict(base_channels=8, channel_multipliers=(1, 2), lr=1e-3, alpha=0.8,
            batch_size=3, min_epochs=1, max_epochs=2, patience=1, verbose=False)

# [5] every architecture trains end to end under the identical protocol
tmp = tempfile.mkdtemp(prefix="exxa-arch-")
try:
    results = {}
    for arch in supported():
        ck = os.path.join(tmp, f"{arch}.pth")
        cfg = dict(BASE)
        if arch == "vae":
            cfg.update(latent_dim=8, kl_weight=1e-4)
        res = train_unet(ds, ds, device, arch=arch, ckpt_path=ck, seed=1, **cfg)
        assert np.isfinite(res["psnr"]) and np.isfinite(res["mse"]), (arch, res)
        assert res["epochs_run"] >= 1
        results[arch] = res
        print(f"[5] {arch:<12} trained: PSNR {res['psnr']:.3f}, "
              f"{res['epochs_run']} epochs, best ep {res['best_epoch']}")

    # [6] kl_weight must change the VAE objective, and 0 must disable it
    torch.manual_seed(0)
    r_off = train_unet(ds, ds, device, arch="vae", latent_dim=8, kl_weight=0.0,
                       seed=3, **BASE)
    torch.manual_seed(0)
    r_on = train_unet(ds, ds, device, arch="vae", latent_dim=8, kl_weight=5.0,
                      seed=3, **BASE)
    assert r_off["best_val_loss"] != r_on["best_val_loss"], \
        "kl_weight had no effect on the objective"
    print(f"[6] kl_weight active: val loss {r_off['best_val_loss']:.4f} (w=0) "
          f"vs {r_on['best_val_loss']:.4f} (w=5)")

    # [7] one metric function scores every architecture
    loader = torch.utils.data.DataLoader(ds, batch_size=3)
    for arch in supported():
        m = build_model(arch, base_channels=8, channel_multipliers=(1, 2), latent_dim=8)
        met = val_metrics(m, loader, device, arch=arch)
        assert set(met) == {"psnr", "ssim", "mse"} and all(np.isfinite(v) for v in met.values()), \
            (arch, met)
    print("[7] val_metrics dispatches per architecture and returns the same keys")

    # [8] checkpoints identify the architecture; the U-Net keeps its legacy label
    ck_unet = torch.load(os.path.join(tmp, "unet.pth"), map_location="cpu",
                         weights_only=False)
    ck_vae = torch.load(os.path.join(tmp, "vae.pth"), map_location="cpu",
                        weights_only=False)
    assert ck_unet["arch"] == "UNet" and ck_unet["arch_key"] == "unet", ck_unet["arch"]
    assert ck_vae["arch_key"] == "vae" and ck_vae["latent_dim"] == 8, ck_vae["arch_key"]
    print(f"[8] checkpoints tagged: unet arch={ck_unet['arch']!r} (legacy label kept), "
          f"vae arch_key={ck_vae['arch_key']!r}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# [9] the KL term must not scale with batch size, or the OOM guard halving the
# batch would silently change the effective kl_weight mid-sweep.
mu = torch.randn(8, 4, 5, 5)
lv = torch.randn(8, 4, 5, 5) * 0.1
full = float(_kl_term({"mu": mu, "log_var": lv}))
half = float(_kl_term({"mu": mu[:4], "log_var": lv[:4]}))
assert abs(full - half) / max(abs(full), 1e-9) < 0.5, (full, half)
print(f"[9] KL is batch-size invariant: {full:.3f} (n=8) vs {half:.3f} (n=4)")

# every architecture must advertise a non-empty search space
for arch in supported():
    sp = search_space(arch)
    assert sp and "lr" in sp, (arch, sp)
assert "kl_weight" in search_space("vae"), "VAE space must include kl_weight"

print("\n" + "=" * 68)
print("All multi-architecture tests PASSED")
print("=" * 68)
