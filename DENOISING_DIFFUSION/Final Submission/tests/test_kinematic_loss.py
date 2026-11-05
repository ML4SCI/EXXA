"""
The velocity-aware objective (KinematicLoss).

Built because the U-Net was measured on 2026-08-28 to improve pixel metrics and M0 while
DEGRADING the GI wiggle: residual correlation 0.804 against 0.891 for leaving the dirty cube
alone. It optimises pixel accuracy, and the wiggle is a sub-channel velocity perturbation that
per-channel smoothing shifts. This adds that quantity to the objective.

    PYTHONPATH=. python3 tests/test_kinematic_loss.py
"""
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from astropy.io import fits

from src.utils.losses import spectral_moment1, KinematicLoss, HybridLoss
from src.data.fits_cube_dataset import FITSChannelDataset

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


print("=" * 70)
print("Velocity-aware objective")
print("=" * 70)

C = 15
velax = torch.linspace(-1.0, 1.0, C)
g = torch.arange(C).float()

# A Gaussian line at a known channel, per spaxel.
def line(centre_ch, amp=1.0, width=2.0, hw=8):
    prof = amp * torch.exp(-0.5 * ((g - centre_ch) / width) ** 2)
    return prof.view(1, C, 1, 1).expand(1, C, hw, hw).clone()


print("\ncase 1  spectral_moment1 measures velocity, not brightness")
base = line(7.0)
m1 = spectral_moment1(base, velax)
check("a centred line gives M1 near the middle of the axis",
      abs(float(m1.mean())) < 0.05, f"{float(m1.mean()):+.4f}")
check("M1 tracks a shifted line",
      float(spectral_moment1(line(9.0), velax).mean()) > float(m1.mean()) + 0.1,
      f"{float(spectral_moment1(line(9.0), velax).mean()):+.4f} vs {float(m1.mean()):+.4f}")
check("M1 is invariant to amplitude (it is a velocity, not a flux)",
      float((spectral_moment1(line(7.0, amp=5.0), velax) - m1).abs().max()) < 1e-5)
check("no NaN on an empty spectrum",
      bool(torch.isfinite(spectral_moment1(torch.zeros(1, C, 4, 4), velax)).all()))

print("\ncase 2  the loss penalises a velocity shift that MSE barely sees")
tgt = line(7.0)
shifted = line(7.6)                      # same shape, shifted centre
smoothed = tgt * 0.9 + 0.05              # amplitude change, centre unchanged

kin = KinematicLoss(alpha=0.8, beta=0.2, gamma=1.0, velax=velax)
_, _, _, k_shift = kin(shifted.clamp(0, 1), tgt.clamp(0, 1))
_, _, _, k_scale = kin(smoothed.clamp(0, 1), tgt.clamp(0, 1))
check("a velocity shift is penalised", float(k_shift) > 0.01, f"kin={float(k_shift):.4f}")
check("a pure amplitude change is NOT penalised as velocity error",
      float(k_scale) < 0.01, f"kin={float(k_scale):.2e}")
check("the velocity term separates the two cases",
      float(k_shift) > 10 * max(float(k_scale), 1e-9),
      f"{float(k_shift):.4f} vs {float(k_scale):.2e}")

_, _, _, k_same = kin(tgt.clamp(0, 1), tgt.clamp(0, 1))
check("identical input and target give zero velocity loss", float(k_same) < 1e-6)

print("\ncase 3  gamma=0 reduces to the pixel-only objective")
kin0 = KinematicLoss(alpha=0.8, beta=0.2, gamma=0.0, velax=velax)
hyb = HybridLoss(alpha=0.8, beta=0.2)
t_kin = float(kin0(shifted.clamp(0, 1), tgt.clamp(0, 1))[0])
# HybridLoss expects (B,1,H,W); compare on a single channel so the SSIM window matches
t_hyb = float(hyb(shifted[:, :1].clamp(0, 1), tgt[:, :1].clamp(0, 1))[0])
# The velocity term is returned UNWEIGHTED as a diagnostic, so it can be monitored even when
# it is not being optimised. What gamma=0 must guarantee is that it contributes nothing to the
# total.
tot0, mse0, ssim0, kin_diag = kin0(shifted.clamp(0, 1), tgt.clamp(0, 1))
check("gamma=0 contributes nothing to the TOTAL",
      abs(float(tot0) - (0.8 * float(mse0) + 0.2 * float(ssim0))) < 1e-6,
      f"total {float(tot0):.6f} vs pixel-only {0.8*float(mse0)+0.2*float(ssim0):.6f}")
check("the velocity term is still reported for monitoring at gamma=0",
      float(kin_diag) > 0.01, f"kin diagnostic {float(kin_diag):.4f}")
check("gamma>0 raises the total by exactly gamma * kin",
      abs(float(kin(shifted.clamp(0,1), tgt.clamp(0,1))[0]) - (float(tot0) + 1.0*float(kin_diag))) < 1e-6)
check("gamma=0 total is finite and positive", 0 < t_kin < 100, f"{t_kin:.4f}")

print("\ncase 4  the dataset can supply matched channel stacks")
with tempfile.TemporaryDirectory() as td:
    rng = np.random.default_rng(0)
    cube = (rng.random((30, 32, 32)).astype(np.float32)
            * np.arange(1, 31, dtype=np.float32)[:, None, None])
    fits.PrimaryHDU(cube).writeto(f"{td}/c.fits")
    fits.PrimaryHDU((cube + rng.normal(0, .05, cube.shape)).astype(np.float32)).writeto(f"{td}/d.fits")
    kw = dict(channel_sampler_fn=lambda n_channels, seed: list(range(n_channels)),
              target_size=32, verbose=False, n_neighbors=3)
    d1, c1 = FITSChannelDataset([(f"{td}/c.fits", f"{td}/d.fits")], stack_target=True, **kw)[10]
    d0, c0 = FITSChannelDataset([(f"{td}/c.fits", f"{td}/d.fits")], **kw)[10]
    check("stack_target gives matching input/target shapes", d1.shape == c1.shape,
          f"{tuple(d1.shape)} vs {tuple(c1.shape)}")
    check("default behaviour is unchanged (centre-channel target)", c0.shape[0] == 1)
    check("the stacked target's centre equals the default target",
          torch.allclose(c1[3:4], c0), "centre channel must be the same data")

print("\n" + "-" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
sys.exit(0)
