#!/usr/bin/env python
"""
Tests for the two no-retraining moment improvements.

The point of each test is to stop a cheap win turning into a fake one: an ensemble that
does not actually reduce error, a smoother that is really just doing the model's job, or a
sigma quietly tuned on the holdout.
"""

import numpy as np

from src.evaluation.moment_maps import generate_moment_maps, moment_improvement
from src.evaluation.postprocess import (ensemble_cubes, spectral_smooth,
                                        tune_spectral_sigma)

print("=" * 60)
print("Post-processing Tests")
print("=" * 60)

C, H, W = 80, 160, 160
rng = np.random.default_rng(5)
vel = np.linspace(-1000, 1000, C).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W]
r = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
disk = 20 * np.exp(-r ** 2 / (2 * 25.0 ** 2))
vf = 300 * (xx - W / 2) / (W / 2)
wd = 120 + 80 * np.exp(-r ** 2 / (2 * 18.0 ** 2))
clean = np.zeros((C, H, W), np.float32)
for i, v in enumerate(vel):
    clean[i] = disk * np.exp(-((v - vf) ** 2) / (2 * wd ** 2))
dirty = clean + rng.normal(0, 1.0, clean.shape).astype(np.float32)
models = [clean + rng.normal(0, 0.5, clean.shape).astype(np.float32) for _ in range(3)]

cm = generate_moment_maps(None, data_velax=(clean, vel))
dm = generate_moment_maps(None, data_velax=(dirty, vel))
sc = lambda cube: moment_improvement(cm, dm, generate_moment_maps(None, data_velax=(cube, vel)))

# [1] the ensemble must beat its own members on every moment, not just on average
singles = [sc(m) for m in models]
ens = sc(ensemble_cubes(models))
for nm in ("M0", "M1", "M2"):
    worst_single = max(s[nm] for s in singles)
    assert ens[nm] > worst_single, (nm, ens[nm], worst_single)
print("[1] ensemble beats the BEST individual model on all three moments: " +
      ", ".join(f"{nm} {max(s[nm] for s in singles):.1f}%->{ens[nm]:.1f}%"
                for nm in ("M0", "M1", "M2")))

# [2] smoothing touches only the spectral axis -- a per-channel spatial statistic that
#     depends on no neighbouring channel must be unchanged in shape
sm = spectral_smooth(models[0], 2.0)
assert sm.shape == models[0].shape
assert not np.allclose(sm, models[0]), "smoothing did nothing"
flat = np.repeat(models[0][:1], C, axis=0)          # identical in every channel
assert np.allclose(spectral_smooth(flat, 2.0), flat, atol=1e-4), \
    "a spectrally flat cube must survive spectral smoothing untouched"
print("[2] smoothing is spectral-only: a spectrally flat cube is unchanged")

# [3] THE CONTROL: what does a smoother alone achieve, with no model at all? Written to
#     record the real answer rather than a flattering one.
#
#     On M0 and M1 the model wins outright. On M2 it does NOT: the ensemble alone only
#     matches a spectrally smoothed dirty cube. M2 is a line WIDTH, and blurring along
#     velocity is close to a direct estimator of it, so the baseline is genuinely strong
#     there. Only ensemble+smoothing clears it on all three. This is exactly the trap the
#     classical baselines in notebook 07 exist to catch, and it must be reported, not
#     smoothed over.
smooth_only = {nm: max(sc(spectral_smooth(dirty, s))[nm] for s in (1.0, 2.0, 3.0, 4.0, 6.0))
               for nm in ("M0", "M1", "M2")}
for nm in ("M0", "M1"):
    assert ens[nm] > smooth_only[nm] + 5.0, (nm, ens[nm], smooth_only[nm])
print(f"[3] model beats smoothing-only on M0 ({smooth_only['M0']:.1f}%->{ens['M0']:.1f}%) "
      f"and M1 ({smooth_only['M1']:.1f}%->{ens['M1']:.1f}%)")

# the honest part: on M2 the ensemble alone does not clear the baseline
assert ens["M2"] <= smooth_only["M2"] + 5.0, (
    "ensemble now clearly beats smoothing on M2 -- good news, but this assertion and the "
    "caveat in postprocess.py and the report must be updated together")
print(f"[3] on M2 the ensemble alone only MATCHES smoothing-only "
      f"({smooth_only['M2']:.1f}% vs {ens['M2']:.1f}%) -- a real caveat, not a bug")

# [3b] the combination is what actually clears the baseline everywhere
combo = sc(spectral_smooth(ensemble_cubes(models), 2.0))
for nm in ("M0", "M1", "M2"):
    assert combo[nm] > smooth_only[nm] + 5.0, (nm, combo[nm], smooth_only[nm])
print("[3b] ensemble + smoothing clears smoothing-only on all three: " +
      ", ".join(f"{nm} {smooth_only[nm]:.1f}%->{combo[nm]:.1f}%" for nm in ("M0", "M1", "M2")))

# [4] sigma is chosen on validation data, and a flat optimum must not buy blurring
picked = tune_spectral_sigma(
    [{"i": 0}], denoise=lambda e: models[0], score=lambda c, e: sc(c)["M2"],
    sigmas=(0.0, 1.0, 2.0), verbose=False)
assert picked in (0.0, 1.0, 2.0)
flat_pick = tune_spectral_sigma(
    [{"i": 0}], denoise=lambda e: models[0], score=lambda c, e: 1.0,   # every sigma equal
    sigmas=(0.0, 1.0, 2.0), verbose=False)
assert flat_pick == 0.0, f"a tie must prefer no smoothing, got {flat_pick}"
print(f"[4] sigma tuned on validation: picked {picked}; ties prefer no smoothing")

print("\n" + "=" * 60)
print("All post-processing tests PASSED")
print("=" * 60)
