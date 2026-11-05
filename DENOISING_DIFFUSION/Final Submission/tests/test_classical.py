#!/usr/bin/env python
"""
Smoke tests for the classical denoising baselines:
  1. every filter denoises a noisy channel better than leaving it alone
     (a baseline that cannot beat "do nothing" is not a baseline).
  2. "none" is exactly the identity, so the dirty control is a true floor.
  3. shapes and dtype survive every filter.
  4. Wiener's flat-patch NaNs are repaired rather than propagated.
  5. an unknown method name fails loudly instead of silently no-op-ing.
  6. tuning picks the parameter with the best mean PSNR, and that optimum is
     interior to the grid (so the grid actually brackets the optimum).
  7. tuning on a set where over-smoothing hurts prefers a smaller sigma than on
     a noisier set -- i.e. the sweep responds to the data, not to a constant.
  8. `denoise_cube` filters channel-by-channel and preserves cube shape.
  9. `summarise_improvements` reproduces mean/std with ddof=1 and ignores NaNs.
"""

import numpy as np

from src.evaluation.classical import (
    DEFAULT_GRIDS,
    apply_filter,
    channel_metrics,
    denoise_cube,
    summarise_improvements,
    tune_on_validation,
)

print("=" * 60)
print("Classical Baseline Smoke Test")
print("=" * 60)

H = 64
yy, xx = np.mgrid[0:H, 0:H]
rng = np.random.default_rng(0)


def disk(amp=1.0, s=9.0):
    """A smooth extended source -- the regime a low-pass filter should help."""
    return amp * np.exp(-((yy - H / 2) ** 2 + (xx - H / 2) ** 2) / (2 * s ** 2))


clean = disk().astype(np.float32)
dirty = (clean + rng.normal(0, 0.08, (H, H))).astype(np.float32)

# [1] each filter must beat doing nothing
base = channel_metrics(clean, dirty)
for method in ("gaussian", "median", "wiener"):
    param = {"gaussian": 1.5, "median": 5, "wiener": 5}[method]
    m = channel_metrics(clean, apply_filter(dirty, method, param))
    assert m["psnr"] > base["psnr"], (method, m["psnr"], base["psnr"])
    print(f"[1] {method:<9} PSNR {m['psnr']:6.2f} > dirty {base['psnr']:6.2f} OK")

# [2] "none" is the identity -> the dirty control is exact
assert np.array_equal(apply_filter(dirty, "none", None), dirty)
print("[2] 'none' is the identity (dirty control exact)")

# [3] shape/dtype preserved everywhere
for method, param in (("gaussian", 2.0), ("median", 3), ("wiener", 7), ("none", None)):
    out = apply_filter(dirty, method, param)
    assert out.shape == dirty.shape and out.dtype == np.float32, (method, out.shape, out.dtype)
print("[3] shape and float32 dtype preserved by all filters")

# [4] Wiener on a perfectly flat patch divides by zero variance -> must not leak NaN
flat = np.zeros((H, H), dtype=np.float32)
flat[10:20, 10:20] = 1.0
out = apply_filter(flat, "wiener", 5)
assert np.isfinite(out).all(), "Wiener leaked non-finite values"
print("[4] Wiener flat-patch NaNs repaired")

# [5] unknown method must raise, not silently pass the image through
try:
    apply_filter(dirty, "bilateral", 3)
    raise AssertionError("unknown method should have raised")
except ValueError as e:
    print(f"[5] unknown method rejected: {e}")

# [6] tuning selects the best-PSNR parameter, and it is interior to the grid
class _DS:
    """Minimal stand-in for FITSChannelDataset: yields (dirty, clean) (1,H,W)."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        d, c = self.pairs[i]
        return d[None], c[None]


pairs = []
for _ in range(6):
    c = disk()
    d = c + rng.normal(0, 0.08, (H, H))
    pairs.append((d.astype(np.float32), c.astype(np.float32)))

res = tune_on_validation(_DS(pairs), methods=("gaussian",), verbose=False)
g = res["gaussian"]
psnrs = [m["psnr"] for _, m in g["all"]]
assert g["psnr"] == max(psnrs), (g["psnr"], psnrs)
grid = list(DEFAULT_GRIDS["gaussian"])
assert g["param"] not in (grid[0], grid[-1]), (
    f"optimum {g['param']} sits on a grid edge {grid} -- grid does not bracket it")
assert res["none"]["psnr"] < g["psnr"], (res["none"]["psnr"], g["psnr"])
print(f"[6] tuning picked sigma={g['param']} (interior to {grid}), "
      f"PSNR {g['psnr']:.2f} > dirty {res['none']['psnr']:.2f}")

# [7] the sweep must respond to noise level: more noise -> more smoothing wanted
quiet = []
for _ in range(6):
    c = disk()
    d = c + rng.normal(0, 0.01, (H, H))
    quiet.append((d.astype(np.float32), c.astype(np.float32)))
sigma_quiet = tune_on_validation(_DS(quiet), methods=("gaussian",), verbose=False)["gaussian"]["param"]
assert sigma_quiet <= g["param"], (sigma_quiet, g["param"])
print(f"[7] sweep tracks noise: sigma {sigma_quiet} (low noise) <= {g['param']} (high noise)")

# [8] cube filtering is per-channel and shape-preserving
cube = np.stack([dirty, dirty * 0.5, dirty * 2.0]).astype(np.float32)
out = denoise_cube(cube, "gaussian", 1.5)
assert out.shape == cube.shape and out.dtype == np.float32
assert np.allclose(out[0], apply_filter(cube[0], "gaussian", 1.5))
print(f"[8] denoise_cube preserved {cube.shape} and matches per-channel filtering")

# [9] improvement summary: ddof=1 std, NaNs dropped, n reported honestly
rows = [
    {"imp_M0": 70.0, "imp_M1": 10.0, "imp_M2": 5.0},
    {"imp_M0": 60.0, "imp_M1": 20.0, "imp_M2": float("nan")},
    {"imp_M0": 80.0, "imp_M1": 30.0, "imp_M2": 15.0},
]
s = summarise_improvements(rows)
assert abs(s["M0"]["mean"] - 70.0) < 1e-9, s
assert abs(s["M0"]["std"] - 10.0) < 1e-9, s          # ddof=1 over (60,70,80)
assert s["M2"]["n"] == 2, s                           # the NaN row is dropped
print(f"[9] summary OK: M0 {s['M0']['mean']:+.1f} +/- {s['M0']['std']:.1f} "
      f"(n={s['M0']['n']}), M2 n={s['M2']['n']} after dropping NaN")

print("\n" + "=" * 60)
print("All classical-baseline tests PASSED")
print("=" * 60)
