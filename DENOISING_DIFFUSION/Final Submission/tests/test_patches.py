#!/usr/bin/env python
"""
Tests for patch training and overlap-blended tiled inference.

The failure mode that matters is silent: naive tiling produces a plausible image with a
hard discontinuity at every tile join, landing mid-field where the moment maps are
computed. These check the seam is actually gone, not merely that the code runs.
"""

import numpy as np
import torch

from src.data.patches import PatchPairDataset, tiled_denoise, tile_grid

print("=" * 60)
print("Patch / Tiling Tests")
print("=" * 60)

H = W = 200
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ [1] identity round trip
def _identity(batch):
    return batch


img = rng.normal(0, 1, (H, W)).astype(np.float32)
out = tiled_denoise(img, _identity, tile=64, overlap=16)
assert out.shape == img.shape, out.shape
# A perfect denoiser must be reproduced exactly: the Hann weights are normalised by their
# own sum, so an identity function has to survive tiling untouched.
err = np.abs(out - img).max()
assert err < 1e-4, f"identity not preserved through tiling, max err {err}"
print(f"[1] identity survives tiling: max |err| {err:.2e}")


# ------------------------------------------------------------------ [2] the seam test
# A denoiser that adds a constant offset PER TILE is the worst case for tiling: naive
# stitching leaves a step at every join. Blending must smooth it into a gradient.
_call = {"n": 0}


def _per_tile_offset(batch):
    out = batch.clone()
    for k in range(out.shape[0]):
        out[k] += (_call["n"] % 2) * 1.0     # alternate tiles get +1
        _call["n"] += 1
    return out


flat = np.zeros((H, W), dtype=np.float32)
blend = tiled_denoise(flat, _per_tile_offset, tile=64, overlap=32)
# neighbouring-pixel jumps measure seam sharpness
d = np.abs(np.diff(blend, axis=1)).max()
assert d < 0.35, f"visible seam: max horizontal jump {d:.3f}"
print(f"[2] per-tile offsets blend smoothly: max neighbour jump {d:.3f} (a hard seam is 1.0)")


# ------------------------------------------------------------------ [3] full coverage
# Every pixel must be written by at least one tile, including the far edges when the span
# is not an exact multiple of the step.
seen = tiled_denoise(np.ones((H, W), np.float32), _identity, tile=64, overlap=16)
assert np.allclose(seen, 1.0, atol=1e-4), (seen.min(), seen.max())
for (h, w, t, o) in ((200, 200, 64, 16), (600, 600, 256, 64), (137, 91, 64, 8)):
    g = tile_grid(h, w, t, o)
    cov = np.zeros((h, w), bool)
    for (y, x) in g:
        cov[y:y + t, x:x + t] = True
    assert cov.all(), f"uncovered pixels for {h}x{w} tile={t} overlap={o}"
print(f"[3] full coverage incl. edges; 600x600 @256/64 needs {len(tile_grid(600,600,256,64))} tiles")


# ------------------------------------------------------------------ [4] patch dataset shape
class _Fake:
    """Yields a disk-like clean channel so the signal-bias path is exercised."""
    def __init__(self, n=6, size=128):
        self.n, self.size = n, size
        yy, xx = np.mgrid[0:size, 0:size]
        self.disk = np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * 12.0 ** 2))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        clean = torch.from_numpy(self.disk.astype(np.float32))[None]
        # Deterministic noise: a fresh randn here would make the DIRTY channel differ on
        # every call, so the determinism check below would fail on the fixture rather than
        # on the patch positions it is meant to test.
        g = torch.Generator().manual_seed(1000 + i)
        dirty = clean + torch.randn(clean.shape, generator=g) * 0.1
        return dirty, clean


ds = PatchPairDataset(_Fake(), patch_size=32, n_patches=8, seed=1, signal_bias=0.5)
p, lbl = ds[0]
assert p.shape == (8, 2, 32, 32), p.shape
assert lbl == 0
print(f"[4] patch batch shape {tuple(p.shape)} -- the 5-D layout _flatten_patches folds")

# deterministic: the same item must give the same patches on every epoch, or validation
# loss would drift for reasons unrelated to training
p2, _ = ds[0]
assert torch.equal(p, p2), "patch positions are not deterministic across calls"
print("[4] patch positions deterministic across epochs")

# ------------------------------------------------------------------ [5] signal bias works
# With bias, patches should land on the source far more often than uniform placement would.
biased = PatchPairDataset(_Fake(), patch_size=32, n_patches=200, seed=2, signal_bias=1.0)
uniform = PatchPairDataset(_Fake(), patch_size=32, n_patches=200, seed=2, signal_bias=0.0)
b_mean = biased[0][0][:, 1].mean().item()      # clean channel brightness inside patches
u_mean = uniform[0][0][:, 1].mean().item()
assert b_mean > u_mean * 1.5, (b_mean, u_mean)
print(f"[5] signal bias lands patches on the source: mean clean flux "
      f"{u_mean:.4f} (uniform) -> {b_mean:.4f} (biased)")

# ------------------------------------------------------------------ [6] flat patch view
# train_unet takes one (dirty, clean) pair per item and convolves a 4-D batch; the stacked
# view above would hand it 5-D. The flat view must expose the SAME patches, or a
# comparison between the two training views measures the crop rather than the batching.
from src.data.patches import FlatPatchDataset

_base = _Fake()
_flat = FlatPatchDataset(_base, patch_size=32, n_patches=8, seed=1, signal_bias=0.5)
_stk = PatchPairDataset(_base, patch_size=32, n_patches=8, seed=1, signal_bias=0.5)
assert len(_flat) == len(_base) * 8, len(_flat)
_d, _c = _flat[0]
assert _d.shape == (1, 32, 32) and _c.shape == (1, 32, 32), (_d.shape, _c.shape)
for _i in range(len(_base)):
    for _k in range(8):
        assert torch.equal(_flat[_i * 8 + _k][0], _stk[_i][0][_k, 0][None]), (_i, _k)
assert torch.equal(_flat[3][0], _flat[3][0]), "flat patches not deterministic"
print(f"[6] flat view: {len(_base)} images x 8 = {len(_flat)} items of {tuple(_d.shape)}, "
      "patch-for-patch identical to the stacked view")

print("\n" + "=" * 60)
print("All patch/tiling tests PASSED")
print("=" * 60)
