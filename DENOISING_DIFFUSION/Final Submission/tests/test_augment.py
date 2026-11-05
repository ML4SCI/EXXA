#!/usr/bin/env python
"""
Tests for the dihedral (D4) augmentation added to `FITSChannelDataset`.

The properties that matter scientifically, in order of how badly a bug would hurt:
  1. dirty and clean receive the SAME transform -- if they ever diverge, every
     augmented example is a corrupted training pair and the model learns noise.
  2. augmentation is lossless: the output is a permutation of the input pixels,
     so no interpolation smooths the noise the model must learn to remove.
  3. the transform actually varies across calls (otherwise it is a no-op dressed
     up as augmentation), and covers more than one orientation.
  4. it is reproducible under a fixed torch seed (results stay comparable).
  5. it is OFF by default, so V12 and every committed result stay reproducible.
  6. shape and dtype are preserved.
  7. all 8 group elements are reachable.
"""

import numpy as np
import torch

from src.data.fits_cube_dataset import FITSChannelDataset

print("=" * 60)
print("D4 Augmentation Test")
print("=" * 60)

aug = FITSChannelDataset._augment_pair

# An asymmetric pattern: any rotation or flip gives a distinguishable array, so
# "same transform applied to both" is actually testable.
H = 8
base = torch.arange(H * H, dtype=torch.float32).reshape(1, H, H)
dirty = base.clone()
clean = base.clone() * -1.0            # deliberately different values, same geometry

# [1] identical geometric transform on both members of the pair
torch.manual_seed(0)
mismatches = 0
for _ in range(200):
    d, c = aug(dirty.clone(), clean.clone())
    # clean is exactly -dirty before augmenting, so it must remain exactly -d after
    if not torch.equal(c, -d):
        mismatches += 1
assert mismatches == 0, f"{mismatches}/200 pairs got different transforms"
print("[1] dirty and clean always receive the same transform (200 draws)")

# [2] lossless -- output is a permutation of the input multiset
torch.manual_seed(0)
for _ in range(50):
    d, _ = aug(dirty.clone(), clean.clone())
    assert torch.equal(torch.sort(d.flatten()).values, torch.sort(dirty.flatten()).values), \
        "augmentation altered pixel values (interpolation?)"
print("[2] lossless: output is a pixel permutation, no interpolation")

# [3] the transform varies, and covers multiple distinct orientations
torch.manual_seed(0)
seen = {aug(dirty.clone(), clean.clone())[0].numpy().tobytes() for _ in range(200)}
assert len(seen) > 1, "augmentation never changed the input -- it is a no-op"
print(f"[3] varies across calls: {len(seen)} distinct orientations in 200 draws")

# [4] reproducible under a fixed seed
torch.manual_seed(123)
a = [aug(dirty.clone(), clean.clone())[0].clone() for _ in range(10)]
torch.manual_seed(123)
b = [aug(dirty.clone(), clean.clone())[0].clone() for _ in range(10)]
assert all(torch.equal(x, y) for x, y in zip(a, b)), "not reproducible under a fixed seed"
print("[4] reproducible: same torch seed reproduces the orientation sequence")

# [5] OFF by default -- guards every already-published result
assert "augment: bool = False" in open(
    "src/data/fits_cube_dataset.py").read(), "augment must default to False"
print("[5] augment defaults to False (V12 and committed results unaffected)")

# [6] shape and dtype preserved
d, c = aug(dirty.clone(), clean.clone())
assert d.shape == dirty.shape and c.shape == clean.shape, (d.shape, c.shape)
assert d.dtype == torch.float32 and c.dtype == torch.float32
print(f"[6] shape {tuple(d.shape)} and float32 dtype preserved")

# [7] all 8 group elements are reachable given enough draws
torch.manual_seed(7)
found = {aug(dirty.clone(), clean.clone())[0].numpy().tobytes() for _ in range(4000)}
expected = set()
for k in range(4):
    r = torch.rot90(dirty, k, dims=(-2, -1))
    expected.add(r.contiguous().numpy().tobytes())
    expected.add(torch.flip(r, dims=(-1,)).contiguous().numpy().tobytes())
assert found == expected, f"reached {len(found)} of {len(expected)} orientations"
print(f"[7] all {len(expected)} dihedral orientations reachable")

# sanity: augmentation must not silently shift the mean (it is a permutation)
torch.manual_seed(0)
means = [float(aug(dirty.clone(), clean.clone())[0].mean()) for _ in range(50)]
assert np.allclose(means, float(dirty.mean())), "mean changed under augmentation"

print("\n" + "=" * 60)
print("All D4 augmentation tests PASSED")
print("=" * 60)
