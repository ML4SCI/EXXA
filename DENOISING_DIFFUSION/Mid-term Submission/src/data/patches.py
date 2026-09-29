#!/usr/bin/env python
"""
src/data/patches.py
===================
Patch-based training and overlap-blended tiled inference for line-emission cubes.

Two problems, one idea.

**Training.** The line-emission set is 14 cubes from 11 simulations, and only 6 RunIDs are
trained on. A DDPM learns a distribution, which needs far more independent samples than a
regressor needs to learn a conditional mean: that gap is the leading explanation for the
DDPM trailing the U-Net by ~19 dB here. Cutting each 256x256 channel into patches does not
create new disks, but it does turn one whole-image sample into many distinct local
structures (ring edge, inner cavity, background, line core), which is what a denoiser
actually has to model. Denoising is local: cleaning a pixel needs its neighbourhood, not
the far side of the disk.

**Inference.** Evaluation currently downsamples 600 -> 256, denoises, then upsamples back,
so every reported number is limited by that round trip. `tiled_denoise` runs the model at
its native tile size across the full 600x600 field instead, with no resampling at all.

Naive tiling leaves visible seams: each tile is denoised in ignorance of its neighbours
and the discontinuity lands mid-image, exactly where moment maps are computed. Tiles are
therefore overlapped and blended with a separable Hann window, so contributions cross-fade
and sum to one everywhere.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Training: patch pairs                                                       #
# --------------------------------------------------------------------------- #
class PatchPairDataset(Dataset):
    """
    Wrap a ``(dirty, clean)`` dataset and yield stacked PATCH batches.

    Returns per item ``(patches, 0)`` where ``patches`` is
    ``(n_patches, 2, patch_size, patch_size)``, channel 0 = dirty, 1 = clean --
    the 5-D layout ``src.training.diffusion._flatten_patches`` already folds into the
    batch axis, so the DDPM trainer needs no change to consume it.

    Args:
        base: dataset returning ``(dirty, clean)``, each ``(1, H, W)``.
        patch_size: side of each square patch.
        n_patches: patches drawn per item.
        seed: base RNG seed. Patch positions are drawn per ``(item, epoch-free)`` index
            deterministically, so an item yields the same patches on every epoch --
            otherwise the validation loss would move for reasons unrelated to training.
        signal_bias: fraction of patches biased toward the brightest region of the CLEAN
            channel rather than placed uniformly. Purely uniform placement wastes most
            patches on empty sky, because the disk covers a small part of the field.
    """

    def __init__(self, base: Dataset, patch_size: int = 64, n_patches: int = 8,
                 seed: int = 42, signal_bias: float = 0.5):
        self.base = base
        self.patch_size = int(patch_size)
        self.n_patches = int(n_patches)
        self.seed = int(seed)
        self.signal_bias = float(signal_bias)

    def __len__(self):
        return len(self.base)

    def _positions(self, clean: torch.Tensor, idx: int) -> List[Tuple[int, int]]:
        _, h, w = clean.shape
        p = self.patch_size
        if h < p or w < p:
            raise ValueError(f"patch_size {p} exceeds image {h}x{w}")
        rng = np.random.default_rng(self.seed + idx)
        n_sig = int(round(self.n_patches * self.signal_bias))

        pos: List[Tuple[int, int]] = []
        if n_sig:
            # Centre-of-mass of the clean channel, with a spread that keeps patches on the
            # source without collapsing them all onto one pixel.
            arr = clean[0].detach().cpu().numpy()
            arr = np.clip(arr - float(np.percentile(arr, 50)), 0, None)
            tot = float(arr.sum())
            if tot > 0:
                ys, xs = np.nonzero(arr)
                wgt = arr[ys, xs].astype(np.float64)
                wgt /= wgt.sum()
                pick = rng.choice(len(ys), size=n_sig, p=wgt)
                for k in pick:
                    i = int(np.clip(ys[k] - p // 2, 0, h - p))
                    j = int(np.clip(xs[k] - p // 2, 0, w - p))
                    pos.append((i, j))
            else:                                   # a genuinely empty channel
                n_sig = 0
        for _ in range(self.n_patches - len(pos)):
            pos.append((int(rng.integers(0, h - p + 1)), int(rng.integers(0, w - p + 1))))
        return pos

    def __getitem__(self, idx: int):
        dirty, clean = self.base[idx]
        p = self.patch_size
        out = []
        for (i, j) in self._positions(clean, idx):
            out.append(torch.cat([dirty[:, i:i + p, j:j + p],
                                  clean[:, i:i + p, j:j + p]], dim=0))
        return torch.stack(out, dim=0), 0           # (n_patches, 2, p, p)


class FlatPatchDataset(Dataset):
    """Patch pairs as ``(dirty, clean)``, one patch per item: the U-Net's input shape.

    ``PatchPairDataset`` returns all of an image's patches stacked as
    ``(n_patches, 2, p, p)``, which is what the DDPM trainer folds into its batch axis.
    ``train_unet`` instead expects each item to be a single ``(dirty, clean)`` pair of
    ``(1, H, W)`` tensors, and would receive a 5-D batch it cannot convolve.

    This flattens the same sampling: item ``i`` is patch ``i % n_patches`` of image
    ``i // n_patches``, so ``len`` is ``len(base) * n_patches`` and every patch is one
    training example. Positions come from the same seeded draw, so the two views see
    identical patches and a comparison between them isolates the batching, not the crop.
    """

    def __init__(self, base: Dataset, patch_size: int = 64, n_patches: int = 8,
                 seed: int = 42, signal_bias: float = 0.5):
        self._inner = PatchPairDataset(base, patch_size=patch_size, n_patches=n_patches,
                                       seed=seed, signal_bias=signal_bias)
        self.n_patches = int(n_patches)
        self.patch_size = int(patch_size)

    def __len__(self):
        return len(self._inner) * self.n_patches

    def __getitem__(self, idx: int):
        img_i, patch_i = divmod(int(idx), self.n_patches)
        dirty, clean = self._inner.base[img_i]
        p = self.patch_size
        i, j = self._inner._positions(clean, img_i)[patch_i]
        return dirty[:, i:i + p, j:j + p], clean[:, i:i + p, j:j + p]


# --------------------------------------------------------------------------- #
# Inference: overlap-blended tiling                                           #
# --------------------------------------------------------------------------- #
def _hann2d(size: int, device=None) -> torch.Tensor:
    """Separable Hann window, floored so edge pixels are never weighted exactly zero."""
    w = torch.hann_window(size, periodic=False, device=device).clamp_min(1e-3)
    return torch.outer(w, w)


def tiled_denoise(
    img: np.ndarray,
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    tile: int = 256,
    overlap: int = 64,
    device: Optional[str] = None,
    batch: int = 4,
) -> np.ndarray:
    """
    Denoise a full-resolution image by running the model over overlapping tiles.

    Args:
        img: ``(H, W)`` array at native resolution.
        denoise_fn: ``(B, 1, tile, tile) -> (B, 1, tile, tile)``, both in the model's range.
        tile: tile side; use the size the model was trained at.
        overlap: pixels shared between neighbouring tiles. Larger removes seams more
            firmly and costs proportionally more tiles.
        batch: tiles pushed through ``denoise_fn`` at once.

    Returns:
        ``(H, W)`` denoised array, same dtype family as the input.

    Contributions are accumulated with a Hann weight and divided by the accumulated weight,
    so overlapping regions cross-fade rather than showing the join. Without the window a
    tile boundary lands as a hard edge in the middle of the field, and moment maps are
    computed straight over it.
    """
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {img.shape}")
    H, W = img.shape
    tile = int(min(tile, H, W))
    step = max(1, tile - int(overlap))

    # Always include the final row/column so the far edges are covered even when the span
    # is not an exact multiple of the step.
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)

    dev = torch.device(device) if device is not None else torch.device("cpu")
    win = _hann2d(tile, device=dev)
    acc = torch.zeros((H, W), dtype=torch.float32, device=dev)
    wsum = torch.zeros((H, W), dtype=torch.float32, device=dev)
    src = torch.from_numpy(np.ascontiguousarray(img)).float().to(dev)

    coords = [(y, x) for y in ys for x in xs]
    for b0 in range(0, len(coords), batch):
        chunk = coords[b0:b0 + batch]
        stack = torch.stack([src[y:y + tile, x:x + tile] for (y, x) in chunk], dim=0)[:, None]
        out = denoise_fn(stack).to(dev)
        if out.shape[-2:] != (tile, tile):
            raise ValueError(f"denoise_fn returned {tuple(out.shape[-2:])}, expected {(tile, tile)}")
        for k, (y, x) in enumerate(chunk):
            acc[y:y + tile, x:x + tile] += out[k, 0] * win
            wsum[y:y + tile, x:x + tile] += win

    return (acc / wsum.clamp_min(1e-8)).cpu().numpy().astype(img.dtype, copy=False)


def tile_grid(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int]]:
    """Tile origins `tiled_denoise` would use: for cost estimates and for figures."""
    tile = int(min(tile, H, W))
    step = max(1, tile - int(overlap))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)
    return [(y, x) for y in ys for x in xs]
