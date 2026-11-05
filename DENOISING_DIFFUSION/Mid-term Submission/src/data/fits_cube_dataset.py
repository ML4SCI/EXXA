#!/usr/bin/env python
"""
src/data/fits_cube_dataset.py
=============================
PyTorch Dataset for line-emission FITS cubes: full-image, channel-by-channel.

Design:
  - Full-image input (no patches); downsample to `target_size` (default 256) for memory.
  - Train on individual velocity channels, sampled per cube via the Gaussian sampler
    (`channel_sampler.sample_channel_indices`): ~center 100, ~75% in [50,150], extremes avoided.
  - Each (cube, channel) pair is one dataset item, so __len__ = n_cubes * channels_per_cube.
  - __getitem__ reads ONLY the requested channel slice via astropy memmap (the full
    201x600x600 cube is never materialised).
  - Per-CHANNEL min-max normalization to [0,1] (channels have very different intensity
    ranges; the dirty cubes also contain negative noise so per-channel min-max is needed).

Returns (dirty_channel, clean_channel), each shape (1, target_size, target_size), float32 in [0,1].
"""

import os
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from torch.utils.data import Dataset

# package-relative imports with script fallback
try:
    from .channel_sampler import sample_channel_indices
except ImportError:  # run as a script
    from channel_sampler import sample_channel_indices


def _to_native_float32(arr: np.ndarray) -> np.ndarray:
    """FITS data is often big-endian (>f4); convert to native-endian float32 for torch."""
    return np.ascontiguousarray(arr).astype(np.float32)


def beam_features_of(header) -> np.ndarray:
    """
    4-feature beam vector from a FITS header:
    [sin(2*BPA), cos(2*BPA), BMAJ*3600, BMIN*3600].

    BPA is stored in degrees (convert to radians before sin/cos; the 2x makes the
    encoding invariant to the beam's 180-degree ambiguity). BMAJ/BMIN are FWHM in
    degrees -> *3600 gives arcsec (O(0.1) here, so no further scaling needed).
    Teaches the model the spatial scale/orientation of the noise (the beam).

    Missing beam keys -> zero vector (clean model-input, and a natural "no beam
    info" null for cubes without beam metadata).
    """
    if header is None or "BPA" not in header or "BMAJ" not in header or "BMIN" not in header:
        return np.zeros(4, dtype=np.float32)
    bpa = np.deg2rad(float(header["BPA"]))
    return np.array([np.sin(2 * bpa), np.cos(2 * bpa),
                     float(header["BMAJ"]) * 3600.0,
                     float(header["BMIN"]) * 3600.0], dtype=np.float32)


def continuum_of(cube: np.ndarray, n: int) -> np.ndarray:
    """
    2D continuum estimate = mean of the first `n` and last `n` (line-free, high-velocity)
    channels of an in-memory cube (C, H, W). Subtracting this from every channel removes the
    static disk continuum and isolates the line emission .

    Array-in variant of `FITSChannelDataset._compute_continuum` (which reads from a path), so
    notebooks that already hold the cube in memory can reuse the exact same definition.
    """
    C = cube.shape[0]
    n = max(1, min(n, C // 2))
    edges = np.concatenate([np.asarray(cube[:n]), np.asarray(cube[C - n:])], axis=0)
    return _to_native_float32(edges.mean(axis=0))


class FITSChannelDataset(Dataset):
    """
    Channel-level dataset over a list of line-emission cubes.

    Args:
        cubes: list of cube dicts {folder, run_id, clean, dirty} (from `cube_split.split_cubes`),
            or a list of (clean_path, dirty_path) tuples.
        channel_sampler_fn: callable(n_channels, seed) -> list[int]. Defaults to a partial of
            `sample_channel_indices` with the default settings.
        n_samples: channels to sample per cube (default 50).
        target_size: output H=W after bilinear downsample (default 256).
        seed: base seed; each cube gets a deterministic per-cube seed so the channel set is
            fixed across epochs (reproducible) but differs between cubes.
        augment: apply random dihedral (D4) augmentation to each (dirty, clean) pair.
            TRAIN SPLITS ONLY: enabling it on val/holdout makes evaluation
            non-deterministic. Off by default; see `_augment_pair`.
        cache_channel_stats: unused placeholder for future per-channel stat caching.
    """

    def __init__(
        self,
        cubes: List,
        channel_sampler_fn: Optional[Callable] = None,
        n_samples: int = 50,
        target_size: int = 256,
        seed: int = 42,
        subtract_continuum: bool = False,
        continuum_n: int = 5,
        return_beam: bool = False,
        augment: bool = False,
        verbose: bool = True,
    ):
        self.target_size = target_size
        self.seed = seed
        # Dihedral (D4) augmentation: 8 lossless orientations, applied identically
        # to dirty and clean. OFF by default so every existing checkpoint and result
        # (V12 included) stays reproducible; TRAIN splits only, never val/holdout,
        # or the evaluation stops being deterministic. See `_augment_pair`.
        self.augment = augment
        # Beam conditioning: expose the observation's beam
        # (noise scale/orientation) as a 4-vector model input. OFF by default so
        # existing notebooks keep their (dirty, clean) item shape.
        self.return_beam = return_beam
        # Continuum subtraction: the first/last high-velocity channels are
        # line-free, so their mean estimates the static continuum (disk dust behind the line).
        # Subtracting it before normalization isolates the line emission. OFF by default so the
        # original notebook behavior is unchanged; the continuum notebook flips it on.
        self.subtract_continuum = subtract_continuum
        self.continuum_n = continuum_n

        # normalise `cubes` into a list of (clean_path, dirty_path, tag)
        self.cube_paths = []
        for c in cubes:
            if isinstance(c, dict):
                self.cube_paths.append((c["clean"], c["dirty"], c.get("folder", "")))
            else:
                clean, dirty = c
                self.cube_paths.append((clean, dirty, os.path.basename(os.path.dirname(dirty))))

        if channel_sampler_fn is None:
            channel_sampler_fn = lambda n_channels, seed: sample_channel_indices(
                n_channels=n_channels, n_samples=n_samples, seed=seed, verbose=False
            )
        self.channel_sampler_fn = channel_sampler_fn

        # build the flat (cube_idx, channel_idx) index list
        self.index = []          # list of (cube_idx, channel_idx)
        self.per_cube_channels = []
        for ci, (clean, dirty, tag) in enumerate(self.cube_paths):
            n_channels = self._cube_n_channels(dirty)
            # deterministic per-cube seed so channel sets are stable across epochs but vary by cube
            chans = self.channel_sampler_fn(n_channels=n_channels, seed=seed + ci)
            self.per_cube_channels.append(chans)
            for ch in chans:
                self.index.append((ci, ch))

        # precompute per-cube continuum (dirty + clean) once, if enabled: each cube subtracts
        # its OWN continuum (the dirty and clean baselines differ).
        self.dirty_continuum = []
        self.clean_continuum = []
        if self.subtract_continuum:
            for ci, (clean, dirty, tag) in enumerate(self.cube_paths):
                self.dirty_continuum.append(self._compute_continuum(dirty, self.continuum_n))
                self.clean_continuum.append(self._compute_continuum(clean, self.continuum_n))

        # precompute per-cube beam vector from the DIRTY header (the beam belongs to
        # the observation); torch tensors so __getitem__ is a cheap index
        self.beam_vectors = []
        if self.return_beam:
            for clean, dirty, tag in self.cube_paths:
                with fits.open(dirty, memmap=True) as hdul:
                    self.beam_vectors.append(torch.from_numpy(beam_features_of(hdul[0].header)))
            if verbose and len(self.beam_vectors) > 1:
                stacked = torch.stack(self.beam_vectors)
                if torch.allclose(stacked, stacked[0].expand_as(stacked)):
                    print("[FITSChannelDataset] WARNING: beam vector identical across all cubes: "
                          "beam conditioning carries no information for this split")

        if verbose:
            cont = (f" | continuum-subtracted (mean of first/last {continuum_n} channels)"
                    if self.subtract_continuum else "")
            aug = " | D4 augmentation ON" if self.augment else ""
            print(f"[FITSChannelDataset] {len(self.cube_paths)} cubes x "
                  f"~{n_samples} channels = {len(self.index)} items | target {target_size}x{target_size}{cont}{aug}")

    @staticmethod
    def _cube_n_channels(path: str) -> int:
        """Read just the channel count from the FITS header (no data load)."""
        with fits.open(path, memmap=True) as hdul:
            shape = hdul[0].shape           # (C, H, W)
        return shape[0]

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _load_channel(path: str, ch: int) -> np.ndarray:
        """Memmap-open the cube and read ONLY channel `ch` (a 2D slice)."""
        with fits.open(path, memmap=True) as hdul:
            plane = hdul[0].data[ch]        # lazy slice -> only this plane is read
            plane = _to_native_float32(plane)
        return plane

    @staticmethod
    def _compute_continuum(path: str, n: int) -> np.ndarray:
        """
        2D continuum estimate = mean of the first `n` and last `n` (line-free, high-velocity)
        channels. Subtracting this from every channel removes the static disk continuum and
        isolates the line emission .
        """
        with fits.open(path, memmap=True) as hdul:
            return continuum_of(hdul[0].data, n)

    @staticmethod
    def _minmax_norm_shared(dirty: np.ndarray, clean: np.ndarray):
        """
        Per-channel min-max using the DIRTY channel's (min,max) for BOTH dirty and clean.

        Critical for invertibility at inference: at test time only the dirty (min,max) is
        known (the clean cube is the unknown we predict). Normalizing the clean target by its
        OWN (min,max) makes the model output live in a clean-specific scale that cannot be
        decoded from the dirty scale -> a negative DC floor (decode background = dirty_min < 0
        vs clean background = 0) that destroys Moment-0. Sharing the dirty scale keeps the
        background floor consistent and the un-normalization exact.

        Clean is NOT clipped: its peak can exceed the dirty max, so normalized clean may slightly
        exceed 1 (the model uses a linear output head, not sigmoid, to represent that).
        """
        lo, hi = float(dirty.min()), float(dirty.max())
        if hi > lo:
            d = (dirty - lo) / (hi - lo)
            c = (clean - lo) / (hi - lo)
            return d, c
        return np.zeros_like(dirty), np.zeros_like(clean)

    def _resize(self, x: np.ndarray) -> torch.Tensor:
        """(H,W) np -> (1, target, target) bilinear-resized tensor."""
        t = torch.from_numpy(x)[None, None]           # (1,1,H,W)
        if t.shape[-1] != self.target_size or t.shape[-2] != self.target_size:
            t = F.interpolate(t, size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)
        return t[0]                                    # (1, target, target)

    def __getitem__(self, i: int):
        cube_idx, ch = self.index[i]
        clean_path, dirty_path, _ = self.cube_paths[cube_idx]

        dirty = self._load_channel(dirty_path, ch)
        clean = self._load_channel(clean_path, ch)

        # continuum subtraction (before norm), each cube subtracts its own line-free baseline
        if self.subtract_continuum:
            dirty = dirty - self.dirty_continuum[cube_idx]
            clean = clean - self.clean_continuum[cube_idx]

        # per-channel min-max using the DIRTY (min,max) for BOTH -> invertible at inference,
        # consistent background floor (see _minmax_norm_shared). NOT independent per array.
        dirty, clean = self._minmax_norm_shared(dirty, clean)

        dirty_t = self._resize(dirty).float()          # (1, target, target)
        clean_t = self._resize(clean).float()

        # dihedral augmentation, TRAIN ONLY (see _augment_pair)
        if self.augment:
            dirty_t, clean_t = self._augment_pair(dirty_t, clean_t)

        if self.return_beam:
            return dirty_t, clean_t, self.beam_vectors[cube_idx]
        return dirty_t, clean_t

    @staticmethod
    def _augment_pair(dirty_t: torch.Tensor, clean_t: torch.Tensor):
        """
        Apply one random element of the dihedral group D4 to BOTH channels alike.

        The group is the 4 multiples of 90 degrees times an optional horizontal
        flip: 8 orientations, each an exact array permutation. Arbitrary-angle
        rotation would need interpolation, which smooths the very pixel noise the
        model is being trained to remove, so only the lossless subgroup is used.

        The identical transform is applied to dirty and clean, which is what keeps
        the pair a valid training example. Sampling uses torch's global RNG, so
        `DataLoader` worker seeding gives every worker a distinct but reproducible
        stream, and the orientation varies across epochs (the point of augmenting,
        as opposed to just enlarging a fixed dataset).

        Why this is safe here: a channel map's noise is beam-shaped and therefore
        oriented, but rotating dirty and clean together only ever shows the network
        the same denoising problem at a new orientation. With 14 cubes total the
        regularisation is worth far more than the orientation prior being diluted,
        and the sweep found beam orientation carries little signal anyway.
        """
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            dirty_t = torch.rot90(dirty_t, k, dims=(-2, -1))
            clean_t = torch.rot90(clean_t, k, dims=(-2, -1))
        if bool(torch.randint(0, 2, (1,)).item()):
            dirty_t = torch.flip(dirty_t, dims=(-1,))
            clean_t = torch.flip(clean_t, dims=(-1,))
        return dirty_t.contiguous(), clean_t.contiguous()


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from .cube_split import split_cubes
    except ImportError:
        from cube_split import split_cubes

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/Line Emission Data")
    ap.add_argument("--target-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--out", default="results/line_emission_sample.png")
    args = ap.parse_args()

    train, val, holdout = split_cubes(data_dir=args.data_dir, verbose=True)

    ds = FITSChannelDataset(train, n_samples=args.n_samples, target_size=args.target_size)
    print(f"\nDataset length (train): {len(ds)}")

    dirty_t, clean_t = ds[0]
    print(f"sample 0: dirty {tuple(dirty_t.shape)} range [{dirty_t.min():.3f}, {dirty_t.max():.3f}]")
    print(f"sample 0: clean {tuple(clean_t.shape)} range [{clean_t.min():.3f}, {clean_t.max():.3f}]")
    ci, ch = ds.index[0]
    print(f"sample 0: cube '{ds.cube_paths[ci][2]}' channel {ch}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(dirty_t[0].numpy(), cmap="inferno"); ax[0].set_title(f"dirty (ch {ch})"); ax[0].axis("off")
    ax[1].imshow(clean_t[0].numpy(), cmap="inferno"); ax[1].set_title(f"clean (ch {ch})"); ax[1].axis("off")
    fig.suptitle(f"Line-emission sample: {ds.cube_paths[ci][2]}, {args.target_size}x{args.target_size}",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig(args.out, dpi=140)
    print(f"\nsaved -> {args.out}")
