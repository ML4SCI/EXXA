#!/usr/bin/env python
"""
src/data/channel_sampler.py
===========================
Velocity-channel sampling for line-emission cubes.

Don't train on every channel. Sample channels
from a Gaussian centered near index 100 with ~75% of samples in [50, 150], and
avoid extreme high-velocity channels (near 0 and near n_channels) which are mostly
continuum with little signal and would confuse the model.

Note on `std`: for a Gaussian centered at 100, the band [50, 150] is center +/- 2*std
when std=25, which captures ~95% of mass: far above the 75% target. To actually hit
~75% in [50, 150] the std must be larger (~43). This module therefore auto-calibrates
std toward the target fraction by default and reports the adjustment, rather than
silently using a std that misses the target.
"""

import argparse
import os
from typing import List, Optional

import numpy as np

BAND_LOW, BAND_HIGH = 50, 150          # the [50, 150] target band
TARGET_FRACTION = 0.75                  # ~75% of samples should land in the band


def _fraction_in_band(std, center, n_channels, low_cutoff, high_cutoff,
                      band=(BAND_LOW, BAND_HIGH), n=200_000, seed=0):
    """Empirical fraction of clipped Gaussian draws landing in `band`."""
    rng = np.random.default_rng(seed)
    x = rng.normal(center, std, n)
    x = np.clip(np.round(x), 0, n_channels - 1).astype(int)
    if low_cutoff is not None:
        x = x[x >= low_cutoff]
    if high_cutoff is not None:
        x = x[x <= high_cutoff]
    if x.size == 0:
        return 0.0
    return float(np.mean((x >= band[0]) & (x <= band[1])))


def _calibrate_std(center, n_channels, low_cutoff, high_cutoff,
                   target=TARGET_FRACTION, tol=0.02, seed=0):
    """Binary-search std so ~`target` of (clipped) draws land in [50, 150].

    Fraction-in-band decreases monotonically as std grows, so we bisect on std.
    """
    lo_std, hi_std = 1.0, 200.0
    best_std, best_frac = None, None
    for _ in range(40):
        mid = 0.5 * (lo_std + hi_std)
        frac = _fraction_in_band(mid, center, n_channels, low_cutoff, high_cutoff, seed=seed)
        if best_std is None or abs(frac - target) < abs(best_frac - target):
            best_std, best_frac = mid, frac
        if abs(frac - target) <= tol:
            return mid, frac
        if frac > target:      # too concentrated -> widen
            lo_std = mid
        else:                  # too spread -> narrow
            hi_std = mid
    return best_std, best_frac


def sample_channel_indices(
    n_channels: int = 201,
    n_samples: int = 50,
    center: int = 100,
    std: float = 25,
    low_cutoff: Optional[int] = None,
    high_cutoff: Optional[int] = None,
    seed: int = 42,
    auto_adjust_std: bool = True,
    target_fraction: float = TARGET_FRACTION,
    verbose: bool = True,
) -> List[int]:
    """
    Sample unique velocity-channel indices from a Gaussian centered at `center`.

    Args:
        n_channels: total channels in the cube (valid indices 0..n_channels-1).
        n_samples: number of unique channel indices to return.
        center: mean of the sampling Gaussian (default about 100).
        std: starting standard deviation. If `auto_adjust_std`, this is calibrated
            so ~`target_fraction` of samples land in [50, 150].
        low_cutoff / high_cutoff: if given, reject indices outside [low_cutoff,
            high_cutoff] to hard-avoid extreme channels. None = no hard cutoff
            (the Gaussian already suppresses extremes).
        seed: RNG seed for reproducibility.
        auto_adjust_std: calibrate std toward `target_fraction` in [50, 150].
        target_fraction: desired fraction of samples inside [50, 150].
        verbose: print the std adjustment and achieved band fraction.

    Returns:
        Sorted list of unique integer channel indices (length n_samples, or fewer
        only if the feasible pool is smaller than n_samples).
    """
    if n_samples > n_channels:
        raise ValueError(f"n_samples ({n_samples}) > n_channels ({n_channels})")

    used_std = float(std)
    if auto_adjust_std:
        cal_std, cal_frac = _calibrate_std(
            center, n_channels, low_cutoff, high_cutoff, target=target_fraction, seed=seed
        )
        if verbose and abs(cal_std - std) > 1e-6:
            print(f"[channel_sampler] std auto-adjusted {std} -> {cal_std:.1f} "
                  f"to hit ~{target_fraction:.0%} in [{BAND_LOW},{BAND_HIGH}] "
                  f"(std={std} would give ~{_fraction_in_band(std, center, n_channels, low_cutoff, high_cutoff, seed=seed):.0%})")
        used_std = cal_std

    rng = np.random.default_rng(seed)

    # feasible pool size (so we never spin forever if cutoffs are very tight)
    pool_lo = 0 if low_cutoff is None else max(0, low_cutoff)
    pool_hi = (n_channels - 1) if high_cutoff is None else min(n_channels - 1, high_cutoff)
    feasible = pool_hi - pool_lo + 1
    target_n = min(n_samples, feasible)

    chosen = set()
    attempts = 0
    while len(chosen) < target_n and attempts < 10_000:
        # draw a batch, round to int, clip to valid range
        batch = rng.normal(center, used_std, size=max(target_n * 4, 64))
        batch = np.clip(np.round(batch), 0, n_channels - 1).astype(int)
        # reject extremes if hard cutoffs provided
        if low_cutoff is not None:
            batch = batch[batch >= low_cutoff]
        if high_cutoff is not None:
            batch = batch[batch <= high_cutoff]
        for v in batch:
            chosen.add(int(v))
            if len(chosen) >= target_n:
                break
        attempts += 1

    result = sorted(chosen)
    if verbose:
        in_band = sum(1 for v in result if BAND_LOW <= v <= BAND_HIGH)
        frac = in_band / len(result) if result else 0.0
        print(f"[channel_sampler] returned {len(result)} unique indices "
              f"(std={used_std:.1f}); {in_band}/{len(result)} = {frac:.1%} in [{BAND_LOW},{BAND_HIGH}]; "
              f"range [{result[0]}, {result[-1]}]")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sample line-emission channel indices")
    ap.add_argument("--n-channels", type=int, default=201)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--center", type=int, default=100)
    ap.add_argument("--std", type=float, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/channel_sampling_distribution.png")
    args = ap.parse_args()

    idx = sample_channel_indices(
        n_channels=args.n_channels, n_samples=args.n_samples,
        center=args.center, std=args.std, seed=args.seed,
    )

    in_band = sum(1 for v in idx if BAND_LOW <= v <= BAND_HIGH)
    pct = 100.0 * in_band / len(idx)
    print(f"\nSampled {len(idx)} unique channel indices:")
    print(idx)
    print(f"\n{in_band}/{len(idx)} = {pct:.1f}% fall in [{BAND_LOW}, {BAND_HIGH}]")
    print(f"min index = {idx[0]}, max index = {idx[-1]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(idx, bins=range(0, args.n_channels + 1, 5), color="#4C9EEB",
            edgecolor="white", alpha=0.9)
    ax.axvspan(BAND_LOW, BAND_HIGH, color="#3BA55C", alpha=0.12,
               label=f"[{BAND_LOW}, {BAND_HIGH}] target band ({pct:.1f}%)")
    ax.axvline(args.center, color="#E8715A", ls="--", lw=1.5, label=f"center={args.center}")
    ax.set_xlabel("velocity channel index")
    ax.set_ylabel("count")
    ax.set_title(f"Sampled channel distribution: {len(idx)} channels, "
                 f"{pct:.1f}% in [{BAND_LOW},{BAND_HIGH}]")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"\nhistogram saved -> {args.out}")
