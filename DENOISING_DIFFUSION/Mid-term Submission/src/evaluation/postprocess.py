#!/usr/bin/env python
"""
src/evaluation/postprocess.py
=============================
Two cheap, legitimate ways to improve the moment maps without retraining.

Why they work is structural. M0 is an intensity **sum** along the spectral axis and scores
well (~+70%). M1 and M2 are spectral **shape** statistics: a weighted mean and a weighted
width, and they lag badly. The models denoise each channel independently and have no
mechanism enforcing consistency along the spectral axis, which is exactly the axis M1 and M2
are computed over. Anything that restores spectral coherence therefore helps them most.

Measured on a synthetic benchmark (`clean` known, three independent model outputs), against
the control that matters: what a smoother achieves with no model at all:

    dirty + spectral smooth 3.0        M0 60.5%   M1 58.4%   M2 67.7%   <- no model
    3-model ensemble                   M0 80.0%   M1 82.1%   M2 67.0%
    ensemble + spectral smooth 2.0     M0 89.8%   M1 85.8%   M2 89.4%

Read that M2 column carefully. On M0 and M1 the model wins outright, by ~20 points. **On M2
the ensemble alone does not beat a spectrally smoothed dirty cube**: 67.0% against 67.7%.
M2 is a line width and blurring along velocity is close to a direct estimator of it, so the
baseline is strong there for a real reason. Only ensemble+smoothing clears it on all three.

That caveat must travel with any M2 claim: the model's advantage on dispersion is not
established on its own, only in combination. Notebook 07 should carry spectral smoothing as
a classical baseline for exactly this reason.

`spectral_smooth` is a post-process and must be reported as one, with its sigma chosen on
the validation cubes (never on the holdout) via `tune_spectral_sigma`.
"""

from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np


def ensemble_cubes(cubes: Iterable[np.ndarray]) -> np.ndarray:
    """
    Average several models' denoised cubes.

    Plain variance reduction: independently seeded models make independent residual errors,
    so averaging cancels part of them. Notebook 08 measured a seed spread of ~1.0 dB PSNR
    on a fixed split, which is exactly the variance this removes. Costs inference only
    the checkpoints already exist.
    """
    acc, n = None, 0
    for c in cubes:
        a = np.asarray(c, dtype=np.float64)
        acc = a.copy() if acc is None else acc + a
        n += 1
    if not n:
        raise ValueError("no cubes to ensemble")
    return (acc / n).astype(np.float32)


def spectral_smooth(cube: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian smoothing along the SPECTRAL axis only (axis 0). Spatial structure untouched.

    A real emission line is spectrally continuous and the noise is not, so smoothing along
    velocity suppresses noise while preserving the line: provided sigma stays well below
    the line width. Choose it with `tune_spectral_sigma`; too large will wash out narrow
    lines and bias M2 upward, which is the failure mode to watch for.
    """
    if sigma <= 0:
        return np.asarray(cube)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.asarray(cube, dtype=np.float32), sigma, axis=0)


def tune_spectral_sigma(
    val_cubes: Sequence[dict],
    denoise: Callable[[dict], np.ndarray],
    score: Callable[[np.ndarray, dict], float],
    sigmas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    verbose: bool = True,
) -> float:
    """
    Pick the spectral-smoothing sigma on the VALIDATION cubes.

    Tuning it on the holdout would be fitting the test set: the same mistake the classical
    baselines in notebook 07 originally made by tuning at one resolution and applying at
    another. Returns the sigma with the best mean score; ties go to the smaller sigma, so a
    flat optimum never buys unnecessary blurring.

    Args:
        val_cubes: validation cube entries (never holdout).
        denoise: cube entry -> denoised cube, called once per cube and reused for all sigmas.
        score: (cube, entry) -> scalar, higher is better.
    """
    denoised = [(denoise(e), e) for e in val_cubes]
    best_sigma, best_val = 0.0, -np.inf
    for s in sigmas:
        vals = [score(spectral_smooth(c, s), e) for c, e in denoised]
        m = float(np.mean(vals))
        if verbose:
            print(f"    sigma {s:>4.1f} -> {m:+.2f}")
        if m > best_val + 1e-9:
            best_sigma, best_val = float(s), m
    if verbose:
        print(f"    chosen sigma {best_sigma:.1f} (validation, not holdout)")
    return best_sigma
