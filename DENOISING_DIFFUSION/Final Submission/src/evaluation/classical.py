#!/usr/bin/env python
"""
src/evaluation/classical.py
===========================
Classical (non-learned) denoising baselines on line-emission channels.

Why this module exists
----------------------
The project's premise is that a learned denoiser beats "traditional data
processing algorithms". Every comparison run so far has been learned-vs-learned
(V7 / V9 / V12 / beam / sweep) or learned-vs-raw-dirty. No classical filter had
ever been scored on the line-emission data, which means the central claim was
unsupported on the project's own dataset: "how much better than a Gaussian
filter?" had no answer.

This module answers it on exactly the protocol the U-Net is judged on -- the
all-5-holdout moment-map improvement -- so the numbers are directly comparable
to the V12 line (M0 +69.8 +/- 15.2 %).

Fairness
--------
The comparison is meant to be generous to the classical side, so that a win for
the network is a real win and not a straw man. The first real run showed the setup
was accidentally doing the opposite, in two ways, both now fixed:

  * **tune at the resolution you evaluate at.** Parameters were swept on 256x256
    validation channels and then applied to the native 600x600 holdout cubes. A
    sigma is in PIXELS, so sigma=4 tuned at 256 smooths 2.3x too little at 600.
    The symptom was unmistakable: pushing the filter through the network's own
    256 round trip scored M0 +36.5% against +11.7% at native resolution -- the
    "penalty" control came out strongly positive, because the round trip happened
    to restore the resolution the filter was tuned for.
  * **the grid must bracket the optimum.** Tuning selected sigma=4.0, the top of
    the old grid, so the true optimum was never reachable. `tune_on_validation`
    prints the whole sweep precisely so an edge hit is visible; the grids are now
    wide enough that it is interior.

  * **each filter's parameter is tuned**, not guessed -- swept on the VALIDATION
    cubes with the same fixed metric used to score sweep runs, then frozen for
    the holdout evaluation. Tuning on validation (never on holdout) is the same
    discipline the learned models follow.
  * filters are applied to the continuum-subtracted cube in physical units, which
    is what an astronomer would actually do. Gaussian and median are equivariant
    under the affine shared-dirty-scale normalisation, so for those two this is
    identical to filtering in normalised space; Wiener is not affine-equivariant,
    and physical units is the more natural choice for it.

No GPU and no training are involved, so a full classical evaluation runs in a
CPU-only session and costs zero GPU quota.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener

# Parameter grids swept on validation. Ranges cover "barely smooths" to
# "visibly over-smooths" so the optimum is interior, not at an edge.
#
# WIDENED after the first real run: tuning picked gaussian sigma=4.0, the old grid's
# top value, so the grid never bracketed the optimum and the filter was scored below
# its best. Every parameter here is in PIXELS, so the useful range depends on image
# size -- these now span far enough to bracket the optimum at native 600x600, where
# the holdout evaluation runs. A sigma of 4 px at 256 is the same physical scale as
# 9.4 px at 600, which the old grid could not even express.
DEFAULT_GRIDS: Dict[str, Sequence] = {
    "gaussian": (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0),   # sigma, pixels
    "median": (3, 5, 7, 9, 13, 17, 21),                              # square window, px
    "wiener": (3, 5, 7, 9, 13, 17, 21),                              # square window, px
}


def apply_filter(img: np.ndarray, method: str, param) -> np.ndarray:
    """
    Denoise one 2D channel with a classical filter.

    Args:
        img: 2D array (any scale; filters are applied in the units given).
        method: "gaussian" | "median" | "wiener" | "none".
        param: sigma for gaussian, window size for median/wiener, ignored for none.

    Returns:
        Filtered 2D float32 array, same shape as `img`.

    "none" returns the input unchanged -- the dirty control, whose improvement
    over dirty is 0 by construction and therefore a useful pipeline sanity check.

    `scipy.signal.wiener` divides by a local variance estimate and emits NaN on
    perfectly flat patches; those pixels fall back to the input value.
    """
    a = np.asarray(img, dtype=np.float32)
    if method == "none":
        return a
    if method == "gaussian":
        return gaussian_filter(a, sigma=float(param)).astype(np.float32)
    if method == "median":
        return median_filter(a, size=int(param)).astype(np.float32)
    if method == "wiener":
        out = wiener(a, mysize=int(param))
        out = np.asarray(out, dtype=np.float32)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = a[bad]
        return out
    raise ValueError(f"unknown method {method!r}; expected gaussian/median/wiener/none")


def channel_metrics(clean: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    PSNR / SSIM / MSE for one channel, matching `sweep.val_metrics` conventions.

    Both arrays are expected on the shared dirty-scale normalisation (data range
    1.0, clean may slightly exceed 1). The prediction is clamped to [0, 1] exactly
    as the sweep metric clamps model output, so the two are comparable.
    """
    from skimage.metrics import structural_similarity as ssim_fn

    c = np.asarray(clean, dtype=np.float64)
    p = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    mse = float(np.mean((c - p) ** 2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
    ssim = float(ssim_fn(c, p, data_range=1.0))
    return {"psnr": float(psnr), "ssim": ssim, "mse": mse}


def tune_on_validation(
    val_ds,
    methods: Optional[Iterable[str]] = None,
    grids: Optional[Dict[str, Sequence]] = None,
    max_channels: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Sweep each filter's parameter on the validation set; return the best per method.

    Args:
        val_ds: a `FITSChannelDataset` (continuum-subtracted, shared-scale
            normalised) yielding `(dirty, clean)` tensors of shape (1, H, W).
        methods: subset of DEFAULT_GRIDS keys; defaults to all three.
        grids: override parameter grids.
        max_channels: cap the number of validation channels scored (the full set
            is a few hundred channels x a handful of parameters; median at window
            9 on 256x256 is the slow corner).
        verbose: print a per-parameter table.

    Returns:
        {method: {"param": best, "psnr": ..., "ssim": ..., "mse": ...,
                  "all": [(param, metrics), ...]}}

    Selection is by mean PSNR, the same fixed metric the hyperparameter sweep
    scores on, so "best classical" and "best learned" are chosen the same way.
    """
    methods = list(methods) if methods is not None else list(DEFAULT_GRIDS)
    grids = grids or DEFAULT_GRIDS

    n = len(val_ds) if max_channels is None else min(max_channels, len(val_ds))
    pairs = []
    for i in range(n):
        d, c = val_ds[i][0], val_ds[i][1]
        pairs.append((np.asarray(d[0]), np.asarray(c[0])))

    if verbose:
        print(f"tuning classical filters on {n} validation channels "
              f"({'all' if max_channels is None else 'capped'})")

    results: Dict[str, dict] = {}
    for method in methods:
        rows = []
        for param in grids[method]:
            m = [channel_metrics(c, apply_filter(d, method, param)) for d, c in pairs]
            agg = {k: float(np.mean([x[k] for x in m])) for k in ("psnr", "ssim", "mse")}
            rows.append((param, agg))
            if verbose:
                print(f"  {method:<9} param={param:<4} "
                      f"PSNR {agg['psnr']:7.4f} | SSIM {agg['ssim']:.4f} | MSE {agg['mse']:.6f}")
        best_param, best = max(rows, key=lambda r: r[1]["psnr"])
        results[method] = {"param": best_param, **best, "all": rows}
        if verbose:
            print(f"  -> best {method}: param={best_param} PSNR {best['psnr']:.4f}\n")

    # dirty control: no filtering at all, for a floor reference
    m = [channel_metrics(c, d) for d, c in pairs]
    results["none"] = {
        "param": None,
        **{k: float(np.mean([x[k] for x in m])) for k in ("psnr", "ssim", "mse")},
        "all": [],
    }
    if verbose:
        print(f"  dirty (unfiltered): PSNR {results['none']['psnr']:.4f} | "
              f"SSIM {results['none']['ssim']:.4f}")
    return results


def denoise_cube(cube_csub: np.ndarray, method: str, param) -> np.ndarray:
    """
    Apply a classical filter channel-by-channel to a continuum-subtracted cube.

    Args:
        cube_csub: (C, H, W) continuum-subtracted cube in physical units.
        method / param: as `apply_filter`.

    Returns:
        (C, H, W) float32 filtered cube, native resolution preserved.

    Native resolution is deliberate: no 256x256 round trip, so the filter is
    evaluated at its best (see module docstring on fairness).
    """
    out = np.empty_like(np.asarray(cube_csub, dtype=np.float32))
    for ch in range(out.shape[0]):
        out[ch] = apply_filter(cube_csub[ch], method, param)
    return out


def summarise_improvements(rows: List[dict], moments=("M0", "M1", "M2")) -> Dict[str, dict]:
    """
    Mean/std of per-cube moment improvements, mirroring the notebooks' summary.

    Args:
        rows: per-cube dicts holding "imp_M0" / "imp_M1" / "imp_M2" percentages.

    Returns:
        {moment: {"mean": ..., "std": ..., "n": ...}} with the sample (ddof=1)
        standard deviation, matching how the V12 and beam error bars were computed.
    """
    out = {}
    for m in moments:
        vals = [r["imp_" + m] for r in rows
                if r.get("imp_" + m) is not None and np.isfinite(r["imp_" + m])]
        out[m] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out
