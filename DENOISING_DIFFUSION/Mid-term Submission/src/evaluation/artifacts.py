#!/usr/bin/env python
"""
src/evaluation/artifacts.py
===========================
Per-channel artifact diagnostics for a denoised line-emission channel.

Three artifacts have been observed across V7/V9/V12 but only ever documented
from a single channel (channel 100), which the V7/V9 variance lesson says is not
enough to characterise anything:

  * **peak overshoot**: denoised peak brighter than clean (1.151x at ch 100).
  * **negative floor leak**: denoised background dips below clean's ~0 floor.
  * **invented structure** ("hallucination"), the model asserts signal where
    the ground truth has none. Scientifically the worst of the three: in a
    protoplanetary-disk map, invented structure reads as a false detection.

`channel_artifacts` scores one channel so a caller can run it over every
validation channel and report distributions instead of anecdotes. It is
model-agnostic (takes arrays, not a model), so the same numbers can be produced
for the U-Net and the DDPM and compared directly.
"""

from typing import Dict

import numpy as np
from scipy import ndimage

# Defaults chosen so "background" and "asserted signal" are separated by a clear
# margin: a pixel the truth puts below 10% of peak, the model puts above 20%.
FLOOR_FRAC = 0.10   # clean below this fraction of its own peak counts as background
INVENT_FRAC = 0.20  # denoised above this fraction of clean's peak = asserted signal
BLOB_MIN_PX = 20    # ignore single-pixel specks; count only structures this large


def channel_artifacts(
    clean: np.ndarray,
    dirty: np.ndarray,
    denoised: np.ndarray,
    *,
    floor_frac: float = FLOOR_FRAC,
    invent_frac: float = INVENT_FRAC,
    blob_min_px: int = BLOB_MIN_PX,
) -> Dict[str, float]:
    """
    Score overshoot / floor leak / invented structure for one channel.

    All three inputs are 2D arrays on the same scale (the shared dirty-scale
    normalisation, so clean may exceed 1).

    Returns a dict with:
        snr:            clean's peak-above-floor / std of dirty over clean's
                        background pixels. NaN only when the background is genuinely
                        empty or noiseless.
        overshoot:      (denoised.max() - floor) / (clean.max() - floor); 1.0 is
                        perfect, >1 overshoots the true peak.
        floor_leak:     denoised.min() - clean's floor; negative means the
                        prediction dips below the true background.
        invented_frac:  fraction of background pixels pushed above the assert
                        threshold.
        invented_blobs: count of connected invented regions of >= blob_min_px,
                        i.e. how many distinct fake structures.
        n_background_px: size of the background mask. Report it: an empty or tiny
                        mask makes `invented_*` vacuously zero, which is how the
                        first run's "no invented structure" result arose.

    Raises ValueError if clean has no finite peak or no positive dynamic range.
    """
    cmax = float(clean.max())
    if not np.isfinite(cmax):
        raise ValueError("clean channel has no finite peak; cannot score artifacts")

    # Everything below is measured relative to the channel's OWN floor and span,
    # never as an absolute fraction of the peak.
    #
    # The original version used `clean < floor_frac * cmax` and reported
    # `denoised.min()` directly, both of which assume the clean background sits at
    # ~0. Under the shared dirty-scale normalisation it does not: continuum
    # subtraction gives DIRTY a negative minimum, and normalising CLEAN by dirty's
    # (min,max) maps clean's zero background to -lo/(hi-lo), a strongly positive
    # number: 0.32 on a representative channel. The background mask then selected
    # ZERO pixels, which silently made SNR NaN, `invented_frac` 0 by construction
    # rather than by evidence, and turned `floor_leak` into a restatement of where
    # the background happens to land. Measuring against the floor makes all four
    # quantities invariant to any affine rescaling of the channel.
    floor = float(np.percentile(clean, 1))       # robust floor, not a lone hot/cold pixel
    span = cmax - floor
    if not np.isfinite(span) or span <= 0:
        raise ValueError("clean channel has no positive dynamic range; cannot score artifacts")

    background = clean < floor + floor_frac * span
    # Noise proxy from DIRTY over clean-background pixels: measures what the model
    # had to see through, not what it produced.
    noise = float(dirty[background].std()) if background.any() else float("nan")
    # peak-above-floor over background noise: a genuine amplitude SNR
    snr = span / noise if np.isfinite(noise) and noise > 0 else float("nan")

    invented = background & (denoised > floor + invent_frac * span)
    labels, n_labels = ndimage.label(invented)
    if n_labels:
        sizes = ndimage.sum(invented, labels, range(1, n_labels + 1))
        n_blobs = int((np.asarray(sizes) >= blob_min_px).sum())
    else:
        n_blobs = 0

    return {
        "snr": snr,
        # both ratios are taken above the floor, so a shared positive offset cannot
        # compress them toward 1 and disguise a real deviation
        "overshoot": float((float(denoised.max()) - floor) / span),
        "floor_leak": float(float(denoised.min()) - floor),
        "invented_frac": float(invented.sum() / max(int(background.sum()), 1)),
        "invented_blobs": n_blobs,
        "n_background_px": int(background.sum()),
    }


def summarise(rows) -> Dict[str, float]:
    """
    Aggregate `channel_artifacts` dicts into report-ready statistics.

    Splits the invented-structure and overshoot rates at the median SNR, which is
    what answers a question we cared about: is invented structure a
    low-SNR-specific failure, or does it happen everywhere?
    """
    if not rows:
        return {}
    ov = np.array([r["overshoot"] for r in rows], dtype=float)
    leak = np.array([r["floor_leak"] for r in rows], dtype=float)
    inv = np.array([r["invented_frac"] for r in rows], dtype=float)
    blob = np.array([r["invented_blobs"] for r in rows], dtype=float)
    snr = np.array([r["snr"] for r in rows], dtype=float)

    nbg = np.array([r.get("n_background_px", np.nan) for r in rows], dtype=float)
    out = {
        "n_channels": len(rows),
        # If this is not ~1.0 the invented-structure numbers are not trustworthy:
        # an empty background mask makes them zero by construction.
        "frac_channels_with_background": float(np.mean(nbg > 0)) if np.isfinite(nbg).any() else float("nan"),
        "snr_finite_frac": float(np.mean(np.isfinite([r["snr"] for r in rows]))),
        "overshoot_mean": float(ov.mean()),
        "overshoot_median": float(np.median(ov)),
        "overshoot_p90": float(np.percentile(ov, 90)),
        "overshoot_max": float(ov.max()),
        "frac_over_10pct": float((ov > 1.10).mean()),
        "floor_leak_mean": float(leak.mean()),
        "floor_leak_min": float(leak.min()),
        "invented_frac_mean": float(inv.mean()),
        "invented_frac_max": float(inv.max()),
        "frac_channels_with_blob": float((blob > 0).mean()),
        "blobs_per_channel": float(blob.mean()),
    }

    ok = np.isfinite(snr)
    if ok.sum() > 4:
        thr = float(np.median(snr[ok]))
        lo, hi = ok & (snr <= thr), ok & (snr > thr)
        out.update({
            "snr_median": thr,
            "low_snr_blobs_per_channel": float(blob[lo].mean()),
            "high_snr_blobs_per_channel": float(blob[hi].mean()),
            "low_snr_invented_frac": float(inv[lo].mean()),
            "high_snr_invented_frac": float(inv[hi].mean()),
            "low_snr_overshoot": float(ov[lo].mean()),
            "high_snr_overshoot": float(ov[hi].mean()),
        })
    return out
