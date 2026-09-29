#!/usr/bin/env python
"""
src/evaluation/moment_maps.py
=============================
Moment-map generation for line-emission cubes via the `bettermoments` package.

Moment maps are the actual scientific data product : an individual
denoised channel means little; the value is in the collapsed cube. We generate:
  - Moment 0 : integrated intensity            (collapse_zeroth)
  - Moment 1 : intensity-weighted velocity     (collapse_first), the rotation/kinematics map
  - Moment 2 : intensity-weighted dispersion    (collapse_second)

The ultimate denoising test: build moment maps from a DIRTY cube vs the model's DENOISED cube
and compare against the CLEAN-cube moment maps. This module provides the generation + a
clean/dirty comparison visualization to confirm the package works.

bettermoments API (v1.10):
  data, velax = bm.load_cube(path)              # data (C,H,W), velax (C,)
  rms          = bm.estimate_RMS(data, N=...)   # noise from N edge channels
  Mx, dMx      = bm.collapse_{zeroth,first,second}(velax, data, rms)
"""

import os
from typing import Optional, Tuple

import numpy as np


def generate_moment_maps(
    fits_path: str,
    rms_n_channels: int = 5,
    save_path: Optional[str] = None,
    data_velax: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: Optional[str] = None,
    clip_sigma: float = 3.0,
    tile_bytes: int = 8 << 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Moment 0/1/2 maps for a line-emission FITS cube.

    Args:
        fits_path: path to the FITS cube (ignored if `data_velax` is supplied).
        rms_n_channels: number of edge channels used to estimate the noise RMS.
        save_path: if given, save a 3-panel (M0|M1|M2) visualization here.
        data_velax: optional pre-loaded (data, velax) to collapse instead of reading
            from disk: used to make moment maps from an in-memory DENOISED cube.
        title: optional suptitle for the figure.
        clip_sigma: channels below `clip_sigma * rms` are zeroed before collapsing.
            `bettermoments.collapse_{first,second}` divide by the per-pixel flux sum
            with no threshold of their own; on a spectral background where flux
            oscillates near zero, that sum crosses zero and the moment blows up
            (M2 is a sqrt of a ratio, so it is unbounded, not just noisy). One cube's
            worth of these pixels is enough to saturate a shared colour scale and to
            dominate a mean-absolute-difference metric taken over the whole map. Set
            to 0 to disable and reproduce the old unclipped behaviour.
        tile_bytes: size of one horizontal strip. The collapse is done in strips because
            `bettermoments` is extremely allocation-hungry. Measured on a 201x600x600
            float32 cube (0.27 GiB): `collapse_zeroth` peaks at 2x the cube,
            `collapse_first` at 8x and `collapse_second` at 12x, for 3.54 GiB of transient
            per call, and there are three calls per cube. That, not the cubes themselves,
            is what exhausted the Kaggle host and killed the kernel with no traceback
            across three runs; trimming the 289 MB cubes around it was never going to be
            enough.

            All three collapses are per-pixel along the spectral axis, so strips are exact
            rather than an approximation. Peak scales linearly with this value (it is
            amplified ~13x by the collapse), and strips are also *faster*: better cache
            locality:

                untiled   3.54 GiB peak   5.24 s
                8 MB      0.11 GiB peak   1.97 s      <- default

            M0 comes out bit-identical and the NaN pattern is unchanged. Over the brightest
            10% of pixels (the region anything is reported from), M1 agrees to 9e-6
            relative and M2 to 5e-11. Larger relative differences occur only on near-zero
            background pixels, where the denominator is ~0 and the value is meaningless
            anyway; that is the instability `clip_sigma` exists to suppress.

            Set to 0 to collapse in one shot.

    Returns:
        (moment0, moment1, moment2) as numpy arrays, each shape (H, W).
    """
    import bettermoments as bm

    if data_velax is not None:
        data, velax = data_velax
    else:
        data, velax = bm.load_cube(fits_path)

    data = np.asarray(data)
    velax = np.asarray(velax)
    C, H, W = data.shape

    # One global noise estimate from the line-free edge channels. It must be computed on
    # the whole cube and shared by every tile: a per-tile rms would make the clip
    # threshold, and therefore the moments, depend on how the image was subdivided.
    rms = bm.estimate_RMS(data=data, N=rms_n_channels)

    rows = H if tile_bytes <= 0 else max(1, min(H, int(tile_bytes // (C * W * 4))))

    moment0 = np.empty((H, W), dtype=np.float64)
    moment1 = np.empty((H, W), dtype=np.float64)
    moment2 = np.empty((H, W), dtype=np.float64)

    for y0 in range(0, H, rows):
        y1 = min(y0 + rows, H)
        blk = np.asarray(data[:, y0:y1, :])
        if clip_sigma > 0:
            blk = np.where(np.abs(blk) >= clip_sigma * rms, blk, 0.0)
        moment0[y0:y1], _ = bm.collapse_zeroth(velax=velax, data=blk, rms=rms)
        moment1[y0:y1], _ = bm.collapse_first(velax=velax, data=blk, rms=rms)
        moment2[y0:y1], _ = bm.collapse_second(velax=velax, data=blk, rms=rms)
        del blk

    if save_path is not None:
        _plot_moments(moment0, moment1, moment2, save_path, title=title or os.path.basename(fits_path))

    return moment0, moment1, moment2


def _plot_moments(m0, m1, m2, save_path, title=""):
    """Save a 3-panel M0/M1/M2 figure with appropriate colormaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (m0, "Moment 0 (integrated intensity)", "inferno"),
        (m1, "Moment 1 (velocity)", "RdBu_r"),          # diverging: rotation red/blue
        (m2, "Moment 2 (dispersion)", "viridis"),
    ]
    for a, (m, name, cmap) in zip(ax, panels):
        im = a.imshow(m, origin="lower", cmap=cmap)
        a.set_title(name); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def _plot_clean_vs_dirty(clean_maps, dirty_maps, save_path, tag=""):
    """2x3 grid: clean (top) vs dirty (bottom) moment maps, shared scale per column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    names = ["Moment 0 (intensity)", "Moment 1 (velocity)", "Moment 2 (dispersion)"]
    cmaps = ["inferno", "RdBu_r", "viridis"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    for col in range(3):
        cm, dm = clean_maps[col], dirty_maps[col]
        # shared color scale per column so clean/dirty are directly comparable
        vmin = float(np.nanmin([np.nanmin(cm), np.nanmin(dm)]))
        vmax = float(np.nanmax([np.nanmax(cm), np.nanmax(dm)]))
        for row, m in [(0, cm), (1, dm)]:
            im = ax[row, col].imshow(m, origin="lower", cmap=cmaps[col], vmin=vmin, vmax=vmax)
            ax[row, col].axis("off")
            fig.colorbar(im, ax=ax[row, col], fraction=0.046, pad=0.04)
        ax[0, col].set_title(names[col])
    ax[0, 0].set_ylabel("CLEAN", fontsize=13)
    ax[1, 0].set_ylabel("DIRTY", fontsize=13)
    # ylabels get cleared by axis('off'); add text labels instead
    fig.text(0.02, 0.74, "CLEAN", fontsize=14, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.26, "DIRTY", fontsize=14, fontweight="bold", rotation=90, va="center")
    fig.suptitle(f"Moment maps: clean vs dirty  {tag}", fontweight="bold", fontsize=14)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    plt.savefig(save_path, dpi=140)
    plt.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test bettermoments on a held-out clean/dirty pair")
    ap.add_argument("--clean", default="data/Line Emission Data/run_0002_00560_rt_00/run_0002_00560_rt_00_clean.fits")
    ap.add_argument("--dirty", default="data/Line Emission Data/run_0002_00560_rt_00/run_0002_00560_rt_00_dirty.fits")
    ap.add_argument("--out", default="results/moment_maps_test.png")
    args = ap.parse_args()

    print("Generating moment maps (clean)...")
    cm = generate_moment_maps(args.clean)
    print("Generating moment maps (dirty)...")
    dm = generate_moment_maps(args.dirty)

    # quantitative sanity: how different are clean vs dirty moment maps?
    def stats(name, c, d):
        diff = np.abs(c - d)
        denom = np.nanmax(np.abs(c)) or 1.0
        print(f"  {name}: clean[min={np.nanmin(c):.4g}, max={np.nanmax(c):.4g}]  "
              f"dirty[min={np.nanmin(d):.4g}, max={np.nanmax(d):.4g}]  "
              f"mean|diff|={np.nanmean(diff):.4g} ({100*np.nanmean(diff)/denom:.1f}% of clean max)")

    print("\nClean vs dirty moment-map comparison:")
    stats("M0", cm[0], dm[0])
    stats("M1", cm[1], dm[1])
    stats("M2", cm[2], dm[2])

    tag = os.path.basename(os.path.dirname(args.clean))
    _plot_clean_vs_dirty(cm, dm, args.out, tag=tag)
    print(f"\nsaved comparison -> {args.out}")


# --------------------------------------------------------------------------- #
# Scoring: where a moment map should actually be compared                      #
# --------------------------------------------------------------------------- #
SIGNAL_FRAC = 0.05   # of the clean M0 peak


def signal_mask(clean_m0: np.ndarray, frac: float = SIGNAL_FRAC) -> np.ndarray:
    """Pixels where the TRUTH has real emission.

    Defined from the clean cube alone, so it is identical for every method being
    compared and cannot be tuned per model. This uses ground truth and is therefore a
    benchmark construct: on real data the mask would have to come from the observation.
    """
    finite = np.isfinite(clean_m0)
    if not finite.any():
        return finite
    peak = float(np.nanmax(np.abs(clean_m0[finite])))
    if not np.isfinite(peak) or peak <= 0:
        return finite
    return finite & (np.abs(clean_m0) > frac * peak)


def moment_improvement(clean, dirty, denoised, frac: float = SIGNAL_FRAC) -> dict:
    """
    Percent improvement of `denoised` over `dirty` against `clean`, per moment.

    `clean`, `dirty`, `denoised` are each a (M0, M1, M2) triple.

    Scored over the signal mask rather than the whole map. Averaging over every finite
    pixel (the previous behaviour) lets empty sky dominate, and dispersion in a pixel
    with no line is not a quantity anyone reports. It penalised M2 hardest because M2 is
    a ratio whose denominator vanishes exactly there.

    The mask does not change which method wins. On a synthetic benchmark with three
    denoisers of known quality the ranking is identical at every threshold from 0.5% to
    10% of peak; only the scale moves (M2 for the best model, 54.1% unmasked vs 68-77%
    masked). `frac` is deliberately taken from a plateau rather than its maximum.

    Returns per-moment improvements plus `n_px`, and the unmasked values under
    `M0_all`/`M1_all`/`M2_all` so the effect of masking is always visible alongside it.
    """
    mask = signal_mask(clean[0], frac)
    out = {"n_px": int(mask.sum()), "frac": frac}
    for i, nm in enumerate(("M0", "M1", "M2")):
        cl, di, no = np.asarray(clean[i]), np.asarray(dirty[i]), np.asarray(denoised[i])
        base = np.isfinite(cl) & np.isfinite(di) & np.isfinite(no)
        for suffix, m in ((nm, base & mask), (nm + "_all", base)):
            if not m.any():
                out[suffix] = float("nan")
                continue
            dd = float(np.abs(cl[m] - di[m]).mean())
            nn = float(np.abs(cl[m] - no[m]).mean())
            out[suffix] = 100.0 * (1.0 - nn / dd) if dd > 0 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Publication figure: clean / dirty / denoised on one honest scale             #
# --------------------------------------------------------------------------- #
MOMENT_LABELS = ("Moment 0: integrated intensity",
                 "Moment 1: line-of-sight velocity",
                 "Moment 2: velocity dispersion")


def _limits(maps, mask, kind, lo=1.0, hi=99.0):
    """Shared colour limits for one moment, taken from masked CLEAN + DIRTY + DENOISED.

    Percentiles rather than min/max: a single hot pixel otherwise sets the ceiling and
    flattens everything else. Velocity is forced symmetric about the systemic value so
    the diverging colormap's white point means "not rotating" instead of landing wherever
    the data happens to average.
    """
    vals = np.concatenate([m[mask & np.isfinite(m)].ravel() for m in maps])
    if vals.size == 0:
        return None, None
    if kind == "velocity":
        centre = float(np.median(vals))
        span = float(np.percentile(np.abs(vals - centre), hi)) or 1.0
        return centre - span, centre + span
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def _intensity_norm(vmin, vmax, stretch="asinh"):
    """Normalisation for M0. Linear wastes the ramp on the bright ring.

    Disk emission is centrally concentrated and spans orders of magnitude, so under a
    linear scale the peak saturates a small area and the extended outer disk: the part a
    denoiser either recovers or invents: sits in the bottom few percent of the colormap
    and reads as black. `asinh` is the standard radio/mm choice: near-linear through zero,
    so noise around the background is not exaggerated the way `log` exaggerates it, and
    compressive at the bright end.
    """
    if stretch != "asinh":
        return None
    try:
        from astropy.visualization import AsinhStretch, ImageNormalize
    except ImportError:                      # astropy always present on Kaggle; be safe
        return None
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=0.15), clip=False)


def plot_channel_triptych(rows, save_path=None, title="", model_label="Denoised",
                          cmap="inferno", min_range=0.03):
    """
    dirty | denoised | clean, one row per validation channel, black background.

    `rows`: list of ``(dirty, denoised, clean, label)``, each image ``(H, W)``.

    Used by both notebook 05 (U-Net) and notebook 06 (DDPM) for their validation-channel
    figures, so the two are visually comparable rather than each carrying its own
    normalisation.

    **The bug this exists to fix.** A validation channel can be near-empty: little or no
    line signal, which happens on the low-SNR end of the sampling Gaussian. Its clean
    channel is then nearly *constant*, so ``vmin`` and ``vmax`` (taken from the clean
    channel's low percentile and max, matching `plot_moment_comparison`) collapse onto each
    other. `imshow` then normalises every pixel in the row to ~0.5, and inferno(0.5) is
    ``(0.74, 0.22, 0.33)``: a solid magenta-pink panel, not the black background an empty
    channel should show. It hit both notebooks: the U-Net figure's row 1 (denoised) and row
    3 (all three panels), and the DDPM figure's row 3, all show the exact same fill colour
    for the exact same reason.

    The guard: when ``hi - lo`` falls below `min_range`, extend ``hi`` upward from the same
    ``lo`` instead of leaving the window degenerate. Extending only upward, rather than
    widening around the floor, matters: `lo` is what the asinh stretch anchors to black, and
    a signal-bearing channel already keeps it there (background floor -> black, peak ->
    bright). Centring the widened window on the floor instead would put the floor at the
    *middle* of the colour scale, so an empty channel would render as a flat mid-tone (still
    wrong, just a different wrong colour) rather than the black an empty region should read
    as. The added span comes from the *dirty* channel's own 1st-99th percentile spread, not
    an arbitrary constant: a window narrower than the real noise amplitude trades the
    magenta wash-out for a different failure, clipping every pixel to the colormap's two
    endpoints and rendering salt-and-pepper speckle instead of real noise texture.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(rows), 3, figsize=(10, 3.4 * len(rows)), squeeze=False)
    for col, label in enumerate(("Dirty", model_label, "Clean GT")):
        ax[0, col].set_title(label, fontweight="bold")
    for r, (dirty, denoised, clean, row_label) in enumerate(rows):
        lo = float(np.nanpercentile(clean, 1))
        hi = float(np.nanmax(clean))
        if hi - lo < min_range:
            dspread = float(np.nanpercentile(dirty, 99) - np.nanpercentile(dirty, 1))
            hi = lo + max(dspread, min_range)
        norm = _intensity_norm(lo, hi, "asinh") or dict(vmin=lo, vmax=hi)
        kw = {"norm": norm} if not isinstance(norm, dict) else norm
        for col, im in enumerate((dirty, denoised, clean)):
            a = ax[r, col]
            a.set_facecolor("black")           # backstop: NaNs/transparency never show as white
            a.imshow(im, cmap=cmap, interpolation="nearest", **kw)
            a.set_xticks([]); a.set_yticks([])
        # On the left, not above: row 0's label and the "Dirty" column title both want the
        # space over the first panel, and in 05 v20 they printed on top of each other.
        ax[r, 0].set_ylabel(row_label, fontsize=8, labelpad=6)
    fig.suptitle(title, fontweight="bold")
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_moment_comparison(clean, dirty, denoised, save_path=None, tag="",
                           frac: float = SIGNAL_FRAC, show_mask: bool = True,
                           stretch: str = "asinh", contours: bool = True):
    """
    3x3 moment-map figure: rows = dirty / denoised / clean truth, columns = M0/M1/M2.

    Three things make this readable where a plain ``imshow`` is not:

    * **Off-source pixels are blanked.** Outside the disk there is no line, so M1 and M2
      are fits to noise and take arbitrary values across the whole range. Left in, they
      set the colour limits and squeeze the disk itself into a few shades: the real
      reason these maps look like static. The mask is ``signal_mask`` on the CLEAN M0,
      the same one ``moment_improvement`` scores over, so the figure shows exactly the
      region the reported numbers describe.
    * **One scale per column, shared by all three rows.** Autoscaling each panel
      separately makes a denoised map that lost half its dynamic range look identical to
      the truth. Sharing the scale is what lets the eye do the comparison.
    * **Velocity centred.** ``RdBu_r`` is diverging; centring on the systemic velocity
      makes red/blue mean receding/approaching rather than "above/below the mean of
      whatever was in frame".

    Returns the Matplotlib figure, so a notebook can display it inline. Pass
    ``save_path`` to also write it.
    """
    import matplotlib.pyplot as plt          # no backend forced: notebooks display inline

    mask = signal_mask(np.asarray(clean[0]), frac)
    rows = [("Dirty (ALMA-like input)", dirty),
            ("Denoised (U-Net)", denoised),
            ("Clean (ground truth)", clean)]
    cmaps = ("inferno", "RdBu_r", "viridis")
    kinds = ("intensity", "velocity", "dispersion")

    fig, ax = plt.subplots(3, 3, figsize=(13.5, 12.5), constrained_layout=True)
    for col in range(3):
        maps = [np.asarray(r[1][col]) for r in rows]
        vmin, vmax = _limits(maps, mask, kinds[col])
        cmap = plt.get_cmap(cmaps[col]).copy()
        cmap.set_bad("#0d0d0d")              # blanked sky reads as background, not data
        norm = _intensity_norm(vmin, vmax, stretch) if col == 0 else None
        for row, (label, _) in enumerate(rows):
            m = np.where(mask, maps[row], np.nan) if col > 0 else maps[row]
            kw = ({"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax})
            im = ax[row, col].imshow(m, origin="lower", cmap=cmap,
                                     interpolation="bilinear", **kw)
            ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
            for sp in ax[row, col].spines.values():
                sp.set_color("#444444")
            if col == 0:
                if contours:
                    # Intensity contours at fixed fractions of the CLEAN peak, identical on
                    # all three rows. This is what makes over- and under-smoothing legible:
                    # the same contour sitting at a different radius is a structural error,
                    # which a colour difference alone is easy to talk yourself out of.
                    #
                    # Contour a lightly smoothed COPY: the displayed image is untouched.
                    # Tracing a level through per-pixel noise produces spaghetti that hides
                    # the shape it was drawn to show; smoothing first is the usual practice
                    # and the levels still come from the unsmoothed clean peak.
                    peak = float(np.nanmax(maps[2])) or 1.0
                    src = maps[row]
                    try:
                        from scipy.ndimage import gaussian_filter
                        src = gaussian_filter(np.nan_to_num(src), sigma=2.0)
                    except ImportError:
                        pass
                    ax[row, col].contour(src, levels=[l * peak for l in (0.1, 0.3, 0.6)],
                                         colors="white", linewidths=0.7, alpha=0.6)
                if show_mask:
                    # outline the scored region, where the full field is still shown
                    ax[row, col].contour(mask.astype(float), levels=[0.5],
                                         colors="cyan", linewidths=0.9, alpha=0.9)
            if col == 0:
                ax[row, col].set_ylabel(label, fontsize=11, fontweight="bold")
            if row == 0:
                ax[row, col].set_title(MOMENT_LABELS[col], fontsize=11)
        fig.colorbar(im, ax=ax[:, col], fraction=0.046, pad=0.02, location="bottom")

    n_px = int(mask.sum())
    bits = [f"M1/M2 over the scored region only ({n_px:,} px, >{frac:.0%} of clean M0 peak)",
            "one colour scale per column"]
    if stretch == "asinh":
        bits.append("M0 on an asinh stretch")
    if contours:
        bits.append("contours at 10/30/60% of clean peak")
    fig.suptitle(f"Moment maps  {tag}".strip() + "\n" + "; ".join(bits),
                 fontweight="bold", fontsize=13)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    return fig
