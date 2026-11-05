#!/usr/bin/env python
"""
Tests for the moment-map comparison figure.

The failure this guards against is silent and cosmetic-looking but not cosmetic: if the
off-source pixels are allowed into the colour limits, M1 and M2 are fits to noise out
there and span tens of km/s, so the disk gets a few percent of the ramp and the figure
reads as static. Nothing errors -- it just stops showing the result.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np

from src.evaluation.moment_maps import (SIGNAL_FRAC, _limits, plot_moment_comparison,
                                        signal_mask)

print("=" * 60)
print("Moment Figure Tests")
print("=" * 60)

# A ring disk with solid-body-ish rotation, systemic velocity deliberately NOT zero so the
# centring logic is actually exercised.
N = 128
SYSTEMIC = 4.2
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:N, 0:N]
r = np.hypot(yy - N / 2, xx - N / 2)
th = np.arctan2(yy - N / 2, xx - N / 2)

m0 = np.exp(-((r - 28) ** 2) / (2 * 9.0 ** 2))
on_source = m0 > SIGNAL_FRAC * m0.max()
m1 = np.where(on_source, SYSTEMIC + 3.0 * np.cos(th) * (r / 30), rng.uniform(-40, 40, (N, N)))
m2 = np.where(on_source, 0.4 + 0.25 * np.exp(-r / 40), rng.uniform(0, 25, (N, N)))
clean = (m0, m1, m2)
dirty = tuple(a + rng.normal(0, s, (N, N)) for a, s in zip(clean, (0.08, 1.2, 0.5)))
den = tuple(a + rng.normal(0, s, (N, N)) for a, s in zip(clean, (0.02, 0.4, 0.15)))

mask = signal_mask(clean[0])


# ------------------------------------------------------------------ [1] masking matters
# This is the whole point: the masked scale must be dramatically tighter than the range
# the raw data spans, or the disk is being squeezed into a sliver of the colormap.
for name, kind, i, min_gain in (("M1", "velocity", 1, 3.0), ("M2", "dispersion", 2, 3.0)):
    maps = [clean[i], dirty[i], den[i]]
    lo, hi = _limits(maps, mask, kind)
    full = np.concatenate([m.ravel() for m in maps])
    gain = (full.max() - full.min()) / (hi - lo)
    assert gain > min_gain, f"{name}: masking only tightened the scale {gain:.1f}x"
    print(f"[1] {name}: masked span {hi - lo:6.2f} vs raw {full.max() - full.min():7.2f} "
          f"-> {gain:.1f}x more of the colormap spent on the disk")


# ------------------------------------------------------------------ [2] velocity centred
# RdBu_r is diverging; its white point must sit at the systemic velocity, otherwise
# red/blue does not mean receding/approaching.
lo, hi = _limits([clean[1], dirty[1], den[1]], mask, "velocity")
centre = (lo + hi) / 2
assert abs(centre - SYSTEMIC) < 0.5, f"velocity scale centred at {centre:.2f}, not {SYSTEMIC}"
print(f"[2] velocity scale centred at {centre:.2f} km/s (systemic {SYSTEMIC}), symmetric")


# ------------------------------------------------------------------ [3] outliers clipped
# One hot pixel must not set the ceiling.
spiked = clean[0].copy()
spiked[0, 0] = 1e6
lo_s, hi_s = _limits([spiked], mask, "intensity")
assert hi_s < 100, f"a single 1e6 pixel set the ceiling to {hi_s}"
print(f"[3] a 1e6 spike does not set the ceiling (vmax {hi_s:.2f}) -- percentiles, not max")


# ------------------------------------------------------------------ [4] figure builds
fig = plot_moment_comparison(clean, dirty, den, tag="unit test")
# 9 map panels + 3 colorbars
assert len(fig.axes) == 12, len(fig.axes)
images = [im for a in fig.axes for im in a.images]
assert len(images) == 9, len(images)
# every column shares one scale across all three rows, or the eye cannot compare them
for col in range(3):
    clims = {images[row * 3 + col].get_clim() for row in range(3)}
    assert len(clims) == 1, f"column {col} has {len(clims)} different colour scales"
print("[4] 3x3 panels + 3 colorbars; each column shares one scale across all rows")


# ------------------------------------------------------------------ [5] degenerate input
# An empty clean map must not raise -- it happens on a channel with no emission.
blank = (np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)))
fig2 = plot_moment_comparison(blank, blank, blank, tag="empty")
assert len(fig2.axes) == 12
print("[5] an all-zero (no-emission) cube renders instead of raising")

# ------------------------------------------------------------------ [6] channel triptych
# The bug this guards: a validation channel can be near-empty (low-SNR end of the sampling
# Gaussian). Its clean channel is then nearly constant, vmin/vmax collapse together, and
# imshow normalises everything to ~0.5 -- inferno(0.5) is a solid magenta-pink panel, not
# the black an empty channel should show. Hit both the U-Net and DDPM validation-channel
# figures in production, in different rows of each.
from src.evaluation.moment_maps import plot_channel_triptych

_N = 64
_yy, _xx = np.mgrid[0:_N, 0:_N]
_r = np.hypot(_yy - _N / 2, _xx - _N / 2)
_disk = np.exp(-((_r - 14) ** 2) / (2 * 5.0 ** 2)).astype(np.float32)
_dirty_sig = (_disk + rng.normal(0, 0.15, (_N, _N))).astype(np.float32)
_den_sig = (_disk + rng.normal(0, 0.02, (_N, _N))).astype(np.float32)

_floor = 0.301
_clean_empty = np.full((_N, _N), _floor, np.float32)          # a truly empty channel
_dirty_empty = (_clean_empty + rng.normal(0, 0.15, (_N, _N))).astype(np.float32)
_den_empty = (_clean_empty + rng.normal(0, 0.02, (_N, _N))).astype(np.float32)

_fig = plot_channel_triptych(
    [(_dirty_sig, _den_sig, _disk, "signal"),
     (_dirty_empty, _den_empty, _clean_empty, "empty")],
    title="test")

_magenta = np.array([0.7357, 0.2159, 0.3302])
_black = np.array([0.0015, 0.0005, 0.0139])
_panels = [im for a in _fig.axes for im in a.images]
assert len(_panels) == 6, len(_panels)
for _i, _im in enumerate(_panels):
    _rgba = _im.cmap(_im.norm(_im.get_array()))[..., :3]
    _frac_magenta = (np.abs(_rgba - _magenta).sum(axis=-1) < 0.15).mean()
    assert _frac_magenta < 0.5, f"panel {_i} collapsed to the inferno(0.5) magenta fill"
# the truly-empty clean panel (index 5: row 1, column 2) must render as solid black
_empty_clean_frac_black = (np.abs(_panels[5].cmap(_panels[5].norm(_panels[5].get_array()))[..., :3]
                                  - _black).sum(axis=-1) < 0.15).mean()
assert _empty_clean_frac_black > 0.99, f"empty clean channel not black: {_empty_clean_frac_black:.2f}"
print(f"[6] no panel collapses to magenta; a truly-empty clean channel renders "
      f"{_empty_clean_frac_black:.0%} black")


print("\n" + "=" * 60)
print("All moment figure tests PASSED")
print("=" * 60)
