#!/usr/bin/env python
"""
Smoke test for the M2 (dispersion) noise-clip fix.

Unmasked, bettermoments' collapse_second computes an intensity-weighted variance over
the WHOLE velocity axis with no threshold of its own. Fed pure noise, that estimator does
not blow up to infinity -- it converges to roughly the RMS width of the velocity axis
itself (~577 m/s here, for a +/-1000 m/s axis), because a zero-mean noise weighting still
samples the full axis roughly evenly. That value is uniform across the whole background,
not a few outlier pixels, and it sits far ABOVE a real disk's actual line width (150 m/s
injected here) -- which is what turned architecture_moment_maps.png's entire M2 column
(everything except the noiseless clean truth) into a solid saturated block: real disk
structure and pure background noise both compare in the hundreds of m/s, so a colour scale
that has to show both loses the real structure entirely.
"""

import numpy as np

from src.evaluation.moment_maps import generate_moment_maps

print("=" * 60)
print("Moment Map Clip Smoke Test")
print("=" * 60)

H, C = 32, 40
rng = np.random.default_rng(0)
velax = np.linspace(-1000, 1000, C).astype(np.float32)
axis_rms = float(np.std(velax))

yy, xx = np.mgrid[0:H, 0:H]
r2 = (yy - H / 2) ** 2 + (xx - H / 2) ** 2
disk = 20.0 * np.exp(-r2 / (2 * 6.0 ** 2))       # 20-sigma peak: well above the noise
line_center = velax[C // 2]
line_width = 150.0

data = np.zeros((C, H, H), dtype=np.float32)
for i, v in enumerate(velax):
    data[i] = disk * np.exp(-((v - line_center) ** 2) / (2 * line_width ** 2))
data += rng.normal(0, 1.0, data.shape).astype(np.float32)

# [1] unclipped: PURE noise (no disk) must read a dispersion comparable to the velocity
# axis itself -- this is the bug. A metric that reports ~axis-width for empty sky is not
# measuring dispersion, it is measuring how wide the observing band is.
noise_only = rng.normal(0, 1.0, data.shape).astype(np.float32)
_, _, m2_noise = generate_moment_maps(None, data_velax=(noise_only, velax), clip_sigma=0)
assert 0.5 * axis_rms < np.nanmedian(m2_noise) < 1.5 * axis_rms, \
    f"expected pure-noise M2 near the axis RMS ({axis_rms:.0f}), got {np.nanmedian(m2_noise):.0f}"
print(f"[1] unclipped pure noise reads M2 {np.nanmedian(m2_noise):.0f} "
      f"~= axis RMS {axis_rms:.0f} -- indistinguishable from a real wide line")

# [2] same field: unclipped, the disk's OWN M2 is dragged far off its true injected width
# by noise in its wings (channels where the line is faint), toward the background's
# axis-RMS value -- a ~2x inflation despite this pixel being a 20-sigma detection.
cy, cx = H // 2, H // 2
_, _, m2_raw = generate_moment_maps(None, data_velax=(data, velax), clip_sigma=0)
disk_raw = float(np.mean(m2_raw[cy - 2:cy + 2, cx - 2:cx + 2]))
bg_raw = float(np.mean(m2_raw[:4, :4]))
assert disk_raw > 1.5 * line_width, \
    f"expected unclipped disk M2 ({disk_raw:.0f}) inflated well past the true width ({line_width:.0f})"
print(f"[2] unclipped disk-centre M2 {disk_raw:.0f} vs true width {line_width:.0f} "
      f"(background corner reads {bg_raw:.0f}) -- noise wings inflate even a strong detection")

# [3] clipped: background must drop out (masked to NaN, not a false dispersion reading),
# and the disk must recover close to its TRUE injected width.
_, _, m2_clip = generate_moment_maps(None, data_velax=(data, velax), clip_sigma=3.0)
bg_clip = m2_clip[:4, :4]
assert np.isnan(bg_clip).mean() > 0.5, \
    f"clip should mask most of the noise-only background, only {np.isnan(bg_clip).mean():.0%} masked"
disk_clip = float(np.nanmean(m2_clip[cy - 2:cy + 2, cx - 2:cx + 2]))
assert abs(disk_clip - line_width) < 60, \
    f"clipped disk M2 ({disk_clip:.0f}) should be near the injected width ({line_width:.0f})"
print(f"[3] clipped: background {np.isnan(bg_clip).mean():.0%} masked, "
      f"disk-centre M2 {disk_clip:.0f} ~= injected width {line_width:.0f}")

print("\n" + "=" * 60)
print("All moment-map clip tests PASSED")
print("=" * 60)


# ---------------------------------------------------------------------------
# [4-6] Strip-wise collapse: the fix for the Kaggle host-RAM kills.
#
# bettermoments is allocation-hungry out of all proportion to its input -- on a
# 201x600x600 float32 cube (0.27 GiB) collapse_second alone peaks at 12x the cube. Three
# collapses per cube is what exhausted the host, not the cubes being held around it.
# All three collapses are per-pixel along the spectral axis, so strips are exact.
import tracemalloc

Cb, Hb, Wb = 60, 200, 200          # small enough to run in a test, same shape of problem
big = rng.normal(0, 1.0, (Cb, Hb, Wb)).astype(np.float32)
yyb, xxb = np.mgrid[0:Hb, 0:Wb]
vel_b = np.linspace(-1000, 1000, Cb).astype(np.float32)
big += (20 * np.exp(-((yyb - Hb / 2) ** 2 + (xxb - Wb / 2) ** 2) / (2 * 25.0 ** 2))
        )[None] * np.exp(-(vel_b ** 2) / (2 * 200.0 ** 2))[:, None, None]
one_cube = Cb * Hb * Wb * 4


def _peak(tb):
    tracemalloc.start()
    m = generate_moment_maps(None, data_velax=(big, vel_b), tile_bytes=tb)
    _, pk = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return m, pk


full, pk_full = _peak(0)
strip, pk_strip = _peak(1 << 18)

# [4] the whole point: a bounded working set
assert pk_strip < pk_full / 3, (pk_strip, pk_full)
print(f"[4] strips bound the working set: {pk_full / one_cube:.1f}x cube -> "
      f"{pk_strip / one_cube:.1f}x  ({pk_full / pk_strip:.1f}x less)")

# [5] M0 must be bit-identical, and NaNs must land in exactly the same places --
#     a different NaN pattern would mean strips changed which pixels are reportable
assert np.array_equal(full[0], strip[0], equal_nan=True), "M0 changed under strips"
for nm, a, b in zip(("M0", "M1", "M2"), full, strip):
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), f"{nm} NaN pattern changed"
print("[5] M0 bit-identical; NaN pattern unchanged for M0/M1/M2")

# [6] on the pixels anything is actually reported from -- the bright ones -- M1 and M2
#     must agree to floating-point noise. Near-zero background is excluded on purpose:
#     there the denominator is ~0 and the value is meaningless, which is the very
#     instability clip_sigma exists to suppress.
bright = np.isfinite(full[0]) & (np.abs(full[0]) > np.nanpercentile(np.abs(full[0]), 90))
for nm, a, b in zip(("M1", "M2"), full[1:], strip[1:]):
    ok = bright & np.isfinite(a) & np.isfinite(b)
    rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(a[ok]), 1e-12)
    assert rel.max() < 1e-4, (nm, rel.max())
    print(f"[6] {nm} over the brightest 10%: max relative difference {rel.max():.2e}")

print("\n" + "=" * 60)
print("Strip-collapse tests PASSED")
print("=" * 60)


# ---------------------------------------------------------------------------
# [7-9] Signal-masked scoring.
#
# The improvement metric averaged over every finite pixel, so empty sky dominated it.
# Dispersion in a pixel with no line is not a quantity anyone reports, and M2 suffered
# most because it is a ratio whose denominator vanishes exactly there.
from src.evaluation.moment_maps import moment_improvement, signal_mask

Cs, Hs, Ws = 80, 160, 160
vs = np.linspace(-1000, 1000, Cs).astype(np.float32)
ys, xs = np.mgrid[0:Hs, 0:Ws]
rr = np.sqrt((ys - Hs / 2) ** 2 + (xs - Ws / 2) ** 2)
src_ = 20 * np.exp(-rr ** 2 / (2 * 20.0 ** 2))          # compact: most of the map is sky
vf = 300 * (xs - Ws / 2) / (Ws / 2)
wd = 120 + 80 * np.exp(-rr ** 2 / (2 * 15.0 ** 2))
clean_c = np.zeros((Cs, Hs, Ws), np.float32)
for i, v in enumerate(vs):
    clean_c[i] = src_ * np.exp(-((v - vf) ** 2) / (2 * wd ** 2))
dirty_c = clean_c + rng.normal(0, 1.0, clean_c.shape).astype(np.float32)

cm = generate_moment_maps(None, data_velax=(clean_c, vs))
dm = generate_moment_maps(None, data_velax=(dirty_c, vs))
qual = {"good": 0.2, "ok": 0.45, "poor": 0.8}
pred = {k: generate_moment_maps(
            None, data_velax=(clean_c + rng.normal(0, s, clean_c.shape).astype(np.float32), vs))
        for k, s in qual.items()}

# [7] the mask must select the source and exclude the sky, not most of the map
msk = signal_mask(cm[0])
assert 0.01 < msk.mean() < 0.5, f"mask covers {msk.mean():.1%} of the map"
print(f"[7] signal mask keeps {msk.mean():.1%} of pixels (compact source, mostly sky)")

# [8] masking must not change WHICH method wins -- otherwise it moves the goalposts
res = {k: moment_improvement(cm, dm, pred[k]) for k in qual}
for nm in ("M0", "M1", "M2"):
    masked = sorted(res, key=lambda k: res[k][nm], reverse=True)
    unmask = sorted(res, key=lambda k: res[k][nm + "_all"], reverse=True)
    assert masked == unmask == ["good", "ok", "poor"], (nm, masked, unmask)
print("[8] ranking identical masked vs unmasked for M0/M1/M2: good > ok > poor")

# [9] and it should lift M2, which the sky was dragging down
assert res["good"]["M2"] > res["good"]["M2_all"], (res["good"]["M2"], res["good"]["M2_all"])
print(f"[9] M2 for the good model: {res['good']['M2_all']:.1f}% over all pixels "
      f"-> {res['good']['M2']:.1f}% over signal")

print("\n" + "=" * 60)
print("Signal-mask scoring tests PASSED")
print("=" * 60)
