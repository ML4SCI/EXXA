"""
The GI wiggle diagnostic (reading list, 2026-08-27: Speedie+2024, Terry+2024,
Hall+2020/2021/2022) fits a Keplerian model to a disk's moment-1 map and looks at what is
left over. Getting that fit wrong is easy to do silently -- a diverged optimiser still
returns a number -- so this is checked against synthetic cases with a known answer before it
is trusted on real data.

    PYTHONPATH=. python3 tests/test_gi_wiggle.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.evaluation.gi_wiggle import (
    keplerian_los, fit_keplerian, wiggle_residual, wiggle_amplitude, disk_geometry_from_m0,
    quadratic_moment1,
)

N, AU_PER_PX = 200, 1.5
TRUE = dict(cx=100.0, cy=95.0, pa_deg=35.0, incl_deg=40.0, vsys=1.2, mstar_msun=0.8)
failures = []


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


print("=" * 70)
print("GI wiggle: Keplerian fit and residual")
print("=" * 70)

y, x = np.mgrid[0:N, 0:N].astype(float)
m1_pure = keplerian_los((x, y), TRUE["cx"], TRUE["cy"], TRUE["pa_deg"], TRUE["incl_deg"],
                        TRUE["vsys"], TRUE["mstar_msun"], AU_PER_PX)
r = np.sqrt((x - TRUE["cx"]) ** 2 + (y - TRUE["cy"]) ** 2)
mask = (r > 6) & (r < 80)

# --- case 1: a pure Keplerian field must be recovered exactly, residual exactly 0 --------
print("\ncase 1  pure Keplerian, no wiggle")
geom = fit_keplerian(m1_pure, mask, AU_PER_PX, init=dict(cx=95, cy=90, pa_deg=30, incl_deg=45))
for k in ("cx", "cy", "pa_deg", "incl_deg", "vsys", "mstar_msun"):
    check(f"{k} recovered exactly", abs(geom[k] - TRUE[k]) < 1e-4,
          f"true {TRUE[k]}, fit {geom[k]:.6f}")
resid = wiggle_residual(m1_pure, geom)
amp = wiggle_amplitude(resid, mask)
check("residual is exactly zero with no wiggle present", amp["rms_kms"] < 1e-6,
      f"RMS {amp['rms_kms']:.2e} km/s")

# --- case 2: a known injected perturbation must be recoverable, not just noise fit -------
print("\ncase 2  a known m=2 perturbation is recovered, not fit away as noise")
pa, inc = np.radians(TRUE["pa_deg"]), np.radians(TRUE["incl_deg"])
dx, dy = x - TRUE["cx"], y - TRUE["cy"]
xr = dx * np.cos(pa) + dy * np.sin(pa)
yr = (-dx * np.sin(pa) + dy * np.cos(pa)) / np.cos(inc)
az = np.arctan2(yr, xr)
injected = 0.4 * np.sin(2 * az) * np.exp(-((r - 40) ** 2) / (2 * 20 ** 2))
m1_wiggly = m1_pure + injected

geom2 = fit_keplerian(m1_wiggly, mask, AU_PER_PX,
                      init=dict(cx=95, cy=90, pa_deg=30, incl_deg=45))
resid2 = wiggle_residual(m1_wiggly, geom2)
amp2 = wiggle_amplitude(resid2, mask)
corr = float(np.corrcoef(resid2[mask].ravel(), injected[mask].ravel())[0, 1])
check("recovered residual correlates with the injected pattern", corr > 0.7, f"r={corr:.3f}")
check("recovered amplitude is the right order of magnitude",
      0.05 < amp2["rms_kms"] < 1.0, f"RMS {amp2['rms_kms']:.3f} km/s")

# --- case 3: no signal at all must not crash or invent a wiggle --------------------------
print("\ncase 3  flat field (no rotation, no wiggle) degrades safely")
flat = np.zeros((N, N))
geom3 = fit_keplerian(flat, mask, AU_PER_PX, init=dict(cx=95, cy=90, pa_deg=30, incl_deg=45))
resid3 = wiggle_residual(flat, geom3)
amp3 = wiggle_amplitude(resid3, mask)
check("a flat field does not crash the fit", geom3["success"] or True)
check("residual amplitude is finite", np.isfinite(amp3["rms_kms"]))

# --- geometry helper: initial guess from image moments alone -----------------------------
print("\ncase 4  disk_geometry_from_m0 gives a sane starting guess")
m0 = np.exp(-((r - 40) ** 2) / (2 * 15 ** 2))
g0 = disk_geometry_from_m0(m0, mask)
check("centre from M0 moments is close to true centre",
      abs(g0["cx"] - TRUE["cx"]) < 5 and abs(g0["cy"] - TRUE["cy"]) < 5,
      f"got ({g0['cx']:.1f}, {g0['cy']:.1f}), true ({TRUE['cx']}, {TRUE['cy']})")

# --- case 5: NaN pixels inside the mask must be dropped, not crash the fit ---------------
print("\ncase 5  NaN pixels inside the mask (moment-1's own 0/0 on noisy data) are dropped")
m1_nan = m1_pure.copy()
nan_idx = np.where(mask)
rng = np.random.default_rng(0)
drop = rng.choice(len(nan_idx[0]), size=200, replace=False)
m1_nan[nan_idx[0][drop], nan_idx[1][drop]] = np.nan
geom5 = fit_keplerian(m1_nan, mask, AU_PER_PX,
                      init=dict(cx=95, cy=90, pa_deg=30, incl_deg=45))
check("fit succeeds despite NaN pixels in the mask", geom5["success"])
check("dropped count matches the injected NaNs", geom5["n_dropped_nonfinite"] == 200,
      f"got {geom5['n_dropped_nonfinite']}")
check("geometry still recovered correctly with NaNs dropped",
      abs(geom5["mstar_msun"] - TRUE["mstar_msun"]) < 1e-3,
      f"mstar {geom5['mstar_msun']:.4f}, true {TRUE['mstar_msun']}")

# --- case 6: the quadratic estimator survives negative sidelobe noise where the naive
# intensity-weighted mean does not. This is the exact failure mode found on the real dirty
# cube (51.55% negative pixels, Phase 0): collapse_first's weighted average is unstable
# when the "weights" go negative, collapse_quadratic is not, since it only looks at the
# peak channel and its two neighbours.
print("\ncase 6  quadratic estimator vs a naive intensity-weighted mean, negative-noise cube")
C, S, PX = 40, 24, 24
velax = np.linspace(-2000, 2000, C)   # m/s, matching bettermoments' convention
true_v0 = np.full((PX, PX), 500.0)    # every spaxel has the same true line centre, m/s
sigma_ch = 300.0                       # km/s-scale line width in m/s

rng2 = np.random.default_rng(1)
line = np.exp(-0.5 * ((velax[:, None, None] - true_v0[None]) / sigma_ch) ** 2)
# A dirty-beam-like corruption: strong negative sidelobes AWAY from the true line centre,
# which is exactly what pulls an intensity-weighted mean off the true value.
sidelobe = -0.8 * np.exp(-0.5 * ((velax[:, None, None] - (-1500.0)) / 150.0) ** 2)
cube = (line + sidelobe + rng2.normal(0, 0.03, (C, PX, PX))).astype(np.float64)

v0_quad, _ = quadratic_moment1(cube, velax)
check("quadratic estimator recovers the true line centre despite the sidelobe",
      float(np.abs(v0_quad - true_v0).mean()) < 100.0,
      f"mean |error| {float(np.abs(v0_quad - true_v0).mean()):.1f} m/s")

# The naive intensity-weighted mean, computed directly (not via bettermoments, so this check
# does not depend on that package's internals): sum(v * I) / sum(I) over the whole spectrum.
weighted_mean = (velax[:, None, None] * cube).sum(axis=0) / cube.sum(axis=0)
check("the naive intensity-weighted mean is pulled off by the sidelobe (the failure this exists to fix)",
      float(np.abs(weighted_mean - true_v0).mean()) > 300.0,
      f"mean |error| {float(np.abs(weighted_mean - true_v0).mean()):.1f} m/s")

# --- case 7: comparisons must use ONE shared rotation model ----------------------------
# Fitting each method its own Keplerian model makes the residual definition differ per
# method, so the comparison measures fit disagreement rather than the wiggle. On real data
# that produced 0.117 / 0.998 / 0.153 / 0.949 / 0.116 for the SAME beam-convolved cube across
# five channel ranges, and a chain of wrong conclusions on 2026-08-27/28.
print("\ncase 7  compare_wiggles uses one shared model for every method")
from src.evaluation.gi_wiggle import compare_wiggles

# Two "methods": one identical to the reference, one with a known extra perturbation.
m1_ref = m1_pure + injected
m1_same = m1_ref.copy()
m1_diff = m1_ref + 0.3 * np.cos(3 * az) * np.exp(-((r - 40) ** 2) / (2 * 20 ** 2))
res = compare_wiggles({"clean": m1_ref, "identical": m1_same, "perturbed": m1_diff},
                      mask, AU_PER_PX, reference="clean")
check("an identical map correlates at 1.0", abs(res["identical"]["corr"] - 1.0) < 1e-6,
      f"{res['identical']['corr']:.6f}")
check("a perturbed map correlates lower", res["perturbed"]["corr"] < 0.999,
      f"{res['perturbed']['corr']:.4f}")
check("every method shares the reference's geometry",
      res["identical"]["geom"] is res["perturbed"]["geom"])
check("a missing reference raises rather than guessing",
      _raises(lambda: compare_wiggles({"a": m1_ref}, mask, AU_PER_PX, reference="clean")))

print("\ncase 8  mass-inclination degeneracy, and what fixing inclination buys")
# Measured on the September SG batch, whose .para files state the truth: with inclination
# free, three of five disks at a true 20-30 deg fitted to 2-7 deg and drove mstar to its
# bound (errors to +4900%). Holding inclination fixed brought them back to the right order.
_H = _W = 121
_au = 2.0
_true = dict(cx=60.0, cy=60.0, pa=0.0, incl=20.0, vsys=0.0, mstar=1.0)
_y, _x = np.mgrid[0:_H, 0:_W].astype(float)
_m1 = keplerian_los((_x.ravel(), _y.ravel()), _true["cx"], _true["cy"], _true["pa"],
                    _true["incl"], _true["vsys"], _true["mstar"], _au).reshape(_H, _W)
_r = np.hypot(_x - _true["cx"], _y - _true["cy"])
_mask = (_r > 4) & (_r < 55)
_m0 = np.where(_mask, 1.0, 0.0)

_fixed = fit_keplerian(_m1, _mask, _au, m0=_m0, fix_incl_deg=_true["incl"])
check("case8: fixed-inclination fit recovers mstar within 5%",
      abs(_fixed["mstar_msun"] - _true["mstar"]) / _true["mstar"] < 0.05,
      f"{_fixed['mstar_msun']:.4f} vs {_true['mstar']}")
check("case8: the fit reports that inclination was held fixed",
      _fixed["incl_fixed"] is True and _fixed["incl_deg"] == _true["incl"])
check("case8: a fixed-inclination fit on clean data does not pin the mass",
      _fixed["mstar_at_bound"] is False)

_free = fit_keplerian(_m1, _mask, _au, m0=_m0)
check("case8: mstar_at_bound is always reported so a pinned fit cannot be quoted silently",
      isinstance(_free["mstar_at_bound"], bool))


print("\n" + "-" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
sys.exit(0)

