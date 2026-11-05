"""pv_profile recovers a known radial wiggle on a synthetic Keplerian + wiggle M1 map along the minor axis (the rotation term vanishes there);
pv_scores gives ~1 for the clean map with small noise, a clearly lower value for a noisy copy, and ~0 for pure noise; the NaN mask is respected.
Run: PYTHONPATH=. python3 tests/test_pv_curve.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import pv_curve as pv

n, au = 241, 2.0
geom = {"cx": 120.0, "cy": 120.0, "pa_deg": 35.0, "incl_deg": 40.0, "vsys": 5.0}
pa, inc = np.radians(geom["pa_deg"]), np.radians(geom["incl_deg"])
yy, xx = np.indices((n, n), dtype=np.float64)
dx, dy = xx - geom["cx"], yy - geom["cy"]
xr = dx * np.cos(pa) + dy * np.sin(pa)
yr = (-dx * np.sin(pa) + dy * np.cos(pa)) / np.cos(inc)
R = np.hypot(xr, yr) * au
az = np.arctan2(yr, xr)
A, lam = 0.060, 80.0                                           # km/s, au: a 60 m/s wiggle
wig = A * np.sin(2 * np.pi * R / lam)
kepler = 3.0 * np.sqrt(100.0 / np.maximum(R, 20.0)) * np.sin(inc) * np.cos(az)   # vanishes on the minor axis
m1 = geom["vsys"] + kepler + wig

r, (hp, hm) = pv.pv_profile(m1, geom, au, 40.0, 200.0)
truth = A * 1000 * np.sin(2 * np.pi * r / lam)
for h in (hp, hm):
    assert np.nanmax(np.abs(h - truth)) < 8.0, np.nanmax(np.abs(h - truth))   # Kepler term ~0 on the minor axis; interpolation error only
assert pv.pv_scores({"clean": [hp, hm], "same": [hp, hm]})["same"]["pv_r"] > 0.999

rng = np.random.default_rng(0)
noisy = m1 + 0.04 * rng.standard_normal(m1.shape)
_, (np_, nm_) = pv.pv_profile(noisy, geom, au, 40.0, 200.0)
_, (wp, wm) = pv.pv_profile(noisy, geom, au, 40.0, 200.0, width_px=5)
sc = pv.pv_scores({"clean": [hp, hm], "noisy": [np_, nm_], "slit5": [wp, wm]})
assert sc["noisy"]["pv_r"] < 0.9, sc                            # noise hurts the 1 px slice
assert sc["slit5"]["pv_r"] > sc["noisy"]["pv_r"], sc            # the 5 px slit averages part of it away
noise_only = 0.04 * rng.standard_normal(m1.shape) + geom["vsys"]
_, (qp, qm) = pv.pv_profile(noise_only, geom, au, 40.0, 200.0)
assert abs(pv.pv_scores({"clean": [hp, hm], "n": [qp, qm]})["n"]["pv_r"]) < 0.3

mask = np.hypot(dx, dy) * au < 120.0                           # disc ends at 120 au: outside it the profile is NaN, not a padded 0
_, (mp, mm) = pv.pv_profile(m1, geom, au, 40.0, 200.0, mask=mask)
assert np.isnan(mp[-1]) and np.isfinite(mp[0]), (mp[0], mp[-1])
mag = pv.pv_scores({"clean": [hp, hm]})["clean"]["pv_mag_ms"]
assert 30.0 < mag < 45.0, mag                                  # a 60 m/s sinusoid has sd 42.4 m/s; the quadratic detrend over ~2 periods removes a little
print(f"pv magnitude {mag:.1f} m/s")
print(f"pv_curve ok: max error {np.nanmax(np.abs(hp - truth)):.1f} m/s; pv_r noisy {sc['noisy']['pv_r']:.2f}, slit5 {sc['slit5']['pv_r']:.2f}")
