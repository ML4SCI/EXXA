"""
Analytic GI disk: (1) with no disc mass the rotation curve is Keplerian, (2) a disc mass speeds it up,
(3) the quadratic line centre of a rendered, UNBEAMED cube recovers the exact v_los to within a few m/s
(so the estimator is not what limits the scorecard), (4) the spiral perturbation is non-axisymmetric.
Run: PYTHONPATH=. python3 tests/test_analytic_gi.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import analytic_gi as ag

# 1/2: Keplerian limit and self-gravity speed-up
r = np.array([30.0, 100.0, 250.0])
vk = np.sqrt(ag.G * 1.0 / r)
v0 = ag.rotation_curve(r, md=1e-6, p=-1.0, rin=10, rout=290, ms=1.0)
assert np.allclose(v0, vk, rtol=2e-3), (v0, vk)
v1 = ag.rotation_curve(r, md=0.35, p=-1.0, rin=10, rout=290, ms=1.0)
assert np.all(v1 > vk), (v1, vk)
print("rotation curve: Keplerian at md->0 (max err %.1e), faster by %s at md=0.35" %
      (np.max(np.abs(v0 / vk - 1)), np.round(v1 / vk - 1, 3)))

# 3: line-centre truth recovered by the quadratic estimator on an unbeamed cube
d = ag.AnalyticGI(md=0.35, incl_deg=30.0, n_radii=120)
n = 121
cube, vt = ag.render_cube(d, n=n, au_per_px=4.0, n_chan=161, dv=0.05, line_sigma=0.25, beam_px=(0.01, 0.01, 0.0))
vel = (np.arange(161) - 80) * 0.05
C = cube.shape[0]
idx = np.clip(np.argmax(cube, 0), 1, C - 2)
yy, xx = np.indices(idx.shape)
ym, y0, yp = cube[idx - 1, yy, xx], cube[idx, yy, xx], cube[idx + 1, yy, xx]
off = np.clip(0.5 * (ym - yp) / (ym - 2 * y0 + yp), -0.5, 0.5)
vq = vel[idx] + off * 0.05
c = (np.arange(n) - n // 2) * 4.0
s0, s1 = np.meshgrid(c, c, indexing="ij")
r_dep = np.hypot(s0 / np.cos(d.incl), s1)
ok = (r_dep > 30) & (r_dep < 250) & (cube.max(0) > 0.05 * cube.max()) & (np.abs(vt) < 3.5)
err = np.abs(vq - vt)[ok]
assert np.median(err) < 0.005 and np.percentile(err, 95) < 0.02, (np.median(err), np.percentile(err, 95))
print("quadratic M1 vs exact v_los (unbeamed): median %.1f m/s, 95%% %.1f m/s over %d px" %
      (np.median(err) * 1e3, np.percentile(err, 95) * 1e3, ok.sum()))

# 4: perturbation present and non-axisymmetric
res = d.v_los(s0, s1) - d.v_los(s0, s1, perturb=False)
assert np.std(res[ok]) > 0.05, np.std(res[ok])
phi = np.arctan2(s0 / np.cos(d.incl), s1)
ring = ok & (np.abs(r_dep - 100) < 10)
assert np.std(res[ring]) > 0.5 * np.std(res[ok]) * 0.3
print("spiral perturbation: rms %.2f km/s inside the mask" % np.std(res[ok]))
print("PASS")
