"""The no-wiggle twin of an analytic GI disc (render_cube perturb=False) has the pure rotation field; the GI disc does not. Run: PYTHONPATH=. python3 tests/test_analytic_twin.py"""
import numpy as np
from src.data import analytic_gi as ag

d = ag.AnalyticGI(ms=1.0, md=0.48, p=-1.0, m=4, beta=4.0, pitch_deg=15, incl_deg=26.0)
c = (np.arange(64) - 32) * 6.0
s0, s1 = np.meshgrid(c, c, indexing="ij")
v_rot = d.v_los(s0, s1, perturb=False)
_, v_twin = ag.render_cube(d, n=64, au_per_px=6.0, n_chan=21, dv=0.3, perturb=False)
_, v_gi = ag.render_cube(d, n=64, au_per_px=6.0, n_chan=21, dv=0.3)
assert np.allclose(v_twin, v_rot)
assert np.abs(v_gi - v_rot).max() > 0.01, np.abs(v_gi - v_rot).max()
print("analytic twin ok: max |v_GI - v_rot| =", round(float(np.abs(v_gi - v_rot).max()), 3), "km/s")
