"""alma_unified.line_free: the case that broke the old summed-spectrum sigma clip (a line over ~40 % of the band with a broad, flat-topped integrated
profile, plus a weak residual continuum ramp) must give exactly the planes outside the line, grown by the margin; pure noise gives all planes.
Run: PYTHONPATH=. python3 tests/test_line_free.py"""
import os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import alma_unified as au

rng = np.random.default_rng(1)
C, n = 301, 96
noise = ndimage.gaussian_filter(rng.standard_normal((C, n, n)).astype(np.float32), (0, 2.5, 2.5))
noise /= noise.std()
cube = noise.copy()
yy, xx = np.indices((n, n)); r = np.hypot(yy - n / 2, xx - n / 2)
for c in range(90, 211):                                   # an iso-velocity strip per channel, moving across a disc of radius 30 px
    x0 = n / 2 + 25 * (c - 150) / 60
    cube[c] += 8.0 * np.exp(-0.5 * ((xx - x0) / 3.0) ** 2) * (r < 30)
cube += np.linspace(0, 0.3, C)[:, None, None]             # a slow residual-continuum ramp
lf, _ = au.line_free(cube, beam_px=6.0)
line = np.setdiff1d(np.arange(C), lf)
assert line.min() <= 90 and line.max() >= 210, (line.min(), line.max())
assert line.min() >= 80 and line.max() <= 220, (line.min(), line.max())      # only the 5-channel margin (plus edge detections) beyond the line
lf0, _ = au.line_free(noise, beam_px=6.0)
assert len(lf0) == C, len(lf0)
print(f"line_free ok: line planes {line.min()}-{line.max()} (true 90-210), {len(lf)} line-free; noise-only cube: all {len(lf0)} line-free")
