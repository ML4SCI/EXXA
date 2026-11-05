"""fov_box finds a square field around an off-centre disc (inside the image, multiple of 8, at least min_px), returns None for pure noise, and write() moves the
WCS reference pixel by the crop offsets. Run: PYTHONPATH=. python3 tests/test_alma_fov.py"""
import os, sys, tempfile
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import alma_unified as au

rng = np.random.default_rng(0)
C, n, lf = 40, 512, np.r_[0:8, 32:40]


def disc_cube(cy, cx, rad):
    yy, xx = np.indices((n, n), dtype=np.float32)
    r = np.hypot(yy - cy, xx - cx)
    m0 = np.exp(-0.5 * (r / rad) ** 2) * (r < 2.5 * rad)
    prof = np.exp(-0.5 * ((np.arange(C) - 20) / 5.0) ** 2).astype(np.float32)
    return prof[:, None, None] * m0[None] * 3.0 + 0.5 * rng.standard_normal((C, n, n)).astype(np.float32)


# an off-centre disc: the box contains it, is square, inside the image, a multiple of 8
cube = disc_cube(300, 180, 30)
y0, y1, x0, x1 = au.fov_box(cube, lf, 6.0)
assert y1 - y0 == x1 - x0 and (y1 - y0) % 8 == 0 and 128 <= y1 - y0 < n, (y0, y1, x0, x1)
assert 0 <= y0 and y1 <= n and 0 <= x0 and x1 <= n
assert y0 <= 300 - 45 and y1 >= 300 + 45 and x0 <= 180 - 45 and x1 >= 180 + 45, (y0, y1, x0, x1)    # holds the disc out to 1.5 sigma-radii

# a tiny disc still gets min_px; a disc in the corner stays inside the image
y0, y1, x0, x1 = au.fov_box(disc_cube(256, 256, 6), lf, 6.0)
assert y1 - y0 == 128, (y0, y1)
y0, y1, x0, x1 = au.fov_box(disc_cube(20, 20, 20), lf, 6.0)
assert y0 == 0 and x0 == 0 and y1 <= n, (y0, y1, x0, x1)

# no disc: nothing to crop to
assert au.fov_box(0.5 * rng.standard_normal((C, n, n)).astype(np.float32), lf, 6.0) is None

# WCS: the reference pixel moves with the centre crop (o) plus the field crop; without the keys the legacy centre is kept
hdr = fits.Header(); hdr["CRPIX1"], hdr["CRPIX2"], hdr["CDELT1"], hdr["CDELT2"], hdr["CRVAL1"], hdr["CRVAL2"] = 513.0, 513.0, -1e-5, 1e-5, 10.0, 20.0
hdr["EXXAOX"], hdr["EXXAOY"] = 100 + 40, 100 + 70
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "x.fits"); au.write(p, np.zeros((3, 152, 152), np.float32), hdr)
    h = fits.getheader(p)
    assert (h["CRPIX1"], h["CRPIX2"]) == (513.0 - 140, 513.0 - 170), (h["CRPIX1"], h["CRPIX2"])
    del hdr["EXXAOX"]; au.write(p, np.zeros((3, 152, 152), np.float32), hdr)
    assert fits.getheader(p)["CRPIX1"] == 152 / 2 + 0.5
print("alma fov ok")

# continuum_check: subtract only a structure present on BOTH far sides of the line (a cube never continuum subtracted); not line wings, not noise
C2, n2 = 60, 96
lf2 = np.r_[0:20, 40:60]                                             # line in planes 20-39
yy, xx = np.indices((n2, n2), dtype=np.float32)
blob = np.exp(-0.5 * (np.hypot(yy - 48, xx - 48) / 6.0) ** 2).astype(np.float32)
noise = lambda: 1.0 * rng.standard_normal((C2, n2, n2)).astype(np.float32)
c = noise(); c[20:40] += 5 * blob                                     # line only, already subtracted: no continuum
assert au.continuum_check(c, lf2)["decision"] == "none", au.continuum_check(c, lf2)
c = noise(); c[20:40] += 5 * blob; c += 2.0 * blob[None]              # continuum never subtracted: the same blob in every plane
r = au.continuum_check(c, lf2)
assert r["decision"] == "subtract" and r["blue_peak_on_red_sigma"] > 3 and r["red_peak_on_blue_sigma"] > 3, r
assert abs(float(r["map"].max()) - 2.0) < 0.5, float(r["map"].max())  # the map removed is the continuum itself
c = noise(); c[20:40] += 5 * blob; c[17:20] += 3 * blob; c[40:43] += 3 * blob   # line wings leaking into the planes next to the line only
assert au.continuum_check(c, lf2)["decision"] == "none", au.continuum_check(c, lf2)
off = np.exp(-0.5 * (np.hypot(yy - 30, xx - 70) / 4.0) ** 2).astype(np.float32)
c = noise(); c[20:40] += 5 * blob; c[0:8] += 3 * blob; c[52:60] += 3 * off      # strong far-plane structure on both sides but in DIFFERENT places: not continuum
assert au.continuum_check(c, lf2)["decision"] == "none", au.continuum_check(c, lf2)
fp = sum(au.continuum_check(noise(), lf2)["decision"] == "subtract" for _ in range(10))   # pure noise never triggers it
assert fp == 0, fp
print("continuum_check ok")

# spec_up / spec_down: a linear ramp in velocity survives the round trip exactly in the interior; original channels sit at c * r; a constant stays constant
C3, r = 9, 4
ramp = (np.arange(C3, dtype=np.float32)[:, None, None] * np.ones((1, 3, 3), np.float32))
up = au.spec_up(ramp, r)
assert up.shape[0] == (C3 - 1) * r + 1 and np.allclose(up[::r, 0, 0], np.arange(C3)), up.shape
back = au.spec_down(up, r)
assert back.shape == ramp.shape and np.allclose(back[1:-1], ramp[1:-1], atol=1e-5), (back[:, 0, 0], ramp[:, 0, 0])
assert np.allclose(au.spec_down(au.spec_up(np.full((5, 2, 2), 3.0, np.float32), 3), 3), 3.0)
print("spec interp ok")

# synth_noise: the planes it draws have the source's variance, spatial spectrum and lag-1 channel correlation
from scipy import ndimage
n4, C4, rho_true = 64, 40, 0.5
src = np.empty((C4, n4, n4), np.float32); prev = None
for c in range(C4):
    e = ndimage.gaussian_filter(rng.standard_normal((n4, n4)), 1.5) * 4.0
    prev = e if prev is None else rho_true * prev + np.sqrt(1 - rho_true ** 2) * e
    src[c] = prev
lf4 = np.arange(C4)
syn, rho_hat = au.synth_noise(src, lf4, 60, seed=1)
assert abs(syn.std() / src.std() - 1) < 0.1, (syn.std(), src.std())
assert abs(rho_hat - rho_true) < 0.1, rho_hat
sl = lambda x: np.mean([np.corrcoef(x[i].ravel(), x[i + 1].ravel())[0, 1] for i in range(len(x) - 1)])
assert abs(sl(syn) - rho_true) < 0.12, sl(syn)
rp = lambda x: np.mean(np.abs(np.fft.rfft2(x - x.mean((1, 2), keepdims=True))) ** 2, axis=0)
ratio = rp(syn)[1:12, 1:12].mean() / rp(src)[1:12, 1:12].mean(); assert 0.7 < ratio < 1.4, ratio     # low-k power (where the spatial correlation lives)
print("synth_noise ok")

# run_scaled ("zoom in", --scale): the model gets the finer grid with the beam scaled to it, and a smooth cube survives the round trip
_seen = {}
def _fake_run(a, model, arm, d, w, dv, beam_px):
    _seen["n"], _seen["beam"] = d.shape[-1], beam_px
    return d
_orig_run, au.run_model = au.run_model, _fake_run
smooth = ndimage.gaussian_filter(rng.standard_normal((5, 120, 120)), (0, 4, 4)).astype(np.float32)
import argparse as _ap
z = au.run_scaled(_ap.Namespace(scale=1.6667), None, None, smooth, smooth, 0.1, 6.0)
assert _seen["n"] == 200 and abs(_seen["beam"] - 10.0) < 0.01, _seen
assert z.shape == smooth.shape and np.abs(z - smooth).max() < 0.1 * np.abs(smooth).max()
au.run_model = _orig_run
print("run_scaled ok")

# target_scale (--target-beam-px): beam to the target, never zoom out, cap on the grid
assert abs(au.target_scale(6.0, 512, 10.0, 1024) - 10 / 6) < 1e-9
assert au.target_scale(12.0, 512, 10.0, 1024) == 1.0
assert au.target_scale(3.0, 768, 10.0, 1024) == 1024 / 768
print("target_scale ok")

# beam_of: header keywords, or the median of a CASA BEAMS table (DSHARP HD 163296 has no BMAJ keyword)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "mb.fits")
    cols = [fits.Column(name="BMAJ", format="E", unit="arcsec", array=np.array([0.10, 0.12, 0.11], np.float32)),
            fits.Column(name="BMIN", format="E", unit="arcsec", array=np.array([0.08, 0.09, 0.07], np.float32))]
    bt = fits.BinTableHDU.from_columns(cols); bt.name = "BEAMS"
    fits.HDUList([fits.PrimaryHDU(np.zeros((3, 8, 8), np.float32)), bt]).writeto(p)
    bm = au.beam_of(p, fits.getheader(p))
    assert abs(bm[0] - 0.11) < 1e-6 and abs(bm[1] - 0.08) < 1e-6, bm
    h = fits.Header(); h["BMAJ"], h["BMIN"] = 0.15 / 3600, 0.1 / 3600
    assert np.allclose(au.beam_of(p, h), (0.15, 0.1))
print("beam_of ok")

# clean_noise_planes: a line-free plane with a compact source is dropped, pure-noise planes are kept
rng2 = np.random.default_rng(5)
nc = rng2.standard_normal((6, 128, 128)).astype(np.float32)
nc[2, 60:66, 60:66] += 6.0                                  # a compact 6-sigma-per-pixel knot (about 30 sigma beam-smoothed)
keep = au.clean_noise_planes(nc, np.arange(6), 4.0)
assert list(keep) == [0, 1, 3, 4, 5], keep
print("clean_noise_planes ok")

# compact_truth: bright centre, Keplerian velocity sign flips across the minor axis, finite band; roll_noise keeps the statistics; flux_by_radius of truth = 1
ct, cv = au.compact_truth(64, 4.0, 4.0, 0.1)
m0c = ct.sum(0)
cy, cx = np.unravel_index(np.argmax(m0c), m0c.shape)
assert abs(cy - 32) <= 3 and abs(cx - 32) <= 3, (cy, cx)
assert ct.shape[0] == len(cv) and cv[0] < 0 < cv[-1]
rn = np.random.default_rng(0).standard_normal((3, 64, 64)).astype(np.float32)
assert np.allclose(np.sort(au.roll_noise(rn, 3).ravel()), np.sort(rn.ravel())) and not np.allclose(au.roll_noise(rn, 3), rn) and au.roll_noise(rn, 0) is rn
fr = au.flux_by_radius(ct, ct, 4.0)
assert all(x is None or abs(x - 1) < 1e-6 for x in fr), fr
print("compact_truth / roll_noise / flux_by_radius ok")

# alma_s7.smooth_to: a uniform map in Jy/beam scales by the beam-area ratio (flux per area conserved)
import alma_s7
u = np.ones((2, 64, 64), np.float32)
su = alma_s7.smooth_to(u, (0.15, 0.15), (0.30, 0.30), 0.025)
assert abs(float(su[:, 20:44, 20:44].mean()) - 4.0) < 1e-3, su.mean()
print("smooth_to ok")
