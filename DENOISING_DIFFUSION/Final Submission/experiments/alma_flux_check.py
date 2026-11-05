"""
Is the S1 flux ratio (denoised M0 / raw M0 over the signal mask) biased by its reference? S1 uses 3-sigma clipped moment-0 maps; on a faint disc the
clip drops faint line emission from the noisy raw cube but not from the denoised one. The unclipped sum over the line channels inside the same mask has
zero-mean noise, so the raw value is an unbiased reference there. Prints both ratios (the clipped one must reproduce the run's report).

Usage: python experiments/alma_flux_check.py --cube raw.fits --run-dir <folder with report.json and *_denoised.fits> --crop 512
"""
import argparse, glob, json, math, os, sys
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alma_unified as au

ap = argparse.ArgumentParser()
ap.add_argument("--cube", required=True); ap.add_argument("--run-dir", required=True); ap.add_argument("--crop", type=int, required=True)
a = ap.parse_args()
rep = json.load(open(os.path.join(a.run_dir, "report.json")))
assert not rep.get("fov") and not (rep.get("continuum") or {}).get("subtracted"), "only plain-crop, unsubtracted runs"
cube, velax, hdr, pix, beam = au.read_cube(a.cube, a.crop)
pred = fits.getdata(glob.glob(os.path.join(a.run_dir, "*_denoised.fits"))[0]).astype(np.float32)
assert pred.shape == cube.shape
lf, _ = au.line_free(cube, beam[0] / pix)
C, dv = cube.shape[0], float(velax[1] - velax[0])
sig = float(cube[lf].std())
assert abs(sig / rep["sigma_raw"] - 1) < 0.01
m0r, m0d = au.moments(cube, velax, 3.0, sig)[0], au.moments(pred, velax, 3.0, sig)[0]
mask = m0r > 5 * sig * dv * math.sqrt(max(1, C - len(lf)))
line = np.setdiff1d(np.arange(C), lf)
raw_u, den_u = float(cube[line][:, mask].sum()), float(pred[line][:, mask].sum())
err_u = sig * math.sqrt(len(line) * mask.sum())                   # white-noise error of the raw unclipped sum (an underestimate: beam-correlated)
beam_px2 = math.pi / (4 * math.log(2)) * beam[0] * beam[1] / pix ** 2
print(f"{os.path.basename(a.cube)}: mask {mask.sum()} px, {len(line)} line channels")
print(f"  clipped (S1):  den / raw = {m0d[mask].sum() / m0r[mask].sum():.3f}  (report {rep['S1']['m0_total_ratio']:.3f})")
print(f"  unclipped sum: den / raw = {den_u / raw_u:.3f}  (raw sum noise, beam-correlated: +-{err_u * math.sqrt(beam_px2) / raw_u:.3f})")
print(f"  clipped / unclipped (1 = the clip loses nothing): raw {m0r[mask].sum() / (raw_u * abs(dv)):.3f}, den {m0d[mask].sum() / (den_u * abs(dv)):.3f}")
