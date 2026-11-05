"""
End-to-end smoke test of the real command line, experiments/alma_unified.py (the script that takes a new cube through recipe B), on CPU.

A tiny synthetic CLEAN-style line cube (a Keplerian disc, beam 4 px, peak SNR about 15) goes through every step the script does on a real cube: read, line-free
channel finder, continuum decision, zoom to a 10 px beam, Wiener, network, gate, second pass, resample back, moment maps, rotation-curve / PV checks, report,
FITS output and the --boot error map. The network is random-initialised and shaped like recipe B's (K = 15, wiener arm); its last convolution is zero-initialised,
so its output is the Wiener step.

This tests the plumbing and the output files. It says nothing about denoising quality: that is what the Kaggle runs under results/ are for. Run it before
spending Kaggle quota on a change to the pipeline:

    PYTHONPATH=. python3 tests/test_alma_unified_smoke.py        (about half a minute on CPU)
"""
import importlib.util, json, os, runpy, shutil, sys, tempfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
need = [m for m in ("numpy", "scipy", "astropy", "torch", "bettermoments") if importlib.util.find_spec(m) is None]
if need:
    print(f"SKIPPED: needs {', '.join(need)} (not installed here)")
    sys.exit(0)

import numpy as np
import torch
from astropy.io import fits
from scipy import ndimage

sys.path[:0] = [PROJECT, os.path.join(PROJECT, "experiments")]
torch.set_num_threads(2)
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False           # CPU on every machine; MPS lacks ops the zoom path needs
import train_unified as tu

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail and not cond else ''}")
    if not cond:
        failures.append(name)


tmp = tempfile.mkdtemp(prefix="exxa_smoke_")
rng = np.random.default_rng(0)
C, n, dv = 80, 64, 0.1
v = (np.arange(C) - C / 2) * dv
yy, xx = np.indices((n, n))
x, y = xx - n / 2, yy - n / 2
r = np.hypot(x, y) + 1.0
inten = np.exp(-r / 14.0) * (r < 26)
vlos = 2.5 * (x / r) * (r / 8.0) ** -0.5
cube = np.stack([inten * np.exp(-0.5 * ((vc - vlos) / 0.25) ** 2) for vc in v]).astype(np.float32)
cube = np.stack([ndimage.gaussian_filter(c, 4 / 2.355) for c in cube])
cube = (cube / cube.max()).astype(np.float32)
noise = np.stack([ndimage.gaussian_filter(rng.standard_normal((n, n)), 4 / 2.355) for _ in range(C)]).astype(np.float32)
cube = cube + noise * (1.0 / 15.0) / noise.std()
hdr = fits.Header()
for k, val in {"CTYPE1": "RA---SIN", "CTYPE2": "DEC--SIN", "CTYPE3": "VRAD", "CUNIT3": "m/s", "CDELT1": -0.02 / 3600, "CDELT2": 0.02 / 3600, "CRVAL3": float(v[0] * 1000),
               "CDELT3": dv * 1000, "CRPIX3": 1.0, "CRPIX1": n / 2 + 0.5, "CRPIX2": n / 2 + 0.5, "CRVAL1": 0.0, "CRVAL2": 0.0, "BMAJ": 0.08 / 3600, "BMIN": 0.08 / 3600,
               "BPA": 0.0, "BUNIT": "JY/BEAM", "RESTFRQ": 3.307e11, "OBJECT": "SYNTH"}.items():
    hdr[k] = val
fits.writeto(f"{tmp}/synth_cube.fits", cube, hdr, overwrite=True)

tu.K = 15
net = tu.build_net("wiener", 16)
torch.save({"K": 15, "arm": "wiener", "base": 16, "model_state_dict": net.state_dict(), "args": {"norm": "robust", "clip_in": 10.0}}, f"{tmp}/synth_net.pth")

argv_before = sys.argv
sys.argv = ["alma_unified.py", "--cube", f"{tmp}/synth_cube.fits", "--ckpt", f"{tmp}/synth_net.pth", "--out", f"{tmp}/run", "--crop", "64",
            "--target-beam-px", "10", "--incl", "40", "--pa", "90", "--dist", "150", "--boot", "2"]
try:
    print("alma_unified.py on a synthetic cube (CPU):")
    runpy.run_path(os.path.join(PROJECT, "experiments", "alma_unified.py"), run_name="__main__")
    ran = True
except SystemExit as e:
    ran = not e.code
finally:
    sys.argv = argv_before

print("\nchecks:")
check("the script ran to the end", ran)
out = f"{tmp}/run"
for name in ("report.json", "synth_cube_denoised.fits", "synth_cube_m0_den.fits", "synth_cube_m1_den.fits", "synth_cube_m2_den.fits", "synth_cube_m1_boot_std.fits"):
    check(f"output written: {name}", os.path.exists(f"{out}/{name}"))
if os.path.exists(f"{out}/report.json"):
    rep = json.load(open(f"{out}/report.json"))
    check("report shape equals the input cube's", rep["shape"] == [C, n, n], str(rep["shape"]))
    check("line-free channels found", rep["line_free_planes"] >= 20, str(rep["line_free_planes"]))
    check("zoomed to a larger grid (beam 4 px to about 10 px)", rep["scale"] > 2.0, str(rep["scale"]))
    check("gate in its allowed range", 0.25 <= rep["gate_a"] <= 1.0, str(rep["gate_a"]))
    check("rotation-curve / PV checks computed", "S6" in rep and "failed" not in rep["S6"], str(rep.get("S6", {}).get("failed")))
    check("error map ran with the draws asked for", rep.get("U", {}).get("draws") == 2, str(rep.get("U")))
if os.path.exists(f"{out}/synth_cube_denoised.fits"):
    den = fits.getdata(f"{out}/synth_cube_denoised.fits")
    check("denoised cube has the input's shape", den.shape == cube.shape, str(den.shape))
    check("denoised cube is finite", bool(np.isfinite(den).all()))

shutil.rmtree(tmp, ignore_errors=True)
print("\nPASSED" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)
