"""
Local S6 / PV rescore of a finished alma_unified.py run whose geometry fit converged wrong (DSHARP HD 142666, 2026-10-18: M* 10 Msun, PA 66 deg off).
No GPU: the raw cube (re-downloaded) and the run's own `*_denoised.fits` (kernel Output) are re-read exactly as the run read them (same crop, same
continuum decision from its report.json), and S6 is recomputed through alma_unified.s6 with the published PA added as a starting point of the fit,
both redshifted sides (fit convention: sky PA + 90 and sky PA - 90; alma_table.sky_pa), the lowest-cost fit kept (the original start included).

Usage: python experiments/alma_refit.py --cube raw.fits --run-dir <folder with report.json and *_denoised.fits> --crop 512 --incl 62.22 --dist 148 --pa 162.11
"""
import argparse, glob, json, math, os, sys
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alma_unified as au


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True); ap.add_argument("--run-dir", required=True)
    ap.add_argument("--crop", type=int, required=True); ap.add_argument("--incl", type=float, required=True)
    ap.add_argument("--dist", type=float, required=True); ap.add_argument("--pa", type=float, required=True, help="published sky PA (deg, east of north)")
    a = ap.parse_args()
    rep = json.load(open(os.path.join(a.run_dir, "report.json")))
    assert not rep.get("fov"), "runs with a field-of-view crop are not handled here"
    cube, velax, hdr, pix, beam = au.read_cube(a.cube, a.crop)
    lf, _ = au.line_free(cube, beam[0] / pix)
    if (rep.get("continuum") or {}).get("subtracted"):
        chk = au.continuum_check(cube, lf)
        cube = cube - (chk["map"] if chk.get("map") is not None else cube[lf].mean(0))[None]
    pred = fits.getdata(glob.glob(os.path.join(a.run_dir, "*_denoised.fits"))[0]).astype(np.float32)
    assert pred.shape == cube.shape, f"denoised {pred.shape} vs raw {cube.shape}: not the same crop"
    C, dv = cube.shape[0], float(velax[1] - velax[0])
    sigma_raw = float(cube[lf].std())
    assert abs(sigma_raw / rep["sigma_raw"] - 1) < 0.01, f"raw cube differs from the run's: rms {sigma_raw:.4g} vs {rep['sigma_raw']:.4g}"
    m0r = au.moments(cube, velax, 3.0, sigma_raw)[0]                     # the S1 mask exactly as main() builds it
    mask = m0r > 5 * sigma_raw * dv * math.sqrt(max(1, C - len(lf)))
    out = os.path.join(a.run_dir, "refit"); os.makedirs(out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.cube))[0]
    new = au.s6(name, cube, pred, velax, pix, beam, mask, a.incl, a.dist, a.pa, out, pa_inits=[a.pa + 90.0, a.pa - 90.0])
    json.dump({"original_S6": rep.get("S6"), "refit_S6": new}, open(os.path.join(out, "refit.json"), "w"), indent=1, default=float)
    g0, g1 = (rep.get("S6") or {}).get("geom", {}), new.get("geom", {})
    print(f"{name}: original fit PA {g0.get('pa_deg_fit')} M* {g0.get('mstar')} cost {g0.get('fit_cost')} | refit PA {g1.get('pa_deg_fit')} M* {g1.get('mstar')} "
          f"cost {g1.get('fit_cost')} | S6 {new.get('rc_r_denoised_vs_raw')} PV5 r {(new.get('pv') or {}).get('pv5_r')}")


if __name__ == "__main__":
    main()
