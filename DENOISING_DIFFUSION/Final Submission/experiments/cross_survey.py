"""
HD 143006 across two independent observations of the same line: exoALMA 12CO 3-2 (25 mas pixels, 0.15" beam) and DSHARP 12CO 2-1 (4 mas, 0.066"
beam). If denoising recovers real structure, the two DENOISED results should agree with each other at least as well as the two RAW ones; a feature in
both denoised maps is independently confirmed. Compared in au, each on its own fitted geometry, so no regridding:
  (1) rotation curves v_phi(R) (wiggle_scorecard.rotation_curve), against each other and against exoALMA VI's published 12CO curve;
  (2) minor-axis PV curves (pv_curve.pv_profile, 5 px slit) on a common radius grid, quadratically detrended, correlated between the surveys.
Light: 2D M0 / M1 maps from the kernel Outputs.
Usage: python experiments/cross_survey.py <exoALMA 12CO run dir> <DSHARP run dir> <published 12CO rotation txt> [refit json for DSHARP]
"""
import glob, json, os, sys
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wiggle_scorecard as ws
import pv_curve as pv


def load(d, refit=None):
    rep = json.load(open(os.path.join(d, "report.json")))
    s6 = json.load(open(refit))["refit_S6"] if refit else rep["S6"]
    g = s6["geom"]
    geom = {"cx": g["cx_fit"], "cy": g["cy_fit"], "pa_deg": g["pa_deg_fit"], "incl_deg": g["incl"], "vsys": g["vsys_fit"]}
    m0 = fits.getdata(glob.glob(os.path.join(d, "*_m0_raw.fits"))[0]).astype(np.float64)
    m1 = {k: fits.getdata(glob.glob(os.path.join(d, f"*_m1_{k}.fits"))[0]).astype(np.float64) for k in ("raw", "den")}
    return geom, g["au_per_px"], rep["beam_px"][0] * g["au_per_px"], np.isfinite(m0) & (m0 > 0.05 * np.nanmax(m0)), m1, g


def detrend(y):
    x = np.arange(len(y)); ok = np.isfinite(y)
    return y - np.polyval(np.polyfit(x[ok], y[ok], 2), x) if ok.sum() > 4 else np.full_like(y, np.nan)


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else float("nan")


def main(exo_dir, ds_dir, pub_file, ds_refit=None):
    E = load(exo_dir); D = load(ds_dir, ds_refit)
    pubr = np.loadtxt(pub_file); pubr = pubr[np.isfinite(pubr[:, 1])]
    half = lambda S: 0.5 * min(S[4]["raw"].shape) * S[1] * 0.9          # 90 percent of each map's half-width, in au
    r_lo = 2 * max(E[2], D[2]); r_hi = min(pubr[:, 0].max(), half(E), half(D))
    grid = np.arange(r_lo, r_hi, max(E[1], D[1], 5.0))
    out = {"geometry": {"exoalma": {k: E[5][k] for k in ("pa_deg_fit", "vsys_fit", "mstar")}, "dsharp": {k: D[5][k] for k in ("pa_deg_fit", "vsys_fit", "mstar")}},
           "r_au": [float(r_lo), float(r_hi)]}
    rc = {}
    for name, (geom, au, beam_au, mask, m1, _) in (("exo", E), ("ds", D)):
        for k in ("raw", "den"):
            c, v = ws.rotation_curve(m1[k], mask, geom, au, max(au, 5.0))
            rc[f"{name}_{k}"] = np.interp(grid, c, np.abs(v), left=np.nan, right=np.nan)
    vpub = np.interp(grid, pubr[:, 0], pubr[:, 1])
    for k in ("raw", "den"):
        out[f"rc_{k}"] = {"exo_vs_ds_median_abs_ms": float(np.nanmedian(np.abs(rc[f"exo_{k}"] - rc[f"ds_{k}"])) * 1000),
                          "exo_vs_pub_ms": float(np.nanmedian(np.abs(rc[f"exo_{k}"] - vpub)) * 1000),
                          "ds_vs_pub_ms": float(np.nanmedian(np.abs(rc[f"ds_{k}"] - vpub)) * 1000),
                          "substructure_r_exo_vs_ds": corr(detrend(rc[f"exo_{k}"]), detrend(rc[f"ds_{k}"]))}
    for k in ("raw", "den"):
        prof = {}
        for name, (geom, au, beam_au, mask, m1, _) in (("exo", E), ("ds", D)):
            r_g, halves = pv.pv_profile(m1[k], geom, au, r_lo, r_hi, width_px=5, mask=mask)
            prof[name] = [np.interp(grid, r_g, h, left=np.nan, right=np.nan) for h in halves]
        # the two surveys' minor-axis halves can be swapped (PA conventions); take the better pairing, report both
        same = np.nanmean([corr(detrend(prof["exo"][i]), detrend(prof["ds"][i])) for i in (0, 1)])
        swap = np.nanmean([corr(detrend(prof["exo"][i]), detrend(prof["ds"][1 - i])) for i in (0, 1)])
        out[f"pv_{k}"] = {"r_same_halves": float(same), "r_swapped_halves": float(swap)}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:5])
