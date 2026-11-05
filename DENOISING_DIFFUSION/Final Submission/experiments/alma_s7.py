"""
S7 (see results/05-real-alma/CHECKS.md): is the denoised cube closer to an independent, lower-noise image of the same disc than the raw cube is? Reference: exoALMA's
"High Surface Brightness Sensitivity" 13CO cubes (`<disc>_13CO_low-res.image.fits`, Harvard Dataverse doi:10.7910/DVN/EFBHNU, about 0.3" beam).
The fiducial raw cube and recipe B's denoised cube (same crop) are smoothed to the low-res beam (Gaussian kernel of width sqrt(b_lr^2 - b_fid^2),
Jy/beam rescaled by the beam-area ratio) and each compared with the low-res cube inside its own signal mask.
Caveat (reported with every number): the low-res image is made from the same visibilities with a taper, so its noise is partly the raw cube's
noise; that favours the raw cube in this comparison, so a denoised win is conservative.

Metrics, inside the low-res mask (M0 above 5 sigma) and over its line channels: rms of (x - lowres) in units of the low-res noise; flux ratio x / lowres
within 1 / 1-3 / >3 low-res beams of the centre; median |M1(x) - M1(lowres)| on the bright half of the mask (m/s).
Usage (Kaggle): python experiments/alma_s7.py --fid <fiducial.fits> --lowres <low-res.fits> --den <denoised.fits> --crop 512 --out <json>
"""
import argparse, json, math, os, sys
import numpy as np
from astropy.io import fits
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alma_unified as au
import wiggle_scorecard as ws
import train_unified as tu


def smooth_to(cube, beam_from, beam_to, pix):
    """Convolve to a larger circularised beam and rescale Jy/beam by the beam-area ratio."""
    bf, bt = math.sqrt(beam_from[0] * beam_from[1]), math.sqrt(beam_to[0] * beam_to[1])
    s = math.sqrt(max(bt ** 2 - bf ** 2, 0.0)) / 2.355 / pix
    out = np.stack([ndimage.gaussian_filter(c, s) for c in cube]).astype(np.float32)
    return out * np.float32((beam_to[0] * beam_to[1]) / (beam_from[0] * beam_from[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", required=True); ap.add_argument("--lowres", required=True); ap.add_argument("--den", required=True)
    ap.add_argument("--crop", type=int, default=512); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw, v_f, h_f, pix_f, b_f = au.read_cube(a.fid, a.crop)
    den = fits.getdata(a.den).astype(np.float32)
    assert den.shape == raw.shape, f"denoised {den.shape} vs raw {raw.shape}"
    c_l = int(round(a.crop * pix_f / abs(float(fits.getheader(a.lowres)["CDELT1"]) * 3600))); c_l -= c_l % 2
    low, v_l, h_l, pix_l, b_l = au.read_cube(a.lowres, c_l)
    if low.shape[-1] != raw.shape[-1]:                       # same sky field, put on the fiducial pixel grid
        low = tu.resample(low, raw.shape[-1])
    if len(v_l) != len(v_f) or np.abs(v_l - v_f).max() > 1e-3:   # put the low-res cube on the fiducial channels (linear in velocity)
        idx = np.interp(v_f, v_l, np.arange(len(v_l)))
        i0 = np.clip(np.floor(idx).astype(int), 0, len(v_l) - 2); w1 = (idx - i0).astype(np.float32)
        low = (low[i0] * (1 - w1)[:, None, None] + low[i0 + 1] * w1[:, None, None]).astype(np.float32)
    lf, _ = au.line_free(low, b_l[0] / pix_f)
    line = np.setdiff1d(np.arange(low.shape[0]), lf)
    sig_l = float(low[lf].std())
    rs, ds = smooth_to(raw, b_f, b_l, pix_f), smooth_to(den, b_f, b_l, pix_f)
    dv = abs(float(v_f[1] - v_f[0]))
    m0l = au.moments(low, v_f, 3.0, sig_l)[0]
    mask = m0l > 5 * sig_l * dv * math.sqrt(max(1, len(line)))
    rep = {"beam_fid_arcsec": b_f, "beam_lowres_arcsec": b_l, "pix_fid_mas": pix_f * 1000, "sigma_lowres": sig_l,
           "sigma_raw_smoothed": float(rs[lf].std()), "sigma_den_smoothed": float(ds[lf].std()), "mask_px": int(mask.sum()),
           "caveat": "the low-res image shares visibilities (and so part of its noise) with the raw fiducial cube; this favours raw"}
    for k, x in (("raw", rs), ("den", ds)):
        diff = (x[line][:, mask] - low[line][:, mask]) / sig_l
        rep[f"rms_diff_{k}_in_sigma_lowres"] = float(np.sqrt(np.mean(diff ** 2)))
    n = raw.shape[-1]; yy, xx = np.indices((n, n)); rad = np.hypot(xx - n / 2, yy - n / 2) / (b_l[0] / pix_f)
    for k, x in (("raw", rs), ("den", ds)):
        rr = []
        for lo, hi in ((0, 1), (1, 3), (3, 1e9)):
            b = mask & (rad >= lo) & (rad < hi)
            rr.append(float(x[line][:, b].sum() / low[line][:, b].sum()) if b.any() else None)
        rep[f"flux_ratio_{k}_vs_lowres_by_radius"] = rr
    m1l = ws.quad_m1(low, v_f)
    bright = mask & (m0l >= np.median(m0l[mask]))
    for k, x in (("raw", rs), ("den", ds)):
        rep[f"m1_abs_diff_{k}_vs_lowres_ms"] = float(np.nanmedian(np.abs(ws.quad_m1(x, v_f) - m1l)[bright]) * 1000)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in rep.items() if k.startswith(("rms", "flux", "m1", "sigma"))}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
