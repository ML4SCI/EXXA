"""
S6 against published kinematics: our 13CO rotation curves (from the raw and the denoised M1 maps of the final exoALMA runs) against exoALMA VI's
published 13CO rotation curves (Stadler et al. 2025, Harvard Dataverse doi:10.7910/DVN/FIDKQV, files <disc>_13co_*_radial_profile_rotation.txt:
radius [au], v_rot [km/s], dv_phi, ddv_phi).

Ours: wiggle_scorecard.rotation_curve (v_phi per ring by least squares of M1 - vsys against sin(i) cos(az), thin disc), on the run's own fitted
geometry (the refit where the kernel's fit failed), ring width = the published radial spacing, mask = raw M0 above 5 percent of its peak, radii beyond 2 beam FWHM.
exoALMA VI fit an elevated emission surface, so a thin-disc curve is biased; compare raw and denoised against the same reference.

Per disc: median |v_ours - v_pub| for raw and denoised (m/s), the signed median bias, and the correlation of the two curves after removing a
quadratic in log R from each (the rotation-curve substructure exoALMA VI studies: pressure bumps).
Light: 2D maps only. Usage: python experiments/alma_vs_published.py <maps dir with alma/<disc>_13CO/> <published dir> [archive dir with refits]
"""
import glob, json, os, sys
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wiggle_scorecard as ws

PUB = {"AA_Tau": "aatau", "CQ_Tau": "cqtau", "DM_Tau": "dmtau", "HD_135344B": "hd135344", "HD_143006": "hd143006", "HD_34282": "hd34282",
       "J1604": "j1604", "J1615": "j1615", "J1842": "j1842", "J1852": "j1852", "LkCa_15": "lkca15", "MWC_758": "mwc758", "PDS_66": "pds66",
       "SY_Cha": "sycha", "V4046_Sgr": "v4046"}


def detrend_logr(r, v):
    x = np.log(r)
    return v - np.polyval(np.polyfit(x, v, 2), x)


def compare(d, pub_file, refit=None):
    rep = json.load(open(os.path.join(d, "report.json")))
    s6 = json.load(open(refit))["refit_S6"] if refit else rep["S6"]
    g = s6["geom"]
    geom = {"cx": g["cx_fit"], "cy": g["cy_fit"], "pa_deg": g["pa_deg_fit"], "incl_deg": g["incl"], "vsys": g["vsys_fit"]}
    m0 = fits.getdata(glob.glob(os.path.join(d, "*_m0_raw.fits"))[0]).astype(np.float64)
    mask = np.isfinite(m0) & (m0 > 0.05 * np.nanmax(m0))
    pub = np.loadtxt(pub_file)
    r_pub, v_pub = pub[:, 0], pub[:, 1]
    bin_au = float(np.median(np.diff(r_pub)))
    out = {}
    for k in ("raw", "den"):
        m1 = fits.getdata(glob.glob(os.path.join(d, f"*_m1_{k}.fits"))[0]).astype(np.float64)
        c, v = ws.rotation_curve(m1, mask, geom, g["au_per_px"], bin_au)
        v = np.abs(v)
        sel = (r_pub >= c.min()) & (r_pub <= c.max()) & (r_pub >= 2 * rep["beam_px"][0] * g["au_per_px"]) & np.isfinite(v_pub)   # inside 2 beams the curve is beam-smeared
        vi = np.interp(r_pub[sel], c, v)
        dv = (vi - v_pub[sel]) * 1000
        r = float(np.corrcoef(detrend_logr(r_pub[sel], vi), detrend_logr(r_pub[sel], v_pub[sel]))[0, 1]) if sel.sum() > 6 else float("nan")
        out[k] = {"n": int(sel.sum()), "r_range": [float(r_pub[sel].min()), float(r_pub[sel].max())] if sel.any() else None,
                  "median_abs_ms": float(np.median(np.abs(dv))) if sel.any() else float("nan"), "median_bias_ms": float(np.median(dv)) if sel.any() else float("nan"),
                  "substructure_r": r}
    return out


def main(maps, pubdir, archive=None):
    print("| disc | radii (au), n | median abs dv raw / den (m/s) | bias raw / den (m/s) | substructure r raw / den |")
    print("|---|---|---|---|---|")
    res = {}
    for d in sorted(glob.glob(os.path.join(maps, "alma", "*_13CO"))):
        disc = os.path.basename(d).replace("_13CO", "")
        pf = glob.glob(os.path.join(pubdir, f"{PUB[disc]}_13co_*radial_profile_rotation.txt"))
        pf = [p for p in pf if "b0p30" not in p and "highres" not in p]
        if not pf:
            continue
        rf = glob.glob(os.path.join(archive, "**", f"{disc}_13CO", "refit", "refit.json"), recursive=True) if archive else []
        o = compare(d, pf[0], rf[0] if rf else None); res[disc] = o
        a, b = o["raw"], o["den"]
        rr = f"{a['r_range'][0]:.0f} to {a['r_range'][1]:.0f}, {a['n']}" if a["r_range"] else "-"
        print(f"| {disc}{' (refit)' if rf else ''} | {rr} | {a['median_abs_ms']:.0f} / {b['median_abs_ms']:.0f} | {a['median_bias_ms']:+.0f} / {b['median_bias_ms']:+.0f} | "
              f"{a['substructure_r']:.2f} / {b['substructure_r']:.2f} |")
    return res


if __name__ == "__main__":
    r = main(*sys.argv[1:4])
    if len(sys.argv) > 4:
        json.dump(r, open(sys.argv[4], "w"), indent=1)
