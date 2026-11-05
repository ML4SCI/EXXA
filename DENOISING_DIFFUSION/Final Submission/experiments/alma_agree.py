"""
S3 / S4: agreement between two runs of alma_unified.py on the same real cube, the reference being recipe B's final run.
S3 compares model recipes (A, R6 against B), S4 the pixel scale (beam at 7.5 / 12.5 px against 10). No truth on real data, so this measures how much
the answer depends on the choice, next to the size of the effect being measured.

Per disc: flux ratio (S1 M0 total, variant / reference), median |M1 variant - M1 reference| on the bright half of the signal mask (m/s), PV agreement
(correlation of the quadratically detrended minor-axis M1 curves, 5 px slit, both on the reference's fitted geometry), each run's S6 rc_r and PV5 r
against the raw cube, and S5 GI rc_r where injected.

Usage: python experiments/alma_agree.py --ref <dir of B runs: <disc>/report.json + *_m1_den.fits + *_m0_raw.fits> --var <dir> [--var <dir> ...]
"""
import argparse, glob, json, os, sys
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_curve as pv


def load(d):
    rep = json.load(open(os.path.join(d, "report.json")))
    rf = os.path.join(d, "refit", "refit.json")
    s6 = json.load(open(rf))["refit_S6"] if os.path.exists(rf) else rep.get("S6", {})
    m1 = fits.getdata(glob.glob(os.path.join(d, "*_m1_den.fits"))[0]).astype(np.float64)
    m0 = fits.getdata(glob.glob(os.path.join(d, "*_m0_raw.fits"))[0]).astype(np.float64)
    return rep, s6, m1, m0


def compare(ref_dir, var_dir):
    rr, rs6, rm1, rm0 = load(ref_dir)
    vr, vs6, vm1, _ = load(var_dir)
    g = rs6["geom"]
    mask = np.isfinite(rm0) & (rm0 > 0.05 * np.nanmax(rm0))
    bright = mask & (rm0 >= np.nanmedian(rm0[mask]))
    ok = bright & np.isfinite(rm1) & np.isfinite(vm1)
    geom = {"cx": g["cx_fit"], "cy": g["cy_fit"], "pa_deg": g["pa_deg_fit"], "incl_deg": g["incl"], "vsys": g["vsys_fit"]}
    prof = {k: pv.pv_profile(m, geom, g["au_per_px"], g["r_in_au"], g["r_out_au"], width_px=5, mask=mask)[1] for k, m in (("clean", rm1), ("var", vm1))}
    s5 = lambda r: " / ".join(f"{v['model']['rc_r']:.2f}" for k, v in sorted((r.get("S5") or {}).items(), key=lambda kv: float(kv[0].split("snr")[1])) if k.startswith("GI"))
    return {"flux_ratio_var_over_ref": vr["S1"]["m0_total_ratio"] / rr["S1"]["m0_total_ratio"],
            "m1_abs_diff_median_ms": float(np.median(np.abs(vm1 - rm1)[ok]) * 1000),
            "pv5_r_var_vs_ref": pv.pv_scores(prof)["var"]["pv_r"],
            "s6_ref": rs6.get("rc_r_denoised_vs_raw"), "s6_var": vs6.get("rc_r_denoised_vs_raw"),
            "pv5_vs_raw_ref": (rs6.get("pv") or {}).get("pv5_r"), "pv5_vs_raw_var": (vs6.get("pv") or {}).get("pv5_r"),
            "s5_gi_ref": s5(rr), "s5_gi_var": s5(vr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--var", action="append", required=True)
    a = ap.parse_args()
    refs = {os.path.basename(os.path.dirname(p)): os.path.dirname(p) for p in glob.glob(os.path.join(a.ref, "**", "report.json"), recursive=True)
            if "refit" not in p}
    print("| variant | disc | flux var / ref | median abs dM1 (m/s) | PV5 r var vs ref | S6 rc_r ref / var | PV5 r vs raw ref / var | S5 GI rc_r ref / var |")
    print("|---|---|---|---|---|---|---|---|")
    for vdir in a.var:
        for p in sorted(glob.glob(os.path.join(vdir, "**", "report.json"), recursive=True)):
            d = os.path.dirname(p); disc = os.path.basename(d)
            if disc not in refs or "refit" in p:
                continue
            c = compare(refs[disc], d)
            f = lambda x: "nan" if x is None else f"{x:.2f}"
            print(f"| {os.path.basename(os.path.dirname(d))} | {disc} | {c['flux_ratio_var_over_ref']:.3f} | {c['m1_abs_diff_median_ms']:.0f} | {f(c['pv5_r_var_vs_ref'])} | "
                  f"{f(c['s6_ref'])} / {f(c['s6_var'])} | {f(c['pv5_vs_raw_ref'])} / {f(c['pv5_vs_raw_var'])} | {c['s5_gi_ref'] or '-'} / {c['s5_gi_var'] or '-'} |")


if __name__ == "__main__":
    main()
