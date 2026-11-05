"""
One table from many alma_unified.py reports: flux, noise reduction, S2, S6, PV of the real cube, fitted vs published position angle, and S5 where it ran.
Usage: python experiments/alma_table.py <dir with */report.json> [...] > table.md
PA: exoALMA PAs (discminer) run 0-360, DSHARP's (continuum) 0-180, and the fit's convention is its own, so the difference is shown modulo 180.
"""
import glob, json, os, sys


def f(x, fmt="{:.2f}"):
    return "nan" if x is None or x != x else fmt.format(x)


def sky_pa(fit):
    """gi_wiggle's pa_deg is the major axis' angle from the image +x axis toward +y. With RA increasing to the left (CDELT1 < 0, north up; read_cube
    flips no spatial axis) +x is west and +y north, so the astronomical PA (east of north) is fit - 90. Checked on AS 209: 84.9 vs Huang 2018's 85.8."""
    return None if fit is None else (fit - 90.0) % 180.0


def pa_diff(fit, lit):
    if fit is None or lit is None:
        return None
    d = (fit - lit) % 180.0
    return min(d, 180.0 - d)


def rows(paths):
    for p in sorted(paths):
        r = json.load(open(p))
        s1, s6 = r["S1"], r.get("S6", {})
        rf = os.path.join(os.path.dirname(p), "refit", "refit.json")
        if os.path.exists(rf):                         # alma_refit.py: the kernel's fit converged wrong; its rescore replaces S6
            s6 = json.load(open(rf))["refit_S6"]
        g, pv = s6.get("geom", {}), s6.get("pv", {})
        s5 = r.get("S5") or {}
        gi = {k.split("@snr")[1]: v["model"]["rc_r"] for k, v in s5.items() if k.startswith("GI")}
        yield {"disc": os.path.basename(os.path.dirname(p)), "snr": r.get("snr_est"), "beam_px": r["beam_px"][0], "scale": r.get("scale", 1.0),
               "flux": s1["m0_total_ratio"], "nr": s1["noise_reduction"], "s2": r["S2"]["frac_above_4sigma_den_lineplanes"],
               "s6": s6.get("rc_r_denoised_vs_raw"), "pv5_r": pv.get("pv5_r"), "pv5_mag": pv.get("pv5_mag_ms"), "pv5_mag_raw": s6.get("pv5_mag_raw_ms"),
               "pa_fit": sky_pa(g.get("pa_deg_fit")), "pa_lit": s6.get("pa_lit_deg"), "dpa": pa_diff(sky_pa(g.get("pa_deg_fit")), s6.get("pa_lit_deg")),
               "mstar": g.get("mstar"), "at_bound": g.get("mstar_at_bound"), "s5_gi": " / ".join(f(gi[k]) for k in sorted(gi, key=float)),
               "cont": (r.get("continuum") or {}).get("subtracted"), "fail": s6.get("failed")}


def main(dirs):
    paths = [p for d in dirs for p in glob.glob(os.path.join(d, "**", "report.json"), recursive=True)]
    print("| disc | blind SNR | beam px (zoom) | flux | noise red. | S2 | S6 rc_r | PV5 r | PV5 mag den / raw (m/s) | PA sky fit / lit (diff mod 180) | M* | S5 GI rc_r | cont. sub |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows(paths):
        print(f"| {x['disc']} | {f(x['snr'], '{:.0f}')} | {f(x['beam_px'], '{:.1f}')} ({f(x['scale'])}) | {f(x['flux'], '{:.3f}')} | {f(x['nr'], '{:.0f}')}x | "
              f"{f(x['s2'], '{:.0e}')} | {f(x['s6'])} | {f(x['pv5_r'])} | {f(x['pv5_mag'], '{:.0f}')} / {f(x['pv5_mag_raw'], '{:.0f}')} | "
              f"{f(x['pa_fit'], '{:.1f}')} / {f(x['pa_lit'], '{:.1f}')} ({f(x['dpa'], '{:.1f}')}) | {f(x['mstar'])}{' BOUND' if x['at_bound'] else ''} | "
              f"{x['s5_gi'] or '-'} | {x['cont']} |" + (f" S6 failed: {x['fail']}" if x["fail"] else ""))


if __name__ == "__main__":
    assert pa_diff(240.3, 60.0) < 0.5 and abs(pa_diff(10.0, 170.0) - 20.0) < 1e-9 and pa_diff(None, 5.0) is None
    main(sys.argv[1:])
