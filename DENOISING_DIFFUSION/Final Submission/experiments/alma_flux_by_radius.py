"""
Where the denoised-over-raw flux excess (S1 M0 ratio) sits radially: flux share and den / raw ratio within 1, 1-3, 3-6 and beyond 6 beams of the
fitted disc centre (the refit geometry where the kernel's fit failed), inside S1's mask, plus the ratio outside 3 beams. Finding:
the excess is concentrated on the unresolved centre, which the truth-based injection (Speedie disc, central cavity) does not cover.
Light: 2D M0 maps only. Usage: python experiments/alma_flux_by_radius.py <maps dir with alma/<disc>_13CO/> <archive dir with refits>
"""
import glob, json, math, os, sys
import numpy as np
from astropy.io import fits


def main(maps, archive):
    print("| disc | blind SNR | S1 flux ratio | flux share <1 / 1-3 / 3-6 / >6 beams | den / raw in those annuli | den / raw outside 3 beams |")
    print("|---|---|---|---|---|---|")
    for d in sorted(glob.glob(os.path.join(maps, "alma", "*_13CO"))):
        disc = os.path.basename(d)
        rep = json.load(open(os.path.join(d, "report.json")))
        rf = glob.glob(os.path.join(archive, "**", disc, "refit", "refit.json"), recursive=True)
        g = (json.load(open(rf[0]))["refit_S6"] if rf else rep["S6"])["geom"]
        C = rep["shape"][0]; s_m0 = rep["sigma_raw"] * abs(rep["dv_ms"] / 1000) * math.sqrt(max(1, C - rep["line_free_planes"]))
        r = fits.getdata(glob.glob(os.path.join(d, "*_m0_raw.fits"))[0]).astype(np.float64)
        n = fits.getdata(glob.glob(os.path.join(d, "*_m0_den.fits"))[0]).astype(np.float64)
        mask = r / s_m0 > 5
        yy, xx = np.indices(r.shape); rad = np.hypot(xx - g["cx_fit"], yy - g["cy_fit"]) / rep["beam_px"][0]
        tot = r[mask].sum(); sh, ra = [], []
        for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 1e9)):
            b = mask & (rad >= lo) & (rad < hi)
            sh.append(r[b].sum() / tot); ra.append(n[b].sum() / r[b].sum() if r[b].sum() > 0 else float("nan"))
        out = mask & (rad >= 3)
        print(f"| {disc[:-5]}{' (refit)' if rf else ''} | {rep['snr_est']:.0f} | {n[mask].sum() / tot:.3f} | " + " / ".join(f"{x:.2f}" for x in sh) + " | "
              + " / ".join(f"{x:.2f}" for x in ra) + f" | {n[out].sum() / r[out].sum():.3f} |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
