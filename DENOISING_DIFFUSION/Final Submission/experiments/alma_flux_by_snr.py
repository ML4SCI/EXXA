"""
Where does the denoised-over-raw flux excess (S1 M0 ratio) come from? Splits the ratio by each pixel's own M0 signal-to-noise (raw M0 over the M0
noise S1 uses: sigma_raw * dv * sqrt(line channels)), inside S1's mask (raw M0 above 5 times that noise). The truth-based test says the model adds about 2 percent at peak SNR 12 and above and up to 10 percent at SNR 6; a disc whose flux sits mostly in faint
pixels should then read high even at a high peak SNR. Light: 2D M0 maps only.
Usage: python experiments/alma_flux_by_snr.py <maps dir with alma/<disc>_13CO/>
"""
import glob, json, math, os, sys
import numpy as np
from astropy.io import fits

EDGES = [5, 10, 20, 40, 1e9]


def main(maps):
    print("| disc | blind SNR | S1 flux ratio | flux share in M0-SNR bins 5-10 / 10-20 / 20-40 / >40 | den / raw in those bins |")
    print("|---|---|---|---|---|")
    for d in sorted(glob.glob(os.path.join(maps, "alma", "*_13CO"))):
        rep = json.load(open(os.path.join(d, "report.json")))
        C = rep["shape"][0]; lf = rep["line_free_planes"]
        s_m0 = rep["sigma_raw"] * abs(rep["dv_ms"] / 1000) * math.sqrt(max(1, C - lf))
        r = fits.getdata(glob.glob(os.path.join(d, "*_m0_raw.fits"))[0]).astype(np.float64)
        n = fits.getdata(glob.glob(os.path.join(d, "*_m0_den.fits"))[0]).astype(np.float64)
        snr = r / s_m0
        mask = snr > 5
        tot = r[mask].sum()
        share, ratio = [], []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            b = mask & (snr >= lo) & (snr < hi)
            share.append(r[b].sum() / tot if b.any() else 0.0)
            ratio.append(n[b].sum() / r[b].sum() if b.any() and r[b].sum() > 0 else float("nan"))
        print(f"| {os.path.basename(d).replace('_13CO', '')} | {rep['snr_est']:.0f} | {n[mask].sum() / tot:.3f} | "
              + " / ".join(f"{x:.2f}" for x in share) + " | " + " / ".join(f"{x:.3f}" for x in ratio) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
