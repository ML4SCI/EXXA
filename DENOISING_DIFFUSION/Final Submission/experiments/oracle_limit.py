"""How much wiggle is recoverable at this noise? Oracle LINEAR filters that know clean's power spectrum (not clean itself)."""
import sys, os, json, numpy as np, scipy.fft as sfft
sys.path.insert(0, "experiments"); import wiggle_scorecard as ws

def oracle3d(d):
    clean, dirty, H = d["clean"].astype(np.float64), d["dirty"].astype(np.float64), d["H"]
    C, n, _ = dirty.shape
    kr = np.hypot(np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None]); nb = 120
    rb = np.minimum((kr / kr.max() * nb).astype(int), nb - 1); cnt = np.maximum(np.bincount(rb.ravel(), minlength=nb), 1)
    S = sfft.fftn(clean, workers=-1); D = sfft.fftn(dirty, workers=-1)
    N = D - S * H[None]                                   # noise in Fourier space (dirty - clean*beam)
    out = np.empty_like(D)
    for i in range(C):                                    # per velocity frequency, radially averaged powers: best shift-invariant linear filter
        Ps = ws._radial_avg(np.abs(S[i]) ** 2, rb, cnt, nb); Pn = ws._radial_avg(np.abs(N[i]) ** 2, rb, cnt, nb)
        out[i] = D[i] * np.conj(H) * Ps / (np.abs(H) ** 2 * Ps + Pn + 1e-30)
    return sfft.ifftn(out, workers=-1).real.astype(np.float32)

ws.METHODS["oracle3d"] = oracle3d
name = sys.argv[1]
d = ws.load(name)
res = ws.score_cube(name, d, ["dirty", "beam_only", "wiener3d_a2", "oracle3d"], sys.argv[2])
for k, v in res["methods"].items():
    print(f"{name} {k:12s} rc_r {v['rc_r']:.3f}  dv_med {v['dv_med_ms']:.0f} m/s")
