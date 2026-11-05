"""
One denoiser for all three kinds of data the project has, judged on everything at once: PSNR, M0/M1/M2 (signal-masked),
and the GI wiggle (rc_r, Terry spine), at 256 / 300 / 320 / 480 / 600 px.

Why: the line-emission models score well on PSNR and moments but lose the GI wiggle; on the
wiggle metric that scores the fingers (rc_r) no checkpoint gets near the noise-free beam, and on high-SNR SG v2 every learned
model falls below dirty. They were trained on one kind of cube at one resolution, one channel at a time.

Data (fresh noise every round; a cube is never seen with the same noise twice):
  line     9 line-emission cubes (cube_split train+val; the standard 5 holdouts are never trained on). dirty = clean + noise,
           noise drawn from that cube's own measured 2D noise power spectrum (beam-correlated, as in a CLEAN image and in
           real ALMA products). Continuum removed from both, as checkpoint_eval does.
  hydro    sg_synth 9015/9019/9025/9032 (9074 held out). dirty = beam (*) clean + noise, noise spectrum measured from the pair.
  analytic an01-an06 (Longarini model; an07/an08 held out). Same forward model as hydro.
Every draw is resampled to a random size from SIZES before the Wiener step, so the model meets each disk at every pixel scale
it is scored at, and the Wiener filter (and its noise estimate, from the line-free edge channels) runs at that size, exactly as
at test time.

Model (arm):
  wiener  out = Wiener(dirty) + UNet([dirty window, Wiener window, noise map])   last conv zero-initialised: step 0 == Wiener
  direct  out = dirty + UNet([dirty window, noise map])                           ablation: same data, no Wiener prior
Windows are 7 channels spanning +-0.3 km/s whatever the channel width. Loss: L1 on three adjacent centres + L1 on the first
velocity difference (line centre, M1, the wiggle) + L1 on the second difference (line width, M2).

Scoring (every holdout x every size): checkpoint_eval.score (PSNR, SSIM, M0/M1/M2, artifacts) and wiggle_scorecard.score_cube
(rc_r, coherence, line-centre error, Terry spine) against dirty and Wiener on the same resampled cube.

Fine-tuning existing checkpoints (--init PATH, arm "legacy"): the checkpoint keeps ITS OWN input/output format (dirty window with
min-max shared from the centre channel; 1 centre channel out, or the whole window out for the kin_gamma0 family), and is trained
further on the same data mix, losses and sizes. ALMA-matched data: real ALMA CLEAN products have the beam inside the clean target, so
the task is noise-only (A = I). Every hydro / analytic training cube is therefore drawn either as the SG deconvolution task (dirty =
beam (*) clean + noise) or, with probability --p-ai, as the noise-only task (dirty = clean + noise, same measured noise spectrum).

Run: the Kaggle scripts in kernels/.  Local smoke: PYTHONPATH=. python3 experiments/train_unified.py --smoke
"""
import os, sys, json, time, math, argparse, glob
import numpy as np
import scipy.fft as sfft
from scipy import ndimage
if not hasattr(np, "trapezoid"):          # bettermoments on numpy < 2 (local only; Kaggle has numpy 2)
    np.trapezoid = np.trapz

ROOT = os.environ.get("EXXA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from astropy.io import fits
import wiggle_scorecard as ws
from src.models.unet import UNet
from src.data.cube_split import split_cubes
from src.evaluation import checkpoint_eval as ce

FP32_STREAM = False                          # --fp32-stream: see _enable_fp32_stream
K = 3                                        # window half-width in 0.1 km/s channels; set by --k (kin_gamma0 used 15)
SIZES = (256, 300, 320, 480, 600)
ALPHA = 2.0
N_EDGE = 8                                   # line-free channels at each end: continuum and noise estimate
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")      # mps: local smoke tests on the Apple GPU
rdev = "cuda" if dev == "cuda" else "cpu"        # resampling: MPS cannot area-pool 600 -> 256 (non-divisible)
LE_DIR = os.environ.get("LE_DIR") or os.path.join(ROOT, "Line Emission Data")
SG_DIR = os.path.join(ROOT, "self-gravitating cube and dirty cube")
# Realistic-noise training:
# peak SNR drawn log-uniform in REAL["snr"], neighbouring-channel noise correlation [a, 1, a] with a ~ U(0, chan_a) (a = 0.5 is
# Hanning: lag-1 +0.67, lag-2 +0.17; Speedie measured +0.64 / +0.15), beam broadened by r ~ U(1, beam_max) in the A = I tasks.
REAL = {"snr": None, "chan_a": 0.0, "beam_max": 1.0, "beam_px": None}   # beam_px: --train-beam-px (lo, hi)
DEADLINE = float("inf")   # wall-clock time (epoch s) by which this session must have saved everything: EXXA_T0 (kernel start) + --budget-h
GATE = None          # --gate lo,hi,floor: do-no-harm gate (see gate_alpha). None = off, so every earlier run reproduces
GATE_SWEEP = []      # --gate-sweep: mix levels a in [0, 1] of  a * model + (1 - a) * Wiener  scored next to the plain model
NORM = "wiener"      # --norm: "wiener" = divide by the Wiener cube's peak (all runs to 2026-10-11); "robust" = see scale_of
INFER_BEAM = 0.0     # --infer-beam-px: run the network where the beam spans this many px (see infer); 0 = on the cube as given (all runs to 2026-10-14)
INFER_MODE = "corr"  # --infer-mode
WIENER2 = False      # --wiener2: evaluation, second pass with prior_wiener (noise-only cases)
CENTRAL_FIX = False  # --central-fix: candidate A (central_flux_fix after inference); off = recipe B as frozen
SAVE_PRED = ("", set())  # --save-pred DIR --save-pred-cases a,b: M1 maps and the central channels of clean / dirty / Wiener / model for those cases (planet check)
CLIP_IN = 0.0        # --clip-in: clamp the network input to +-CLIP_IN (k15 v3: inputs of +-35-40 vs the usual +-4 overflowed fp16 and the run diverged)
HYDRO_TRAIN = ["9015", "9019", "9025", "9032"]
ANALYTIC_TRAIN = ["an01", "an02", "an03", "an04", "an05", "an06"]


# ------------------------------------------------------------------ shared helpers
def edge_idx(C):
    return np.r_[0:N_EDGE, C - N_EDGE:C]


def continuum_sub(cube):
    return cube - cube[edge_idx(cube.shape[0])].mean(0)[None]


def empirical_noise2d(cube, chans=None):
    """Per-channel noise power |FFT|^2 from the line-free edge channels (or `chans`), after removing their per-pixel mean
    (continuum / static pattern). Blind: no clean data. Matches how SG v2's noise was measured (wiggle_scorecard.load)."""
    e = cube[edge_idx(cube.shape[0]) if chans is None else chans].astype(np.float64)
    e = e - e.mean(0)[None]
    m = e.shape[0]
    return np.mean(np.abs(sfft.fft2(e, workers=-1)) ** 2, axis=0) * m / max(m - 1, 1)


def quiet_noise2d(cube, m=16):
    """Noise power from the m channels with the least total power (works when the edge channels carry line flux, as on 9074)."""
    pw = (cube.reshape(cube.shape[0], -1).astype(np.float64) ** 2).sum(1)
    return empirical_noise2d(cube, np.argsort(pw)[:m])


def is_ai(kind):
    """A = I regime (the beam is already inside clean): real line emission and the noise-only variants of the simulations."""
    return kind == "line" or kind.endswith("_ai")


def resample(cube, S):
    """(C, n, n) -> (C, S, S): area average going down (flux per pixel kept), bilinear going up."""
    n = cube.shape[-1]
    if S == n:
        return np.ascontiguousarray(cube, dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(cube, dtype=np.float32))
    out = []
    for i in range(0, t.shape[0], 32):
        x = t[i:i + 32, None].to(rdev)
        y = Fn.interpolate(x, size=(S, S), mode="area") if S < n else Fn.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
        out.append(y[:, 0].float().cpu())
    return torch.cat(out).numpy()


_BEAM = {}


def beam_otf(n):
    """Recovered SG beam (unit gain) on an n x n grid: the forward model of sg_synth, the analytic disks and SG v2."""
    if n not in _BEAM:
        b = fits.getdata(ws.BEAM).astype(np.float64)
        _BEAM[n] = ws.beam_otf(b / b.sum(), n).astype(np.complex64)
    return _BEAM[n]


def velocity_corr(cube, frac=0.2):
    """Blind lag-1 / lag-2 noise correlation between channels, from the faintest `frac` of pixels (off-source: noise only).
    The faint pixels are chosen on a map smoothed far beyond the beam: ranking pixels by their own noise picks the ones whose
    noise happens to be smooth and biases the correlation low (lag-2 0.10 for a true 0.17 at frac 0.2)."""
    m = ndimage.gaussian_filter(np.abs(cube).mean(0), cube.shape[-1] / 20)
    x = cube[:, m <= np.quantile(m, frac)].astype(np.float64)
    x -= x.mean(0)
    v = max((x * x).mean(), 1e-300)
    r1, r2 = ((x[:-k] * x[k:]).mean() / v for k in (1, 2))
    return float(np.clip(r1, 0.0, 0.95)), float(np.clip(r2, -0.5, 0.95))


def vel_spectrum(C, r1, r2):
    """Noise power along the velocity-frequency axis for lag correlations r1, r2 (mean 1, fftfreq order)."""
    f = np.fft.fftfreq(C)
    return np.clip(1 + 2 * r1 * np.cos(2 * np.pi * f) + 2 * r2 * np.cos(4 * np.pi * f), 0.02, None).astype(np.float32)


def chan_kernel(a):
    """[a, 1, a], normalised so the noise variance per channel is unchanged."""
    return np.array([a, 1.0, a], np.float32) / np.float32(np.sqrt(1 + 2 * a * a))


def gauss_otf(n, fwhm_px):
    """rfft2-grid transfer function of a circular Gaussian with this FWHM in pixels."""
    s2 = (fwhm_px / 2.3548) ** 2
    k2 = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.rfftfreq(n)[None, :] ** 2
    return np.exp(-2 * np.pi ** 2 * s2 * k2).astype(np.float32)


def native_wiener(dirty, kind, noise_chans=None):
    """3D Wiener at the cube's NATIVE grid, where the beam and the noise are defined, blind (no clean data).
    line: no deconvolution (the beam is already in clean), noise from the line-free edge channels (continuum already removed).
    sg/analytic: beam deconvolution, noise = beam-shaped white noise with its level from the quietest channels at high k
    (wiggle_scorecard.blind_noise2d); SG v2: noise measured on its far line-free channels.
    Running it on a resampled cube instead amplifies the resampling's own artifacts in the deconvolution (PSNR 1.3 dB on an07
    at 256 px, 2026-10-09), and a real cube is never resampled, so the test sizes resample AFTER this step."""
    n = dirty.shape[-1]
    vs = vel_spectrum(dirty.shape[0], *velocity_corr(dirty))      # blind; ~flat on channel-independent noise
    if kind == "line":                     # noise from the line-free edge channels, or from `noise_chans` (a real cube: chosen from its spectrum)
        return wiener(dirty, None, empirical_noise2d(dirty, noise_chans), vspec=vs)
    if kind.endswith("_ai"):
        return wiener(dirty, None, quiet_noise2d(dirty), vspec=vs)
    H = beam_otf(n)
    nz = empirical_noise2d(dirty, noise_chans) if noise_chans is not None else ws.blind_noise2d(dirty, H)
    return wiener(dirty, H, nz, vspec=vs)


def wiener(dirty, H, noise2d, alpha=ALPHA, vspec=None):
    S = dirty.shape[-1]
    return ws.wiener3d({"dirty": dirty, "H": H if H is not None else np.ones((S, S), np.complex64), "noise2d": noise2d,
                        "vspec": vspec}, alpha)


def scale_of(dS, wS):
    """Normalisation scale of a cube pair. "wiener": the Wiener peak, which COLLAPSES at low SNR (the filter shrinks a peak-SNR-3 cube to ~1/16 of the
    clean peak), so targets reach 16x and inputs 29x the usual range and training diverges (`unified-r4-k15-long`, 2026-10-12). "robust": the larger of
    the Wiener peak and the 99.99th percentile of the dirty cube, which tracks the clean peak to within ~20 % from SNR 3 to 150 and is blind (test time too)."""
    s = float(wS.max())
    if NORM == "robust":
        s = max(s, float(np.percentile(dS[::2, ::2, ::2], 99.99)))
    return max(s, 1e-30)


def gate_alpha(snr_est, gate=None):
    """Do-no-harm gate. Mix of the model and the Wiener output, a * model + (1 - a) * Wiener, chosen from the blind peak-SNR estimate:
    a = 1 up to `lo`, falling linearly to `floor` at `hi`, `floor` above. Measured (eval-gate, 2026-10-13, R4n-long seeds 3 + 11): at estimate 121-133 (true peak SNR
    ~135) the plain model damages M0 / M2 (-76 / -257 %); a = 0.25 gives -4 / +50 % with the wiggle at dirty's level; at estimates 9-27 (true SNR 12-36 and SG v2)
    the plain model has the best wiggle, so a stays 1. The ramp between 40 and 120 is an interpolation: no cube of that estimate was measured."""
    g = gate or GATE
    if not g:
        return 1.0
    lo, hi, floor = g
    return float(np.clip(1.0 - (1.0 - floor) * (snr_est - lo) / max(hi - lo, 1e-9), floor, 1.0))


def noise_level(dS, wS):
    """Noise-map value fed to the network: rms of (dirty - Wiener) at the working size. Same blind definition in training and at
    test time, whatever the data kind or resampling."""
    return float(np.std(dS.astype(np.float64) - wS.astype(np.float64)))


def offsets(dv):
    s = max(1, int(round(0.1 / abs(dv))))
    return np.arange(-K, K + 1) * s, s


# ------------------------------------------------------------------ training cubes
def pair_noise_amp(clean, dirty, conv):
    """sqrt of the pair's measured noise power (dirty - conv), rfft2 grid, so white N(0,1) (*) amp reproduces its spectrum."""
    C, n = clean.shape[0], clean.shape[-1]
    r = (dirty - conv)[:: max(1, C // 48)].astype(np.float64)
    P = np.mean(np.abs(sfft.rfft2(r - r.mean((1, 2), keepdims=True), workers=-1)) ** 2, axis=0)
    return (np.sqrt(P) / n).astype(np.float32)


class TrainCube:
    """family: line | hydro | analytic. Simulations are drawn as the deconvolution task or, with p_ai, the noise-only task."""

    def __init__(self, name, family, clean, dirty, dv, p_ai=0.0, fwhm_px=None):
        self.name, self.family, self.dv, self.p_ai, self.kind = name, family, float(dv), p_ai, family
        self.fwhm = fwhm_px if fwhm_px is not None else sim_beam_fwhm()      # native beam (geometric mean FWHM, pixels)
        clean, dirty = clean.astype(np.float32, copy=False), dirty.astype(np.float32, copy=False)
        line_like = family in ("line", "sph")   # sph: rendered SPH discs, beam already in clean (A = I), its own sampling family
        if line_like:                            # real continuum; the simulations (hydro, analytic) have none; sph edges are line-free (~0)
            clean, dirty = continuum_sub(clean), continuum_sub(dirty)
        self.C, self.n = clean.shape[0], clean.shape[-1]
        self.clean = clean
        if line_like:
            self.conv = clean                                              # A = I: the beam is already in clean
        else:
            H = beam_otf(self.n)
            self.conv = sfft.ifft2(sfft.fft2(clean, workers=-1) * H[None], workers=-1).real.astype(np.float32)
        self.amp = pair_noise_amp(clean, dirty, self.conv)
        w = 0.1 + clean.reshape(self.C, -1).max(1) / max(clean.max(), 1e-30)
        self.ch_w = w / w.sum()
        self.ready = False

    def redraw(self, rng, sizes):
        line_like = self.family in ("line", "sph")
        ai = line_like or rng.random() < self.p_ai
        self.kind = "line" if line_like else self.family + ("_ai" if ai else "")
        clean, spec = self.clean, self.amp
        r = 1.0
        if ai and REAL["beam_max"] > 1:              # a coarser beam: inside clean AND shaping the noise, as in a CLEAN product
            r = rng.uniform(1.0, REAL["beam_max"])
            G = gauss_otf(self.n, self.fwhm * math.sqrt(r * r - 1))
            clean = sfft.irfft2(sfft.rfft2(clean, workers=-1) * G[None], s=(self.n, self.n), workers=-1).astype(np.float32)
            spec = spec * G
        white = rng.standard_normal(self.clean.shape, dtype=np.float32)
        if REAL["chan_a"] > 0:
            white = ndimage.convolve1d(white, chan_kernel(rng.uniform(0.0, REAL["chan_a"])), axis=0, mode="wrap")
        noise = sfft.irfft2(sfft.rfft2(white, workers=-1) * spec[None], s=(self.n, self.n), workers=-1).astype(np.float32)
        sky = clean if ai else self.conv
        if REAL["snr"]:
            lo, hi = REAL["snr"]
            noise *= np.float32(float(sky.max()) / math.exp(rng.uniform(math.log(lo), math.log(hi))) / max(float(noise.std()), 1e-30))
        dirty = sky + noise
        del white, noise
        w = native_wiener(dirty, self.kind)
        S = int(rng.choice(sizes))
        if REAL["beam_px"]:                          # train where inference runs (infer: the beam spans ~INFER_BEAM px), not over every image size
            S = max(64, int(round(self.n * rng.uniform(*REAL["beam_px"]) / (self.fwhm * r) / 8)) * 8)
        self.S = S
        cS, dS, wS = resample(clean, S), resample(dirty, S), resample(w, S)
        del dirty, w, clean
        # normalise BEFORE the float16 cast: 9019 is in different units (~1e10 peak; every other cube < 0.3), which overflows float16
        s = scale_of(dS, wS)
        self.sig = noise_level(dS, wS) / s
        dn = dS / s
        self.lo, self.hi = dn.reshape(self.C, -1).min(1), dn.reshape(self.C, -1).max(1)    # per-channel min-max (legacy models)
        self.c, self.d, self.w = (cS / s).astype(np.float16), dn.astype(np.float16), (wS / s).astype(np.float16)
        self.ready = True


_SIM_FWHM = []


def load_sgv2_pair():
    """SG v2 as a (clean, dirty) training pair in common units. Measured 2026-10-10: in the bright channels dirty = g * (beam (*) clean) with
    g ~ 344 (the two cubes are in different units) and no extra blur; clean also carries a continuum floor (~0.43 summed per plane) that
    the dirty cube lacks after its static pattern is removed. So: dirty = block - static pattern (from the line-free planes); clean = block
    - its own edge mean, then scaled by g fitted on the brightest channels (a single scalar for the whole cube, not per channel)."""
    p = os.path.join(SG_DIR, "kinematic_data_v2")
    blk = np.arange(180, 421)
    with fits.open(f"{p}/clean_sg.fits", memmap=True) as h:
        hdr, cl = h[0].header, h[0].data[blk].astype(np.float32)
    with fits.open(f"{p}/dirty_sg.fits", memmap=True) as h:
        dr = h[0].data[blk].astype(np.float32)
        static = h[0].data[np.r_[0:60, 541:601]].astype(np.float32).mean(0)
    dr -= static[None]
    cl -= cl[:8].mean(0)[None]
    n = cl.shape[-1]
    H = beam_otf(n)
    top = np.argsort(cl.reshape(len(cl), -1).sum(1))[-20:]
    conv = sfft.ifft2(sfft.fft2(cl[top], workers=-1) * H[None], workers=-1).real
    g = float((dr[top] * conv).sum() / (conv * conv).sum())
    return cl * np.float32(g), dr, abs(float(hdr["CDELT3"])), g


def sim_beam_fwhm():
    """Geometric-mean FWHM (pixels) of the recovered SG beam that sg_synth, the analytic disks and SG v2 share."""
    if not _SIM_FWHM:
        b = fits.getdata(ws.BEAM).astype(np.float64); b = np.clip(b, 0, None); b /= b.sum()
        y, x = np.indices(b.shape)
        my, mx = (b * y).sum(), (b * x).sum()
        cov = [[(b * (y - my) ** 2).sum(), (b * (y - my) * (x - mx)).sum()], [(b * (y - my) * (x - mx)).sum(), (b * (x - mx) ** 2).sum()]]
        _SIM_FWHM.append(2.3548 * float(np.linalg.det(cov)) ** 0.25)
    return _SIM_FWHM[0]


def load_train_cubes(args):
    cubes = []
    tr, va, _ = split_cubes(data_dir=LE_DIR, n_holdout=3, val_fraction=0.2, seed=42)
    le = (tr + va)[: args.n_line] if args.n_line else tr + va
    for c in le:
        cl, dr = fits.getdata(c["clean"]).astype(np.float32), fits.getdata(c["dirty"]).astype(np.float32)
        h = fits.getheader(c["clean"])
        fwhm = math.sqrt(float(h["BMAJ"]) * float(h["BMIN"])) / abs(float(h["CDELT1"]))
        cubes.append(TrainCube(c["folder"], "line", cl, dr, float(h["CDELT3"]), fwhm_px=fwhm))
        print(f"  line  {c['folder']}  {cl.shape}", flush=True)
    if args.train_sgv2:
        cl, dr, dv, g = load_sgv2_pair()
        cubes.append(TrainCube("sgv2", "hydro", cl, dr, dv, p_ai=args.p_ai))
        print(f"  hydro    sgv2  {cl.shape}  (clean -> dirty units: x{g:.0f}; SG v2 is in TRAINING: its scores are not held-out)", flush=True)
        del cl, dr
    for folder in (sorted(glob.glob(os.path.join(args.sph_dir, "run_young_*"))) if args.sph_dir else [])[: args.sph_max or None]:
        meta = json.load(open(os.path.join(folder, "view.json")))
        if meta["split"] != "train":
            continue
        base = os.path.join(folder, os.path.basename(folder))
        h = fits.getheader(base + "_clean.fits")
        cl, dr = fits.getdata(base + "_clean.fits").astype(np.float32), fits.getdata(base + "_dirty.fits").astype(np.float32)
        cubes.append(TrainCube(os.path.basename(folder), "sph", cl, dr, float(h["CDELT3"]), fwhm_px=math.sqrt(meta["bmaj_px"] * meta["bmin_px"])))
        print(f"  sph      {os.path.basename(folder)}  {cl.shape}  mu {meta['mu']:.2f}  i {meta['incl']:.0f}", flush=True)
        del cl, dr
    for name in args.hydro.split(",") + args.analytic.split(","):
        if not name:
            continue
        d = ws.load(name)
        cubes.append(TrainCube(name, "analytic" if name.startswith("an") else "hydro", d["clean"], d["dirty"],
                               d["velax"][1] - d["velax"][0], p_ai=args.p_ai))
        print(f"  {cubes[-1].family:8s} {name}  {d['clean'].shape}", flush=True)
        del d
    return cubes


def make_batch(cubes, probs, batch, crop_max, rng, arm):
    X, B0, Cc = [], [], []
    for _ in range(batch):
        cb = cubes[rng.choice(len(cubes), p=probs)]
        offs, step = offsets(cb.dv)
        crop = min(crop_max, cb.S)
        c = int(np.clip(rng.choice(cb.C, p=cb.ch_w), step, cb.C - 1 - step))
        y0, x0 = int(rng.integers(0, cb.S - crop + 1)), int(rng.integers(0, cb.S - crop + 1))
        sl = (slice(None), slice(y0, y0 + crop), slice(x0, x0 + crop))
        for cen in (c - step, c, c + step):
            nb = np.clip(cen + offs, 0, cb.C - 1)
            parts = [cb.d[nb][sl]]
            if arm == "wiener":
                parts.append(cb.w[nb][sl])
            parts.append(np.full((1, crop, crop), cb.sig, np.float16))
            X.append(np.concatenate(parts).astype(np.float32))           # cached arrays are already in units of the cube's Wiener peak
            B0.append((cb.w if arm == "wiener" else cb.d)[cen][sl[1:]].astype(np.float32))
            Cc.append(cb.c[cen][sl[1:]].astype(np.float32))
    # crops differ in size between cubes when a cube is smaller than crop_max: pad to the batch max
    h = max(x.shape[-1] for x in X)
    pad = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, h - a.shape[-2]), (0, h - a.shape[-1])])
    t = lambda a: torch.from_numpy(np.stack([pad(x) for x in a])).float().to(dev)
    x = t(X)
    return (x.clamp_(-CLIP_IN, CLIP_IN) if CLIP_IN else x), t(B0), t(Cc)


def make_batch_legacy(cubes, probs, batch, crop_max, rng, kk, oc):
    """Checkpoint-native I/O: input = dirty window (2kk+1), min-max shared from the centre channel; target = clean in the same scale
    (1 centre channel, in triplets for the velocity-difference loss, or the whole window when the model outputs a stack)."""
    X, T = [], []
    for _ in range(batch):
        cb = cubes[rng.choice(len(cubes), p=probs)]
        step = max(1, int(round(0.1 / abs(cb.dv))))
        offs = np.arange(-kk, kk + 1) * step
        crop = min(crop_max, cb.S)
        c = int(np.clip(rng.choice(cb.C, p=cb.ch_w), step, cb.C - 1 - step))
        y0, x0 = int(rng.integers(0, cb.S - crop + 1)), int(rng.integers(0, cb.S - crop + 1))
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        for cen in ((c - step, c, c + step) if oc == 1 else (c,)):
            nb = np.clip(cen + offs, 0, cb.C - 1)
            lo, hi = float(cb.lo[cen]), float(cb.hi[cen])
            span = hi - lo if hi > lo else 1.0
            X.append((cb.d[nb][(slice(None),) + sl].astype(np.float32) - lo) / span)
            T.append((cb.c[[cen] if oc == 1 else nb][(slice(None),) + sl].astype(np.float32) - lo) / span)
    h = max(x.shape[-1] for x in X)
    pad = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, h - a.shape[-2]), (0, h - a.shape[-1])])
    t = lambda a: torch.from_numpy(np.stack([pad(x) for x in a])).float().to(dev)
    return t(X), t(T)


def build_legacy(path, fresh=False):
    """Load a shipped checkpoint with its own config. Returns (net, kk, oc). fresh=True keeps the architecture and discards the weights
    (same net, random init), so fine-tuning and training from scratch can be compared on one architecture."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    kk, oc = int(ck.get("n_neighbors", 0)), int(ck.get("out_channels", 1))
    base = int(ck["base_channels"])
    net = UNet(in_channels=int(ck.get("in_channels", 2 * kk + 1)), out_channels=oc, base_channels=base,
               channel_multipliers=list(ck["channel_multipliers"]), time_emb_dim=128, num_res_blocks=2,
               groups=math.gcd(8, base), beam_dim=int(ck.get("beam_dim", 0))).to(dev)
    if not fresh:
        net.load_state_dict(ck["model_state_dict"], strict=True)
    return net, kk, oc


def loss_legacy(pred, tgt, oc, w_d1=1.0, w_d2=1.0):
    if oc == 1:
        p, c = pred[:, 0].view(-1, 3, *pred.shape[-2:]), tgt[:, 0].view(-1, 3, *tgt.shape[-2:])
        l_pix = (p - c).abs().mean()
        l_d1 = ((p[:, 2] - p[:, 0]) - (c[:, 2] - c[:, 0])).abs().mean()
        l_d2 = ((p[:, 2] - 2 * p[:, 1] + p[:, 0]) - (c[:, 2] - 2 * c[:, 1] + c[:, 0])).abs().mean()
    else:                                         # the model predicts a stack: adjacent output channels are adjacent velocities
        l_pix = (pred - tgt).abs().mean()
        l_d1 = ((pred[:, 2:] - pred[:, :-2]) - (tgt[:, 2:] - tgt[:, :-2])).abs().mean()
        l_d2 = ((pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]) - (tgt[:, 2:] - 2 * tgt[:, 1:-1] + tgt[:, :-2])).abs().mean()
    return l_pix + w_d1 * l_d1 + w_d2 * l_d2, (l_pix.item(), l_d1.item(), l_d2.item())


def denoise_legacy(net, dirty, dv, kk, oc, bs=8):
    """nb16's path for these checkpoints: window from the full cube, min-max from the centre channel, centre of the output."""
    C, n, _ = dirty.shape
    step = max(1, int(round(0.1 / abs(dv))))
    offs = np.arange(-kk, kk + 1) * step
    pad = (-n) % 8
    bsz = bs if n <= 320 else max(1, bs // 2)
    out = np.empty_like(dirty)
    net.eval()
    with torch.no_grad():
        for st in range(0, C, bsz):
            idx = np.arange(st, min(st + bsz, C))
            nb = np.clip(idx[:, None] + offs[None, :], 0, C - 1)
            ref = dirty[idx].reshape(len(idx), -1)
            lo, hi = ref.min(1), ref.max(1)
            span = np.where(hi > lo, hi - lo, 1.0)
            x = torch.from_numpy(((dirty[nb] - lo[:, None, None, None]) / span[:, None, None, None]).astype(np.float32)).to(dev)
            if pad:
                x = Fn.pad(x, (0, pad, 0, pad), mode="replicate")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                p = net(x, torch.zeros(len(idx), dtype=torch.long, device=dev), None)
            cen = p.float()[:, 0 if oc == 1 else kk, :n, :n].cpu().numpy()
            out[idx] = cen * span[:, None, None] + lo[:, None, None]
    return out


def _enable_fp32_stream():
    """The U-Net's residual stream is not normalised (out = h + shortcut(x), models/unet.py), so it grows with training and
    then overflows fp16 in whichever conv consumes it raw: unified-wiener-real v1 (K=3, 15 inputs) was healthy for six rounds
    (L1 0.0139) and then went non-finite at step 2013 on ordinary inputs (-1..3); its saved weights are finite and a forward pass
    on ordinary data reaches 1.27e5 / 9.9e4 in dec_blocks.2.0 / .3.0 .shortcut (fp16 max 65504). Fix for NEW runs only (changes
    nothing for loaded checkpoints): the layers that read the raw stream (1x1 shortcuts, strided / transposed down- and
    up-sampling convs) and the residual add run in fp32, the stream stays fp32, and everything fed by GroupNorm stays fp16."""
    from src.models import unet as U

    def rb_forward(self, x, t_emb):
        h = self.conv1(Fn.silu(self.norm1(x)))
        h = h + self.time_emb_proj(t_emb)[:, :, None, None]
        h = self.conv2(Fn.silu(self.norm2(h)))
        if isinstance(self.shortcut, nn.Identity):
            sc = x.float()
        else:
            with torch.autocast(device_type="cuda", enabled=False):
                sc = self.shortcut(x.float())
        return h.float() + sc

    def conv_fp32(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            return self.conv(x.float())

    U.ResidualBlock.forward = rb_forward
    U.Downsample.forward = conv_fp32
    U.Upsample.forward = conv_fp32


def build_net(arm, base):
    cin = (2 if arm == "wiener" else 1) * (2 * K + 1) + 1
    net = UNet(in_channels=cin, out_channels=1, base_channels=base, channel_multipliers=(1, 2, 4, 8), time_emb_dim=128,
               num_res_blocks=2, groups=math.gcd(8, base), beam_dim=0).to(dev)
    nn.init.zeros_(net.conv_out.weight); nn.init.zeros_(net.conv_out.bias)        # step 0 == the base (Wiener or dirty)
    return net


def make_m1_batch(cubes, probs, n, crop_max, band, rng):
    """`n` crops, each with ALL 2*band+1 channels around a centre (cube's own 0.1 km/s step), so the line centre (M1) of the prediction can be
    compared with clean's inside the band. Same inputs as make_batch (arm wiener)."""
    X, B0, Cc = [], [], []
    for _ in range(n):
        cb = cubes[rng.choice(len(cubes), p=probs)]
        offs, step = offsets(cb.dv)
        crop = min(crop_max, cb.S)
        lo_c, hi_c = band * step, cb.C - 1 - band * step
        c = int(np.clip(rng.choice(cb.C, p=cb.ch_w), lo_c, hi_c))
        y0, x0 = int(rng.integers(0, cb.S - crop + 1)), int(rng.integers(0, cb.S - crop + 1))
        sl = (slice(None), slice(y0, y0 + crop), slice(x0, x0 + crop))
        for cen in c + np.arange(-band, band + 1) * step:
            nb = np.clip(cen + offs, 0, cb.C - 1)
            X.append(np.concatenate([cb.d[nb][sl], cb.w[nb][sl], np.full((1, crop, crop), cb.sig, np.float16)]).astype(np.float32))
            B0.append(cb.w[cen][sl[1:]].astype(np.float32))
            Cc.append(cb.c[cen][sl[1:]].astype(np.float32))
    h = max(x.shape[-1] for x in X)
    pad = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, h - a.shape[-2]), (0, h - a.shape[-1])])
    t = lambda a: torch.from_numpy(np.stack([pad(x) for x in a])).float().to(dev)
    x = t(X)
    return (x.clamp_(-CLIP_IN, CLIP_IN) if CLIP_IN else x), t(B0), t(Cc)


def loss_m1(corr, b0, cc, n, band, thr=0.05):
    """Line-centre error, the quantity the wiggle is made of (rc_r correlates M1 minus the axisymmetric rotation curve with clean's). Intensity-weighted
    centroid over the band (positive part) for prediction and clean, L1 in channel units, on pixels where clean has a line peaking inside the band
    (not within 2 channels of its edge) and brighter than `thr` of the normalised peak. Invented structure on a disc without a wiggle is an M1 error
    too, so this penalises it directly."""
    nb = 2 * band + 1
    p = (b0 + corr[:, 0]).view(n, nb, *b0.shape[-2:]).clamp(min=0)
    c = cc.view(n, nb, *cc.shape[-2:]).clamp(min=0)
    v = (torch.arange(nb, device=p.device, dtype=p.dtype) - band).view(1, nb, 1, 1)
    m1p = (p * v).sum(1) / (p.sum(1) + 1e-6)
    m1c = (c * v).sum(1) / (c.sum(1) + 1e-6)
    pk = c.argmax(1)
    mask = ((c.amax(1) > thr) & (pk >= 2) & (pk <= nb - 3)).float()
    return ((m1p - m1c).abs() * mask).sum() / mask.sum().clamp(min=1.0), float(mask.mean().item())


def loss_amp(corr, b0, cc, n, band, thr=0.05, box=31):
    """Candidate D: wiggle-amplitude term. The model keeps only 60 to 70 percent of a wiggle's amplitude (9074; an15 23 of 93 m/s), which an
    L1 loss favours under uncertainty. Same crops and line centres as loss_m1; the smooth part of each M1 map (masked `box` px mean) is removed and the
    rms of what is left is matched to clean's: |rms_pred / rms_clean - 1|, on crops where at least 20 percent of pixels carry a line. Symmetric, so an
    over-strong (invented) ripple costs as much as a lost one."""
    nb = 2 * band + 1
    p = (b0 + corr[:, 0]).view(n, nb, *b0.shape[-2:]).clamp(min=0)
    c = cc.view(n, nb, *cc.shape[-2:]).clamp(min=0)
    v = (torch.arange(nb, device=p.device, dtype=p.dtype) - band).view(1, nb, 1, 1)
    m1p = (p * v).sum(1) / (p.sum(1) + 1e-6)
    m1c = (c * v).sum(1) / (c.sum(1) + 1e-6)
    pk = c.argmax(1)
    mask = ((c.amax(1) > thr) & (pk >= 2) & (pk <= nb - 3)).float()
    pool = lambda a: Fn.avg_pool2d(a[:, None], box, 1, box // 2, count_include_pad=False)[:, 0]
    sm = lambda m: pool(m * mask) / pool(mask).clamp(min=1e-3)
    npx = mask.sum((1, 2)).clamp(min=1.0)
    rms = lambda m: torch.sqrt((((m - sm(m)) * mask) ** 2).sum((1, 2)) / npx + 1e-8)
    ok = mask.mean((1, 2)) > 0.2
    if not ok.any():
        return torch.zeros((), device=p.device)
    return ((rms(m1p) / rms(m1c) - 1.0).abs())[ok].mean()


def loss_fn(corr, b0, cc, w_d1=1.0, w_d2=1.0):
    p = (b0 + corr[:, 0]).view(-1, 3, *b0.shape[-2:])
    c = cc.view(-1, 3, *cc.shape[-2:])
    l_pix = (p - c).abs().mean()
    l_d1 = ((p[:, 2] - p[:, 0]) - (c[:, 2] - c[:, 0])).abs().mean()
    l_d2 = ((p[:, 2] - 2 * p[:, 1] + p[:, 0]) - (c[:, 2] - 2 * c[:, 1] + c[:, 0])).abs().mean()
    return l_pix + w_d1 * l_d1 + w_d2 * l_d2, (l_pix.item(), l_d1.item(), l_d2.item())


# ------------------------------------------------------------------ inference
def denoise(net, arm, dirty, w, dv, bs=8):
    """Whole cube at its working size: `dirty` and `w` (the native-grid Wiener, resampled) at the same size. Returns pred."""
    C, n, _ = dirty.shape
    offs, _ = offsets(dv)
    s = scale_of(dirty, w)
    sig = noise_level(dirty, w) / s
    base = w if arm == "wiener" else dirty
    pad = (-n) % 8
    bsz = bs if n <= 320 else max(1, bs // 2)
    out = np.empty_like(w)
    net.eval()
    with torch.no_grad():
        for st in range(0, C, bsz):
            idx = np.arange(st, min(st + bsz, C))
            nb = np.clip(idx[:, None] + offs[None, :], 0, C - 1)
            parts = [dirty[nb] / s] + ([w[nb] / s] if arm == "wiener" else []) + [np.full((len(idx), 1, n, n), sig, np.float32)]
            x = torch.from_numpy(np.concatenate(parts, 1).astype(np.float32)).to(dev)
            if CLIP_IN:
                x = x.clamp_(-CLIP_IN, CLIP_IN)
            if pad:
                x = Fn.pad(x, (0, pad, 0, pad), mode="replicate")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                corr = net(x, torch.zeros(len(idx), dtype=torch.long, device=dev), None)
            out[idx] = base[idx] + corr.float()[:, 0, :n, :n].cpu().numpy() * s
    if GATE and arm == "wiener":
        a_ = gate_alpha(1.0 / max(sig, 1e-30))               # snr_est = scale / noise level = 1 / sig, the network's own noise-map input
        out = (a_ * out + (1.0 - a_) * w).astype(np.float32)
    return out


def prior_wiener(d, p):
    """Noise-only regime (A = I): a Wiener filter of `d` whose signal power is the radially averaged power of `p` (a first pass's output) and whose noise
    power is that of d - p, per velocity frequency. The blind Wiener estimates the signal power as the dirty cube's minus a noise model, which is poor at low
    SNR (9074: blind 0.16, oracle 0.38; oracle_limit.py); the network starts from that estimate."""
    C, n, _ = d.shape
    kr = np.hypot(np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None]); nb = 120
    rb = np.minimum((kr / kr.max() * nb).astype(int), nb - 1); cnt = np.maximum(np.bincount(rb.ravel(), minlength=nb), 1)
    D = sfft.fftn(d.astype(np.float32), workers=-1); P = sfft.fftn(p.astype(np.float32), workers=-1)
    for i in range(C):
        Ps = ws._radial_avg(np.abs(P[i]) ** 2, rb, cnt, nb); Pn = ws._radial_avg(np.abs(D[i] - P[i]) ** 2, rb, cnt, nb)
        D[i] *= (Ps / (Ps + Pn + 1e-30)).astype(np.float32)
    del P
    return sfft.ifftn(D, workers=-1).real.astype(np.float32)


def infer(model, arm, dS, w, dv, beam_px, ai=True):
    """The model on the cube as given, or (INFER_BEAM > 0 and the beam wider than 1.2 x INFER_BEAM px) on a coarser grid where the beam spans INFER_BEAM px.
    Why: the same line cubes lose wiggle from 256 to 600 px (beam ~9 -> ~20 px; run_0026 0.90 -> 0.66, below Wiener's 0.85), while Speedie at 1024 px
    (beam 12 x 9 px) is fine, so the network works at a beam scale, not an image size. INFER_MODE "corr": only the network's correction is computed on the
    coarse grid and upsampled onto the full-resolution Wiener; "full": the coarse output itself is upsampled."""
    f = (lambda d_, w_: model.denoise(arm, d_, w_, dv)) if isinstance(model, Ensemble) else (lambda d_, w_: denoise(model, arm, d_, w_, dv))
    S = dS.shape[-1]
    if not INFER_BEAM or not ai or beam_px <= 1.2 * INFER_BEAM:   # ai: only the noise-only regime (A = I); beam deconvolution cubes keep the cube as given (combo-a / b: analytic -0.05, SG v2 M1/M2 -6/-8)
        return f(dS, w)
    s0 = max(64, int(round(S * INFER_BEAM / beam_px / 8)) * 8)
    print(f"  infer: beam {beam_px:.1f} px > 1.2 x {INFER_BEAM:g}: network on a {s0} px grid (image {S} px, mode {INFER_MODE})", flush=True)
    d0, w0 = resample(dS, s0), resample(w, s0)
    p0 = f(d0, w0)
    if INFER_MODE == "corr":
        return (w + resample(p0 - w0, S)).astype(np.float32)
    return resample(p0, S)


class Ensemble:
    """Mean of several checkpoints' outputs, optionally over the 4 flips of the sky plane (test-time augmentation). Flipping is a
    symmetry of the task (the model is applied per pixel window; noise is isotropic in distribution), so each pass sees the same
    data with independent network errors. Averaging can only reduce the variance part of the error."""

    def __init__(self, nets, tta):
        self.nets, self.flips = nets, ([(), (-1,), (-2,), (-2, -1)] if tta else [()])

    def denoise(self, arm, dirty, w, dv):
        acc, n = None, 0
        for net in self.nets:
            for ax in self.flips:
                f = (lambda a: np.ascontiguousarray(np.flip(a, ax))) if ax else (lambda a: a)
                out = f(denoise(net, arm, f(dirty), f(w), dv))
                acc = out.astype(np.float64) if acc is None else acc + out
                n += 1
        return (acc / n).astype(np.float32)


# ------------------------------------------------------------------ evaluation cases (native arrays; resampled per size)
def eval_cases(args):
    """Yields (name, kind, clean, dirty, velax_kms, au_native, incl, beam_au, beam_vec, noise_chans, score_slice)."""
    from src.data.fits_cube_dataset import beam_features_of
    _, _, hold = split_cubes(data_dir=LE_DIR, n_holdout=3, val_fraction=0.2, seed=42)
    for ho in ([] if args.n_line_eval < 0 else (hold[: args.n_line_eval] if args.n_line_eval else hold)):
        hdr = fits.getheader(ho["clean"])
        cl, dr = fits.getdata(ho["clean"]).astype(np.float32), fits.getdata(ho["dirty"]).astype(np.float32)
        pdir = os.path.dirname(ho["clean"])
        dist = ce._para_value(pdir, "distance (pc)") or float(hdr.get("DIST_PC", 140.0))
        incl = ce._para_value(pdir, "RT: imin")
        velax = hdr["CRVAL3"] + (np.arange(cl.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
        yield (ho["folder"], "line", continuum_sub(cl), continuum_sub(dr), velax, abs(hdr["CDELT1"]) * 3600 * dist, incl,
               float(hdr["BMAJ"]) * 3600 * dist, beam_features_of(hdr), None, slice(None))
    for folder in (sorted(glob.glob(os.path.join(args.sph_dir, "run_young_*"))) if args.sph_dir and args.eval_sph else []):
        meta = json.load(open(os.path.join(folder, "view.json")))     # held-out WHOLE simulations (Young et al. 2026), never trained on
        if meta["split"] != args.eval_sph_split:                        # "test" (and TEST2) by default; "test3" = the fresh set
            continue
        base = os.path.join(folder, os.path.basename(folder))
        h = fits.getheader(base + "_clean.fits")
        cl, dr = fits.getdata(base + "_clean.fits").astype(np.float32), fits.getdata(base + "_dirty.fits").astype(np.float32)
        velax = h["CRVAL3"] + np.arange(cl.shape[0]) * h["CDELT3"]
        yield (os.path.basename(folder), "line", cl, dr, velax, float(h["AU_PX"]), float(h["INCL_DEG"]), meta["bmaj_px"] * float(h["AU_PX"]),
               np.zeros(4, np.float32), None, slice(None))
    for name in [x for x in args.eval_speedie.split(",") if x]:
        # Speedie et al. 2024 simulated AB Aur GI disk (and its Keplerian twin) through syndisk to ALMA beam, channels and noise. The beam is already in
        # clean (noise-only regime, kind "line"); contsub is built in (the model has none), so no continuum step. Native pixels: beam 12 x 9 px.
        hdr = fits.getheader(os.path.join(args.speedie_dir, f"{name}_clean.fits"))
        cl = fits.getdata(os.path.join(args.speedie_dir, f"{name}_clean.fits")).astype(np.float32)
        dr = fits.getdata(os.path.join(args.speedie_dir, f"{name}_dirty.fits")).astype(np.float32)
        velax = (hdr["CRVAL3"] + (np.arange(cl.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) / 1000.0
        dist, incl = 155.9, 23.2           # the release's own MCFOST parameter file; the first prep kernel wrote 162.9 pc / 23.5 deg into the header
        yield (f"speedie_{name}", "line", cl, dr, velax, abs(hdr["CDELT1"]) * 3600 * dist, incl,
               float(hdr["BMAJ"]) * 3600 * dist, np.zeros(4, np.float32), None, slice(None))
    # SNR ladder: the Speedie cubes with their own measured noise (dirty - clean, so its spatial and Hanning channel
    # correlation are kept) rescaled so that peak(clean) / rms(noise) = each target. Truth, geometry and beam are unchanged: the same disc seen
    # at exoALMA's SNR range (fiducial 13-50, high-resolution 3-10). Native pixels, like the Speedie cases.
    for name in [x for x in args.ladder_base.split(",") if x and args.eval_ladder]:
        cl = fits.getdata(os.path.join(args.speedie_dir, f"{name}_clean.fits")).astype(np.float32)
        nz = fits.getdata(os.path.join(args.speedie_dir, f"{name}_dirty.fits")).astype(np.float32) - cl
        hdr = fits.getheader(os.path.join(args.speedie_dir, f"{name}_clean.fits"))
        velax = (hdr["CRVAL3"] + (np.arange(cl.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) / 1000.0
        for snr in [float(v) for v in args.eval_ladder.split(",")]:
            f = float(cl.max()) / snr / max(float(nz.std()), 1e-30)
            for k in range(max(1, args.ladder_seeds)):     # error bars: the same noise shifted relative to the disc (statistics kept exactly); k = 0 as before
                nzk = nz if k == 0 else np.roll(nz, tuple(int(x) for x in np.random.default_rng(2000 + k).integers(nz.shape[-1] // 4, 3 * nz.shape[-1] // 4, 2)), axis=(1, 2))
                yield (f"speedie_ladder_{name}_snr{snr:g}" + (f"_s{k}" if k else ""), "line", cl, cl + f * nzk, velax, abs(hdr["CDELT1"]) * 3600 * 155.9, 23.2,
                       float(hdr["BMAJ"]) * 3600 * 155.9, np.zeros(4, np.float32), None, slice(None))
    for name in [x for x in args.eval_sg.split(",") if x]:
        if name == "sgv2":
            p = os.path.join(SG_DIR, "kinematic_data_v2")
            blk = np.arange(180, 421)
            with fits.open(f"{p}/clean_sg.fits", memmap=True) as h:
                hdr, cl = h[0].header, h[0].data[blk].astype(np.float32)
            with fits.open(f"{p}/dirty_sg.fits", memmap=True) as h:
                dr = h[0].data[blk].astype(np.float32)
                lf = h[0].data[np.r_[0:60, 541:601]].astype(np.float32)
            static = lf.mean(0)
            dr = np.concatenate([dr - static[None], lf - static[None]])      # line-free planes ride along for the noise estimate
            cl = np.concatenate([cl - cl[:8].mean(0)[None], np.zeros_like(lf)])
            velax = hdr["CRVAL3"] + (np.r_[blk, np.zeros(len(lf))] + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
            noise_ch = np.arange(len(blk), len(blk) + len(lf))
            yield ("sgv2", "hydro", cl, dr, velax, abs(hdr["CDELT1"]) * 3600 * 140.0, 30.0,
                   float(hdr["BMAJ"]) * 3600 * 140.0, beam_features_of(hdr), noise_ch, slice(60, 181))
        else:
            d = ws.load(name)
            hdr_beam = np.zeros(4, np.float32)
            fam = "analytic" if name.startswith("an") else "hydro"
            yield (name, fam, d["clean"], d["dirty"], d["velax"], d["au"], d["incl"], d["beam_au"], hdr_beam, None, slice(None))
            if args.eval_ai:
                # the ALMA-matched version: dirty = clean + noise with the pair's own measured noise spectrum (no extra beam, no deconvolution)
                cl, n = d["clean"].astype(np.float32), d["clean"].shape[-1]
                conv = sfft.ifft2(sfft.fft2(cl, workers=-1) * beam_otf(n)[None], workers=-1).real.astype(np.float32)
                amp = pair_noise_amp(cl, d["dirty"], conv)
                white = np.random.default_rng(777).standard_normal(cl.shape, dtype=np.float32)
                noise = sfft.irfft2(sfft.rfft2(white, workers=-1) * amp[None], s=(n, n), workers=-1).astype(np.float32)
                yield (name + "_ai", fam + "_ai", cl, cl + noise, d["velax"], d["au"], d["incl"], d["beam_au"], hdr_beam, None, slice(None))
            if args.sg_ladder and name in [x for x in args.sg_ladder_base.split(",") if x]:
                # hydro SNR ladder: the same disc with its own measured noise rescaled, so peak SNR AFTER the beam = each target.
                # Both regimes: deconvolution (dirty = beam (*) clean + f * the pair's own noise) and noise-only (_ai: clean + f * noise of the pair's spectrum).
                # Answers whether a disc the model fails at its native SNR (9074, about 11) is recovered at ALMA-like SNR, and, run on a TRAINING disc (9025),
                # whether a failure is the noise or a generalisation gap. A training disc's scores are a diagnostic, never a held-out result.
                cl, n = d["clean"].astype(np.float32), d["clean"].shape[-1]
                conv = sfft.ifft2(sfft.fft2(cl, workers=-1) * beam_otf(n)[None], workers=-1).real.astype(np.float32)
                nz_dec = (d["dirty"] - conv).astype(np.float32)
                amp = pair_noise_amp(cl, d["dirty"], conv)
                white = np.random.default_rng(778).standard_normal(cl.shape, dtype=np.float32)
                nz_ai = sfft.irfft2(sfft.rfft2(white, workers=-1) * amp[None], s=(n, n), workers=-1).astype(np.float32)
                for snr in [float(v) for v in args.sg_ladder.split(",") if v]:
                    f_dec = float(conv.max()) / snr / max(float(nz_dec.std()), 1e-30)
                    f_ai = float(cl.max()) / snr / max(float(nz_ai.std()), 1e-30)
                    yield (f"{name}_ladder_snr{snr:g}", fam, cl, conv + f_dec * nz_dec, d["velax"], d["au"], d["incl"], d["beam_au"], hdr_beam, None, slice(None))
                    yield (f"{name}_ai_ladder_snr{snr:g}", fam + "_ai", cl, cl + f_ai * nz_ai, d["velax"], d["au"], d["incl"], d["beam_au"], hdr_beam, None, slice(None))


def central_flux_fix(pred, dirty, beam_px, r_in=2.0, r_out=4.0):
    """Candidate A: the model over-brightens a compact bright centre by 30 to 40 percent while
    the noisy cube's flux there is unbiased. Within r_in beams of the disc centre, rescale each pixel's denoised spectrum so its line flux matches the
    beam-smoothed flux of the input cube; taper to no change at r_out beams. Only amplitudes change, so the line centre (M1) and the wiggle are untouched.
    Centre: the peak of the beam-smoothed denoised M0 within the central half of the image. Returns (corrected cube, info dict)."""
    from scipy import ndimage
    tot = pred.reshape(pred.shape[0], -1).sum(1)
    line = tot > 1e-3 * tot.max()
    s = beam_px / 2.355
    mp = ndimage.gaussian_filter(pred[line].sum(0).astype(np.float64), s)
    md = ndimage.gaussian_filter(dirty[line].sum(0).astype(np.float64), s)
    n = pred.shape[-1]; q = n // 4
    cy, cx = np.unravel_index(np.argmax(mp[q:n - q, q:n - q]), (n - 2 * q, n - 2 * q)); cy += q; cx += q
    yy, xx = np.indices((n, n)); rb = np.hypot(yy - cy, xx - cx) / beam_px
    w = np.clip((r_out - rb) / (r_out - r_in), 0.0, 1.0); w = 0.5 - 0.5 * np.cos(np.pi * w)      # 1 inside r_in, smooth to 0 at r_out
    ratio = np.where(mp > 0.05 * mp.max(), md / np.maximum(mp, 1e-30), 1.0)
    ratio = np.clip(ratio, 0.5, 1.5)
    f = (1.0 + w * (ratio - 1.0)).astype(np.float32)
    return pred * f[None], {"centre": [int(cy), int(cx)], "factor_at_centre": float(f[cy, cx]), "mean_factor_r_in": float(f[rb <= r_in].mean())}


_NATIVE = {}


def score_case(model, arm, case, S, figdir):
    name, kind, cl, dr, velax, au, incl, beam_au, beam_vec, noise_ch, sl = case
    n = cl.shape[-1]
    key = (name, "w")
    if key not in _NATIVE:                         # the native Wiener (and noise-free beam oracle) once per case, not per size
        # SG v2's dirty carries the recovered beam on top of clean (that is how the beam was recovered), like sg_synth / analytic
        wn = native_wiener(dr, kind, noise_ch)
        if noise_ch is not None:                   # SG v2: the line-free planes only fed the noise estimate
            keep = noise_ch[0]
            cl, dr, wn, velax = cl[:keep], dr[:keep], wn[:keep], velax[:keep]
        oracle = None if is_ai(kind) else sfft.ifft2(sfft.fft2(cl, workers=-1) * beam_otf(n)[None], workers=-1).real.astype(np.float32)
        _NATIVE.clear()
        _NATIVE[key] = (cl, dr, wn, velax, oracle)
    cl, dr, wn, velax, oracle = _NATIVE[key]
    cS, dS, w = resample(cl, S), resample(dr, S), resample(wn, S)
    dv = float(velax[1] - velax[0])
    if arm == "legacy":
        net, kk, oc = model
        pred = denoise_legacy(net, dS, dv, kk, oc)
    else:
        pred = infer(model, arm, dS, w, dv, beam_au / (au * n / S), is_ai(kind))
        if WIENER2 and is_ai(kind) and arm == "wiener":   # second pass from a Wiener built on the first pass (prior_wiener); the "wiener" row stays the blind one
            pred = infer(model, arm, dS, prior_wiener(dS, pred), dv, beam_au / (au * n / S), True)
    if CENTRAL_FIX:                                # candidate A: central flux matched to the input cube within 2 beams of the centre
        pred, _cf = central_flux_fix(pred, dS, beam_au / (au * n / S))
    cS, dS, pred, w, v = cS[sl], dS[sl], pred[sl], w[sl], velax[sl]
    if SAVE_PRED[0] and name in SAVE_PRED[1]:     # float16 central +-40 channels (+-4 km/s at 0.1 km/s) and every method's quadratic M1
        c0, h = cS.shape[0] // 2, 40
        np.savez_compressed(os.path.join(SAVE_PRED[0], f"{name}_{S}.npz"), velax=v, au_px=au * n / S, incl=incl if incl is not None else np.nan,
                            beam_au=beam_au, **{f"m1_{k}": ws.quad_m1(c, v).astype(np.float32) for k, c in (("clean", cS), ("dirty", dS), ("wiener", w), ("model", pred))},
                            m0_clean=cS.sum(0).astype(np.float32), chan_lo=max(0, c0 - h),
                            **{f"ch_{k}": c[max(0, c0 - h):c0 + h + 1].astype(np.float16) for k, c in (("clean", cS), ("dirty", dS), ("model", pred))})
    # 1) PSNR / SSIM / M0 / M1 / M2 / artifacts: checkpoint_eval, the nb16 definitions
    case_ce = ce.Case(f"{name}@{S}", "line_emission" if is_ai(kind) else "sg", cS.copy(), dS.copy(), v * 1000.0,
                      au * n / S, beam_vec, incl, {})
    prep = ce.prepare(case_ce)
    res = {"model": ce.score(prep, pred)[0], "wiener": ce.score(prep, w)[0]}
    res["dirty_psnr"] = prep.psnr_dirty
    # blind peak-SNR estimate, the quantity a gate would switch on: Wiener-normalisation scale over the rms of what the Wiener removed (same
    # numbers the network gets as its noise-map input); logged for every case so its relation to the true peak SNR can be read off.
    res["snr_est"] = float(scale_of(dS, w) / max(noise_level(dS, w), 1e-30))
    res["gate_a"] = gate_alpha(res["snr_est"]) if GATE else 1.0
    # 2) the wiggle: rotation-curve residual, coherence, line-centre error, Terry spine
    d = {"clean": cS, "dirty": dS, "velax": v, "au": au * n / S, "incl": incl if incl is not None else 30.0,
         "crop": slice(None), "beam_au": beam_au}
    ws.METHODS["model"] = lambda _d: pred
    ws.METHODS["wiener"] = lambda _d: w
    methods = ["dirty", "wiener", "model"]
    if oracle is not None:                         # noise-free clean (*) beam at this size: the resolution-only reference
        ws.METHODS["beam_oracle"] = lambda _d: resample(oracle, S)[sl]
        methods.append("beam_oracle")
    for a_ in GATE_SWEEP:                          # blend with the Wiener output; a = 1 is the plain model, a = 0 the plain Wiener
        if 0.0 < a_ < 1.0:
            ws.METHODS[f"mix{a_:g}"] = lambda _d, a_=a_: a_ * pred + (1 - a_) * w
            methods.append(f"mix{a_:g}")
    sc = ws.score_cube(f"{name}_{S}", d, methods, figdir)
    res["wiggle"] = sc["methods"]
    if GATE_SWEEP:
        res["mix"] = {}
        for a_ in GATE_SWEEP:
            if 0.0 < a_ < 1.0:
                m_ = ce.score(prep, a_ * pred + (1 - a_) * w)[0]
                res["mix"][f"{a_:g}"] = {k: float(m_[k]) for k in ("psnr", "M0", "M1", "M2") if k in m_}
    for k in ("model", "wiener"):
        res[k] = {kk: (float(vv) if isinstance(vv, (int, float, np.floating, np.integer, bool)) else vv)
                  for kk, vv in res[k].items() if not isinstance(vv, np.ndarray)}
    return res


# ------------------------------------------------------------------ main
def run_eval(model, arm, args, sizes, res_path, figdir, only=None):
    os.makedirs(figdir, exist_ok=True)
    allres = json.load(open(res_path)) if os.path.exists(res_path) else {}
    last = 300.0                                         # seconds the previous case took; first case assumed 5 min
    for case in eval_cases(args):
        if only is not None and case[0] not in only:
            continue
        for S in ([case[2].shape[-1]] if case[0].startswith("speedie_") else sizes):
            key = f"{case[0]}@{S}"
            if key in allres and "failed" not in allres[key]:
                continue
            if time.time() + 1.5 * last > DEADLINE:          # save and stop before the session limit; the next session resumes here
                print(f"DEADLINE: evaluation stopped before {key} ({(DEADLINE - time.time()) / 60:.0f} min left); RESUME_NEEDED", flush=True)
                return None
            te = time.time()
            try:
                r = score_case(model, arm, case, S, figdir)
            except Exception:
                import traceback
                r = {"failed": traceback.format_exc()[-1500:]}
                print(f"{key}: FAILED\n{r['failed']}", flush=True)
            allres[key] = r
            json.dump(allres, open(res_path, "w"), indent=1, default=float)
            if "failed" not in r:
                m, w, g = r["model"], r["wiener"], r["wiggle"]
                print(f"{key:30s} PSNR dirty {r['dirty_psnr']:.2f} wiener {w['psnr']:.2f} MODEL {m['psnr']:.2f} | "
                      f"M0/M1/M2 % wiener {w['M0']:.0f}/{w['M1']:.0f}/{w['M2']:.0f} MODEL {m['M0']:.0f}/{m['M1']:.0f}/{m['M2']:.0f} | "
                      f"rc_r dirty {g['dirty']['rc_r']:.3f} wiener {g['wiener']['rc_r']:.3f} MODEL {g['model']['rc_r']:.3f} | "
                      f"dv_med {g['dirty']['dv_med_ms']:.0f}/{g['wiener']['dv_med_ms']:.0f}/{g['model']['dv_med_ms']:.0f} m/s | "
                      f"{time.time() - te:.0f}s", flush=True)
            last = time.time() - te
        del case
    return allres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="wiener", choices=["wiener", "direct"])
    ap.add_argument("--init", default="", help="fine-tune this checkpoint in its own input/output format (arm 'legacy')")
    ap.add_argument("--fresh", action="store_true", help="with --init: use the checkpoint's architecture only, random weights")
    ap.add_argument("--eval-only", action="store_true", help="score --init as it is (all cases, all sizes): no data loading, no training")
    ap.add_argument("--eval-ckpt", default="", help="with --eval-only: a train_unified checkpoint of the wiener/direct arm (window K and base read from it)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=3, help="spectral window half-width at 0.1 km/s sampling (3: +-0.3 km/s; 15: +-1.5 km/s)")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--hydro", default=",".join(HYDRO_TRAIN))
    ap.add_argument("--analytic", default=",".join(ANALYTIC_TRAIN))
    ap.add_argument("--p-ai", type=float, default=0.5, help="probability a simulation is drawn as the noise-only (ALMA-matched) task")
    ap.add_argument("--n-line", type=int, default=0, help="0 = all 9 training line-emission cubes")
    ap.add_argument("--n-line-eval", type=int, default=0, help="0 = all 5 holdouts")
    ap.add_argument("--eval-sg", default="9074,sgv2,an07,an08")
    ap.add_argument("--no-eval-ai", dest="eval_ai", action="store_false")
    ap.add_argument("--speedie-dir", default="", help="folder with the speedie-prep output ({GI,Kep}_r{0.5,1.5}_{clean,dirty}.fits)")
    ap.add_argument("--eval-speedie", default="", help="e.g. GI_r0.5,Kep_r0.5: scored at NATIVE pixels only (noise-only regime, truth available)")
    ap.add_argument("--before", action="store_true", help="score the starting checkpoint first (subset), same protocol")
    ap.add_argument("--before-cases", default="run_0002_00560_rt_00,run_0026_00005_rt_04,9074,9074_ai,an07,an07_ai,sgv2")
    ap.add_argument("--before-sizes", default="300,600")
    ap.add_argument("--refresh", type=int, default=3, help="cubes redrawn (new noise, new size) per round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/unified/local"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snr", default="", help="peak-SNR range lo,hi drawn log-uniform per redraw (empty: the measured noise level)")
    ap.add_argument("--chan-corr", type=float, default=0.0, help="max a of the [a,1,a] channel kernel on the training noise (0.5 = Hanning)")
    ap.add_argument("--fp32-stream", action="store_true", help="residual stream and its raw consumers in fp32 (fp16 overflow of the unnormalised stream)")
    ap.add_argument("--budget-h", type=float, default=11.0, help="session budget in hours from EXXA_T0 (Kaggle stops GPU sessions at 12 h)")
    ap.add_argument("--eval-reserve-h", type=float, default=1.5, help="time kept free for the evaluation when deciding whether to train another round")
    ap.add_argument("--resume", default="", help="checkpoint of a previous session of this run (full state): continue from its round")
    ap.add_argument("--sph-dir", default="", help="folder of make_sph_cubes.py output (run_young_*); train split trained on, test split scored with --eval-sph")
    ap.add_argument("--sph-max", type=int, default=0, help="at most this many sph folders (0 = all)")
    ap.add_argument("--eval-sph", action="store_true", help="score the held-out sph simulations")
    ap.add_argument("--eval-sph-split", default="test", help="which held-out sph split to score: test (TEST + TEST2) or test3 (the fresh set)")
    ap.add_argument("--central-fix", action="store_true", help="candidate A: central_flux_fix after inference; off = recipe B")
    ap.add_argument("--ladder-seeds", type=int, default=1, help="noise realisations per Speedie ladder SNR (error bars); 1 = as before")
    ap.add_argument("--save-pred", default="", help="folder: save M1 maps and central channels of the --save-pred-cases (npz)")
    ap.add_argument("--save-pred-cases", default="", help="comma list of case names for --save-pred")
    ap.add_argument("--norm", default="wiener", choices=["wiener", "robust"], help="cube normalisation (robust: needed for training SNR down to 3)")
    ap.add_argument("--gate", default="", help="do-no-harm gate lo,hi,floor (e.g. 40,120,0.25); empty = off")
    ap.add_argument("--gate-sweep", default="", help="evaluation: comma list of mix levels a; scores a * model + (1 - a) * Wiener for each")
    ap.add_argument("--tta", action="store_true", help="evaluation: average over the 4 sky-plane flips")
    ap.add_argument("--eval-ladder", default="", help="peak-SNR targets for the Speedie SNR ladder, e.g. 3,6,12,24 (empty: off)")
    ap.add_argument("--ladder-base", default="GI_r0.5,Kep_r0.5")
    ap.add_argument("--sg-ladder", default="", help="peak-SNR targets (after the beam) for a hydro / analytic disc ladder, e.g. 11,20,40,80 (empty: off)")
    ap.add_argument("--sg-ladder-base", default="9074", help="discs from --eval-sg that get the ladder; a training disc (e.g. 9025) is a diagnostic, not a held-out result")
    ap.add_argument("--train-sgv2", action="store_true", help="add SG v2 (Terry setup, q 0.75) to the training set; it is then NOT held out")
    ap.add_argument("--clip-in", type=float, default=0.0, help="clamp the network input to +-this (0 = off)")
    ap.add_argument("--beam-max", type=float, default=1.0, help="max beam broadening factor in the noise-only tasks (1 = off)")
    ap.add_argument("--infer-beam-px", type=float, default=0.0, help="evaluation: run the network on a grid where the beam spans this many px (0 = off)")
    ap.add_argument("--infer-mode", default="corr", choices=["corr", "full"], help="with --infer-beam-px: upsample the correction only, or the whole output")
    ap.add_argument("--wiener2", action="store_true", help="evaluation: second pass from a Wiener built on the first pass's output (prior_wiener)")
    ap.add_argument("--train-beam-px", default="", help="lo,hi: each redraw resamples the cube so its beam spans U(lo, hi) px instead of drawing from --sizes")
    ap.add_argument("--warm", default="", help="start from this wiener-arm checkpoint's weights (K and base from it), fresh optimizer and schedule")
    ap.add_argument("--w-m1", type=float, default=0.0, help="weight of the line-centre (M1) loss, loss_m1 (0 = off)")
    ap.add_argument("--eval-sizes", default="", help="sizes the final evaluation scores (empty: --sizes); training sizes are not changed")
    ap.add_argument("--w-amp", type=float, default=0.0, help="candidate D: weight of the wiggle-amplitude term loss_amp on the M1 crops (0 = off)")
    ap.add_argument("--m1-band", type=int, default=10, help="M1 loss: half-width of the channel band, in 0.1 km/s steps")
    ap.add_argument("--m1-crops", type=int, default=2, help="M1 loss: crops per step (each carries 2 * band + 1 channels)")
    ap.add_argument("--m1-crop", type=int, default=96, help="M1 loss: crop size (px)")
    args = ap.parse_args()
    if args.smoke:
        args.rounds, args.steps, args.batch, args.crop, args.base = 2, 4, 2, 64, 8
        args.sizes, args.hydro, args.analytic, args.n_line, args.n_line_eval = "128,160", "9015", "an01", 1, 1
        args.eval_sg, args.before_sizes, args.before_cases = "an07", "128", "run_0002_00560_rt_00,an07_ai"
    global K, CLIP_IN, NORM, DEADLINE, GATE_SWEEP, GATE, INFER_BEAM, INFER_MODE, WIENER2, SAVE_PRED, CENTRAL_FIX
    K, CLIP_IN, NORM = args.k, args.clip_in, args.norm
    INFER_BEAM, INFER_MODE, WIENER2 = args.infer_beam_px, args.infer_mode, args.wiener2
    GATE = tuple(float(x) for x in args.gate.split(",")) if args.gate else None
    GATE_SWEEP = [float(x) for x in args.gate_sweep.split(",") if x]
    if args.save_pred:
        os.makedirs(args.save_pred, exist_ok=True)
    SAVE_PRED = (args.save_pred, set(x for x in args.save_pred_cases.split(",") if x))
    CENTRAL_FIX = args.central_fix
    DEADLINE = float(os.environ.get("EXXA_T0", time.time())) + args.budget_h * 3600
    if args.fp32_stream:
        _enable_fp32_stream()
    REAL.update(snr=tuple(float(v) for v in args.snr.split(",")) if args.snr else None, chan_a=args.chan_corr, beam_max=args.beam_max,
                beam_px=tuple(float(v) for v in args.train_beam_px.split(",")) if args.train_beam_px else None)
    ens_paths = [p for p in args.eval_ckpt.split(",") if p] if args.eval_ckpt else []
    if args.eval_ckpt:                                        # an arm checkpoint written by this script (comma list = ensemble, same K / arm / base)
        _ck = torch.load(ens_paths[0], map_location="cpu", weights_only=False)
        K, args.arm, args.base, args.eval_only = int(_ck["K"]), _ck["arm"], int(_ck["base"]), True
        CLIP_IN = float((_ck.get("args") or {}).get("clip_in", 0.0))      # a model trained with an input clamp is evaluated with it
        NORM = (_ck.get("args") or {}).get("norm", "wiener")               # and with the normalisation it was trained with
    if args.warm:                                             # fine-tune: architecture from the checkpoint, everything else from the command line
        _wk = torch.load(args.warm, map_location="cpu", weights_only=False)
        K, args.k, args.arm, args.base = int(_wk["K"]), int(_wk["K"]), _wk["arm"], int(_wk["base"])
    legacy = bool(args.init)
    arm = "legacy" if legacy else args.arm
    if args.eval_ckpt:
        legacy = False
    tag = args.tag or (f"eval_{os.path.splitext(os.path.basename(ens_paths[0]))[0]}" if args.eval_ckpt else None) or (f"{'fresh' if args.fresh else ('eval' if args.eval_only else 'ft')}_{os.path.splitext(os.path.basename(args.init))[0]}" if legacy else f"{args.arm}_k{K}")
    sizes = [int(x) for x in args.sizes.split(",")]
    esizes = [int(x) for x in args.eval_sizes.split(",")] if args.eval_sizes else sizes    # --eval-sizes: score at these only (training sizes unchanged)
    os.makedirs(args.out, exist_ok=True)
    print(f"device {dev}  {vars(args)}", flush=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if legacy:
        net, kk, oc = build_legacy(args.init, fresh=args.fresh)
        model = (net, kk, oc)
        print(f"fine-tuning {args.init}: window +-{kk} channels, {oc} output channel(s)", flush=True)
    else:
        net = build_net(args.arm, args.base)
        if args.eval_ckpt:
            net.load_state_dict(_ck["model_state_dict"], strict=True)
            print(f"loaded {ens_paths[0]}: arm {args.arm}, window +-{K}, base {args.base}", flush=True)
        if args.warm:
            net.load_state_dict(_wk["model_state_dict"], strict=True); del _wk
            print(f"warm start from {args.warm}: arm {args.arm}, window +-{K}, base {args.base}", flush=True)
        model = net
        if len(ens_paths) > 1 or args.tta:
            nets = [net]
            for pth in ens_paths[1:]:
                ck = torch.load(pth, map_location="cpu", weights_only=False)
                assert (int(ck["K"]), ck["arm"], int(ck["base"])) == (K, args.arm, args.base), f"{pth}: ensemble members must share K / arm / base"
                assert (ck.get("args") or {}).get("norm", "wiener") == NORM, f"{pth}: ensemble members must share --norm"
                n2 = build_net(args.arm, args.base); n2.load_state_dict(ck["model_state_dict"], strict=True); nets.append(n2)
                print(f"loaded {pth} (ensemble member {len(nets)})", flush=True)
            model = Ensemble(nets, args.tta)
    figdir = os.path.join(args.out, f"figs_{tag}"); os.makedirs(figdir, exist_ok=True)
    if args.before and not args.fresh:
        run_eval(model, arm, args, [int(x) for x in args.before_sizes.split(",")], os.path.join(args.out, f"eval_{tag}_before.json"),
                 figdir + "_before", only=set(args.before_cases.split(",")))
        print("BEFORE-EVAL DONE", flush=True)

    status_path = os.path.join(args.out, f"status_{tag}.json")
    def status(**kw):                                    # what a next session needs to know (the Kaggle script reads it)
        json.dump(dict(tag=tag, **kw), open(status_path, "w"), indent=1)
    if args.eval_only:
        assert legacy or args.eval_ckpt, "--eval-only needs --init or --eval-ckpt"
        done = run_eval(model, arm, args, esizes, os.path.join(args.out, f"eval_{tag}.json"), figdir) is not None
        status(train_complete=True, eval_complete=done)
        print("EVAL DONE" if done else "EVAL INCOMPLETE: RESUME_NEEDED", flush=True)
        return
    t0 = time.time()
    cubes = load_train_cubes(args)
    fams = sorted({c.family for c in cubes})
    probs = np.array([1.0 / (len(fams) * sum(x.family == c.family for x in cubes)) for c in cubes]); probs /= probs.sum()
    print(f"{len(cubes)} training cubes ({', '.join(f'{f}: {sum(c.family == f for c in cubes)}' for f in fams)}), "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.rounds * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05, final_div_factor=30)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    ckpt = os.path.join(args.out, f"unified_{tag}.pth")
    hist, step, n_bad, bad_run, start = [], 0, 0, 0, 0
    if args.resume:                                      # continue exactly where the previous session stopped
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model_state_dict"], strict=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            rng.bit_generator.state = ck["rng_np"]; torch.set_rng_state(ck["rng_torch"])
            if dev == "cuda" and ck.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(ck["rng_cuda"])
        else:
            print("  resume: checkpoint has weights only (older format); optimizer, schedule and random state restart", flush=True)
            for _ in range(int(ck["steps"])):
                sched.step()
        start, step, hist = int(ck["round"]), int(ck["steps"]), list(ck["hist"])
        print(f"RESUMED {args.resume}: round {start}/{args.rounds}, step {step}", flush=True)
    stopped = False
    for rnd in range(start, args.rounds):
        if rnd > start:                                  # time the next round would need, from the rounds of THIS session
            per = (time.time() - t_rounds0) / (rnd - start)
            if time.time() + per + args.eval_reserve_h * 3600 > DEADLINE:
                print(f"DEADLINE: training stopped after round {rnd}/{args.rounds} ({(DEADLINE - time.time()) / 60:.0f} min left, a round takes "
                      f"{per / 60:.1f} min, eval reserve {args.eval_reserve_h:.1f} h); checkpoint {ckpt}; RESUME_NEEDED", flush=True)
                stopped = True
                break
        else:
            t_rounds0 = time.time()
        todo = [c for c in cubes if not c.ready] or list(rng.choice(cubes, size=min(args.refresh, len(cubes)), replace=False))
        tr0 = time.time()
        for c in todo:
            c.redraw(rng, sizes)
        t_draw = time.time() - tr0
        if dev == "cuda":
            torch.cuda.empty_cache()     # redraw resamples on the GPU; return those blocks before training resumes
        net.train()
        acc, n_ok, acc_m1, n_m1 = np.zeros(3), 0, 0.0, 0
        for _ in range(args.steps):
            if legacy:
                x, tgt = make_batch_legacy(cubes, probs, args.batch, args.crop, rng, kk, oc)
            else:
                x, b0, cc = make_batch(cubes, probs, args.batch, args.crop, rng, args.arm)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    out = net(x, torch.zeros(x.shape[0], dtype=torch.long, device=dev), None)
                loss, parts = loss_legacy(out.float(), tgt, oc) if legacy else loss_fn(out.float(), b0, cc)
            except torch.OutOfMemoryError:
                print(f"  step {step + 1}: CUDA OOM on batch {tuple(x.shape)}, skipping", flush=True)
                out = loss = None; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                parts = np.full(3, np.nan)
            if not np.isfinite(parts[0]):
                if not np.isfinite(parts[0]) and loss is not None:       # keep a record of what the bad batch looked like
                    print(f"  step {step + 1}: non-finite loss; x min/max {x.min().item():.3g}/{x.max().item():.3g} shape {tuple(x.shape)}", flush=True)
                # fp16 overflow in the forward pass (inputs finite): skip this step as GradScaler would, abort only if it persists.
                # Round 2 (unified-wiener-k15 v1) died at step 1532 on a single such step when this raised immediately.
                n_bad += 1; bad_run += 1
                if not torch.isfinite(x).all() or bad_run > 50:
                    raise FloatingPointError(f"non-finite loss at step {step + 1} ({bad_run} in a row, {n_bad} total): "
                                             f"inputs finite={bool(torch.isfinite(x).all())}")
                out = loss = None                # free this step's autograd graph now: holding it into the next forward doubled
                opt.zero_grad(set_to_none=True)  # activation memory and caused k15 v2's OOM one step after the skip
                scaler.update(scaler.get_scale() / 2.0)
                sched.step(); step += 1
                continue
            bad_run = 0
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if (args.w_m1 > 0 or args.w_amp > 0) and not legacy:   # second, separate forward/backward (memory): line-centre / amplitude losses on full-band crops
                out = loss = None
                xm, bm, cm = make_m1_batch(cubes, probs, args.m1_crops, args.m1_crop, args.m1_band, rng)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    om = net(xm, torch.zeros(xm.shape[0], dtype=torch.long, device=dev), None)
                lm, mfrac = loss_m1(om.float(), bm, cm, args.m1_crops, args.m1_band)
                la = loss_amp(om.float(), bm, cm, args.m1_crops, args.m1_band) if args.w_amp > 0 else torch.zeros((), device=lm.device)
                if torch.isfinite(lm) and torch.isfinite(la):
                    scaler.scale(args.w_m1 * lm + args.w_amp * la).backward()
                    acc_m1 += lm.item(); n_m1 += 1
                del xm, bm, cm, om, lm
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            acc += parts; n_ok += 1
            if step == 1:
                if legacy:
                    print(f"step 1: L1 {parts[0]:.5f} (the starting checkpoint on the new data mix)", flush=True)
                else:
                    with torch.no_grad():
                        base_l = (b0 - cc).abs().mean().item()
                    print(f"step 1: L1 {parts[0]:.5f} vs base {base_l:.5f} (equal: zero-init identity)", flush=True)
        acc /= max(1, n_ok)
        hist.append([rnd + 1, step, *acc.tolist(), t_draw])
        print(f"round {rnd + 1}/{args.rounds} step {step}  L1 {acc[0]:.5f}  d1 {acc[1]:.5f}  d2 {acc[2]:.5f}  "
              f"redraw {t_draw:.0f}s  sizes {sorted({c.S for c in cubes})}  {(time.time() - t0) / 60:.1f} min"
              + (f"  M1 err {acc_m1 / max(1, n_m1):.4f} ch" if n_m1 else "")
              + (f"  [{n_bad} non-finite steps skipped so far]" if n_bad else ""), flush=True)
        torch.save({"model_state_dict": net.state_dict(), "arm": arm, "base": args.base, "K": K, "alpha": ALPHA, "init": args.init,
                    "n_neighbors": kk if legacy else None, "out_channels": oc if legacy else 1, "in_channels": net.in_channels,
                    "base_channels": net.base_channels, "channel_multipliers": list(net.channel_multipliers),
                    "round": rnd + 1, "steps": step, "hist": hist, "args": vars(args),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),          # full state,
                    "rng_np": rng.bit_generator.state, "rng_torch": torch.get_rng_state(),                       # so a resume is exact
                    "rng_cuda": torch.cuda.get_rng_state_all() if dev == "cuda" else None,
                    "complete": rnd + 1 == args.rounds}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)                  # every round; atomic, a kill mid-write keeps the previous round
    del cubes
    if stopped:
        status(train_complete=False, eval_complete=False, round=len(hist), rounds=args.rounds, checkpoint=os.path.basename(ckpt))
        return
    print(f"TRAINED {step} steps in {(time.time() - t0) / 60:.1f} min -> {ckpt}", flush=True)

    done = run_eval(model, arm, args, esizes, os.path.join(args.out, f"eval_{tag}.json"), figdir) is not None
    status(train_complete=True, eval_complete=done, round=args.rounds, rounds=args.rounds, checkpoint=os.path.basename(ckpt))
    print("EVAL DONE" if done else "EVAL INCOMPLETE: RESUME_NEEDED", flush=True)


if __name__ == "__main__":
    main()
