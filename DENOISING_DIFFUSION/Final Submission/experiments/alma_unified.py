"""
Real exoALMA cube -> the unified denoiser (Wiener + learned correction, train_unified.py) -> denoised cube, moment maps, checks. 

INFERENCE ONLY: exoALMA data are never trained on. There is no truth on a real cube, so this measures what the model CHANGES and whether
it invents anything:
  S1  noise off the line (rms of line-free planes, raw vs denoised), M0 total ratio inside the signal mask, M1 shift on the brightest pixels (m/s)
  S2  invented structure: fraction of line-free-plane pixels above 4 sigma_raw, raw vs denoised
  S6  kinematics: rotation-curve residual of M1 (wiggle_scorecard), raw vs denoised: correlation and rms
Same preprocessing as training and the synthetic tests (continuum is NOT removed again unless --cont-sub: exoALMA is already subtracted), the velocity-aware Wiener on the native grid
with its noise from the line-free planes, the model's own normalisation (--norm and --clip-in read back from the checkpoint), optional resampling to
a pixel scale (--scale) and back. Several checkpoints are averaged (same K / arm / base).

  python experiments/alma_unified.py --cube MWC_758_13CO_fiducial.image.fits --ckpt a.pth[,b.pth] --out results/alma/mwc758_13co
         [--incl 21 --pa 240 --dist 155.9] [--crop 1024] [--auto-fov | --fov-arcsec 6.4] [--cont-sub] [--scale 1.0] [--budget-h 11]
"""
import argparse, json, math, os, sys, time
import numpy as np

ROOT = os.environ.get("EXXA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import torch
from astropy.io import fits
import train_unified as tu
import wiggle_scorecard as ws

C_KMS = 299792.458


def read_cube(path, crop):
    """(C, n, n) float32 with velocity ascending (km/s), the header, pixel (arcsec), beam (arcsec). Accepts a truncated download (whole planes only)."""
    with fits.open(path, memmap=True, ignore_missing_end=True) as h:
        hdr = h[0].header
        off = h[0]._data_offset
    shape = tuple(int(hdr[f"NAXIS{k}"]) for k in range(int(hdr["NAXIS"]), 0, -1))
    dtype = {-32: ">f4", -64: ">f8"}[int(hdr["BITPIX"])]
    plane = int(np.prod(shape[-2:])) * np.dtype(dtype).itemsize
    nplanes = min(int(np.prod(shape[:-2])), (os.path.getsize(path) - off) // plane)   # a truncated download keeps its whole planes
    if nplanes < int(np.prod(shape[:-2])):
        print(f"  {os.path.basename(path)} is truncated: {nplanes} of {int(np.prod(shape[:-2]))} planes present", flush=True)
    data = np.memmap(path, dtype=dtype, mode="r", offset=off, shape=(nplanes,) + shape[-2:])
    cube = np.asarray(data, dtype=np.float32)          # 4D (Stokes, chan, y, x) with one Stokes plane reads as (chan, y, x) here
    C = cube.shape[0]
    nu0 = float(hdr.get("RESTFRQ") or hdr.get("RESTFREQ"))
    ax = float(hdr["CRVAL3"]) + (np.arange(C) + 1 - float(hdr["CRPIX3"])) * float(hdr["CDELT3"])
    if str(hdr["CTYPE3"]).upper().startswith("FREQ"):
        velax = C_KMS * (nu0 - ax) / nu0                                      # radio convention
    else:
        velax = ax / (1.0 if str(hdr.get("CUNIT3", "m/s")).lower().startswith("km") else 1000.0)
    if velax[1] < velax[0]:
        cube, velax = cube[::-1], velax[::-1]
    n = cube.shape[-1]
    c = min(crop or n, n); c -= c % 2
    o = (n - c) // 2
    cube = np.nan_to_num(np.ascontiguousarray(cube[:, o:o + c, o:o + c]), nan=0.0)   # NaN outside the primary beam -> 0
    hdr["EXXAOX"], hdr["EXXAOY"] = o, o                                             # pixel offset of this array inside the file's image, for the output WCS
    pix = abs(float(hdr["CDELT1"])) * 3600
    return cube, np.ascontiguousarray(velax), hdr, pix, beam_of(path, hdr)


def beam_of(path, hdr):
    """(BMAJ, BMIN) in arcsec: the header keywords, or for a CASA per-channel beam cube (DSHARP HD 163296: no BMAJ, a BEAMS table) the median over channels."""
    if "BMAJ" in hdr:
        return float(hdr["BMAJ"]) * 3600, float(hdr["BMIN"]) * 3600
    with fits.open(path, memmap=True, ignore_missing_end=True) as h:
        t = h["BEAMS"]
        to_as = {"deg": 3600.0, "arcsec": 1.0}
        bmaj = np.median(t.data["BMAJ"]) * to_as[str(t.columns["BMAJ"].unit or "arcsec").strip().lower()]
        bmin = np.median(t.data["BMIN"]) * to_as[str(t.columns["BMIN"].unit or "arcsec").strip().lower()]
        print(f"  beam from the BEAMS table ({len(t.data)} channels, median): {bmaj:.3f} x {bmin:.3f} arcsec", flush=True)
    return float(bmaj), float(bmin)


def fov_box(cube, lf, beam_px, margin_beams=2.0, factor=1.25, min_px=128, step=8):
    """(y0, y1, x0, x1) of a square field around the disc, or None if no disc is found. Review of 2026-10-16: the exoALMA field is huge and the beam tiny,
    so crop to the disc before the network, which sees features relative to the cube size. The disc is the connected region (around the brightest pixel) of the
    beam-smoothed line M0 above 5 robust sigma; the field is its larger extent x `factor` plus `margin_beams` beams, rounded up to a multiple of `step`
    (at least `min_px`), centred on it and kept inside the image."""
    from scipy import ndimage
    line = np.setdiff1d(np.arange(cube.shape[0]), lf)
    if len(line) == 0:
        return None
    n = cube.shape[-1]
    m0 = ndimage.gaussian_filter(cube[line].sum(0), max(1.0, beam_px / 2))
    sig = 1.4826 * np.median(np.abs(m0 - np.median(m0)))
    if not np.isfinite(sig) or sig <= 0:
        return None
    lab, _ = ndimage.label(m0 > np.median(m0) + 5 * sig)
    if lab.max() == 0:
        return None
    ys, xs = np.nonzero(lab == lab[np.unravel_index(np.argmax(m0), m0.shape)])
    ext = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    size = int(math.ceil((ext * factor + 2 * margin_beams * beam_px) / step) * step)
    size = min(max(size, min_px), n - n % step)
    cy, cx = (ys.max() + ys.min()) / 2, (xs.max() + xs.min()) / 2
    y0 = int(np.clip(round(cy - size / 2), 0, n - size))
    x0 = int(np.clip(round(cx - size / 2), 0, n - size))
    return y0, y0 + size, x0, x0 + size


def continuum_check(cube, lf, nsig=5.0, match_sig=3.0, min_planes=3):
    """Is there continuum left in this cube? exoALMA images are already continuum subtracted, so the default is NOT to subtract (review of 2026-10-16:
    subtracting again double-subtracts). But a leftover in the line-free mean can be (a) real continuum (a cube that was never subtracted) or (b) faint line
    wings in planes we call line-free. Continuum is flat in frequency: it is the same in the line-free planes FAR from the line on BOTH sides. Line wings are
    not. So: mean of the farther half of the line-free planes on the blue side and on the red side, each in units of its own error; continuum = both peaks
    above `nsig` AND each side's peak reappears on the other side (same sign, above `match_sig`). A whole-map correlation is reported but not used: a compact
    continuum is drowned by the empty sky in it (0.33 for a clear 8 sigma blob in the test). Returns a dict with the decision ("subtract" / "none"), the
    reason, the numbers and, when subtracting, `map`: the far-plane mean to remove once."""
    lf = np.sort(np.asarray(lf)); C = cube.shape[0]
    line = np.setdiff1d(np.arange(C), lf)
    if len(line) == 0 or len(lf) == 0:
        return {"decision": "none", "reason": "no line or no line-free planes"}
    blue, red = lf[lf < line.min()], lf[lf > line.max()]
    blue, red = blue[:max(min_planes, len(blue) // 2)], red[len(red) - max(min_planes, len(red) // 2):]
    if len(blue) < min_planes or len(red) < min_planes:
        return {"decision": "none", "reason": f"too few line-free planes far from the line (blue {len(blue)}, red {len(red)})"}
    sigma = float(cube[lf].std())
    mb, mr = cube[blue].mean(0), cube[red].mean(0)
    eb, er = sigma / math.sqrt(len(blue)), sigma / math.sqrt(len(red))
    ib, ir = np.unravel_index(np.argmax(np.abs(mb)), mb.shape), np.unravel_index(np.argmax(np.abs(mr)), mr.shape)
    sb, sr = float(abs(mb[ib]) / eb), float(abs(mr[ir]) / er)
    b_in_r, r_in_b = float(mr[ib] * np.sign(mb[ib]) / er), float(mb[ir] * np.sign(mr[ir]) / eb)   # each peak, read on the other side, same sign > 0
    corr = float(np.corrcoef(mb.ravel(), mr.ravel())[0, 1])
    out = {"blue_far_planes": int(len(blue)), "red_far_planes": int(len(red)), "blue_peak_sigma": sb, "red_peak_sigma": sr,
           "blue_peak_on_red_sigma": b_in_r, "red_peak_on_blue_sigma": r_in_b, "blue_red_corr": corr}
    if sb > nsig and sr > nsig and b_in_r > match_sig and r_in_b > match_sig:
        out.update(decision="subtract", reason="the same structure on both far sides of the line: continuum left in the cube; removed once",
                   map=(mb * len(blue) + mr * len(red)) / (len(blue) + len(red)))
    else:
        out.update(decision="none", reason="no continuum on both far sides (any leftover is near the line: line wings or noise); cube kept as delivered")
    return out


def line_free(cube, beam_px=6.0, nsig=6.0, grow=5, bridge=3):
    """Planes without line emission. Each plane is tested on its own: the peak of its central half, smoothed over a beam, in units of that plane's robust
    noise (MAD of the smoothed plane). Planes above `nsig` hold line emission; runs of them closer than `bridge` channels are joined, the run with the most
    signal is the line, grown by `grow` channels for faint wings; every plane outside it is line-free. A disc's line covers one contiguous velocity range
    (each channel's iso-velocity strip), so this does not depend on the shape of the integrated spectrum. The previous version sigma-clipped the summed
    spectrum and called 241-301 of 301 planes line-free on the exoALMA 13CO cubes, line planes included (AA Tau: all 301; 2026-10-15)."""
    from scipy import ndimage
    C, n = cube.shape[0], cube.shape[-1]
    q = n // 4
    k = max(3, int(round(beam_px)))
    det = np.empty(C)
    for i in range(C):
        p = ndimage.uniform_filter(cube[i, q:n - q, q:n - q].astype(np.float32), k)
        med = float(np.median(p)); mad = 1.4826 * float(np.median(np.abs(p - med))) + 1e-30
        det[i] = (float(p.max()) - med) / mad
    spec = cube[:, q:n - q, q:n - q].astype(np.float64).sum((1, 2))
    hit = np.where(det > nsig)[0]
    if len(hit) == 0:                                      # no line found: every plane is noise
        return np.arange(C), spec
    runs, cur = [], [hit[0], hit[0]]
    for h in hit[1:]:
        if h - cur[1] <= bridge + 1:
            cur[1] = h
        else:
            runs.append(cur); cur = [h, h]
    runs.append(cur)
    lo, hi = max(runs, key=lambda r: det[r[0]:r[1] + 1].sum())
    lo, hi = max(0, lo - grow), min(C - 1, hi + grow)
    lf = np.r_[0:lo, hi + 1:C]
    return (lf if len(lf) >= 6 else np.r_[0:max(3, C // 10), C - max(3, C // 10):C]), spec


def load_models(paths):
    ck = torch.load(paths[0], map_location="cpu", weights_only=False)
    tu.K, tu.CLIP_IN = int(ck["K"]), float((ck.get("args") or {}).get("clip_in", 0.0))
    tu.NORM = (ck.get("args") or {}).get("norm", "wiener")
    arm, base = ck["arm"], int(ck["base"])
    nets = []
    for p in paths:
        c = torch.load(p, map_location="cpu", weights_only=False)
        assert (int(c["K"]), c["arm"], int(c["base"]), (c.get("args") or {}).get("norm", "wiener")) == (tu.K, arm, base, tu.NORM), f"{p}: members must match"
        net = tu.build_net(arm, base); net.load_state_dict(c["model_state_dict"], strict=True); nets.append(net)
    print(f"models: {len(nets)} x arm {arm}, K {tu.K}, base {base}, norm {tu.NORM}, clip {tu.CLIP_IN}", flush=True)
    return tu.Ensemble(nets, False), arm


def moments(cube, velax, mask_sig, sigma):
    """M0 (K km/s or Jy/beam km/s), quadratic M1 (km/s), M2 (km/s) with a sigma clip, the project's own estimators."""
    dv = float(velax[1] - velax[0])
    clip = np.where(cube > mask_sig * sigma, cube, 0.0)
    m0 = clip.sum(0) * dv
    m1 = ws.quad_m1(cube, velax)
    w = np.clip(clip, 0, None)
    m1w = (w * velax[:, None, None]).sum(0) / np.maximum(w.sum(0), 1e-30)
    m2 = np.sqrt((w * (velax[:, None, None] - m1w) ** 2).sum(0) / np.maximum(w.sum(0), 1e-30))
    return m0, m1, m2


def regrid_velocity(cube, velax, dv_new):
    """Average a cube onto a coarser velocity grid (bins of width dv_new centred on 0); returns (cube, velax)."""
    lo, hi = velax[0], velax[-1]
    edges = np.arange(np.floor(lo / dv_new) - 0.5, np.ceil(hi / dv_new) + 0.5 + 1e-9) * dv_new
    idx = np.digitize(velax, edges) - 1
    k = np.unique(idx)
    out = np.stack([cube[idx == j].mean(0) for j in k]).astype(np.float32)
    return out, ((edges[k] + edges[k + 1]) / 2).astype(np.float64)


def spec_up(cube, r):
    """Linear interpolation along the velocity axis onto a grid r times finer, so original channel c sits at index c * r ((C - 1) * r + 1 planes)."""
    C = cube.shape[0]
    pos = np.arange((C - 1) * r + 1) / r
    i0 = np.minimum(pos.astype(int), C - 2); f = (pos - i0).astype(np.float32)[:, None, None]
    return (cube[i0] * (1 - f) + cube[i0 + 1] * f).astype(np.float32)


def spec_down(up, r):
    """Back to the original channels: the mean of the r fine planes around each original channel (the channel's own width), ends clipped."""
    C = (up.shape[0] - 1) // r + 1
    h = r // 2
    return np.stack([up[max(0, c * r - h):min(up.shape[0], c * r + h + 1)].mean(0) for c in range(C)]).astype(np.float32)


def target_scale(beam_px, n, target, max_px):
    """--scale that puts the beam at `target` px, never below 1 (no zooming out) and never past a `max_px` grid."""
    return float(max(1.0, min(target / beam_px, max_px / n)))


def run_scaled(a, model, arm, d, w, dv, beam_px):
    """run_model on a pixel grid `a.scale` times the cube's (review of 2026-10-16, "zoom in": exoALMA's beam is about 6 px, below the 8 to 20 px the models
    were trained on; --scale 1.67 puts it at about 10 px), then area / bilinear resampling back to the cube's grid. The sky field and the beam's angular size are
    unchanged, only the pixel size changes. Used for the real cube AND for S5, so the injection test measures the same pipeline."""
    n = d.shape[-1]
    if getattr(a, "scale", 1.0) == 1.0:
        return _central_fix(a, run_model(a, model, arm, d, w, dv, beam_px), d, beam_px)
    S = int(round(n * a.scale)); S -= S % 2
    pred = run_model(a, model, arm, tu.resample(d, S), tu.resample(w, S), dv, beam_px * S / n)
    return _central_fix(a, tu.resample(pred, n).astype(np.float32), d, beam_px)


def _central_fix(a, pred, d, beam_px):
    """Candidate A, off unless --central-fix: train_unified.central_flux_fix on the cube's own grid."""
    if not getattr(a, "central_fix", False):
        return pred
    out, info = tu.central_flux_fix(pred, d, beam_px)
    print(f"  central flux fix: centre {info['centre']}, factor at centre {info['factor_at_centre']:.3f}", flush=True)
    return out


def run_model(a, model, arm, d, w, dv, beam_px):
    """The frozen inference recipe: the network on a ~INFER_BEAM-px-beam grid when the beam is wider (train_unified.infer; exoALMA fiducial beams are
    ~6 px, so this is inert there), then a second pass from a Wiener built on the first pass's output (train_unified.prior_wiener), gate inside denoise()."""
    r = int(round(abs(dv) / a.spec_interp)) if getattr(a, "spec_interp", 0) and abs(dv) > 1.5 * a.spec_interp else 1
    if r > 1:      # channels coarser than the 0.1 km/s the models were trained on (DSHARP: 0.35 km/s): run on a velocity grid r times finer, then average back
        d_n, w_n, dv_n = spec_up(d, r), spec_up(w, r), abs(dv) / r
        pred = tu.infer(model, arm, d_n, w_n, dv_n, beam_px)
        if a.wiener2:
            pred = tu.infer(model, arm, d_n, tu.prior_wiener(d_n, pred), dv_n, beam_px)
        return spec_down(pred, r)
    pred = tu.infer(model, arm, d, w, dv, beam_px)
    if a.wiener2:
        pred = tu.infer(model, arm, d, tu.prior_wiener(d, pred), dv, beam_px)
    return pred


def clean_noise_planes(cube, lf, beam_px, nsig=5.5):
    """The line-free planes that hold no compact source: beam-smoothed peak below `nsig` times the smoothed noise (median over the planes). DSHARP
    HD 143006 (2026-10-18): planes 35 to 53 of its 12CO cube, classed line-free, carry a 7 to 11 sigma central source (fast inner-disc gas, red side
    only); used as S5 noise they put a knot of the real disc into every injected cube. Pure noise in a 512 px field peaks at about 4 sigma."""
    from scipy import ndimage
    sm = np.stack([ndimage.gaussian_filter(cube[k].astype(np.float64), beam_px / 2.355) for k in lf])
    s = float(np.median(sm.std(axis=(1, 2))))
    keep = [k for k, p in zip(lf, sm.max(axis=(1, 2))) if p < nsig * s]
    return np.asarray(keep, dtype=int)


def synth_noise(cube, lf, C, seed=0):
    """C planes of noise with the real cube's own statistics, for S5 on cubes whose line fills most of the band (DSHARP HD 142666: 12 line-free planes of 60, the
    injected disc needs 35). Spatial power spectrum = the mean over the line-free planes (white noise convolved with it, as in training: pair_noise_amp), channel
    correlation = the lag-1 correlation of the line-free planes that are neighbouring channels, imposed as an AR(1) along velocity. Same variance as the real noise."""
    noise = cube[lf].astype(np.float64)
    n = cube.shape[-1]
    P = np.mean(np.abs(np.fft.rfft2(noise - noise.mean((1, 2), keepdims=True))) ** 2, axis=0)
    amp = np.sqrt(P) / n
    adj = [i for i in range(len(lf) - 1) if lf[i + 1] == lf[i] + 1]
    rho = float(np.clip(np.mean([np.corrcoef(noise[i].ravel(), noise[i + 1].ravel())[0, 1] for i in adj]), 0.0, 0.95)) if adj else 0.0
    rng = np.random.default_rng(seed)
    out, prev = np.empty((C, n, n), np.float32), None
    for c in range(C):
        e = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((n, n))) * amp, s=(n, n))
        x = e if prev is None else rho * prev + math.sqrt(1 - rho ** 2) * e
        out[c], prev = x, x
    return out, rho


def compact_truth(n, au_px, beam_px, dv, incl=23.2, mstar=1.4, r_c=50.0, r_in=5.0, r_out=400.0, sig_v=0.12, pa_deg=30.0):
    """S5 truth with a bright COMPACT CENTRE (2026-10-19: on real exoALMA discs the denoised flux is 16 to 66 percent above raw within one beam of the
    star, and the Speedie truth has a central cavity, so the injection never tested this). Thin Keplerian disc, I(R) = exp(-R / r_c) from r_in to
    r_out au (r_in = 5 au, far inside the beam, keeps the line band finite), Gaussian line of width sig_v km/s, convolved with the real cube's beam.
    Returns (cube (C, n, n) float32, velocities km/s ascending) on the cube's own pixel scale (au_px) and channel width (dv km/s)."""
    from scipy import ndimage
    yy, xx = np.indices((n, n), dtype=np.float64)
    pa, inc = np.radians(pa_deg), np.radians(incl)
    dx, dy = xx - n / 2, yy - n / 2
    xr = dx * np.cos(pa) + dy * np.sin(pa)
    yr = (-dx * np.sin(pa) + dy * np.cos(pa)) / np.cos(inc)
    R = np.hypot(xr, yr) * au_px
    I = np.where((R >= r_in) & (R <= r_out), np.exp(-R / r_c), 0.0)
    vlos = 29.78 * np.sqrt(mstar / np.maximum(R, r_in)) * np.sin(inc) * np.cos(np.arctan2(yr, xr))     # km/s; 29.78 = sqrt(G Msun / 1 au)
    vmax = float(np.abs(vlos[I > 0]).max()) + 4 * sig_v
    adv = abs(dv)
    v = np.arange(-vmax, vmax + adv / 2, adv)
    cube = np.stack([I * np.exp(-0.5 * ((vc - vlos) / sig_v) ** 2) for vc in v]).astype(np.float32)
    s_px = beam_px / 2.355
    cube = np.stack([ndimage.gaussian_filter(c, s_px) for c in cube]).astype(np.float32)
    return cube, v


def roll_noise(nz, seed):
    """Another realisation of the same real noise: the noise planes shifted spatially relative to the signal (seed 0 = unshifted). Spatial and channel
    correlation are kept exactly; used for S5 error bars (--inject-seeds)."""
    if seed == 0:
        return nz
    rng = np.random.default_rng(1000 + seed)
    n = nz.shape[-1]
    return np.roll(nz, (int(rng.integers(n // 4, 3 * n // 4)), int(rng.integers(n // 4, 3 * n // 4))), axis=(1, 2))


def flux_by_radius(truth, x, beam_px, edges=(0, 1, 3, 6, 1e9)):
    """Unclipped flux of x over the truth's own disc in annuli of beam FWHM around the image centre, divided by the truth's."""
    n = truth.shape[-1]
    yy, xx = np.indices((n, n)); rad = np.hypot(xx - n / 2, yy - n / 2) / beam_px
    m0t = np.clip(truth, 0, None).sum(0); tm = m0t > 0.05 * m0t.max()
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = tm & (rad >= lo) & (rad < hi)
        t = float(truth[:, b].sum())
        out.append(float(x[:, b].sum()) / t if b.any() and t > 0 else None)
    return out


def inject(a, model, arm, cube, lf, velax_real, pix, beam):
    """S5: a cube with KNOWN truth in this observation's REAL noise. Truth = Speedie et al.'s GI disc and its Keplerian twin (speedie_prep clean cubes,
    test-only data), moved to this cube's pixel scale (area resampling) and channel width (velocity binning), centred, then added to blocks of the real
    cube's own line-free planes (real noise, its spatial and channel correlation kept) scaled to each target peak SNR. Same denoising path as the real cube."""
    import speedie_prep  # noqa: F401  (keeps the import surface of the test-only data in one place)
    out = {}
    dv = float(velax_real[1] - velax_real[0])
    n_real = cube.shape[-1]
    lf_clean = clean_noise_planes(cube, lf, beam[0] / pix)
    if len(lf_clean) < len(lf):
        print(f"  S5: {len(lf) - len(lf_clean)} of {len(lf)} line-free planes hold a compact source (> 5.5 sigma, beam-smoothed); noise taken from the other {len(lf_clean)}", flush=True)
    noise_planes = cube[lf_clean]
    sig = float(noise_planes.std())
    for name in [x for x in a.inject_cases.split(",") if x]:
        if name == "KepC":                                                  # analytic compact-centre disc, built on the cube's own grid
            S = n_real; au_px = pix * 155.9
            tr, v = compact_truth(n_real, au_px, beam[0] / pix, dv)
        else:
            base = os.path.join(a.inject_dir, f"{name}_clean.fits")
            h = fits.getheader(base); tr = fits.getdata(base).astype(np.float32)
            v = (h["CRVAL3"] + (np.arange(tr.shape[0]) + 1 - h["CRPIX3"]) * h["CDELT3"]) / 1000.0
            tr, v = regrid_velocity(tr, v, dv)
            occ = np.where(tr.sum((1, 2)) > 1e-3 * tr.sum((1, 2)).max())[0]   # empty edge channels carry no signal; keep the truth short enough for the real cube's line-free planes
            tr, v = tr[occ.min():occ.max() + 1], v[occ.min():occ.max() + 1]
            truth_field = tr.shape[-1] * abs(float(h["CDELT1"])) * 3600      # arcsec
            S = int(round(truth_field / pix))                               # default: the truth keeps its angular size
            if getattr(a, "inject_field_frac", 0):                          # or: the cube's field shows this fraction of the truth's width (exoALMA 512 px: 0.625)
                S = int(round(n_real / a.inject_field_frac))
            S -= S % 2
            au_px = truth_field * 155.9 / S                                 # Speedie's distance; equals pix * 155.9 when the angular size is kept
            tr = tu.resample(tr, S)
            if S >= n_real:
                o = (S - n_real) // 2; tr = tr[:, o:o + n_real, o:o + n_real]
            else:
                o = (n_real - S) // 2; tr = np.pad(tr, ((0, 0), (o, n_real - S - o), (o, n_real - S - o)))
        pad = int(min(20, max(5, (len(noise_planes) - tr.shape[0]) // 2)))   # line-free planes each side, as a real cube has
        tr = np.concatenate([np.zeros((pad,) + tr.shape[1:], np.float32), tr, np.zeros((pad,) + tr.shape[1:], np.float32)])
        v = np.r_[v[0] - dv * np.arange(pad, 0, -1), v, v[-1] + dv * np.arange(1, pad + 1)]
        C = tr.shape[0]
        if len(noise_planes) < C:      # the line fills the band: synthetic noise with the cube's own spectrum and channel correlation, not a skip
            nz, rho = synth_noise(cube, lf_clean, C)
            print(f"  S5 {name}: only {len(noise_planes)} line-free planes for {C} needed; noise drawn from their measured spectrum (lag-1 channel correlation {rho:.2f})", flush=True)
        else:
            nz = noise_planes[:C]
        for snr, seed in [(float(x), k) for x in a.inject_snr.split(",") if x for k in range(max(1, getattr(a, "inject_seeds", 1)))]:
            truth = tr * np.float32(snr * sig / max(float(tr.max()), 1e-30))
            dirty = truth + roll_noise(nz, seed)
            tag = f"{name}@snr{snr:g}" + (f"#s{seed}" if seed else "")
            lf_i, _ = line_free(dirty, beam[0] / pix)
            # no continuum subtraction here: `cube` (and so its line-free planes, the noise) already went through the one continuum decision in main();
            # subtracting each injected cube's own line-free mean again was a second subtraction, applied to the model's input but not to the dirty row (2026-10-17)
            w = tu.native_wiener(dirty, "line", lf_i)
            pred = run_scaled(a, model, arm, dirty, w, dv, beam[0] / pix)
            d = {"clean": truth, "dirty": dirty, "velax": v, "au": au_px, "incl": 23.2, "crop": slice(None), "beam_au": beam[0] / pix * au_px}
            ws.METHODS["wiener"] = lambda _d, w=w: w; ws.METHODS["model"] = lambda _d, p=pred: p
            figdir = a.out if seed == 0 else os.path.join(a.out, "seeds"); os.makedirs(figdir, exist_ok=True)
            sc = ws.score_cube(f"S5_{name}_snr{snr:g}" + (f"_s{seed}" if seed else ""), d, ["dirty", "wiener", "model"], figdir)["methods"]
            m1d, m1p = ws.quad_m1(dirty, v), ws.quad_m1(pred, v)
            m0t = np.clip(truth, 0, None).sum(0); br = m0t >= np.percentile(m0t[m0t > 0], 90)
            row = {k: {"rc_r": sc[k]["rc_r"], "dv_med_ms": sc[k]["dv_med_ms"], "spurious_ms": sc[k]["rc_rms_ms"] - sc[k]["rc_rms_clean_ms"]} for k in ("dirty", "wiener", "model")}
            row["m1_shift_model_vs_dirty_bright_median_ms"] = float(np.nanmedian(np.abs(m1p - m1d)[br]) * 1000)
            # flux against the KNOWN truth (2026-10-18: on faint exoALMA discs S1's model / raw flux ratio was 1.07 to 1.09; is it the model or the noisy
            # reference?): unclipped sums over the truth's own disc (M0 above 5 % of its peak), and S1's clipped ratio model / dirty computed as on real data
            tm = m0t > 0.05 * m0t.max()
            tf = float(truth[:, tm].sum())
            row["flux_vs_truth"] = {k: float(x[:, tm].sum()) / tf for k, x in (("dirty", dirty), ("wiener", w), ("model", pred))}
            sig_i = float(dirty[lf_i].std()); dv_ = abs(dv)
            m0d_ = moments(dirty, v, 3.0, sig_i)[0]; m0p_ = moments(pred, v, 3.0, sig_i)[0]
            s1m = m0d_ > 5 * sig_i * dv_ * math.sqrt(max(1, C - len(lf_i)))
            row["s1_flux_ratio_model_vs_dirty"] = float(m0p_[s1m].sum() / max(m0d_[s1m].sum(), 1e-30)) if s1m.any() else None
            row["s1_flux_ratio_truth_vs_dirty"] = float(moments(truth, v, 3.0, sig_i)[0][s1m].sum() / max(m0d_[s1m].sum(), 1e-30)) if s1m.any() else None
            row["truth_field_frac"] = float(n_real / S); row["au_px"] = float(au_px); row["noise_planes"] = int(len(noise_planes)); row["seed"] = seed
            # flux against the truth by radius (beams from the centre: <1 / 1-3 / 3-6 / >6), the real-data analysis of 2026-10-19 on a known answer
            row["flux_by_radius_vs_truth"] = {k: flux_by_radius(truth, x, beam[0] / pix) for k, x in (("dirty", dirty), ("model", pred))}
            out[tag] = row
            print(f"  S5 {tag}: flux / truth by radius (<1 / 1-3 / 3-6 / >6 beams) model {row['flux_by_radius_vs_truth']['model']}", flush=True)
            print(f"  S5 {name} SNR {snr:g} seed {seed}: rc_r dirty {row['dirty']['rc_r']:.2f} wiener {row['wiener']['rc_r']:.2f} MODEL {row['model']['rc_r']:.2f} | "
                  f"line-centre error vs truth dirty {row['dirty']['dv_med_ms']:.0f} MODEL {row['model']['dv_med_ms']:.0f} m/s | "
                  f"|M1 model - M1 dirty| on bright px {row['m1_shift_model_vs_dirty_bright_median_ms']:.0f} m/s | flux / truth dirty "
                  f"{row['flux_vs_truth']['dirty']:.3f} MODEL {row['flux_vs_truth']['model']:.3f}, S1-style model / dirty {row['s1_flux_ratio_model_vs_dirty']}", flush=True)
    return out


def bootstrap_m1(a, model, arm, cube, pred, lf, velax, pix, beam, mask, sigma_raw):
    """Parametric bootstrap of the M1 error. The denoised cube stands in for the sky; each draw adds a new realisation of this cube's own
    noise (synth_noise: its measured spatial spectrum and lag-1 channel correlation, as S5 when the line fills the band) and goes through the same
    denoising path. The per-pixel std of M1 over the draws is a LOWER bound on the error: it holds the noise term, not the model's bias."""
    C = cube.shape[0]
    dv = float(velax[1] - velax[0])
    lf_clean = clean_noise_planes(cube, lf, beam[0] / pix)
    m1s, shift = [], []
    m1_pred = ws.quad_m1(pred, velax)
    for k in range(1, a.boot + 1):
        nz, rho = synth_noise(cube, lf_clean, C, seed=k)
        d = (pred + nz).astype(np.float32)
        w = tu.native_wiener(d, "line", lf)
        p = run_scaled(a, model, arm, d, w, dv, beam[0] / pix)
        m1 = ws.quad_m1(p, velax)
        m1s.append(m1)
        shift.append(float(np.nanmedian(np.abs(m1 - m1_pred)[mask]) * 1000))
        print(f"  U draw {k}/{a.boot}: noise rms {float(nz.std()):.3g} (raw {sigma_raw:.3g}, lag-1 {rho:.2f}), median |M1 - M1(denoised)| in the mask {shift[-1]:.1f} m/s", flush=True)
    std = np.nanstd(np.stack(m1s), axis=0).astype(np.float32) * 1000          # m/s
    s = std[mask & np.isfinite(std)]
    out = {"draws": a.boot, "mask_px": int(mask.sum()), "m1_std_median_ms": float(np.median(s)), "m1_std_p90_ms": float(np.percentile(s, 90)),
           "median_abs_m1_shift_per_draw_ms": shift, "note": "lower bound: noise term only, model bias not included"}
    print(f"  U: M1 bootstrap std in the mask median {out['m1_std_median_ms']:.1f} m/s, p90 {out['m1_std_p90_ms']:.1f} m/s ({a.boot} draws)", flush=True)
    return out, np.where(mask, std, np.nan).astype(np.float32)


def write(path, data, hdr, keep3d=True):
    h = fits.Header()
    for k in ("CTYPE1", "CTYPE2", "CDELT1", "CDELT2", "CRVAL1", "CRVAL2", "CUNIT1", "CUNIT2", "BMAJ", "BMIN", "BPA", "BUNIT", "OBJECT", "RESTFRQ"):
        if k in hdr:
            h[k] = hdr[k]
    n = data.shape[-1]
    if "CRPIX1" in hdr and "EXXAOX" in hdr:        # the reference pixel of the file, moved by the centre crop and the field crop
        h["CRPIX1"], h["CRPIX2"] = float(hdr["CRPIX1"]) - float(hdr["EXXAOX"]), float(hdr["CRPIX2"]) - float(hdr["EXXAOY"])
    else:
        h["CRPIX1"] = h["CRPIX2"] = n / 2 + 0.5
    h["ORIGIN"] = "EXXA unified denoiser (train_unified.py); inference only"
    fits.writeto(path, data.astype(np.float32), h, overwrite=True)


def s6(name, cube, pred, velax, pix, beam, mask, incl, dist, pa_lit, out, pa_inits=None):
    """S6: rotation-curve residual and PV of the denoised cube against the raw one (no truth on real data), geometry fitted on the raw M1 with the
    inclination fixed. `pa_inits` (fit-convention degrees) adds starting PAs, lowest-cost fit kept (alma_refit.py: a published PA when the default
    start converged wrong). Shared by main() and the local refit so both score the same way."""
    from scipy import ndimage
    au = pix * dist
    # the scorecard builds its geometry mask from the reference M0 (5 % of its peak); with the RAW cube as reference that mask took in noise blobs over
    # the whole field (v1 / v2, 2026-10-13). Restrict both cubes to the S1 signal mask grown by one beam, so the fit, R_out and the spine see the disc only.
    keep = ndimage.binary_dilation(mask, iterations=max(1, int(round(beam[0] / pix))))
    raw_m, pred_m = cube * keep[None], pred * keep[None]
    d = {"clean": raw_m, "dirty": raw_m, "velax": velax, "au": au, "incl": incl, "crop": slice(None), "beam_au": beam[0] * dist, "pa_inits": pa_inits}
    ws.METHODS["denoised"] = lambda _d: pred_m
    try:
        full = ws.score_cube(name, d, ["denoised"], out, ref_label="raw")
        sc = full["methods"]["denoised"]
        return {"rc_r_denoised_vs_raw": sc["rc_r"], "rc_rms_raw_ms": sc["rc_rms_clean_ms"], "rc_rms_den_ms": sc["rc_rms_ms"],
                "pv": {k: sc.get(k) for k in ("pv_r", "pv5_r", "pv_rms_ms", "pv5_rms_ms", "pv_mag_ms", "pv5_mag_ms")},
                "pv_mag_raw_ms": full["methods"]["clean"].get("pv_mag_ms"), "pv5_mag_raw_ms": full["methods"]["clean"].get("pv5_mag_ms"),
                "geom": full["meta"], "pa_lit_deg": pa_lit, "pa_inits": pa_inits,
                "note": "raw is the reference here (no truth): correlation of the denoised residual with the RAW residual"}
    except Exception as e:
        return {"failed": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--ckpt", required=True, help="comma list: averaged")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--scale", type=float, default=1.0, help="resample the cube by this factor before the model (and back); 1 = native pixels")
    ap.add_argument("--target-beam-px", type=float, default=0.0, help="per-cube zoom: set --scale so the beam spans this many px (only zooms in, never out); 0 = off")
    ap.add_argument("--max-scaled-px", type=int, default=1024, help="cap on the zoomed grid size for --target-beam-px (GPU memory)")
    ap.add_argument("--auto-fov", action="store_true", help="crop to a square field around the disc before the model (fov_box); the output cubes carry the matching WCS")
    ap.add_argument("--fov-arcsec", type=float, default=0.0, help="crop to this field (arcsec) centred on the disc instead of the automatic size")
    ap.add_argument("--cont-sub", nargs="?", const="on", default="auto", choices=["auto", "on", "off"],
                    help="continuum: auto (default) = subtract only if continuum_check finds the same structure on both far sides of the line (a cube that was never "
                         "subtracted); off = never (exoALMA is already subtracted: subtracting again double-subtracts, review of 2026-10-16); on = always subtract "
                         "the line-free mean (the old behaviour). The leftover is measured and reported in every mode.")
    ap.add_argument("--incl", type=float, default=None)
    ap.add_argument("--pa", type=float, default=None)
    ap.add_argument("--dist", type=float, default=150.0)
    ap.add_argument("--gate", default="40,120,0.25", help="do-no-harm gate lo,hi,floor (train_unified.gate_alpha); empty = off")
    ap.add_argument("--infer-beam-px", type=float, default=10.0, help="run the network on a grid where the beam spans this many px when it is wider than 1.2x (0 = off)")
    ap.add_argument("--spec-interp", type=float, default=0.0, help="channel width (km/s) of the training data, 0.1: when the cube's channels are coarser (DSHARP 0.35), interpolate "
                    "to that spacing along velocity, run the model there and average back (0 = run on the native channels)")
    ap.add_argument("--no-wiener2", dest="wiener2", action="store_false", help="skip the second pass from a model-informed Wiener (default on)")
    ap.add_argument("--inject-dir", default="", help="S5: folder with speedie_prep clean cubes (GI_r0.5_clean.fits ...); empty = skip")
    ap.add_argument("--inject-cases", default="GI_r0.5,Kep_r0.5")
    ap.add_argument("--inject-snr", default="12,24,36")
    ap.add_argument("--central-fix", action="store_true", help="candidate A: rescale the denoised flux within 2 to 4 beams of the centre to the input cube's")
    ap.add_argument("--inject-seeds", type=int, default=1, help="S5: noise realisations per SNR (the real noise shifted relative to the signal, roll_noise); 1 = as before")
    ap.add_argument("--inject-field-frac", type=float, default=0.0, help="S5: scale the truth so the cube's field shows this fraction of its width "
                    "(0 = keep its angular size; 0.625 = the exoALMA 512 px framing, for small DSHARP fields)")
    ap.add_argument("--boot", type=int, default=0, help="empirical M1 error map from this many re-denoised copies of the denoised cube "
                    "plus new realisations of the cube's own noise (0 = off)")
    a = ap.parse_args()
    tu.GATE = tuple(float(x) for x in a.gate.split(",")) if a.gate else None
    tu.INFER_BEAM, tu.INFER_MODE = a.infer_beam_px, "full"
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cube, velax, hdr, pix, beam = read_cube(a.cube, a.crop)
    C, n = cube.shape[0], cube.shape[-1]
    dv = float(velax[1] - velax[0])
    lf, spec = line_free(cube, beam[0] / pix)
    fov = None
    if a.auto_fov or a.fov_arcsec > 0:
        box = fov_box(cube, lf, beam[0] / pix, min_px=128)
        if box is not None and a.fov_arcsec > 0:                 # a fixed field, centred on the disc the automatic box found
            size = min(n - n % 8, max(8, int(round(a.fov_arcsec / pix / 8)) * 8))
            cy, cx = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
            y0 = int(np.clip(round(cy - size / 2), 0, n - size)); x0 = int(np.clip(round(cx - size / 2), 0, n - size))
            box = (y0, y0 + size, x0, x0 + size)
        if box is not None:
            cube = np.ascontiguousarray(cube[:, box[0]:box[1], box[2]:box[3]])
            hdr["EXXAOY"] += box[0]; hdr["EXXAOX"] += box[2]
            C, n = cube.shape[0], cube.shape[-1]
            lf, spec = line_free(cube, beam[0] / pix)            # again on the cropped field: the disc is now its central half
            fov = {"box_yx": [int(box[0]), int(box[1]), int(box[2]), int(box[3])], "size_px": int(n), "size_arcsec": float(n * pix)}
            print(f"field of view: cropped to {n} px ({n * pix:.2f} arcsec) around the disc, box y {box[0]}:{box[1]} x {box[2]}:{box[3]}", flush=True)
        else:
            print("field of view: no disc found by fov_box, keeping the whole field", flush=True)
    if a.target_beam_px > 0:
        a.scale = target_scale(beam[0] / pix, n, a.target_beam_px, a.max_scaled_px)
        print(f"zoom: beam {beam[0] / pix:.1f} px, scale {a.scale:.4f} (grid {n} to {int(round(n * a.scale))} px)", flush=True)
    cont = cube[lf].mean(0)
    sigma_raw = float(cube[lf].std())
    cont_sig = float(np.abs(cont).max() / (sigma_raw / math.sqrt(len(lf))))
    chk = continuum_check(cube, lf)
    sub_map = cont if a.cont_sub == "on" else (chk.get("map") if a.cont_sub == "auto" and chk["decision"] == "subtract" else None)
    if sub_map is not None:
        cube = cube - sub_map[None]
    cont_rep = {k: v for k, v in chk.items() if k != "map"}
    cont_rep.update(mode=a.cont_sub, subtracted=sub_map is not None)
    print(f"continuum ({a.cont_sub}): {'SUBTRACTED' if sub_map is not None else 'not subtracted'}; {chk['reason']}; line-free mean peak {cont_sig:.1f} sigma_mean"
          + (f", far planes blue {chk['blue_peak_sigma']:.1f} / red {chk['red_peak_sigma']:.1f} sigma, each peak on the other side "
             f"{chk['blue_peak_on_red_sigma']:.1f} / {chk['red_peak_on_blue_sigma']:.1f} sigma" if "blue_peak_sigma" in chk else ""), flush=True)
    print(f"cube {os.path.basename(a.cube)}: {C} x {n} x {n}, {pix * 1000:.1f} mas/px, beam {beam[0]:.3f} x {beam[1]:.3f} arcsec "
          f"({beam[0] / pix:.1f} x {beam[1] / pix:.1f} px), dv {dv * 1000:.0f} m/s, line-free planes {len(lf)}, rms {sigma_raw:.3g}, "
          f"residual continuum peak {cont_sig:.1f} sigma_mean", flush=True)
    model, arm = load_models([p for p in a.ckpt.split(",") if p])
    w = tu.native_wiener(cube, "line", lf)
    pred = run_scaled(a, model, arm, cube, w, dv, beam[0] / pix)      # --scale != 1: on a finer / coarser pixel grid and back (same path as S5)
    print(f"denoised in {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- S1 / S2
    sig_den = float(pred[lf].std())
    m0r, m1r, m2r = moments(cube, velax, 3.0, sigma_raw)
    m0d, m1d, m2d = moments(pred, velax, 3.0, sigma_raw)
    s_m0 = sigma_raw * dv * math.sqrt(max(1, C - len(lf)))
    mask = m0r > 5 * s_m0
    bright = mask & (m0r >= np.percentile(m0r[mask], 90)) if mask.any() else mask
    dm1 = np.abs(m1d - m1r)[bright & np.isfinite(m1d) & np.isfinite(m1r)] * 1000
    inv_raw = float((np.abs(cube[lf]) > 4 * sigma_raw).mean())
    inv_den = float((np.abs(pred[lf]) > 4 * sigma_raw).mean())
    rep = {"gate": a.gate, "wiener2": a.wiener2, "infer_beam_px": a.infer_beam_px, "snr_est": float(tu.scale_of(cube, w) / max(tu.noise_level(cube, w), 1e-30)),
           "gate_a": tu.gate_alpha(float(tu.scale_of(cube, w) / max(tu.noise_level(cube, w), 1e-30))), "cube": os.path.basename(a.cube), "shape": [C, n, n], "pix_mas": pix * 1000, "beam_arcsec": beam, "beam_px": [beam[0] / pix, beam[1] / pix],
           "dv_ms": dv * 1000, "line_free_planes": len(lf), "line_range_kms": [float(velax[np.setdiff1d(np.arange(C), lf)].min()), float(velax[np.setdiff1d(np.arange(C), lf)].max())] if len(lf) < C else None, "sigma_raw": sigma_raw, "residual_continuum_peak_sigma": cont_sig, "continuum": cont_rep, "fov": fov, "scale": a.scale,
           "S1": {"offline_rms_raw": sigma_raw, "offline_rms_den": sig_den, "noise_reduction": sigma_raw / max(sig_den, 1e-30),
                  "m0_total_ratio": float(m0d[mask].sum() / max(m0r[mask].sum(), 1e-30)) if mask.any() else None, "mask_px": int(mask.sum()),
                  "m1_shift_bright_median_ms": float(np.median(dm1)) if dm1.size else None, "m1_shift_bright_p95_ms": float(np.percentile(dm1, 95)) if dm1.size else None},
           "S2": {"frac_above_4sigma_raw_lineplanes": inv_raw, "frac_above_4sigma_den_lineplanes": inv_den}}
    # ---- S6: rotation-curve residual (needs the geometry; the fit uses the raw M1)
    if a.incl is not None:
        rep["S6"] = s6(os.path.splitext(os.path.basename(a.cube))[0], cube, pred, velax, pix, beam, mask, a.incl, a.dist, a.pa, a.out)
    if a.inject_dir:
        rep["S5"] = inject(a, model, arm, cube, lf, velax, pix, beam)
    if a.boot:
        rep["U"], m1_std = bootstrap_m1(a, model, arm, cube, pred, lf, velax, pix, beam, mask, sigma_raw)
        write(os.path.join(a.out, os.path.splitext(os.path.basename(a.cube))[0].replace(".image", "")) + "_m1_boot_std.fits", m1_std, hdr)
    rep["minutes"] = (time.time() - t0) / 60
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
    base = os.path.join(a.out, os.path.splitext(os.path.basename(a.cube))[0].replace(".image", ""))
    write(base + "_denoised.fits", pred, hdr)
    for tag, arr in (("m0_raw", m0r), ("m1_raw", m1r), ("m2_raw", m2r), ("m0_den", m0d), ("m1_den", m1d), ("m2_den", m2d)):
        write(f"{base}_{tag}.fits", arr, hdr)
    figure(base + "_moments.png", cube, pred, velax, m0r, m1r, m2r, m0d, m1d, m2d, lf, rep)
    print(json.dumps({k: rep[k] for k in ("S1", "S2") }, indent=1, default=float), flush=True)
    print("ALMA DONE", flush=True)


def figure(path, raw, den, velax, m0r, m1r, m2r, m0d, m1d, m2d, lf, rep):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    line = np.setdiff1d(np.arange(raw.shape[0]), lf)
    mid = int(line[len(line) // 2]) if len(line) else raw.shape[0] // 2
    vmin1, vmax1 = np.nanpercentile(m1r[m0r > 0], [2, 98]) if (m0r > 0).any() else (-1, 1)
    fig, axs = plt.subplots(2, 5, figsize=(22, 9))
    for r, (lab, cube, m0, m1, m2) in enumerate((("raw", raw, m0r, m1r, m2r), ("denoised", den, m0d, m1d, m2d))):
        axs[r, 0].imshow(cube[mid], origin="lower", cmap="inferno", vmin=0, vmax=np.percentile(raw[mid], 99.9)); axs[r, 0].set_title(f"{lab}: channel {velax[mid]:.2f} km/s")
        axs[r, 1].imshow(m0, origin="lower", cmap="inferno", vmin=0, vmax=np.percentile(m0r, 99.5)); axs[r, 1].set_title(f"{lab}: M0")
        axs[r, 2].imshow(np.where(m0r > 0, m1, np.nan), origin="lower", cmap="RdBu_r", vmin=vmin1, vmax=vmax1); axs[r, 2].set_title(f"{lab}: M1 (quadratic)")
        axs[r, 3].imshow(np.where(m0r > 0, m2, np.nan), origin="lower", cmap="viridis", vmin=0, vmax=np.nanpercentile(m2r[m0r > 0], 98) if (m0r > 0).any() else 1); axs[r, 3].set_title(f"{lab}: M2")
        axs[r, 4].imshow(cube[lf[len(lf) // 2]], origin="lower", cmap="RdBu_r", vmin=-4 * rep["sigma_raw"], vmax=4 * rep["sigma_raw"]); axs[r, 4].set_title(f"{lab}: a line-free plane (+-4 sigma_raw)")
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    s1 = rep["S1"]
    fig.suptitle(f"{rep['cube']}: noise x{s1['noise_reduction']:.1f} lower off-line, M0 total ratio {s1['m0_total_ratio']}, "
                 f"M1 shift on bright px median {s1['m1_shift_bright_median_ms']} m/s; >4 sigma in line-free planes raw {rep['S2']['frac_above_4sigma_raw_lineplanes']:.2e} "
                 f"denoised {rep['S2']['frac_above_4sigma_den_lineplanes']:.2e}")
    fig.savefig(path, dpi=70, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
