"""
GI wiggle scorecard: score any denoising method on every GI cube we have, the same way.

Cubes (the original run_sg_* pairs are left out: run_sg_00019's dirty equals its clean to 3e-12 and the
other four differ only by 0.3 to 30 % rms, i.e. very low noise (measured 2026-10-10); none can separate methods):
  - sg_synth, 5 disks: dirty = beam (*) clean + beam (*) noise (synthesize_sg_pairs.py),
    0.1 km/s channels, ~2 au/px, inclinations from the .para files.
  - an01-an06: analytic GI disks (src/data/analytic_gi.py, Longarini's model) built like sg_synth, with an
    exact line-of-sight truth; `--only an01,...` (not in the default list).
  - SG v2 (kinematic_data_v2): q = 0.75, i = 30 deg, 140 pc, 1.33 au/px, 0.033 km/s
    (Terry et al. 2022's own setup). Static noise pattern removed before any method runs.

Metrics, each against the clean cube:
  rc_*           HEADLINE. As resid_r / coh_finger / amp_ratio below, but subtracting the clean
                 cube's own fitted rotation curve v_phi(R) (free per bin, beam/2 wide) instead of
                 a Keplerian: removes the axisymmetric part a massive disk's non-Keplerian rotation
                 leaves behind, so only non-axisymmetric motion (the wiggle) is scored.
  resid_r        Keplerian-subtracted quadratic M1 vs clean's; ONE shared Keplerian fit from
                 the clean M1, true inclination fixed, signal mask frac 0.05.
  coh_finger     Fourier coherence of that residual with clean's at wiggle scales
                 (40-160 au, Terry+2022 Fig. 5 lambda_wiggle range).
  dv_rms_ms      RMS line-centre error |M1 - M1_clean| over the mask, m/s; outlier-dominated, so
  dv_med_ms      also the median (exoALMA features are 10-100 m/s: this wants ~10 m/s) and
  bad_frac       the fraction of mask pixels off by more than 100 m/s (one exoALMA channel).
  amp_ratio      RMS of the residual / clean's (1 = right wiggle strength, <1 = smoothed away).
  spine_*        Terry et al. 2022 Eq. 3 on channels |v - vsys| < 1 km/s: per row across the
                 spine, intensity-weighted centroid of the brightest pixel +/- 6.65 au
                 (their 5 px of 1.33 au), giving y_n(R) for R_in..R_out on both halves of the
                 spine. Then |A(lambda)| by Fourier; A_wiggle = max, lambda_wiggle its wavelength.
                 spine_r / spine_rms_au compare y_n to clean's; A_ratio, lam_ratio compare the
                 per-track A_wiggle and lambda_wiggle (means over tracks).
Deviations from Terry, stated: y_n is linearly detrended before the FFT (their bar at the full
track length is the trend); R_in = 2 beam FWHM (they used h/H < 0.25 from the SPH, which a cube
does not carry); R_out = radius holding 95% of clean M0 (they used 95% of disc mass).

Run:  PYTHONPATH=. python3 experiments/wiggle_scorecard.py [--only 9015,sgv2] [--selftest]
Out:  results/wiggle-scorecard/<date>/{scorecard.json, scorecard.md, <cube>.png}
"""
import os, sys, json, time, argparse, datetime, importlib.util

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import map_coordinates
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gi_wiggle", os.path.join(ROOT, "src/evaluation/gi_wiggle.py"))
gw = importlib.util.module_from_spec(spec); spec.loader.exec_module(gw)
_pvs = importlib.util.spec_from_file_location("pv_curve", os.path.join(ROOT, "experiments/pv_curve.py"))
pv = importlib.util.module_from_spec(_pvs); _pvs.loader.exec_module(pv)

DATA = os.path.join(ROOT, "self-gravitating cube and dirty cube")
BEAM = os.path.join(ROOT, "results/self-gravitating/dirty_beam_recovered_v2.fits")
SYNTH = {"9015": ("run_9015_00370_rt_00", 20.0), "9019": ("run_9019_00019_rt_00", 30.0),
         "9025": ("run_9025_00370_rt_00", 20.0), "9032": ("run_9032_00020_rt_00", 20.0),
         "9074": ("run_9074_00025_rt_00", 20.0)}
FRAC = 0.05
FINGER_AU = (40.0, 160.0)
TERRY_HALF_AU = 6.65          # Terry+2022: 5 px either side at 1.33 au/px
TERRY_DV = 1.0                # km/s, channels scored by the spine method


def beam_otf(beam, n):
    """FFT of the unit-gain beam placed at the centre of an n x n image."""
    pad = np.zeros((n, n)); hb = beam.shape[0] // 2; c = n // 2
    pad[c - hb:c + hb + 1, c - hb:c + hb + 1] = beam
    return sfft.fft2(np.fft.ifftshift(pad))


def blind_noise2d(dirty, H):
    """Per-channel noise power spectrum from the dirty cube alone: beam-shaped white noise, P_n = s^2 |H|^2, with s^2 taken
    from the faintest channels at high k, where the signal is negligible (no clean data)."""
    n = dirty.shape[-1]
    kr = np.hypot(np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None])
    Hm2 = np.abs(H) ** 2
    quiet = np.argsort(np.abs(dirty).reshape(len(dirty), -1).mean(1))[:max(10, len(dirty) // 8)]
    q = dirty[quiet].astype(np.float64)
    Pq = np.mean(np.abs(sfft.fft2(q - q.mean(axis=(1, 2), keepdims=True))) ** 2, axis=0)
    hi = (kr > 0.12) & (kr < 0.35) & (Hm2 > 1e-8 * Hm2.max())
    return float(np.median(Pq[hi] / Hm2[hi])) * Hm2


# ---------------------------------------------------------------- loading
def load(name):
    """Returns dict: clean, dirty (float32, channels x y x x), velax km/s, au, incl, beam_au,
    noise2d (per-channel noise power spectrum of the dirty cube, for blind methods),
    crop (channel slice that is scored)."""
    beam = fits.getdata(BEAM).astype(np.float64)
    beam /= beam.sum()
    if name == "sgv2":
        p = os.path.join(DATA, "kinematic_data_v2")
        blk = np.arange(180, 421)                          # processed; scored on 240-360
        with fits.open(f"{p}/clean_sg.fits", memmap=True) as h:
            hdr, clean = h[0].header, h[0].data[blk].astype(np.float32)
        with fits.open(f"{p}/dirty_sg.fits", memmap=True) as h:
            dirty = h[0].data[blk].astype(np.float32)
            lf = h[0].data[np.r_[0:60, 541:601]].astype(np.float64)
        static = lf.mean(0)                                # 54% of per-channel noise is a fixed pattern
        dirty -= static[None].astype(np.float32)
        noise2d = np.mean(np.abs(sfft.fft2(lf - static)) ** 2, axis=0)
        chans, incl, crop = blk, 30.0, slice(60, 181)
    else:
        if name.startswith("an"):                          # analytic GI disks (make_analytic_gi_cubes.py)
            run, p = name, os.path.join(DATA, "sg_analytic", name)
            incl = float(fits.getheader(f"{p}/{run}_clean.fits")["INCL_DEG"])
        else:
            run, incl = SYNTH[name]
            p = os.path.join(DATA, "sg_synth", run)
        with fits.open(f"{p}/{run}_clean.fits", memmap=True) as h:
            hdr, clean = h[0].header, h[0].data[:].astype(np.float32)
        with fits.open(f"{p}/{run}_dirty.fits", memmap=True) as h:
            dirty = h[0].data[:].astype(np.float32)
        chans, crop, noise2d = np.arange(clean.shape[0]), slice(None), None
    velax = hdr["CRVAL3"] + (chans + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    au = abs(hdr["CDELT1"]) * 3600.0 * float(hdr["DIST_PC"])
    n = clean.shape[-1]
    H = beam_otf(beam, n)
    if noise2d is None:
        noise2d = blind_noise2d(dirty, H)
    return {"clean": clean, "dirty": dirty, "velax": velax, "au": au, "incl": incl, "H": H,
            "noise2d": noise2d, "crop": crop, "beam_au": float(hdr["BMAJ"]) * 3600.0 * float(hdr["DIST_PC"])}


# ---------------------------------------------------------------- methods (blind: no clean data)
def _radial_avg(P, rb, cnt, nb):
    return (np.bincount(rb.ravel(), P.ravel(), nb) / cnt)[rb]


def wiener3d(d, alpha=1.0):
    """Spatial+spectral Wiener deconvolution. Signal power from the dirty cube's own radially
    averaged 3D spectrum minus the noise model. Noise is independent between channels unless d["vspec"] gives its power
    along the velocity-frequency axis (mean 1, np.fft.fftfreq order): real ALMA-like cubes are Hanning-correlated."""
    dirty, H = d["dirty"], d["H"]
    C, n, _ = dirty.shape
    kr = np.hypot(np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None])
    nb = 120
    rb = np.minimum((kr / kr.max() * nb).astype(int), nb - 1)
    cnt = np.maximum(np.bincount(rb.ravel(), minlength=nb), 1)
    Pn = _radial_avg(d["noise2d"], rb, cnt, nb).astype(np.float32)
    Hs = np.where(np.abs(H) > 1e-6 * np.abs(H).max(), H, 1e-6 * np.abs(H).max()).astype(np.complex64)
    D = sfft.fftn(dirty, workers=-1)
    for i in range(C):                              # in place, one spectral plane at a time
        Pd = _radial_avg(np.abs(D[i]) ** 2, rb, cnt, nb) / C
        pn = Pn if d.get("vspec") is None else Pn * d["vspec"][i]
        D[i] *= np.clip(1 - alpha * pn / np.maximum(Pd, 1e-30), 0, 1) / Hs
    return sfft.ifftn(D, workers=-1).real.astype(np.float32)


_NETS = {}
# Disks a checkpoint saw in training or validation (RUNS.md 12-sg-spectral-context: train 9015/9019/9032, val 9025,
# holdout 9074). Those rows are marked with a dagger in the table and must not be read as held-out scores.
SEEN = {"sg_k3_fresh": {"9015", "9019", "9032", "9025"}}


def _learned(name, d):
    """Existing checkpoints, each through ITS OWN training preprocessing (wiggle_domain_split.py's loaders,
    read off the training code). Must run from this folder (relative checkpoint paths)."""
    sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
    import wiggle_domain_split as wd
    x = d["dirty"].astype(np.float32)
    if name == "winner_aug":                 # line-emission trained, 1 channel, per-channel min-max, 256 px
        net = _NETS.setdefault(name, wd.load_winner_aug())
        return wd.denoise_single(net, x).astype(np.float32)
    if name == "sg_k3_fresh":                # SG-trained (nb12), 7-channel stack, 256 px
        net, k = _NETS.setdefault(name, wd.load_k3())
        return wd.denoise_stack_k(net, x, np.arange(len(x)), k).astype(np.float32)
    if name == "kin_gamma0":                 # nb08, 31-channel stack, continuum subtracted then added back
        from src.data.fits_cube_dataset import continuum_of
        net, k = _NETS.setdefault(name, wd.load_kin_gamma0())
        cont = continuum_of(x, wd.CONTINUUM_N)
        return wd.denoise_stack31(net, x - cont[None], np.arange(len(x)), k, cont).astype(np.float32)
    raise KeyError(name)


METHODS = {"dirty": lambda d: d["dirty"],
           "winner_aug": lambda d: _learned("winner_aug", d),
           "sg_k3_fresh": lambda d: _learned("sg_k3_fresh", d),
           "kin_gamma0": lambda d: _learned("kin_gamma0", d),
           # ORACLE (uses clean): noise-free clean (*) beam, the resolution-only loss. Not a method.
           "beam_only": lambda d: sfft.ifft2(sfft.fft2(d["clean"], workers=-1) * d["H"].astype(np.complex64), workers=-1).real.astype(np.float32),
           "wiener3d_a2": lambda d: wiener3d(d, 2.0),    # alpha=2 is the 2026-10-02 LOO number (alpha=1 leaves noise: 9015 0.34)
           "wiener3d_a4": lambda d: wiener3d(d, 4.0)}


# ---------------------------------------------------------------- metrics
def quad_m1(cube, velax):
    C = cube.shape[0]
    idx = np.clip(np.argmax(cube, axis=0), 1, C - 2)
    yy, xx = np.indices(idx.shape)
    ym, y0, yp = cube[idx - 1, yy, xx], cube[idx, yy, xx], cube[idx + 1, yy, xx]
    den = ym - 2 * y0 + yp
    off = np.clip(np.where(den != 0, 0.5 * (ym - yp) / np.where(den != 0, den, 1), 0.0), -0.5, 0.5)
    return velax[idx] + off * (velax[1] - velax[0])


def coherence(x, ref, mask, kmin, kmax):
    def prep(a):
        a = np.where(mask & np.isfinite(a), a, 0.0)
        return a - a[mask].mean() * mask
    n = x.shape[0]
    kr = np.hypot(np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None])
    b = (kr >= kmin) & (kr < kmax)
    X, R = np.fft.fft2(prep(x))[b], np.fft.fft2(prep(ref))[b]
    return float(np.sum(X * np.conj(R)).real / np.sqrt(np.sum(abs(X) ** 2) * np.sum(abs(R) ** 2)))


def rotation_curve_model(m1, mask, geom, au, bin_au):
    """vsys + v_phi(R) sin(i) cos(az), with v_phi fitted freely per radial bin (least squares)
    instead of forced Keplerian. A q = 0.25-0.75 disk is not Keplerian (its own gravity and
    pressure bend the curve), and on 9015 the Keplerian residual of the CLEAN cube is a global
    dipole that swamps the fingers. Subtracting the axisymmetric curve leaves only the
    non-axisymmetric motion, which is what the GI wiggle is."""
    n = m1.shape[0]
    yy, xx = np.indices((n, n), dtype=np.float64)
    pa, inc = np.radians(geom["pa_deg"]), np.radians(geom["incl_deg"])
    dx, dy = xx - geom["cx"], yy - geom["cy"]
    xr = dx * np.cos(pa) + dy * np.sin(pa)
    yr = (-dx * np.sin(pa) + dy * np.cos(pa)) / np.cos(inc)
    R, proj = np.hypot(xr, yr) * au, np.sin(inc) * np.cos(np.arctan2(yr, xr))
    centres, v = rotation_curve(m1, mask, geom, au, bin_au, R, proj)
    return geom["vsys"] + np.interp(R, centres, v) * proj


def rotation_curve(m1, mask, geom, au, bin_au, R=None, proj=None):
    """v_phi(R) (km/s) per radial bin of width bin_au: least squares of (M1 - vsys) against sin(i) cos(az) in each ring. Returns (centres_au, v_phi)
    for the bins with data. Used by rotation_curve_model and by alma_vs_published.py (comparison with exoALMA VI's published curves)."""
    if R is None:
        n = m1.shape[0]
        yy, xx = np.indices((n, n), dtype=np.float64)
        pa, inc = np.radians(geom["pa_deg"]), np.radians(geom["incl_deg"])
        dx, dy = xx - geom["cx"], yy - geom["cy"]
        xr = dx * np.cos(pa) + dy * np.sin(pa)
        yr = (-dx * np.sin(pa) + dy * np.cos(pa)) / np.cos(inc)
        R, proj = np.hypot(xr, yr) * au, np.sin(inc) * np.cos(np.arctan2(yr, xr))
    ok = mask & np.isfinite(m1)
    b = (R[ok] / bin_au).astype(int)
    dv, pj = m1[ok] - geom["vsys"], proj[ok]
    num, den = np.bincount(b, dv * pj), np.bincount(b, pj * pj)
    good = den > 1e-3 * den.max()
    centres = (np.arange(len(num)) + 0.5) * bin_au
    return centres[good], num[good] / den[good]


def spine_tracks(chan, geom, au, r_in_au, r_out_au):
    """Terry+2022 Eq. 3 on one channel map. Returns two tracks (both halves of the spine),
    each y_n in au at R = r_grid (au, deprojected)."""
    pa, cos_i = np.radians(geom["pa_deg"]), np.cos(np.radians(geom["incl_deg"]))
    m = np.array([-np.sin(pa), np.cos(pa)])          # minor axis: where v_los = vsys
    u = np.array([np.cos(pa), np.sin(pa)])           # major axis: across the spine
    dr = au                                          # one sample per pixel along the spine
    r_grid = np.arange(r_in_au, r_out_au, dr)
    t = np.arange(-int(r_out_au / au), int(r_out_au / au) + 1, dtype=np.float64)   # px across
    half = max(1, int(round(TERRY_HALF_AU / au)))
    out = []
    for sgn in (1, -1):
        s = sgn * r_grid * cos_i / au                # projected px along the minor axis
        X = geom["cx"] + s[:, None] * m[0] + t[None, :] * u[0]
        Y = geom["cy"] + s[:, None] * m[1] + t[None, :] * u[1]
        rows = map_coordinates(chan, [Y.ravel(), X.ravel()], order=1, cval=0.0).reshape(X.shape)
        y = np.empty(len(r_grid))
        for k, row in enumerate(rows):
            j = int(np.argmax(row))
            lo, hi = max(0, j - half), min(len(row), j + half + 1)
            w = np.clip(row[lo:hi], 0, None)
            y[k] = (np.sum(w * t[lo:hi]) / w.sum() if w.sum() > 0 else t[j]) * au
        out.append(y)
    return r_grid, out


def wiggle_fourier(y, dr):
    """A_wiggle (au) and lambda_wiggle (au): the strongest Fourier component of the detrended track."""
    if len(y) < 8:                       # no usable spine at this size (fit left no channel near v_sys, or R_out <= R_in)
        return float("nan"), float("nan")
    x = np.arange(len(y))
    y = y - np.polyval(np.polyfit(x, y, 1), x)
    A = 2 * np.abs(np.fft.rfft(y)) / len(y)
    k = 1 + int(np.argmax(A[1:]))
    return float(A[k]), float(len(y) * dr / k)


def score_cube(name, d, methods, outdir, ref_label="clean"):
    t0 = time.time()
    crop, velax = d["crop"], d["velax"][d["crop"]]
    au = d["au"]
    cubes = {"clean": d["clean"]}
    for m in methods:
        cubes[m] = METHODS[m](d)
        print(f"  {name}: {m} built [{time.time() - t0:.0f}s]", flush=True)
    m1 = {k: quad_m1(v[crop], velax) for k, v in cubes.items()}
    m0 = d["clean"][crop].sum(0)
    mask = np.isfinite(m0) & (np.abs(m0) > FRAC * np.abs(m0).max())
    geom = gw.fit_keplerian(m1["clean"], mask, au, m0=m0, fix_incl_deg=d["incl"])
    if d.get("pa_inits"):          # extra starts (e.g. a published PA, both redshifted sides); the lowest-cost fit wins, the default start included
        base = gw.disk_geometry_from_m0(m0, mask)
        for p in d["pa_inits"]:
            g = gw.fit_keplerian(m1["clean"], mask, au, m0=m0, fix_incl_deg=d["incl"], init=dict(base, pa_deg=float(p)))
            if g["cost"] < geom["cost"]:
                geom = g
    res = {k: gw.wiggle_residual(v, geom) for k, v in m1.items()}

    # spine geometry: R_out holds 95% of clean M0 (deprojected radius), R_in = 2 beam FWHM
    n = m0.shape[0]
    yy, xx = np.indices((n, n), dtype=np.float64)
    pa, ci = np.radians(geom["pa_deg"]), np.cos(np.radians(geom["incl_deg"]))
    dx, dy = xx - geom["cx"], yy - geom["cy"]
    rdep = np.hypot(dx * np.cos(pa) + dy * np.sin(pa), (-dx * np.sin(pa) + dy * np.cos(pa)) / ci) * au
    w = np.clip(m0, 0, None).ravel(); o = np.argsort(rdep.ravel())
    r_out = float(rdep.ravel()[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.95)])
    r_in = 2 * d["beam_au"]
    edge = min(geom["cx"], geom["cy"], n - 1 - geom["cx"], n - 1 - geom["cy"]) - 2
    r_out = min(r_out, edge * au / ci)                 # keep the spine inside the image
    rc = rotation_curve_model(m1["clean"], mask, geom, au, d["beam_au"] / 2)
    rres = {k: v - rc for k, v in m1.items()}
    rc_ok = rdep >= r_in                               # inside 2 beams even clean's M1 is beam-smeared
    full_v = d["velax"]
    sel = np.where(np.abs(full_v - geom["vsys"]) < TERRY_DV)[0]
    tracks = {k: [] for k in cubes}
    for c in sel:
        for k, v in cubes.items():
            r_grid, ys = spine_tracks(v[c].astype(np.float64), geom, au, r_in, r_out)
            tracks[k].extend(ys)
    fou = {k: np.array([wiggle_fourier(y, au) for y in tr]).reshape(-1, 2) for k, tr in tracks.items()}

    kmin, kmax = au / FINGER_AU[1], au / FINGER_AU[0]
    ref = res["clean"]
    ok = mask & np.isfinite(ref)
    rows = {}
    for k in cubes:
        r = res[k]; okk = ok & np.isfinite(r)
        have = len(tracks["clean"]) > 0
        yc = np.concatenate([y - y.mean() for y in tracks["clean"]]) if have else np.zeros(2)
        yk = np.concatenate([y - y.mean() for y in tracks[k]]) if have else np.zeros(2)
        rr, rref = rres[k], rres["clean"]
        o2 = okk & rc_ok
        rows[k] = {
            "rc_r": float(np.corrcoef(rref[o2], rr[o2])[0, 1]),
            "rc_coh_finger": coherence(rr, rref, mask & rc_ok, kmin, kmax),
            "rc_amp_ratio": float(np.std(rr[o2]) / np.std(rref[o2])),
            # absolute rotation-curve residual rms in m/s, for the method and for clean: on a disk WITHOUT a wiggle (the Keplerian twin)
            # clean's is the floor and any excess of the method's over it is spurious structure the denoiser invented
            "rc_rms_ms": float(np.sqrt(np.mean(rr[o2] ** 2)) * 1000.0), "rc_rms_clean_ms": float(np.sqrt(np.mean(rref[o2] ** 2)) * 1000.0),
            "resid_r": float(np.corrcoef(ref[okk], r[okk])[0, 1]),
            "coh_finger": coherence(r, ref, mask, kmin, kmax),
            "dv_rms_ms": float(np.sqrt(np.mean((m1[k] - m1["clean"])[okk] ** 2)) * 1000),
            "dv_med_ms": float(np.median(np.abs(m1[k] - m1["clean"])[okk]) * 1000),
            "bad_frac": float(np.mean(np.abs(m1[k] - m1["clean"])[okk] > 0.1)),
            "amp_ratio": float(np.std(r[okk]) / np.std(ref[okk])),
            "spine_r": float(np.corrcoef(yc, yk)[0, 1]) if have else float("nan"),
            "spine_rms_au": float(np.sqrt(np.mean((np.concatenate(tracks[k]) - np.concatenate(tracks["clean"])) ** 2))) if have else float("nan"),
            "A_wiggle_au": float(np.nanmean(fou[k][:, 0])) if have else float("nan"),
            "A_sem": float(np.nanstd(fou[k][:, 0], ddof=1) / np.sqrt(len(fou[k]))) if len(fou[k]) > 1 else float("nan"),
            "lam_wiggle_au": float(np.nanmean(fou[k][:, 1])) if have else float("nan"),
            "lam_sem": float(np.nanstd(fou[k][:, 1], ddof=1) / np.sqrt(len(fou[k]))) if len(fou[k]) > 1 else float("nan"),
        }
    for k in cubes:
        rows[k]["A_ratio"] = rows[k]["A_wiggle_au"] / rows["clean"]["A_wiggle_au"]
        rows[k]["lam_ratio"] = rows[k]["lam_wiggle_au"] / rows["clean"]["lam_wiggle_au"]
    # position-velocity curve along the minor axis (review of 2026-10-16: the field standard); slit 1 px and 5 px, same radii as the spine.
    # Fail-soft: a problem here must never cost a case its other scores (the keys become NaN and the figure is skipped).
    try:
        pv_prof = {wd: {k: pv.pv_profile(m1[k], geom, au, r_in, r_out, width_px=wd, mask=mask)[1] for k in cubes} for wd in (1, 5)}
        pv_r_grid = pv.pv_profile(m1["clean"], geom, au, r_in, r_out)[0]
        pv_sc = {wd: pv.pv_scores(pv_prof[wd]) for wd in pv_prof}
        # hydro discs (9074): clean M1 on the minor axis jumps by hundreds of m/s over less than two beams (two velocity components inside a spiral),
        # which no beam-limited observation keeps; the noise-free beam-convolved clean is then the fair PV truth.
        pv_bo = {wd: pv.pv_scores(pv_prof[wd], ref="beam_oracle") for wd in pv_prof} if "beam_oracle" in cubes else None
        pv.plot_pv(os.path.join(outdir, f"{name}_pv.png"), pv_r_grid, pv_prof, pv_sc,
                   f"{name}: M1 along the minor axis, i={d['incl']:.0f} deg, {au:.2f} au/px, R {r_in:.0f}-{r_out:.0f} au", ref_label=ref_label)
    except Exception as e:
        print(f"  {name}: PV curve failed ({type(e).__name__}: {e}); pv keys are NaN", flush=True)
        pv_sc = {wd: {k: {"pv_r": float("nan"), "pv_rms_ms": float("nan"), "pv_mag_ms": float("nan")} for k in cubes} for wd in (1, 5)}
        pv_bo = None
    for k in cubes:
        if pv_bo is not None:
            rows[k]["pv_r_vs_beam"], rows[k]["pv5_r_vs_beam"] = pv_bo[1][k]["pv_r"], pv_bo[5][k]["pv_r"]
        rows[k]["pv_r"], rows[k]["pv_rms_ms"] = pv_sc[1][k]["pv_r"], pv_sc[1][k]["pv_rms_ms"]
        rows[k]["pv5_r"], rows[k]["pv5_rms_ms"] = pv_sc[5][k]["pv_r"], pv_sc[5][k]["pv_rms_ms"]
        rows[k]["pv_mag_ms"], rows[k]["pv5_mag_ms"] = pv_sc[1][k]["pv_mag_ms"], pv_sc[5][k]["pv_mag_ms"]
    meta = {"au_per_px": au, "incl": d["incl"], "mstar": geom["mstar_msun"], "mstar_at_bound": geom["mstar_at_bound"],
            "pa_deg_fit": float(geom["pa_deg"]), "fit_cost": float(geom["cost"]), "vsys_fit": float(geom["vsys"]), "cx_fit": float(geom["cx"]), "cy_fit": float(geom["cy"]),
            "r_in_au": r_in, "r_out_au": r_out, "n_tracks": len(tracks["clean"]), "n_spine_channels": int(len(sel)),
            "dv_kms": float(full_v[1] - full_v[0]), "seen_in_training": sorted(m for m, v in SEEN.items() if name in v)}
    figure(name, cubes, m0, m1, rres, mask, geom, tracks, r_grid, sel, rows, meta, outdir)
    print(f"  {name}: scored [{time.time() - t0:.0f}s]", flush=True)
    return {"meta": meta, "methods": rows}


def figure(name, cubes, m0, m1, res, mask, geom, tracks, r_grid, sel, rows, meta, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(cubes)
    c0 = sel[len(sel) // 2]                          # the channel nearest vsys
    i0 = 2 * (len(sel) // 2)                          # its first track
    au = meta["au_per_px"]
    vr = np.nanpercentile(np.abs(res["clean"][mask]), 98)
    v1, v2 = np.nanpercentile(m1["clean"][mask], [1, 99])
    fig, ax = plt.subplots(len(keys), 5, figsize=(22, 4.3 * len(keys)))
    for i, k in enumerate(keys):
        mk = cubes[k][c0].astype(np.float64)
        ax[i, 0].imshow(cubes[k].sum(0), origin="lower", cmap="inferno")
        ax[i, 0].set_title(f"{k}: M0")
        ax[i, 1].imshow(np.where(mask, m1[k], np.nan), origin="lower", cmap="RdBu_r", vmin=v1, vmax=v2)
        ax[i, 1].set_title("M1 (quadratic)")
        ax[i, 2].imshow(np.where(mask, res[k], np.nan), origin="lower", cmap="RdBu_r", vmin=-vr, vmax=vr)
        ax[i, 2].set_title(f"M1 - rotation curve  r={rows[k]['rc_r']:.3f}  coh={rows[k]['rc_coh_finger']:.3f}")
        ax[i, 3].imshow(mk, origin="lower", cmap="magma")
        pa = np.radians(geom["pa_deg"]); ci = np.cos(np.radians(geom["incl_deg"]))
        for j, sgn in enumerate((1, -1)):
            s = sgn * r_grid * ci / au; t = tracks[k][i0 + j] / au
            ax[i, 3].plot(geom["cx"] - s * np.sin(pa) + t * np.cos(pa), geom["cy"] + s * np.cos(pa) + t * np.sin(pa), "c-", lw=1)
        ax[i, 3].set_title(f"systemic channel + Terry spine  spine_r={rows[k]['spine_r']:.3f}")
        ax[i, 4].plot(r_grid, tracks["clean"][i0], "k-", lw=2, label="clean")
        if k != "clean":
            ax[i, 4].plot(r_grid, tracks[k][i0], "r-", lw=1.2, label=k)
        ax[i, 4].set_xlabel("R (au)"); ax[i, 4].set_ylabel("y_n (au)"); ax[i, 4].legend(fontsize=8)
        ax[i, 4].set_title(f"A={rows[k]['A_wiggle_au']:.1f}+/-{rows[k]['A_sem']:.1f} au  "
                           f"lam={rows[k]['lam_wiggle_au']:.0f} au  dv_rms={rows[k]['dv_rms_ms']:.0f} m/s")
    for a in ax[:, :4].ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{name}: i={meta['incl']:.0f} deg, {au:.2f} au/px, R {meta['r_in_au']:.0f}-{meta['r_out_au']:.0f} au, "
                 f"{meta['n_tracks']} spine tracks over |v-vsys|<{TERRY_DV} km/s", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- self-test of the spine extractor
def selftest():
    """A spine with a known 12 au, 80 au-wavelength sinusoid must come back as A ~ 12, lambda ~ 80."""
    au, n = 1.33, 401
    geom = {"cx": 200.0, "cy": 200.0, "pa_deg": 0.0, "incl_deg": 0.0}
    yy, xx = np.indices((n, n), dtype=np.float64)
    R = (yy - 200) * au                                # spine along +y (minor axis at pa=0)
    off = 12.0 * np.sin(2 * np.pi * R / 80.0)          # au, across the spine (x)
    chan = np.exp(-0.5 * (((xx - 200) * au - off) / 5.0) ** 2) * (np.abs(R) < 250)
    r_grid, (ya, yb) = spine_tracks(chan, geom, au, 20.0, 240.0)
    A, lam = wiggle_fourier(ya, au)
    truth = 12.0 * np.sin(2 * np.pi * r_grid / 80.0)
    assert np.sqrt(np.mean((ya - truth) ** 2)) < 1.0, "spine centroid off"
    assert abs(A - 12.0) < 2.5 and abs(lam - 80.0) < 10, (A, lam)
    print(f"selftest: A={A:.1f} au (12), lambda={lam:.1f} au (80) PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="9015,9019,9025,9032,9074,sgv2,an01,an02,an03,an04,an05,an06")
    ap.add_argument("--methods", default="beam_only,dirty,wiener3d_a2,wiener3d_a4,winner_aug,sg_k3_fresh,kin_gamma0")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        sys.exit()
    outdir = os.path.join(ROOT, "results/wiggle-scorecard", datetime.date.today().isoformat())
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, "scorecard.json")
    allres = json.load(open(jpath)) if os.path.exists(jpath) else {}
    for name in a.only.split(","):
        print(f"{name} ...", flush=True)
        allres[name] = score_cube(name, load(name), a.methods.split(","), outdir)
        json.dump(allres, open(jpath, "w"), indent=1)
    cols = ["rc_r", "rc_coh_finger", "rc_amp_ratio", "resid_r", "coh_finger", "dv_med_ms", "bad_frac", "dv_rms_ms", "amp_ratio", "spine_r", "spine_rms_au", "A_ratio", "lam_ratio"]
    lines = ["| cube | method | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for name, r in allres.items():
        for k, v in r["methods"].items():
            lines.append(f"| {name} | {k}{chr(0x2020) if name in SEEN.get(k, ()) else ''} | " + " | ".join(f"{v[c]:.3f}" for c in cols) + " |")
    open(os.path.join(outdir, "scorecard.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
