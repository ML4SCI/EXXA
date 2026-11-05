"""
Phase 0 of the physics-informed plan: what actually produced the dirty cubes?

DDRM constrains the reverse diffusion chain with a known forward operator `A` such that
`dirty = A(clean) + noise`. The method is only worth building if `A` is a non-trivial
operator we can write down. Two possibilities with opposite conclusions:

  1. dirty = clean (*) beam + noise   -> a real convolution, DDRM applies, build it.
  2. dirty = clean + noise            -> A = I, DDRM degenerates to the conditional DDPM
                                         that already exists, and there is nothing to gain.

BMAJ/BMIN/BPA sitting in the headers and the MCFOST provenance both point at (1), but that
is an inference and this module is the check.

The discriminator is the azimuthally averaged power spectrum. Write P_d and P_c for the
dirty and clean spectra and |B(k)| for the beam's transfer function:

    convolution:  P_d(k) = |B(k)|^2 P_c(k) + N     -> the ratio DIPS below 1 at mid-k,
                                                      because |B(k)| < 1 there
    additive:     P_d(k) =            P_c(k) + N   -> the ratio only ever RISES above 1

A convolution suppresses intermediate spatial frequencies. Additive noise cannot suppress
anything; it only adds power. That asymmetry is what makes the two easy to tell apart, and
it survives not knowing the noise level, because the noise term is estimated and removed
before the ratio is formed.

There is a third answer the plan cares about. If a convolution is present but its transfer
function is not Gaussian, a true interferometric dirty beam was used, sidelobes and all, and
a Gaussian `A` would make DDRM enforce the wrong constraint. That case needs the PSF image
from whoever generated the cubes, so it is reported separately rather than folded into (1).
"""

from __future__ import annotations

import numpy as np

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # 0.42466

# Relative-residual thresholds for the forward-model fit. See phase0_report for the measured
# separation these sit between; both have well over an order of magnitude of margin.
NO_BEAM_TOL = 0.05   # below this, P_d = A*P_c + N explains the data and there is no beam
GAUSS_TOL = 0.05     # below this, a Gaussian beam explains it; above, a real dirty beam


def pixel_scale_arcsec(header) -> float | None:
    """
    Arcsec per pixel from CDELT1/CDELT2, or CD1_1/CD2_2 for headers written with a CD
    matrix instead. None when the header carries neither.

    Nothing in src/ read CDELT before this: `beam_features_of` takes only BPA/BMAJ/BMIN,
    which is enough to describe a beam in angular units but not to build a kernel in
    pixels.
    """
    if header is None:
        return None
    for k in ("CDELT2", "CDELT1", "CD2_2", "CD1_1"):
        if k in header:
            v = abs(float(header[k])) * 3600.0
            if v > 0:
                return v
    return None


def beam_kernel_of(header, shape=None) -> np.ndarray | None:
    """
    Elliptical Gaussian beam kernel in pixel units, rotated by BPA, normalised to unit sum.

    This is the `A` of Phase 1, and the restoring beam rather than the dirty beam: a real
    interferometric PSF has sidelobes from incomplete uv coverage and a Gaussian has none.
    Whether that approximation is safe here is exactly what `phase0_report` decides.

    BMAJ/BMIN are FWHM in degrees; BPA is degrees east of north, so the major axis runs
    along (-sin BPA, cos BPA) in (x, y). Returns None when the header lacks the beam keys
    or the pixel scale.
    """
    if header is None or not all(k in header for k in ("BMAJ", "BMIN", "BPA")):
        return None
    scale = pixel_scale_arcsec(header)
    if scale is None:
        return None

    sig_maj = float(header["BMAJ"]) * 3600.0 / scale * FWHM_TO_SIGMA
    sig_min = float(header["BMIN"]) * 3600.0 / scale * FWHM_TO_SIGMA
    if sig_maj <= 0 or sig_min <= 0:
        return None

    # Six sigma of support on the major axis, forced odd so the peak sits on a pixel.
    half = max(1, int(np.ceil(3.0 * sig_maj)))
    if shape is not None:
        half = min(half, min(shape) // 2)
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)

    bpa = np.deg2rad(float(header["BPA"]))
    # Rotate into the beam frame: u along the major axis, v along the minor axis.
    u = -x * np.sin(bpa) + y * np.cos(bpa)
    v = x * np.cos(bpa) + y * np.sin(bpa)

    k = np.exp(-0.5 * ((u / sig_maj) ** 2 + (v / sig_min) ** 2))
    return (k / k.sum()).astype(np.float64)


def radial_power_spectrum(img: np.ndarray, n_bins: int = 60):
    """
    Azimuthally averaged power spectrum. Returns (k, power) with k in cycles per pixel,
    running from 0 to 0.5 (Nyquist).

    A Hann window is applied first. Without it the image edges act as a step function and
    leak broadband power across every frequency, which is indistinguishable from the
    high-frequency noise floor this module has to measure.
    """
    img = np.asarray(img, dtype=np.float64)
    img = img - img.mean()

    h, w = img.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    f = np.fft.fftshift(np.abs(np.fft.fft2(img * win)) ** 2)

    ky = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)

    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.digitize(kr.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < n_bins)

    power = np.bincount(idx[valid], weights=f.ravel()[valid], minlength=n_bins)
    count = np.bincount(idx[valid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        power = np.where(count > 0, power / np.maximum(count, 1), np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), power


def transfer_function(clean, dirty, n_bins: int = 60):
    """
    Estimate |B(k)| from a clean/dirty pair, plus the k grid and the noise power removed.

    Both inputs may be 2D (one channel) or 3D (C, H, W), in which case spectra are averaged
    over channels, which is the cheapest way to beat down the variance of a single channel's
    periodogram.

    Returns `(k, b_raw, b, noise, pc, pd)` with two transfer estimates, because the two
    questions Phase 0 asks want different ones:

      b_raw = sqrt(P_d / P_c)                  -- for "is there a convolution at all"
      b     = sqrt(max(P_d - N, 0) / P_c)      -- for "what shape is it"

    `b_raw` carries the noise floor, but it is the honest statistic for the dip test:
    additive noise can only ADD power, so P_d = P_c + N gives a raw ratio at or above 1
    everywhere, no matter how large N is or how badly it was estimated. A dip below 1 is
    therefore proof of suppression rather than an artefact of the noise model.

    `b` removes an estimated floor so the beam's shape can be fitted. N is the median of
    P_dirty over the top decile of k. That estimate is biased high whenever the clean field
    still has power up there, and subtracting a too-large N pushes the ratio below 1 where
    nothing was suppressed -- which is exactly why the verdict must not be read off `b`.
    """
    clean = np.asarray(clean, dtype=np.float64)
    dirty = np.asarray(dirty, dtype=np.float64)
    if clean.ndim == 2:
        clean, dirty = clean[None], dirty[None]

    k = None
    pc = np.zeros(n_bins)
    pd = np.zeros(n_bins)
    for c, d in zip(clean, dirty):
        k, p = radial_power_spectrum(c, n_bins); pc += np.nan_to_num(p)
        _, p = radial_power_spectrum(d, n_bins); pd += np.nan_to_num(p)
    pc /= len(clean)
    pd /= len(dirty)

    tail = k >= 0.45
    noise = float(np.median(pd[tail])) if tail.any() else 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(pc > 0, pd / pc, np.nan)
        ratio = np.where(pc > 0, np.maximum(pd - noise, 0.0) / pc, np.nan)
    return k, np.sqrt(raw), np.sqrt(ratio), noise, pc, pd


def _measurement_band(k, pc, pd, noise, k_max: float = 0.45):
    """
    Bins where BOTH spectra carry real information.

    The dirty-side cut is the obvious one: beyond its noise floor the ratio is noise over
    noise. The clean-side cut is the one that was missing, and it cost a run.

    These cubes are Jy/beam, so the clean map is already beam-convolved and its power falls
    thirteen orders of magnitude by k ~ 0.10, then FLATTENS onto the float32 floor near
    1e-12. Past that point neither spectrum is physical, yet their ratio settles to a
    plausible-looking constant near 2.2. Including that region alongside the real one, where
    the ratio climbs from 1.0 to 58, leaves no (A, N) able to fit either, which is exactly
    how Phase 0's fourth run produced sigma = 0 with a residual of 0.42 to 0.89.

    The clean floor is estimated the same way as the dirty one: the median of the top decile
    of k, where nothing physical survives.
    """
    tail = k >= k_max
    pc_floor = float(np.median(pc[tail])) if tail.any() else 0.0
    return (np.isfinite(pc) & np.isfinite(pd)
            & (pc > 3.0 * pc_floor) & (pd > 3.0 * noise) & (k < k_max))


def fit_forward_model(k, pc, pd, band, noise_basis=None, max_sigma_px: float = 25.0,
                      n_sigma: int = 400):
    """
    Fit the forward model directly instead of thresholding a ratio:

        P_dirty(k) = A * exp(-4 pi^2 sigma^2 k^2) * P_clean(k) + N * noise_basis(k)

    `noise_basis` defaults to 1 (white noise). For a Jy/beam map it should be |B_header(k)|^2:
    the noise in such a map has been through the same beam as the signal, so its power
    spectrum is beam-shaped rather than flat. On this project's cubes a flat basis cannot fit
    -- the implied N runs from 16 down to 2e-9 across the band -- while a beam-shaped one
    holds it within about 1.5 orders.

    Three unknowns. `A` absorbs any intensity-scale factor between the two maps (Jy/beam
    against Jy/pixel is a factor of the beam area, order 200 here), `N` is the noise floor,
    and `sigma` is the beam. Fitting them together is what makes the answer scale-invariant
    by construction rather than by a normalisation that has to be chosen correctly.

    Two earlier versions of this check failed on exactly that choice. Testing whether the raw
    ratio dips below 1 fails when A != 1. Normalising the ratio by its own low-k level then
    taking the minimum over the whole band fails the other way, because for a monotonically
    RISING ratio the lowest bin is by construction below the median of the lowest bins, so a
    dip appears where nothing was suppressed. Both produced confident, wrong verdicts on the
    project's real cubes.

    Given sigma the model is linear in (A, N), so a 1D scan over sigma with a linear solve at
    each step is exact and needs no optimiser. Residuals are relative, because P_clean spans
    several orders of magnitude and an unweighted fit would see only the lowest frequencies.

    Returns sigma_px, A, N, the relative RMS residual, and the residual of the sigma = 0 model
    (pure addition, no convolution) so the two can be compared as hypotheses.
    """
    kb, pcb, pdb = k[band], pc[band], pd[band]
    nb = (np.ones_like(pcb) if noise_basis is None else np.asarray(noise_basis)[band])
    ok = (pdb > 0) & (pcb > 0)
    kb, pcb, pdb, nb = kb[ok], pcb[ok], pdb[ok], nb[ok]
    if kb.size < 6:
        return None

    # Noise gets BOTH a white term and a beam-shaped one, rather than a guess between them.
    # A Jy/beam map's noise has been through the beam, so its spectrum is beam-shaped; a
    # synthetic cube with noise added after blurring has a flat one. Fitting both and letting
    # the data weight them covers either without the caller having to know which it has.
    cols = [np.ones_like(pcb)] if noise_basis is None else [np.ones_like(pcb), nb]

    def solve(sigma):
        g = np.exp(-4.0 * np.pi ** 2 * sigma ** 2 * kb ** 2)
        # rows weighted by 1/pd -> least squares on the RELATIVE error
        M = np.stack([g * pcb] + cols, axis=1) / pdb[:, None]
        # Column scaling before the solve, unscaled after. Without it this breaks whenever
        # P_dirty/P_clean swings over several orders of magnitude across the band -- which
        # is not a hypothetical: on the 2026-08-27 self-gravitating v2 cube the ratio runs
        # from ~1e5 at the lowest k to ~1e-4 by mid-band, and the unscaled solve returned
        # A=2.4e-5 against a naive low-k estimate of A~1e5, nine orders of magnitude off.
        # This is standard Jacobi/Ruiz preconditioning: an ill-conditioned lstsq is a
        # numerical failure, not evidence about the physics, and it must not be read as one.
        scale = np.maximum(np.abs(M).max(axis=0), 1e-300)
        coef_scaled, *_ = np.linalg.lstsq(M / scale, np.ones_like(pdb), rcond=None)
        coef = np.maximum(coef_scaled / scale, 0.0)       # A, N are powers: non-negative
        model = coef[0] * g * pcb + sum(c * col for c, col in zip(coef[1:], cols))
        resid = float(np.sqrt(np.mean(((model - pdb) / pdb) ** 2)))
        return coef, resid

    sigmas = np.linspace(0.0, max_sigma_px, n_sigma)
    best = min((solve(sg) + (sg,) for sg in sigmas), key=lambda t: t[1])
    coef, resid, sigma = best
    coef0, resid0 = solve(0.0)

    return {"sigma_px": float(sigma), "A": float(coef[0]), "N": float(coef[1]),
            "N_beam": float(coef[2]) if len(coef) > 2 else 0.0,
            "residual": resid, "residual_no_convolution": resid0,
            "k_band": kb, "pc_band": pcb, "pd_band": pdb}


def phase0_report(clean, dirty, header=None, n_bins: int = 60) -> dict:
    """
    Settle Phase 0. Returns a dict with a `verdict` of:

      "no_convolution"           -> A = I. DDRM degenerates to the existing conditional
                                    DDPM, and VIREO's consistency term must not ship.
      "gaussian_convolution"     -> DDRM applies and the header's beam is the operator.
      "non_gaussian_convolution" -> a real dirty beam with sidelobes. DDRM still applies but
                                    a Gaussian A enforces the wrong constraint; the PSF image
                                    is needed from whoever generated the cubes.
      "indeterminate"            -> the data cannot decide. Do not build on it.

    The verdict comes from fitting the forward model (see `fit_forward_model`) and comparing
    it against the same model with sigma forced to zero. That is a comparison of two
    hypotheses on equal footing, rather than a threshold on a ratio whose normalisation has
    to be guessed -- two earlier versions of this function each got a confident wrong answer
    out of exactly that guess.
    """
    k, b_raw, b, noise, pc, pd = transfer_function(clean, dirty, n_bins)

    # The measurement is only meaningful where the dirty spectrum still stands above its own
    # noise floor. Beyond that it is noise divided by noise.
    #
    # The band must not be set as a fraction of the PEAK power instead: a disk's spectrum is
    # steeply red, so "1% of the maximum" lands near k = 0.04 and discards exactly the range
    # where a Gaussian beam and a sidelobed one stop looking alike.
    band = _measurement_band(k, pc, pd, noise)
    if band.sum() < 6:
        return {"verdict": "indeterminate",
                "reason": "too few bins where both spectra rise above their own floors",
                "k": k, "transfer": b, "noise_power": noise}

    # A Jy/beam map's noise has been through the beam too, so its spectrum is beam-shaped.
    sigma_hdr_early = _header_sigma_px(header) if header is not None else None
    nb = (np.exp(-4.0 * np.pi ** 2 * sigma_hdr_early ** 2 * k ** 2)
          if sigma_hdr_early else None)
    fit = fit_forward_model(k, pc, pd, band, noise_basis=nb)
    if fit is None:
        return {"verdict": "indeterminate", "reason": "forward-model fit did not converge",
                "k": k, "transfer": b, "noise_power": noise}

    kb = k[band]
    out = {
        "k": k, "transfer": b, "transfer_raw": b_raw, "noise_power": noise,
        "clean_power": pc, "dirty_power": pd,
        "fit_sigma_px": fit["sigma_px"], "scale_factor": fit["A"],
        "fit_noise": fit["N"], "fit_residual": fit["residual"],
        "residual_no_convolution": fit["residual_no_convolution"],
        "band_k_max": float(kb.max()), "band_bins": int(band.sum()),
    }
    # How much better the convolution hypothesis explains the data than "no convolution".
    out["convolution_gain"] = (fit["residual_no_convolution"] / fit["residual"]
                               if fit["residual"] > 0 else float("inf"))

    sigma_hdr = _header_sigma_px(header) if header is not None else None
    if sigma_hdr:
        out["header_sigma_px"] = sigma_hdr
        out["k_half_from_header"] = float(
            np.sqrt(np.log(2.0) / (2.0 * np.pi ** 2 * sigma_hdr ** 2)))

    # --- no convolution -----------------------------------------------------------------
    # The test is whether the NO-BEAM model, P_d = A*P_c + N, is adequate on its own. It is
    # deliberately not "does a Gaussian beat it": a sidelobed beam is not Gaussian, so
    # neither model fits it and the comparison declares a tie, calling a real convolution
    # no_convolution. Measured on synthetic cases (6 channels, ~40 bins):
    #
    #     additive noise, x1 to x25 scale ... 0.0008 to 0.0107
    #     sidelobed beam ..................... 0.8897
    #     Gaussian beam, sigma 1 to 3 ........ 2.86 to 3.21
    #
    # NO_BEAM_TOL sits at 0.05, about 5x the observed scatter floor of the radially averaged
    # spectrum and two orders below anything with a beam in it.
    if fit["residual_no_convolution"] < NO_BEAM_TOL:
        # ...unless the band never reached where the header's own beam would act, in which
        # case the null is uninformative rather than negative.
        if sigma_hdr and out["band_k_max"] < out["k_half_from_header"]:
            out["verdict"] = "indeterminate"
            out["reason"] = (
                f"no beam found, but the band only reaches k = {out['band_k_max']:.3f} and "
                f"the header's beam (sigma {sigma_hdr:.1f} px) would not act until "
                f"k = {out['k_half_from_header']:.3f}, so it could not have been seen")
            return out
        out["verdict"] = "no_convolution"
        out["reason"] = (
            f"P_d = A*P_c + N already fits to {fit['residual_no_convolution']:.4f} with no "
            f"beam at all; dirty = clean + noise, so A = I")
        return out

    # A fit that lands on sigma = 0 found NO beam. If the no-beam model was also inadequate,
    # then neither hypothesis describes the data and the honest answer is that the method does
    # not apply here -- not "a beam with strange shape". Without this guard the Gaussian
    # comparison runs against a flat transfer and reports RMS values of 1e32.
    if out["fit_sigma_px"] < 0.5:
        out["verdict"] = "indeterminate"
        out["reason"] = (
            f"neither model fits: the best beam is {out['fit_sigma_px']:.2f} px (i.e. none) "
            f"yet the no-beam model still misses by {fit['residual_no_convolution']:.4f}. "
            f"P_d is not A*G(k)*P_c + N for any Gaussian G, so something other than a beam "
            f"differs between these two cubes. Run phase0_diagnostics to see the spectra.")
        return out

    # --- a convolution is present; Gaussian or not? --------------------------------------
    # Compare the measured transfer against the fitted Gaussian. A is already divided out, so
    # what is left is |B(k)|^2 in its own right.
    g_meas = np.maximum(fit["pd_band"] - fit["N"], 0.0) / (fit["A"] * fit["pc_band"])
    g_fit = np.exp(-4.0 * np.pi ** 2 * fit["sigma_px"] ** 2 * fit["k_band"] ** 2)
    live = g_fit > 0.02          # where the beam has not already killed the signal
    rms = float(np.sqrt(np.mean((g_meas[live] - g_fit[live]) ** 2))) if live.any() else np.inf
    out["gaussian_rms_error"] = rms
    out["gaussian_sigma_px"] = fit["sigma_px"]

    if fit["residual"] < GAUSS_TOL and rms < 0.05:
        out["verdict"] = "gaussian_convolution"
        out["reason"] = (f"a {fit['sigma_px']:.2f} px Gaussian beam explains the data "
                         f"({out['convolution_gain']:.1f}x better than no beam), matching to "
                         f"{rms:.4f} RMS")
    else:
        out["verdict"] = "non_gaussian_convolution"
        out["reason"] = (f"a beam is present ({out['convolution_gain']:.1f}x better than no "
                         f"beam, best fit {fit['sigma_px']:.2f} px) but the transfer departs "
                         f"from a Gaussian by {rms:.4f} RMS; a real dirty beam, so the PSF "
                         f"image is needed")
    return out


def fit_gaussian_transfer(k: np.ndarray, b: np.ndarray) -> dict:
    """
    Fit |B(k)| = exp(-2 pi^2 sigma^2 k^2) by least squares on log|B|, which is linear in
    k^2. Returns the implied beam sigma in pixels and the RMS error of the fit.

    Only bins with |B| above a floor are used: below it the log is dominated by the noise
    subtraction and would drag the fit.
    """
    m = np.isfinite(b) & (b > 0.05)
    if m.sum() < 4:
        return {"gaussian_sigma_px": float("nan"), "gaussian_rms_error": float("inf")}

    x = k[m] ** 2
    y = np.log(b[m])
    slope = float(np.sum(x * y) / np.sum(x * x))  # forced through log|B(0)| = 0
    sigma_sq = -slope / (2.0 * np.pi ** 2)
    sigma = float(np.sqrt(sigma_sq)) if sigma_sq > 0 else float("nan")

    rms = float(np.sqrt(np.mean((b[m] - np.exp(slope * x)) ** 2)))
    return {"gaussian_sigma_px": sigma, "gaussian_rms_error": rms}


def _header_sigma_px(header) -> float | None:
    """Geometric-mean beam sigma in pixels predicted by the header, for cross-checking."""
    scale = pixel_scale_arcsec(header)
    if scale is None or not all(k in header for k in ("BMAJ", "BMIN")):
        return None
    maj = float(header["BMAJ"]) * 3600.0 / scale * FWHM_TO_SIGMA
    mn = float(header["BMIN"]) * 3600.0 / scale * FWHM_TO_SIGMA
    return float(np.sqrt(maj * mn))


def format_report(r: dict) -> str:
    sig = r.get("gaussian_sigma_px")
    have_sig = sig is not None and np.isfinite(sig)

    lines = [f"verdict: {r['verdict'].upper()}", f"  {r.get('reason', '')}"]
    if r.get("fit_sigma_px") is not None:
        lines.append(f"  best-fit beam sigma  = {r['fit_sigma_px']:.2f} px")
        lines.append(f"  fit residual         = {r['fit_residual']:.4f}  "
                     f"(no-beam model: {r['residual_no_convolution']:.4f}, "
                     f"{r['convolution_gain']:.2f}x worse)")
    if r.get("scale_factor") is not None:
        sc = r["scale_factor"]
        note = ("   <- clean and dirty are NOT on one intensity scale (Jy/beam vs Jy/pixel?)"
                if (sc > 3.0 or sc < 0.33) else "")
        lines.append(f"  fitted amplitude A   = {sc:.3f}{note}")
    if have_sig:
        lines.append(f"  Gaussian match       = {r['gaussian_rms_error']:.4f} RMS")
    if "band_k_max" in r:
        line = (f"  band reaches k = {r['band_k_max']:.3f} cyc/px over {r['band_bins']} bins")
        if r.get("k_half_from_header"):
            kh = r["k_half_from_header"]
            line += (f"; header's beam would act from k = {kh:.3f}"
                     f"{'  <- band too narrow to see it' if r['band_k_max'] < kh else '  <- wide enough'}")
        lines.append(line)
    if r.get("header_sigma_px"):
        lines.append(f"  header beam sigma    = {r['header_sigma_px']:.2f} px")
        if have_sig:
            ratio = sig / r["header_sigma_px"]
            lines.append(f"  measured / header    = {ratio:.2f}"
                         f"{'  <- consistent' if 0.8 < ratio < 1.25 else '  <- MISMATCH'}")
    return "\n".join(lines)


def load_pair(clean_path: str, dirty_path: str, max_channels: int = 8):
    """
    Load a clean/dirty FITS pair and keep the `max_channels` channels with the most SIGNAL.

    Not evenly spaced channels. `np.linspace(0, n-1, k)` includes channel 0 and channel n-1,
    the extreme high-velocity ends, which the sampling note (2026-06-18) calls
    "mostly continuum with little signal". In those the clean power is near zero, so
    P_d / P_c explodes for reasons that have nothing to do with a beam, and averaging them in
    with real channels destroys the fit. That is what made the third Phase 0 run return
    sigma = 0 with a no-beam residual of 0.55 to 0.90: neither model could fit, because the
    input was half empty channels.

    Ranking by the clean channel's standard deviation picks the line-bright channels without
    assuming anything about where in the cube they sit.
    """
    from astropy.io import fits

    with fits.open(clean_path, memmap=False) as hdul:
        clean = np.squeeze(hdul[0].data).astype(np.float64)
        clean_header = hdul[0].header
    with fits.open(dirty_path, memmap=False) as hdul:
        dirty = np.squeeze(hdul[0].data).astype(np.float64)
        header = hdul[0].header

    if clean.ndim == 3 and clean.shape[0] > max_channels:
        strength = clean.reshape(clean.shape[0], -1).std(axis=1)
        idx = np.sort(np.argsort(strength)[-max_channels:])
        clean, dirty = clean[idx], dirty[idx]
    return clean, dirty, header, clean_header


def phase0_from_fits(clean_path: str, dirty_path: str, max_channels: int = 8) -> dict:
    """
    Run Phase 0 on a clean/dirty FITS pair, on its line-bright channels (see `load_pair`).
    """
    clean, dirty, header, _ = load_pair(clean_path, dirty_path, max_channels)
    return phase0_report(clean, dirty, header)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python -m src.evaluation.forward_operator <clean.fits> <dirty.fits>")
        raise SystemExit(2)
    print(format_report(phase0_from_fits(sys.argv[1], sys.argv[2])))


def phase0_diagnostics(clean_path: str, dirty_path: str, max_channels: int = 8) -> str:
    """
    Dump the raw material behind a Phase 0 verdict, so a surprising answer can be diagnosed
    from data instead of guessed at.

    Three versions of the verdict logic were wrong before this existed, each time because a
    summary statistic hid what the spectra were actually doing. Print this alongside any
    result that looks strange.
    """
    clean, dirty, header, clean_header = load_pair(clean_path, dirty_path, max_channels)
    L = []
    L.append(f"clean  shape={clean.shape} range=[{clean.min():.4g}, {clean.max():.4g}] "
             f"BUNIT={clean_header.get('BUNIT', '?')}")
    L.append(f"dirty  shape={dirty.shape} range=[{dirty.min():.4g}, {dirty.max():.4g}] "
             f"BUNIT={header.get('BUNIT', '?')}")
    if clean.shape != dirty.shape:
        L.append("  ** shapes differ -- the k axes are not comparable and nothing below is "
                 "meaningful **")
    for key in ("BMAJ", "BMIN", "BPA", "CDELT1", "CDELT2", "BTYPE", "TELESCOP"):
        cv, dv = clean_header.get(key), header.get(key)
        if cv is not None or dv is not None:
            flag = "  <- differs" if (cv is not None and dv is not None and cv != dv) else ""
            L.append(f"  {key:9s} clean={cv!s:22s} dirty={dv!s}{flag}")

    ps = pixel_scale_arcsec(header)
    L.append(f"  pixel scale = {ps if ps else '?'} arcsec/px | "
             f"header beam sigma = {_header_sigma_px(header)} px")

    k, b_raw, b, noise, pc, pd = transfer_function(clean, dirty)
    L.append("")
    L.append(f"  {'k':>7s} {'P_clean':>12s} {'P_dirty':>12s} {'ratio':>9s} {'sqrt':>7s}")
    for i in range(0, len(k), max(1, len(k) // 18)):
        r = pd[i] / pc[i] if pc[i] > 0 else float("nan")
        L.append(f"  {k[i]:7.4f} {pc[i]:12.4g} {pd[i]:12.4g} {r:9.3f} {np.sqrt(abs(r)):7.3f}")
    L.append(f"  noise floor estimate (median P_dirty above k=0.45) = {noise:.4g}")

    L.append("")
    L.append("  per-channel clean std (which channels were selected):")
    L.append("    " + "  ".join(f"{v:.3g}" for v in clean.reshape(len(clean), -1).std(axis=1)))
    return "\n".join(L)


def beam_cutoff_k(header, level: float = 0.1) -> float | None:
    """
    Spatial frequency where the header's beam has suppressed power to `level`.

    |B(k)|^2 = exp(-4 pi^2 sigma^2 k^2), so |B|^2 = level at
    k = sqrt(ln(1/level) / (4 pi^2 sigma^2)). For this project's beams (sigma 6.2 to
    10.3 px) that is k ~ 0.023 to 0.039 cycles/pixel, i.e. structures below roughly
    25 to 43 pixels across.
    """
    sigma = _header_sigma_px(header)
    if not sigma:
        return None
    return float(np.sqrt(np.log(1.0 / level) / (4.0 * np.pi ** 2 * sigma ** 2)))


def out_of_band_power(clean, denoised, header, level: float = 0.1) -> dict:
    """
    What fraction of a model's ERROR sits above the beam's cutoff.

    Phase 0 established these cubes are already beam-convolved, so the clean map is
    band-limited by construction. A prediction carrying power above the cutoff is asserting
    structure the instrument could not have measured, and a loss can penalise that with one
    FFT per step. This measures whether such a loss would be aimed at the actual failure.

    The measurement is on the RESIDUAL, `denoised - clean`, not on the prediction. A global
    power fraction of the prediction cannot see a local artifact: the clean field's own
    out-of-band share is ~3e-4 of a total dominated by the lowest frequencies, so one
    invented blob is some three orders of magnitude too small to move it. The residual is
    near zero everywhere the model is right, so the error is what its spectrum describes.

      out_of_band_frac >> clean's   the error is sharper than the beam -- a band-limit
                                    penalty targets it directly
      out_of_band_frac ~ clean's    the error is beam-scale -- band-limiting cannot see it,
                                    and a VIREO-style band constraint would be aimed at the
                                    wrong thing

    Inputs are 2D or (C, H, W); spectra are averaged over channels.
    """
    k_cut = beam_cutoff_k(header, level)
    if k_cut is None:
        return {"error": "header has no beam or pixel scale"}

    clean = np.asarray(clean, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)
    if clean.ndim == 2:
        clean, denoised = clean[None], denoised[None]

    k, pc, pr = None, None, None
    for c, d in zip(clean, denoised):
        k, p = radial_power_spectrum(c)
        pc = np.nan_to_num(p) if pc is None else pc + np.nan_to_num(p)
        _, p = radial_power_spectrum(d - c)
        pr = np.nan_to_num(p) if pr is None else pr + np.nan_to_num(p)

    above = k > k_cut
    fc = float(pc[above].sum() / max(pc.sum(), 1e-300))
    fr = float(pr[above].sum() / max(pr.sum(), 1e-300))
    return {"k_cut": k_cut, "beam_sigma_px": _header_sigma_px(header),
            "min_resolvable_px": float(1.0 / k_cut),
            "clean_out_of_band_frac": fc,
            "residual_out_of_band_frac": fr,
            "excess": float(fr / fc) if fc > 0 else float("inf")}


def estimate_beam_from_pair(clean, dirty, crop: int = 129):
    """
    Recover the forward operator from a clean/dirty pair, when one exists.

    Uses the cross-spectrum estimator `B = <D conj(C)> / <|C|^2>` rather than dividing
    channel by channel. The noise in D is uncorrelated with C, so it averages towards zero
    across channels instead of blowing up wherever `C` happens to be small.

    Only meaningful when there IS an operator between the two cubes, which
    `phase0_report` decides. For the line-emission training set (`A = I`) this returns a
    delta function and says nothing.

    Measured on the SG v2 self-gravitating pair (2026-08-21): peak 0.911 at the exact
    centre, sum ~0, FWHM 5 px, negative ring reaching -2.8% of peak from r ~ 26 px. Peak near
    1 with zero total flux is the signature of an interferometric DIRTY beam, and the negative
    ring is why a Gaussian `A` would be the wrong operator. Convolving the clean cube with it
    reproduced held-out channels at correlation 0.994, residual 16.6% of the dirty cube's own
    rms, which is the noise term.

    Returns the beam centred in a `crop` x `crop` window, or the full field when `crop` is
    None. Odd `crop` keeps the peak on a pixel.
    """
    clean = np.asarray(clean, dtype=np.float64)
    dirty = np.asarray(dirty, dtype=np.float64)
    if clean.ndim == 2:
        clean, dirty = clean[None], dirty[None]

    fc = np.fft.fft2(clean, axes=(-2, -1))
    fd = np.fft.fft2(dirty, axes=(-2, -1))
    den = (np.abs(fc) ** 2).mean(axis=0)
    num = (fd * np.conj(fc)).mean(axis=0)
    # Drop frequencies the clean cube never illuminated; there the ratio is 0/0.
    b = np.where(den > 1e-12 * den.max(), num / np.maximum(den, 1e-300), 0.0)

    beam = np.fft.fftshift(np.real(np.fft.ifft2(b)))
    if crop:
        c = beam.shape[0] // 2
        h = crop // 2
        beam = beam[c - h:c + h + 1, c - h:c + h + 1]
    return beam


def apply_beam(cube, beam):
    """Convolve with a beam returned by `estimate_beam_from_pair` (centred, not fftshifted)."""
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim == 2:
        cube = cube[None]
    n = cube.shape[-1]
    pad = np.zeros((n, n))
    h = beam.shape[0] // 2
    c = n // 2
    pad[c - h:c + h + 1, c - h:c + h + 1] = beam
    k = np.fft.fft2(np.fft.ifftshift(pad))
    return np.real(np.fft.ifft2(np.fft.fft2(cube, axes=(-2, -1)) * k[None], axes=(-2, -1)))


def plot_phase0_report(r: dict, save_path: str, title: str = "") -> None:
    """
    Power-spectrum figure behind a Phase 0 verdict: P_clean, P_dirty, and the fitted forward
    model, on the same axes, band-of-fit shaded. The verdict is a single line ("A = I" /
    "gaussian_convolution" / ...); this is the evidence it came from.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k, pc, pd = r["k"], r["clean_power"], r["dirty_power"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].loglog(k, pc, label="P_clean", color="tab:blue")
    ax[0].loglog(k, pd, label="P_dirty", color="tab:orange")
    if r.get("fit_sigma_px") is not None:
        model = (r["scale_factor"] * np.exp(-4 * np.pi ** 2 * r["fit_sigma_px"] ** 2 * k ** 2) * pc
                 + r.get("fit_noise", 0.0))
        ax[0].loglog(k, model, "--", color="k", lw=1.3, label="fitted model")
    if r.get("band_k_max"):
        ax[0].axvspan(0, r["band_k_max"], color="0.9", zorder=0, label="fit band")
    ax[0].set_xlabel("k (cycles/px)")
    ax[0].set_ylabel("power")
    ax[0].legend(fontsize=9)
    ax[0].set_title("Radially averaged power spectrum")

    ax[1].semilogx(k, r["transfer_raw"], color="tab:gray", lw=1, alpha=0.7,
                   label="raw sqrt(P_dirty/P_clean)")
    if r.get("fit_sigma_px") is not None:
        model_b = np.sqrt(np.maximum(
            r["scale_factor"] * np.exp(-4 * np.pi ** 2 * r["fit_sigma_px"] ** 2 * k ** 2), 0))
        model_b = model_b / max(model_b[k < 0.01].mean(), 1e-12)
        ax[1].semilogx(k, model_b, "--", color="k", lw=1.3, label="fitted |B(k)| (normalised)")
    if r.get("k_half_from_header"):
        ax[1].axvline(r["k_half_from_header"], color="crimson", ls=":", lw=1,
                      label="header beam rolloff")
    ax[1].set_xlabel("k (cycles/px)")
    ax[1].set_ylabel("transfer")
    ax[1].legend(fontsize=9)
    ax[1].set_title("Transfer function")

    plt.suptitle(f"{title}\nverdict: {r['verdict'].upper()}  --  {r.get('reason', '')}",
                fontsize=11, wrap=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
