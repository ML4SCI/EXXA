"""
The GI wiggle: a Keplerian-subtracted moment-1 residual, used across the literature (Hall+2020 ApJ, Hall+2021 MNRAS, Hall+2022 ApJL, Terry+2024 A&A, Speedie+2024
Nature) as the kinematic signature of gravitational instability in a self-gravitating disk.
A planet leaves a localised "kink"; GI leaves a global "interlocking fingers" pattern across
the whole disk. Fitting and subtracting the disk's own Keplerian rotation from its moment-1
map is what turns raw velocity into that residual.

None of the geometry needed to do this (inclination, position angle, systemic velocity,
stellar mass) is in either cube's header, so it has to come from a fit.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

GM_SUN_OVER_AU = 29.784677386228  # km/s: sqrt(GM_sun / 1 AU), the Keplerian speed at 1 AU
                                    # around a 1 Msun star -- the one physical constant this
                                    # model needs.


def disk_geometry_from_m0(m0: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Centre, position angle and inclination from the image moments of M0.

    A disk projects to an ellipse. Its second-moment matrix's principal axes give the
    position angle directly, and the axis ratio gives cos(inclination) -- a face-on disk
    is a circle (ratio 1), edge-on collapses to a line (ratio 0). This is a starting point
    for the Keplerian fit below, not the final answer: M0 also responds to the disk's
    surface-density profile, not purely its aspect ratio, so PA and inclination are
    refit as free parameters against the actual velocity field.
    """
    H, W = m0.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    w = np.where(mask, np.abs(m0), 0.0) if mask is not None else np.abs(m0)
    tot = w.sum()
    if tot <= 0:
        return {"cx": W / 2, "cy": H / 2, "pa_deg": 0.0, "incl_deg": 45.0}

    cx = float((w * x).sum() / tot)
    cy = float((w * y).sum() / tot)
    dx, dy = x - cx, y - cy
    ixx = float((w * dx * dx).sum() / tot)
    iyy = float((w * dy * dy).sum() / tot)
    ixy = float((w * dx * dy).sum() / tot)

    theta = 0.5 * np.arctan2(2 * ixy, ixx - iyy)
    common = np.sqrt(((ixx - iyy) / 2) ** 2 + ixy ** 2)
    lam1 = (ixx + iyy) / 2 + common   # major-axis variance
    lam2 = (ixx + iyy) / 2 - common   # minor-axis variance
    ratio = float(np.sqrt(max(lam2, 0) / max(lam1, 1e-12)))
    incl = float(np.degrees(np.arccos(np.clip(ratio, 0.0, 1.0))))

    return {"cx": cx, "cy": cy, "pa_deg": float(np.degrees(theta)) % 180.0, "incl_deg": incl}


def keplerian_los(xy, cx, cy, pa_deg, incl_deg, vsys, mstar, au_per_px, r_min_px=3.0):
    """
    Line-of-sight Keplerian velocity at each (x, y) pixel, in km/s.

    Standard thin-disk deprojection: rotate into the disk's major/minor axes, deproject the
    minor axis by cos(inclination) to get the in-plane radius and azimuth, then project the
    circular velocity back onto the line of sight by sin(inclination) * cos(azimuth).
    `r_min_px` guards the 1/sqrt(r) singularity at the fitted centre, where a real cube also
    has no reliable moment-1 (the beam smears it), so both sides handle it the same way.
    """
    x, y = xy
    pa = np.radians(pa_deg)
    inc = np.radians(incl_deg)

    dx, dy = x - cx, y - cy
    xr = dx * np.cos(pa) + dy * np.sin(pa)
    yr = (-dx * np.sin(pa) + dy * np.cos(pa)) / max(np.cos(inc), 1e-6)

    r_px = np.sqrt(xr ** 2 + yr ** 2)
    r_px = np.maximum(r_px, r_min_px)
    az = np.arctan2(yr, xr)

    r_au = r_px * au_per_px
    v_circ = GM_SUN_OVER_AU * np.sqrt(np.maximum(mstar, 1e-6) / np.maximum(r_au, 1e-6))
    return vsys + v_circ * np.sin(inc) * np.cos(az)


def fit_keplerian(m1: np.ndarray, mask: np.ndarray, au_per_px: float,
                  init: Optional[Dict[str, float]] = None,
                  m0: Optional[np.ndarray] = None,
                  fix_incl_deg: Optional[float] = None) -> Dict[str, float]:
    """
    Least-squares Keplerian fit to a moment-1 map, over `mask`.

    Free parameters: centre (cx, cy), position angle, systemic velocity, stellar mass, and
    inclination unless `fix_incl_deg` is given. `au_per_px` fixes the physical scale (from
    the header's pixel scale and distance) so `mstar` comes out in solar masses rather than
    as a degenerate GM/scale product.

    **Inclination and mass are very nearly degenerate, and at low inclination the fit cannot
    separate them.** The line-of-sight velocity is `sqrt(GM/r) sin(i) cos(theta)`, so mass
    and inclination enter through the single product `sqrt(M) sin(i)`. Only the geometric
    deprojection (which stretches sky coordinates by `1/cos i`) breaks that, and it is a 6%
    effect at i = 20 deg. Measured on the September self-gravitating batch, whose `.para`
    files state the true values: with inclination free, three of five disks at a true 20-30
    deg collapsed to a fitted 2-7 deg and drove `mstar` to its 50 Msun bound, errors up to
    +4900%. Holding inclination at its true value brought the same fits to -29%..+47%, and
    to -5% on the best-conditioned disk.

    So pass `fix_incl_deg` whenever inclination is known independently (a `.para` file,
    continuum imaging, the literature). Check `mstar_at_bound` in the result before quoting
    a mass: a fit that pinned is a failure, not a measurement.

    `m0` supplies the initial geometry. Without it this falls back to the mask's own shape,
    which for a roughly circular mask implies inclination ~0 and starts the optimiser at the
    degenerate end of that valley.
    """
    H, W = m1.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    ys, xs = y[mask], x[mask]
    vs = m1[mask].astype(np.float64)

    # A moment-1 map divides by the per-pixel flux sum, and `mask` comes from the CLEAN
    # cube's M0, not the fitted cube's own signal quality -- so a noisy or dirty-beam cube
    # can have a genuinely undefined M1 (0/0) at pixels the mask still calls "signal". Drop
    # them rather than let a NaN in the initial guess make the whole fit infeasible.
    finite = np.isfinite(vs)
    n_dropped = int((~finite).sum())
    if n_dropped:
        ys, xs, vs = ys[finite], xs[finite], vs[finite]
    if vs.size < 10:
        raise ValueError(f"only {vs.size} finite M1 pixels in the mask, too few to fit")

    # Initial geometry from the EMISSION's shape when M0 is available. Passing a uniform array
    # here (which this did until 2026-09-04) measures the shape of the MASK instead, and a
    # roughly circular mask implies inclination ~0 -- the degenerate end of the mass-inclination
    # valley, and the worst possible place to start.
    if init is None:
        init = disk_geometry_from_m0(m0 if m0 is not None else np.ones_like(m1), mask)
    p0 = init
    incl_guess = p0.get("incl_deg", 45.0)
    if m0 is None and init is None:
        incl_guess = 45.0        # neutral, rather than the mask-shape ~0

    MSTAR_MAX = 50.0

    if fix_incl_deg is None:
        def resid(p):
            cx, cy, pa, inc, vsys, mstar = p
            return keplerian_los((xs, ys), cx, cy, pa, inc, vsys, mstar, au_per_px) - vs

        guess = [p0.get("cx", W / 2), p0.get("cy", H / 2), p0.get("pa_deg", 0.0),
                 incl_guess, float(np.median(vs)), 1.0]
        lo = [0, 0, -360, 1, vs.min() - 5, 1e-3]
        hi = [W, H, 360, 89, vs.max() + 5, MSTAR_MAX]
        guess = [min(max(g, l), h) for g, l, h in zip(guess, lo, hi)]
        fit = least_squares(resid, guess, bounds=(lo, hi), max_nfev=4000)
        cx, cy, pa, inc, vsys, mstar = fit.x
    else:
        inc = float(fix_incl_deg)

        def resid(p):
            cx, cy, pa, vsys, mstar = p
            return keplerian_los((xs, ys), cx, cy, pa, inc, vsys, mstar, au_per_px) - vs

        guess = [p0.get("cx", W / 2), p0.get("cy", H / 2), p0.get("pa_deg", 0.0),
                 float(np.median(vs)), 1.0]
        lo = [0, 0, -360, vs.min() - 5, 1e-3]
        hi = [W, H, 360, vs.max() + 5, MSTAR_MAX]
        guess = [min(max(g, l), h) for g, l, h in zip(guess, lo, hi)]
        fit = least_squares(resid, guess, bounds=(lo, hi), max_nfev=4000)
        cx, cy, pa, vsys, mstar = fit.x

    return {"cx": float(cx), "cy": float(cy), "pa_deg": float(pa) % 180.0,
            "incl_deg": float(inc), "vsys": float(vsys), "mstar_msun": float(mstar),
            "au_per_px": au_per_px, "cost": float(fit.cost), "success": bool(fit.success),
            "n_dropped_nonfinite": n_dropped, "n_fit": int(vs.size),
            "incl_fixed": fix_incl_deg is not None,
            # A pinned mass is a failed fit, not a measurement -- check this before quoting it.
            "mstar_at_bound": bool(mstar > 0.9 * MSTAR_MAX)}


def wiggle_residual(m1: np.ndarray, geom: Dict[str, float]) -> np.ndarray:
    """M1 minus its fitted Keplerian model -- the GI wiggle, if the disk has one."""
    H, W = m1.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    model = keplerian_los((x, y), geom["cx"], geom["cy"], geom["pa_deg"], geom["incl_deg"],
                          geom["vsys"], geom["mstar_msun"], geom["au_per_px"])
    return m1 - model


def wiggle_amplitude(residual: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Summary statistics of the residual over the signal region.

    Not the literature's Fourier-decomposed A_wiggle (Hall+2021 Eq. via channel-map wavelet
    fit, Hall+2022 Eq. 25's integrated geometric distance) -- this is a coarser proxy: RMS
    and max absolute residual within the mask, in km/s. Good enough to say "is there a
    residual pattern at all" and to compare clean vs dirty vs denoised on equal footing;
    not good enough to report a disc-to-star mass ratio q against Hall+2021's A_wiggle =
    50q - 5.4, which needs the real per-channel Fourier amplitude, not this summary.
    """
    r = residual[mask]
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"rms_kms": float("nan"), "max_abs_kms": float("nan"), "n_px": 0}
    return {"rms_kms": float(np.sqrt(np.mean(r ** 2))), "max_abs_kms": float(np.max(np.abs(r))),
            "n_px": int(r.size)}


def quadratic_moment1(data: np.ndarray, velax: np.ndarray, rms: Optional[float] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Line-of-sight velocity via a parabola fit around each spectrum's peak channel
    (Teague & Foreman-Mackey 2018, `bettermoments.collapse_quadratic`), not the
    intensity-weighted mean `collapse_first` uses.

    Why this matters here specifically: `collapse_first`'s weighted average is unstable on
    the dirty cube, which is 51.55% negative pixels. A negative "weight" in an intensity-weighted
    sum is not a small perturbation, it can flip the sign of the average or divide by a total
    near zero, and that is what pinned `fit_keplerian`'s mass at its bound on the dirty cube
    even after both known init bugs were fixed. The quadratic estimator only looks at the
    single brightest channel and its two neighbours, so a negative sidelobe elsewhere in the
    spectrum cannot pull it away from the true line centre.

    Returns (v0, peak) in the units of `velax`, one value per spatial pixel.
    """
    import bettermoments as bm

    if rms is None:
        rms = bm.estimate_RMS(data=data, N=5)
    v0, _, peak, _ = bm.collapse_quadratic(velax=velax, data=data, rms=rms)
    return v0, peak


def compare_wiggles(m1_maps: Dict[str, np.ndarray], mask: np.ndarray, au_per_px: float,
                    reference: str = "clean") -> Dict[str, dict]:
    """
    Compare GI wiggle residuals across methods against ONE shared Keplerian model.

    Fit the reference cube's rotation curve once, subtract THAT SAME model from every map,
    and compare what is left. Never fit a separate model per method.

    Why this matters, measured: fitting each cube its own Keplerian model makes the residual
    definition differ between cubes, so the comparison is dominated by how the fits disagree
    rather than by the wiggle. Beam-convolved clean data scored 0.117, 0.998, 0.153, 0.949 and
    0.116 against the same truth across five channel ranges under per-method fitting -- noise,
    not measurement. Under a shared model it scores 0.920 to 0.999 across all of them.

    That instability produced a chain of wrong conclusions on 2026-08-27/28, including "the
    beam alone erases the wiggle", which is false: the beam preserves it (r >= 0.92, and the
    residual RMS is within 10% of the truth's).

    Args:
        m1_maps: {name: moment-1 map in km/s}. Must contain `reference`.
        mask: pixels to score over, from `signal_mask` on the reference's M0.
        au_per_px: physical scale.
        reference: which entry defines the shared rotation model.

    Returns {name: {corr, rms_kms, geom}} with `geom` the shared fit, repeated for clarity.
    """
    if reference not in m1_maps:
        raise KeyError(f"reference {reference!r} not among {sorted(m1_maps)}")

    geom = fit_keplerian(m1_maps[reference], mask, au_per_px)
    ref_resid = wiggle_residual(m1_maps[reference], geom)

    out = {}
    for name, m1 in m1_maps.items():
        resid = wiggle_residual(m1, geom)          # the SAME model for every method
        ok = np.isfinite(ref_resid[mask]) & np.isfinite(resid[mask])
        corr = (float(np.corrcoef(ref_resid[mask][ok], resid[mask][ok])[0, 1])
                if ok.sum() > 10 else float("nan"))
        out[name] = {"corr": corr, "rms_kms": wiggle_amplitude(resid, mask)["rms_kms"],
                     "geom": geom, "residual": resid}
    return out
