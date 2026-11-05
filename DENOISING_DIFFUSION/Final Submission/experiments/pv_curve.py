"""
Position-velocity (PV) curve along the minor axis: the moment-1 (quadratic) value along the line where a purely rotating disc has v_los = vsys.
There the Keplerian / axisymmetric rotation contributes nothing, so M1 - vsys IS the non-axisymmetric motion, the GI wiggle. This is the
measurement Speedie et al. 2024 and Longarini et al. (Elias 2-27) fit, and the field standard.

pv_profile  M1 map + disc geometry -> (R grid in au, [profile on the +minor half, profile on the -minor half]) in m/s, optional slit width in px
pv_scores   profiles of several methods -> pv_r (correlation with clean after removing a quadratic trend, per half, pooled), pv_rms_ms and
            pv_mag_ms, the wiggle "magnitude" of Speedie et al. 2024: the standard deviation of the quadratically detrended line-centre velocities
            along the minor axis (their 13CO: 37.4 +/- 2.9 m/s, AB Aur, 1 to 5 arcsec). That is the literature number our curves can be compared with.
plot_pv     one panel per half, clean in black, each method a colour

Literature: Speedie 2024 take line centres from a quadratic fit per pixel (bettermoments) and read them in a 0.5 deg wedge on the SOUTHERN
minor axis, 1 to 5 arcsec, detrended with a quadratic (a linear trend fitted worse); Longarini 2024 (Elias 2-27) cut along the southern semi-minor axis of a
Gaussian-fit velocity map and mask the inner two beams. Neither states a slit width; the review asked for a straight slice or a slit of about 4 to 5 px.
Our M1 is already the quadratic line centre, so the profiles here are the same quantity. We score both halves (the scorecard has no cloud to avoid).

Geometry: the same as wiggle_scorecard.spine_tracks (geom has cx, cy, pa_deg, incl_deg, vsys; the minor axis is m = (-sin pa, cos pa), a point at
deprojected radius R lies s = R cos(i) / au pixels from the centre along m).
Run the test: PYTHONPATH=. python3 tests/test_pv_curve.py
"""
import numpy as np
from scipy.ndimage import map_coordinates


def pv_profile(m1, geom, au, r_in_au, r_out_au, width_px=1, mask=None):
    """M1 (km/s) along both halves of the minor axis, minus vsys, in m/s. width_px > 1 averages across the slit (along the major-axis direction).
    Samples whose mask value is < 0.5 (or that are not finite) come back NaN, so a noisy cube is never padded with zeros."""
    pa, cos_i = np.radians(geom["pa_deg"]), np.cos(np.radians(geom["incl_deg"]))
    m = np.array([-np.sin(pa), np.cos(pa)])            # minor axis
    u = np.array([np.cos(pa), np.sin(pa)])             # across the slit
    r_grid = np.arange(r_in_au, r_out_au, au)
    k = max(1, int(width_px))
    t = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    ok = np.isfinite(m1) if mask is None else (np.isfinite(m1) & mask)
    val, w = np.where(ok, m1, 0.0), ok.astype(np.float64)
    out = []
    for sgn in (1, -1):
        s = sgn * r_grid * cos_i / au
        X = geom["cx"] + s[:, None] * m[0] + t[None, :] * u[0]
        Y = geom["cy"] + s[:, None] * m[1] + t[None, :] * u[1]
        v = map_coordinates(val, [Y.ravel(), X.ravel()], order=1, cval=0.0).reshape(X.shape)
        ww = map_coordinates(w, [Y.ravel(), X.ravel()], order=1, cval=0.0).reshape(X.shape)
        good = ww > 0.5
        num = np.where(good, v, 0.0).sum(1)
        den = np.where(good, ww, 0.0).sum(1)
        prof = np.where(den > 0, num / np.maximum(den, 1e-12) - geom["vsys"], np.nan) * 1000.0
        out.append(prof)
    return r_grid, out


def _detrend(y, deg):
    x = np.arange(len(y))
    g = np.isfinite(y)
    if g.sum() <= deg + 1:
        return np.full(len(y), np.nan)
    return y - np.polyval(np.polyfit(x[g], y[g], deg), x)


def pv_scores(profiles, ref="clean", deg=2):
    """profiles: {method: [half+, half-]}. Returns {method: {"pv_r": ..., "pv_rms_ms": ...}} against `ref`.
    pv_r pools both halves after removing a degree-`deg` polynomial from each (the smooth part is not the wiggle); deg 2 as in Speedie 2024.
    pv_mag_ms is the standard deviation of the detrended profile (mean over the two halves), in m/s: the wiggle magnitude of that paper."""
    out = {}
    for k, halves in profiles.items():
        a, b, mag = [], [], []
        for hr, hk in zip(profiles[ref], halves):
            g = np.isfinite(hr) & np.isfinite(hk)
            if g.sum() < deg + 3:
                continue
            a.append(_detrend(np.where(g, hr, np.nan), deg)[g]); b.append(_detrend(np.where(g, hk, np.nan), deg)[g])
            mag.append(float(np.std(b[-1])))
        if not a:
            out[k] = {"pv_r": float("nan"), "pv_rms_ms": float("nan"), "pv_mag_ms": float("nan")}
            continue
        a, b = np.concatenate(a), np.concatenate(b)
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        out[k] = {"pv_r": r, "pv_rms_ms": float(np.sqrt(np.mean((a - b) ** 2))), "pv_mag_ms": float(np.mean(mag))}
    return out


def plot_pv(path, r_grid, profiles_by_width, scores_by_width, title, ref_label="clean"):
    """profiles_by_width: {width_px: {method: [half+, half-]}}; one row per width, one column per half. `ref_label` names the black reference curve
    ("raw" on a real cube, where there is no truth). The y range follows the reference curve (its range plus its own width on each side) so a few noisy
    pixels in a dirty profile (cavity edge, thousands of m/s) cannot flatten the wiggle; points outside are clipped by the axes, not removed."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    widths = list(profiles_by_width)
    fig, ax = plt.subplots(len(widths), 2, figsize=(14, 3.6 * len(widths)), squeeze=False)
    for i, wd in enumerate(widths):
        for j in range(2):
            for k, halves in profiles_by_width[wd].items():
                ax[i, j].plot(r_grid, halves[j], "k-" if k == "clean" else "-", lw=2.2 if k == "clean" else 1.1,
                              label=ref_label if k == "clean" else f"{k} (r={scores_by_width[wd][k]['pv_r']:.2f})")
            ref = np.asarray(profiles_by_width[wd]["clean"][j], dtype=float)
            if np.isfinite(ref).any():
                lo, hi = np.nanmin(ref), np.nanmax(ref); pad = max(hi - lo, 60.0)
                ax[i, j].set_ylim(lo - pad, hi + pad)
            ax[i, j].set_xlabel("R (au)"); ax[i, j].set_ylabel("M1 - vsys along the minor axis (m/s)")
            ax[i, j].set_title(f"{'+' if j == 0 else '-'} half, slit {wd} px"); ax[i, j].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)
