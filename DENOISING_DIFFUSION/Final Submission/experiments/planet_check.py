"""
Planet check (review of 2026-10-16: "a possible planet" in run_0002_00560 and in Young M1 R100 at 7 to 8 o'clock). Input: the npz files that
train_unified.py --save-pred writes (M1 maps of clean / dirty / Wiener / model, clean M0, central channels of clean / dirty / model).

A planet shows in the kinematics as a LOCALISED velocity deviation (a "kink", Pinte et al. 2018) on top of rotation, not as a ring or spiral that runs all
the way round. So: fit the geometry to clean M1 (inclination fixed), subtract clean's own measured rotation curve (wiggle_scorecard.rotation_curve_model,
v_phi free per half-beam ring: a forced Keplerian leaves a rotation-shaped dipole where the beam smears the inner disc) from every method's M1, remove the
azimuthal Fourier series up to m = 3 per ring (ring mean, rotation error, surface-height pattern, smooth spirals; a kink is narrow in azimuth and stays), smooth by the beam, and find the strongest feature of clean outside 3 beams. For each method report whether
it reproduces that feature: correlation with clean within 2 beams of it, amplitude ratio there, and whether its own strongest feature lies within one beam.
Clock positions as the figures show them (imshow origin="lower": 12 o'clock is up, 3 o'clock is right).

Usage: python experiments/planet_check.py out_dir case.npz [...] [--sector 6.5,8.5]
"""
import json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import gi_wiggle as gw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wiggle_scorecard as ws

METHODS = ("clean", "dirty", "wiener", "model")


def clock(dx, dy):
    """Clock hour (1-12) of an offset, 12 up (+y), 3 right (+x)."""
    h = (np.degrees(np.arctan2(dx, dy)) % 360.0) / 30.0
    return 12.0 if h < 0.5 else h


def localised(res, rdep, az, ring, mask, mmax=3):
    """res minus, per ring of width `ring` (au), its least-squares azimuthal Fourier series up to m = mmax (in the disc-plane azimuth `az`).
    m = 0 is the ring mean, m = 1 a rotation / systemic error, m = 2 the emission-surface height pattern of an inclined disc, m = 3 smooth spirals;
    a planet kink is narrow in azimuth, so it survives. NaN outside the mask."""
    out = np.full(res.shape, np.nan)
    good = mask & np.isfinite(res)
    b = (rdep / ring).astype(int)
    for k in np.unique(b[good]):
        sel = good & (b == k)
        if sel.sum() < 4 * (2 * mmax + 1):
            continue
        a = az[sel]
        X = np.column_stack([np.ones_like(a)] + [f(m * a) for m in range(1, mmax + 1) for f in (np.cos, np.sin)])
        coef = np.linalg.lstsq(X, res[sel], rcond=None)[0]
        out[sel] = res[sel] - X @ coef
    return out


def smooth_nan(a, sig):
    w = np.isfinite(a).astype(float)
    num = gaussian_filter(np.where(w > 0, a, 0.0), sig)
    den = gaussian_filter(w, sig)
    return np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)


def analyse(z, name, sector=None):
    m1 = {k: z[f"m1_{k}"].astype(np.float64) for k in METHODS}
    m0 = z["m0_clean"].astype(np.float64)
    au, beam_au, incl = float(z["au_px"]), float(z["beam_au"]), float(z["incl"])
    n = m0.shape[0]
    mask = np.isfinite(m0) & (m0 > 0.05 * np.nanmax(m0))
    geom = gw.fit_keplerian(m1["clean"], mask, au, m0=m0, fix_incl_deg=incl if np.isfinite(incl) else None)
    yy, xx = np.indices((n, n), dtype=np.float64)
    pa, ci = np.radians(geom["pa_deg"]), np.cos(np.radians(geom["incl_deg"]))
    dx, dy = xx - geom["cx"], yy - geom["cy"]
    xr, yr = dx * np.cos(pa) + dy * np.sin(pa), (-dx * np.sin(pa) + dy * np.cos(pa)) / ci
    rdep, az = np.hypot(xr, yr) * au, np.arctan2(yr, xr)
    beam_px = beam_au / au
    sig = beam_px / 2.355
    rc = ws.rotation_curve_model(m1["clean"], mask, geom, au, beam_au / 2)
    loc = {k: smooth_nan(localised((m1[k] - rc) * 1000.0, rdep, az, beam_au / 2, mask), sig) for k in METHODS}
    r_out = np.percentile(rdep[mask], 97)
    ok = mask & (rdep > 3 * beam_au) & (rdep < r_out) & np.isfinite(loc["clean"])
    if sector is not None:                          # only clean features inside this clock range (e.g. the "7 to 8 o'clock" region)
        hr = (np.degrees(np.arctan2(dx, dy)) % 360.0) / 30.0
        hr = np.where(hr < 0.5, hr + 12.0, hr)
        ok = ok & (hr >= sector[0]) & (hr <= sector[1])
    a = np.where(ok, np.abs(loc["clean"]), -1)
    py, px = np.unravel_index(np.argmax(a), a.shape)
    near = ok & (np.hypot(xx - px, yy - py) < 2 * beam_px)
    cand = {"x": int(px), "y": int(py), "R_au": float(rdep[py, px]), "clock": float(clock(px - geom["cx"], py - geom["cy"])),
            "dv_clean_ms": float(loc["clean"][py, px]), "ring_rms_clean_ms": float(np.nanstd(loc["clean"][ok]))}
    rows = {}
    for k in METHODS:
        v = loc[k]
        g = near & np.isfinite(v)
        r = float(np.corrcoef(loc["clean"][g], v[g])[0, 1]) if g.sum() > 10 and np.nanstd(v[g]) > 0 else float("nan")
        b = np.where(ok & np.isfinite(v), np.abs(v), -1)
        qy, qx = np.unravel_index(np.argmax(b), b.shape)
        rows[k] = {"r_near": r, "amp_ratio": float(v[py, px] / loc["clean"][py, px]) if np.isfinite(v[py, px]) else float("nan"),
                   "own_peak_dist_beams": float(np.hypot(qx - px, qy - py) / beam_px), "own_peak_clock": float(clock(qx - geom["cx"], qy - geom["cy"])),
                   "own_peak_R_au": float(rdep[qy, qx]), "own_peak_ms": float(v[qy, qx]), "rms_ms": float(np.nanstd(v[ok]))}
    return {"case": name, "geom": {k: float(geom[k]) for k in ("cx", "cy", "pa_deg", "incl_deg", "vsys", "mstar_msun")}, "beam_px": beam_px,
            "candidate": cand, "methods": rows}, loc, (px, py, beam_px, geom, mask)


def figure(path, z, out, loc, extra):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    px, py, bp, geom, mask = extra
    vr = np.nanpercentile(np.abs(loc["clean"][mask]), 99)
    velax = np.asarray(z["velax"]); lo = int(z["chan_lo"])
    vloc = z["m1_clean"][py, px]
    c = int(np.clip(np.argmin(np.abs(velax[lo:lo + z["ch_clean"].shape[0]] - vloc)), 0, z["ch_clean"].shape[0] - 1))
    h = int(4 * bp); ys, xs = slice(max(0, py - h), py + h), slice(max(0, px - h), px + h)
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.5))
    for i, k in enumerate(METHODS):
        ax[0, i].imshow(np.where(mask, loc[k], np.nan), origin="lower", cmap="RdBu_r", vmin=-vr, vmax=vr)
        ax[0, i].add_patch(plt.Circle((px, py), 1.5 * bp, fill=False, color="k", lw=1.5))
        m = out["methods"][k]
        ax[0, i].set_title(f"{k}: localised M1 residual\nr near {m['r_near']:.2f}, amp {m['amp_ratio']:.2f}, own peak {m['own_peak_dist_beams']:.1f} beams away", fontsize=9)
    for i, k in enumerate(("clean", "dirty", "model")):
        ch = z[f"ch_{k}"][c].astype(np.float32)
        ax[1, i].imshow(ch[ys, xs], origin="lower", cmap="inferno", extent=(xs.start, xs.stop, ys.start, ys.stop))
        ax[1, i].plot(px, py, "c+", ms=14)
        ax[1, i].set_title(f"{k}: channel {velax[lo + c]:.2f} km/s (local clean M1 {vloc:.2f})", fontsize=9)
    ax[1, 3].axis("off")
    cd = out["candidate"]
    ax[1, 3].text(0, 0.5, f"candidate (clean's strongest localised feature)\nR = {cd['R_au']:.0f} au, {cd['clock']:.1f} o'clock\n"
                  f"clean dv {cd['dv_clean_ms']:.0f} m/s (ring rms {cd['ring_rms_clean_ms']:.0f})\nbeam {bp:.1f} px", fontsize=11)
    fig.suptitle(f"{out['case']}: planet check (localised kinematic residual)"); fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main(outdir, paths, sector=None):
    os.makedirs(outdir, exist_ok=True)
    allres = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        z = np.load(p)
        name = name + (f"_sector{sector[0]:g}-{sector[1]:g}" if sector else "")
        out, loc, extra = analyse(z, name, sector)
        figure(os.path.join(outdir, f"{name}_planet.png"), z, out, loc, extra)
        allres.append(out)
        c = out["candidate"]
        print(f"{name}: candidate R {c['R_au']:.0f} au at {c['clock']:.1f} o'clock, clean {c['dv_clean_ms']:.0f} m/s (ring rms {c['ring_rms_clean_ms']:.0f}); "
              + "; ".join(f"{k} r {v['r_near']:.2f} amp {v['amp_ratio']:.2f} own-peak {v['own_peak_dist_beams']:.1f} bm" for k, v in out["methods"].items() if k != "clean"))
    json.dump(allres, open(os.path.join(outdir, "planet_check" + (f"_sector{sector[0]:g}-{sector[1]:g}" if sector else "") + ".json"), "w"), indent=1)


if __name__ == "__main__":
    assert abs(clock(0, 1) - 12) < 1e-9 and abs(clock(1, 0) - 3) < 1e-9 and abs(clock(0, -1) - 6) < 1e-9 and abs(clock(-1, 0) - 9) < 1e-9
    args = sys.argv[1:]
    sector = None
    if "--sector" in args:                          # --sector lo,hi (clock hours)
        i = args.index("--sector"); sector = tuple(float(x) for x in args[i + 1].split(",")); del args[i:i + 2]
    main(args[0], args[1:], sector)
