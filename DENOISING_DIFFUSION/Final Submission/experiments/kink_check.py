"""
 2023, arXiv:2301.05075) survive denoising? Rules fixed before any number.

Statistic: Keplerian mirror asymmetry. An axisymmetric rotating disc, whatever its vertical structure, is unchanged by (reflection across the minor
axis, v_sys + dv -> v_sys - dv). So A(dv) = I(dv) - R I(-dv) is noise only for such a disc, and a localised kink shows up in it. Both channel maps
are interpolated linearly to the exact velocities and beam-smoothed; the reflection axis is fitted blind on the RAW cube (one angle, then the centre
within +-2 px, then the angle again) and used unchanged for the denoised cube. sigma_A = the same statistic built from line-free planes.

Local self-check (synthetic disc with and without a kink): PYTHONPATH=. python3 experiments/kink_check.py --demo
"""
import argparse, json, math, os, sys
import numpy as np
from scipy import ndimage

ROOT = os.environ.get("EXXA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]


def plane_at(cube, velax, v):
    """Channel map at velocity v, linear between the two nearest channels (velax ascending)."""
    f = float(np.interp(v, velax, np.arange(len(velax))))
    i0 = min(int(f), len(velax) - 2)
    t = f - i0
    return (1 - t) * cube[i0] + t * cube[i0 + 1], (i0, t)


def reflect(img, cx, cy, theta_deg):
    """Mirror `img` across the line through (cx, cy) at angle theta (degrees from +x, towards +y)."""
    th = math.radians(theta_deg)
    ux, uy = math.cos(th), math.sin(th)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    dx, dy = xx - cx, yy - cy
    p = dx * ux + dy * uy
    return ndimage.map_coordinates(img, [cy + 2 * p * uy - dy, cx + 2 * p * ux - dx], order=1, mode="constant", cval=0.0)


def asym(cube, velax, vsys, dv, cx, cy, theta, sm):
    a, _ = plane_at(cube, velax, vsys + dv)
    b, _ = plane_at(cube, velax, vsys - dv)
    a, b = ndimage.gaussian_filter(a, sm), ndimage.gaussian_filter(b, sm)
    return a - reflect(b, cx, cy, theta)


def fit_axis(cube, velax, vsys, cx, cy, dvs, sm, mask, theta0=None):
    """Blind minor-axis angle (and centre within +-2 px): least mirror asymmetry over all channel pairs `dvs`, inside `mask`."""
    sm_planes = {dv: (ndimage.gaussian_filter(plane_at(cube, velax, vsys + dv)[0], sm), ndimage.gaussian_filter(plane_at(cube, velax, vsys - dv)[0], sm)) for dv in dvs}
    cost = lambda th, x, y: sum(float(((a - reflect(b, x, y, th))[mask] ** 2).mean()) for a, b in sm_planes.values())
    grid = np.arange(0.0, 180.0, 1.0) if theta0 is None else np.arange(theta0 - 10, theta0 + 10.01, 1.0)
    th = float(min(grid, key=lambda t: cost(t, cx, cy)))
    cx, cy = min(((cx + i, cy + j) for i in range(-2, 3) for j in range(-2, 3)), key=lambda c: cost(th, *c))
    th = float(min(np.arange(th - 2, th + 2.01, 0.1), key=lambda t: cost(t, cx, cy)))
    return th, float(cx), float(cy)


def stats(Ar, Ad, ap, sig):
    r_raw, r_den = float(np.sqrt((Ar[ap] ** 2).mean())), float(np.sqrt((Ad[ap] ** 2).mean()))
    sig_raw_signal = math.sqrt(max(r_raw ** 2 - sig ** 2, 0.0))
    return {"rms_A_raw": r_raw, "rms_A_den": r_den, "S_raw": r_raw / sig, "S_den": r_den / sig,
            "pearson_r": float(np.corrcoef(Ar[ap], Ad[ap])[0, 1]),
            "amp_ratio_debiased": (r_den / sig_raw_signal) if sig_raw_signal > 0 else None}


def verdict(st):
    """readings, in order."""
    if st["S_raw"] < 3:
        return "not testable"
    r, q = st["pearson_r"], st["amp_ratio_debiased"]
    if r >= 0.5 and q is not None and 0.5 <= q <= 1.5:
        return "PRESERVED"
    if r < 0.3 or q is None or q < 0.5:
        return "LOST"
    return "PARTIAL"


def noise_sigma(cube, lf, sm, cx, cy, theta, ap, t, n=12, seed=0):
    """sigma_A from line-free planes: two interpolated noise planes (same weight t as the line channels), smoothed, one mirrored, differenced."""
    rng = np.random.default_rng(seed)
    pairs = [i for i in lf[:-1] if i + 1 in set(lf)]
    out = []
    for _ in range(n):
        i, j = rng.choice(pairs, 2, replace=False)
        a = ndimage.gaussian_filter((1 - t) * cube[i] + t * cube[i + 1], sm)
        b = ndimage.gaussian_filter((1 - t) * cube[j] + t * cube[j + 1], sm)
        out.append(float(np.sqrt(((a - reflect(b, cx, cy, theta))[ap] ** 2).mean())))
    return float(np.median(out))


def run(raw, den, velax, lf, pix_as, cdelt1, cdelt2, beam_px, cx, cy, vsys, dvs_kink, ap_off_as, ap_r_as, out, pa_minor_expected=None):
    sm = beam_px / 2.355
    n = raw.shape[-1]
    yy, xx = np.indices((n, n))
    m0 = np.clip(ndimage.gaussian_filter(raw, (0, sm, sm)), 0, None).sum(0)
    mask = m0 > 0.1 * m0.max()
    fit_dvs = [d for d in np.arange(0.35, 4.01, 0.35) if vsys - d >= velax[0] and vsys + d <= velax[-1]]
    theta, cx, cy = fit_axis(raw, velax, vsys, cx, cy, fit_dvs, sm, mask)
    ax_ = cx + ap_off_as[0] / (cdelt1 * 3600)          # (dRA, dDec) arcsec -> pixels; dRA > 0 is east
    ay_ = cy + ap_off_as[1] / (cdelt2 * 3600)
    ap = np.hypot(xx - ax_, yy - ay_) <= ap_r_as / pix_as
    ap_rot = np.hypot(xx - (2 * cx - ax_), yy - (2 * cy - ay_)) <= ap_r_as / pix_as
    rep = {"theta_minor_deg_image": theta, "centre_px": [cx, cy], "vsys": vsys, "aperture_px": [ax_, ay_, ap_r_as / pix_as], "beam_px": beam_px,
           "fit_channel_pairs": len(fit_dvs), "channels": {}, "controls": {}}
    if pa_minor_expected is not None:
        rep["theta_minor_expected_deg_image"] = pa_minor_expected
    figs = []
    for dv in dvs_kink:
        Ar, Ad = asym(raw, velax, vsys, dv, cx, cy, theta, sm), asym(den, velax, vsys, dv, cx, cy, theta, sm)
        t = plane_at(raw, velax, vsys + dv)[1][1]
        sig = noise_sigma(raw, lf, sm, cx, cy, theta, ap, t)
        st = stats(Ar, Ad, ap, sig); st["sigma_A"] = sig; st["verdict"] = verdict(st)
        st["control_rotated_aperture"] = stats(Ar, Ad, ap_rot, sig)
        rep["channels"][f"{dv:+.2f}"] = st
        figs.append((dv, plane_at(raw, velax, vsys + dv)[0], plane_at(den, velax, vsys + dv)[0], Ar, Ad))
        print(f"dv {dv:+.2f}: S_raw {st['S_raw']:.2f} S_den {st['S_den']:.2f} r {st['pearson_r']:.2f} amp {st['amp_ratio_debiased']} -> {st['verdict']} | "
              f"rotated aperture S_raw {st['control_rotated_aperture']['S_raw']:.2f} r {st['control_rotated_aperture']['pearson_r']:.2f}", flush=True)
    for dv in [d for d in (-3.15, -2.80, 2.80, 3.15) if vsys - abs(d) >= velax[0] and vsys + abs(d) <= velax[-1]]:
        Ar, Ad = asym(raw, velax, vsys, dv, cx, cy, theta, sm), asym(den, velax, vsys, dv, cx, cy, theta, sm)
        sig = noise_sigma(raw, lf, sm, cx, cy, theta, ap, plane_at(raw, velax, vsys + dv)[1][1])
        rep["controls"][f"{dv:+.2f}"] = dict(stats(Ar, Ad, ap, sig), sigma_A=sig)
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "kink.json"), "w"), indent=1, default=float)
    figure(os.path.join(out, "kink.png"), figs, ap, cx, cy, ax_, ay_, ap_r_as / pix_as, theta)
    return rep


def figure(path, figs, ap, cx, cy, ax_, ay_, r, theta):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(figs), 4, figsize=(16, 4.2 * len(figs)), squeeze=False)
    for row, (dv, pr, pd, Ar, Ad) in zip(axs, figs):
        v = max(float(np.percentile(pr, 99.5)), 1e-30); a = max(float(np.percentile(np.abs(Ar), 99.5)), 1e-30)
        for k, (img, title, lim) in enumerate(((pr, f"raw, dv {dv:+.2f} km/s", (0, v)), (pd, "denoised", (0, v)),
                                              (Ar, "mirror asymmetry A, raw", (-a, a)), (Ad, "mirror asymmetry A, denoised", (-a, a)))):
            ax = row[k]
            ax.imshow(img, origin="lower", cmap="magma" if k < 2 else "RdBu_r", vmin=lim[0], vmax=lim[1])
            ax.add_patch(plt.Circle((ax_, ay_), r, fill=False, color="lime", lw=1.2))
            th = math.radians(theta); L = 3 * r
            ax.plot([cx - L * math.cos(th), cx + L * math.cos(th)], [cy - L * math.sin(th), cy + L * math.sin(th)], "w--", lw=0.6)
            ax.set_xlim(cx - 4 * r, cx + 4 * r); ax.set_ylim(cy - 4 * r, cy + 4 * r); ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("HD 142666 12CO: Terry et al. 2023 kink aperture (green), fitted minor axis (dashed)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def demo():
    """Synthetic Keplerian disc (i 60 deg) with and without a localised velocity kick; the statistic must find the kink and not invent one."""
    n, C = 160, 41
    velax = np.linspace(-4, 4, C)
    yy, xx = np.indices((n, n), dtype=np.float64)
    cx = cy = n / 2 - 0.3
    th = math.radians(30.0)                        # major axis at 30 deg, minor at 120
    u, w = (xx - cx) * math.cos(th) + (yy - cy) * math.sin(th), -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
    i = math.radians(60.0)
    r = np.hypot(u, w / math.cos(i)) + 1e-3
    vlos = 6.0 * u / r * (r / 10.0) ** -0.5 * math.sin(i)
    inten = np.exp(-r / 25.0) * (r < 60)
    kick = 0.6 * np.exp(-((xx - (cx + 18)) ** 2 + (yy - (cy + 8)) ** 2) / (2 * 3.0 ** 2))
    rng = np.random.default_rng(1)
    res = {}
    for name, vk in (("keplerian", vlos), ("kink", vlos + kick)):
        cube = np.stack([inten * np.exp(-0.5 * ((v - vk) / 0.25) ** 2) for v in velax])
        cube = np.concatenate([np.zeros((8, n, n)), cube, np.zeros((8, n, n))])
        vel = np.r_[velax[0] - 0.2 * np.arange(8, 0, -1), velax, velax[-1] + 0.2 * np.arange(1, 9)]
        noisy = cube + 0.03 * rng.standard_normal(cube.shape)
        lf = np.r_[0:8, C + 8:C + 16]
        # the "denoised" cube is the noise-free truth: it must come out PRESERVED on the kink, and the Keplerian disc must be "not testable"
        dv_spot = float(vlos[int(round(cy + 8)), int(round(cx + 18))])        # the channel in which the kicked spot emits
        rep = run(noisy, cube, vel, lf, 0.01, -0.01 / 3600, 0.01 / 3600, 4.0, cx + 1, cy - 1, 0.0, [dv_spot],
                  (-0.18, 0.08), 0.12, os.path.join(os.environ.get("TMPDIR", "/tmp"), f"kink_demo_{name}"))
        res[name] = list(rep["channels"].values())[0]
        print(name, "theta", round(rep["theta_minor_deg_image"], 1), "(true 120)", res[name]["verdict"])
    assert abs(rep["theta_minor_deg_image"] - 120.0) < 2.0, rep["theta_minor_deg_image"]
    assert res["keplerian"]["verdict"] == "not testable", res["keplerian"]
    assert res["kink"]["verdict"] == "PRESERVED", res["kink"]
    print("demo OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw"); ap.add_argument("--den"); ap.add_argument("--out", default="kink_out"); ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--cx", type=float, default=261.9); ap.add_argument("--cy", type=float, default=258.0); ap.add_argument("--vsys", type=float, default=4.272)
    ap.add_argument("--dv", default="-1.40,-1.75"); ap.add_argument("--ap-off", default="-0.03,0.15"); ap.add_argument("--ap-r", type=float, default=0.30)
    ap.add_argument("--pa-sky", type=float, default=162.11)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()
    from astropy.io import fits
    import alma_unified as au
    raw, velax, hdr, pix, beam = au.read_cube(a.raw, a.crop)
    den = fits.getdata(a.den).astype(np.float32)
    assert den.shape == raw.shape, (den.shape, raw.shape)
    lf, _ = au.line_free(raw, beam[0] / pix)
    bpx = math.sqrt(beam[0] * beam[1]) / pix
    cd1, cd2 = float(hdr["CDELT1"]), float(hdr["CDELT2"])
    pa_m = math.radians(a.pa_sky - 90.0)                     # minor axis on the sky (PA north through east) -> image angle, a sanity check on the blind fit
    exp_theta = math.degrees(math.atan2(math.cos(pa_m) / (cd2 * 3600), math.sin(pa_m) / (cd1 * 3600))) % 180
    print(f"raw {raw.shape}, {pix * 1000:.1f} mas/px, beam {beam[0]:.3f} x {beam[1]:.3f} arcsec ({bpx:.1f} px), dv {velax[1] - velax[0]:.3f} km/s, line-free planes {len(lf)}", flush=True)
    run(raw, den, velax, lf, pix, cd1, cd2, bpx, a.cx, a.cy, a.vsys, [float(x) for x in a.dv.split(",")],
        tuple(float(x) for x in a.ap_off.split(",")), a.ap_r, a.out, pa_minor_expected=exp_theta)
    print("KINK DONE", flush=True)


if __name__ == "__main__":
    main()
