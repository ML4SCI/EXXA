"""
SPH disc simulation (Phantom dump) -> line-emission channel cube, with radiative transfer along every line of sight.

Why: public GI hydro simulations (Young et al. 2026, 63 runs over Md/M* 0.2-1.4; Rowther et al. 2024;
Longarini et al. 2026; Speedie et al. 2024) are released as particle dumps, not cubes. This turns one dump into many observations
(inclination, position angle, in-plane rotation, spin direction, optical depth) on our pixel and channel grid.

Model, per pixel and channel v (Rayleigh-Jeans brightness temperature, K):
    I(v) = sum_p T_p (1 - exp(-dtau_p(v))) exp(-tau_front_p(v)),   dtau_p(v) = kappa m_p phi(v - v_los,p; sigma_p)
with particles ordered along the line of sight (nearest first), phi a Gaussian of the particle's thermal width for the emitting
molecule. An optically thick line therefore shows the velocity of the layer facing the observer, as 12CO / 13CO do, which is
what makes the GI wiggle visible in the channel maps (Terry et al. 2022). kappa is set from the target optical depth `tau0`
(median peak tau over the disc), so the physical abundance is not needed. What it leaves out: SPH kernel smoothing inside a pixel
(particles are binned; pixel size should be about the emitting layer's smoothing length or larger, and the beam is applied
afterwards), dust, freeze-out / photodissociation, non-LTE. Validated against Speedie et al.'s own MCFOST cube of the same dump before any training use.

Units of the dumps: au, Msun, G = 1 (velocity unit 29.78 km/s).
"""
import math
import numpy as np

V_UNIT_KMS = 29.7847                    # sqrt(G Msun / au) in km/s
K_B, M_H = 1.380649e-16, 1.6735575e-24  # cgs


def load_phantom(path, mu=2.353):
    """Gas particles relative to the star (most massive sink): x, y, z (au), vx, vy, vz (km/s), m (Msun), T (K), mstar (Msun)."""
    import sarracen
    sdf, sinks = sarracen.read_phantom(path, separate_types="sinks")
    if "itype" in sdf.columns:
        sdf = sdf[sdf["itype"] == 1]
    s = sinks.iloc[int(np.argmax(sinks["m"].to_numpy()))]
    p = sdf.params
    m = sdf["mass"].to_numpy(np.float64) if "mass" in sdf.columns else np.full(len(sdf), float(p.get("massoftype", p.get("mass"))))
    if "temperature" in sdf.columns:
        T = sdf["temperature"].to_numpy(np.float64)
    elif "u" not in sdf.columns:          # some dumps carry neither (e.g. Young M0.2_R200_Mu1.2): passively heated disc, T = 20 K (R / 100 au)^-0.5
        R = np.hypot(sdf["x"].to_numpy(np.float64) - s.x, sdf["y"].to_numpy(np.float64) - s.y)
        T = 20.0 * (np.maximum(R, 1.0) / 100.0) ** -0.5
    else:
        u = sdf["u"].to_numpy(np.float64) * (V_UNIT_KMS * 1e5) ** 2               # erg / g
        T = (float(p.get("gamma", 5 / 3)) - 1) * mu * M_H * u / K_B
    g = lambda c: sdf[c].to_numpy(np.float64)
    return {"x": g("x") - s.x, "y": g("y") - s.y, "z": g("z") - s.z,
            "vx": (g("vx") - s.vx) * V_UNIT_KMS, "vy": (g("vy") - s.vy) * V_UNIT_KMS, "vz": (g("vz") - s.vz) * V_UNIT_KMS,
            "m": m, "T": np.maximum(T, 5.0), "mstar": float(s.m), "h": g("h"),
            "T_source": "temperature" if "temperature" in sdf.columns else ("u" if "u" in sdf.columns else "profile 20 K (R/100 au)^-0.5")}


def rescale(d, length=1.0, mass=1.0):
    """Scale-free dynamics (G = 1): lengths x L, masses x M, velocities x sqrt(M/L), temperatures x M/L. Valid for
    beta-cooled / isothermal runs; not for radiatively cooled ones (Young et al.), which carry a physical scale."""
    vs, ts = math.sqrt(mass / length), mass / length
    out = dict(d)
    for k in ("x", "y", "z"):
        out[k] = d[k] * length
    for k in ("vx", "vy", "vz"):
        out[k] = d[k] * vs
    out["m"], out["T"], out["mstar"] = d["m"] * mass, np.maximum(d["T"] * ts, 5.0), d["mstar"] * mass
    if "h" in d:
        out["h"] = d["h"] * length
    return out


def project(d, incl_deg, pa_deg=0.0, phi0_deg=0.0, flip=False):
    """Sky-plane positions (au), depth toward the observer (au) and line-of-sight velocity (km/s, + = receding).
    phi0 rotates the disc in its own plane first; flip mirrors it (spin reversed), a symmetry of the equations."""
    x, y, z, vx, vy, vz = (d[k] for k in ("x", "y", "z", "vx", "vy", "vz"))
    if flip:
        y, vy = -y, -vy
    c, s = math.cos(math.radians(phi0_deg)), math.sin(math.radians(phi0_deg))
    x, y, vx, vy = c * x - s * y, s * x + c * y, c * vx - s * vy, s * vx + c * vy
    ci, si = math.cos(math.radians(incl_deg)), math.sin(math.radians(incl_deg))
    ys, depth = y * ci - z * si, y * si + z * ci                  # depth > 0: toward the observer
    vlos = -(vy * si + vz * ci)                                     # moving toward the observer = blueshifted (negative)
    cp, sp = math.cos(math.radians(pa_deg)), math.sin(math.radians(pa_deg))
    return cp * x - sp * ys, sp * x + cp * ys, depth, vlos


def render(d, n, au_per_px, velax, incl_deg, pa_deg=0.0, phi0_deg=0.0, flip=False, tau0=10.0, mol_mass=29.0, rng=None, smooth=True):
    """Clean cube (C, n, n) in K on the given velocity axis (km/s). mol_mass: 29 for 13CO, 28 for 12CO, 30 for C18O.
    smooth: spread each particle's contribution over its SPH smoothing length (Gaussian, sigma 0.6 h in the sky plane, particles grouped
    into log2 bins of sigma). Plain pixel binning leaves ~10 particles per pixel and its shot noise reads as 60 m/s of rotation-curve
    residual on a Keplerian twin (sph-validation v2, 2026-10-10: 55-63 m/s vs MCFOST's 17.5), as large as the GI wiggle itself."""
    X, Y, depth, v = project(d, incl_deg, pa_deg, phi0_deg, flip)
    half = n * au_per_px / 2
    ix, iy = np.floor((X + half) / au_per_px).astype(np.int64), np.floor((Y + half) / au_per_px).astype(np.int64)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    pix, depth, v, m, T = iy[ok] * n + ix[ok], depth[ok], v[ok], d["m"][ok], d["T"][ok]
    sig_px = 0.6 * d["h"][ok] / au_per_px if (smooth and "h" in d) else None
    sig = np.sqrt(K_B * T / (mol_mass * M_H)) / 1e5                # thermal line width, km/s
    order = np.lexsort((-depth, pix))                               # by pixel, nearest to the observer first
    pix, v, m, T, sig = pix[order], v[order], m[order], T[order], sig[order]
    if sig_px is not None:
        sig_px = sig_px[order]
        grp = np.clip(np.floor(np.log2(np.maximum(sig_px, 0.25) / 0.25)).astype(np.int64), 0, 6)   # 0: < 0.5 px (no smoothing), then x2 steps
        gs = [(g, float(np.median(sig_px[grp == g]))) for g in np.unique(grp)]
        ky, kx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(n)[None, :]
        k2 = ky ** 2 + kx ** 2
        kern = {g: np.exp(-2 * math.pi ** 2 * s_ ** 2 * k2).astype(np.float32) for g, s_ in gs if g > 0}
    first = np.r_[True, pix[1:] != pix[:-1]]
    gstart = np.maximum.accumulate(np.where(first, np.arange(len(pix)), 0))     # index of each particle's group start
    # kappa from the target optical depth: median over occupied pixels of the column's peak tau
    colm = np.bincount(pix, m, minlength=n * n)
    sv = np.sqrt(np.bincount(pix, m * sig ** 2, minlength=n * n) / np.maximum(colm, 1e-300))
    occ = colm > 0
    kappa = tau0 / np.median(colm[occ] / (math.sqrt(2 * math.pi) * np.maximum(sv[occ], 1e-3)))
    norm = kappa * m / (math.sqrt(2 * math.pi) * sig)
    cube = np.zeros((len(velax), n * n), np.float32)
    for j, vj in enumerate(velax):
        dtau = norm * np.exp(-0.5 * ((vj - v) / sig) ** 2)
        cs = np.cumsum(dtau)
        front = cs - dtau - (cs[gstart] - dtau[gstart])            # optical depth of the particles in front, same pixel
        contrib = T * -np.expm1(-dtau) * np.exp(-front)
        if sig_px is None:
            cube[j] = np.bincount(pix, contrib, minlength=n * n)
        else:
            img = np.zeros((n, n), np.float32)
            for g, _ in gs:
                sel = grp == g
                im = np.bincount(pix[sel], contrib[sel], minlength=n * n).reshape(n, n).astype(np.float32)
                img += im if g == 0 else np.fft.irfft2(np.fft.rfft2(im) * kern[g], s=(n, n)).astype(np.float32)
            cube[j] = img.ravel()
    return cube.reshape(len(velax), n, n)


def gaussian_beam(cube, fwhm_px_major, fwhm_px_minor=None, pa_deg=0.0):
    """Convolve every channel with an elliptical Gaussian beam (unit gain; Jy/beam scaling is irrelevant for training)."""
    fwhm_px_minor = fwhm_px_major if fwhm_px_minor is None else fwhm_px_minor
    n = cube.shape[-1]
    ky, kx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(n)[None, :]
    c, s = math.cos(math.radians(pa_deg)), math.sin(math.radians(pa_deg))
    ka, kb = c * kx + s * ky, -s * kx + c * ky
    sa, sb = fwhm_px_major / 2.3548, fwhm_px_minor / 2.3548
    G = np.exp(-2 * math.pi ** 2 * ((sa * ka) ** 2 + (sb * kb) ** 2))
    return np.fft.irfft2(np.fft.rfft2(cube) * G[None], s=(n, n)).astype(np.float32)
