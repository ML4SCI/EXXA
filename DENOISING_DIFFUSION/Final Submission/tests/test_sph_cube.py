"""src/data/sph_cube.py on a synthetic Keplerian particle disc (no dump file needed).

Run: PYTHONPATH=. python3 tests/test_sph_cube.py
"""
import math
import numpy as np
from src.data import sph_cube as sc


def kepler_disc(N=400_000, mstar=1.0, rin=10.0, rout=100.0, hr=0.05, T0=40.0, seed=0):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(rin ** 2, rout ** 2, N))                      # uniform surface density
    ph = rng.uniform(0, 2 * math.pi, N)
    z = rng.standard_normal(N) * hr * r
    vk = math.sqrt(mstar / 1.0) / np.sqrt(r) * sc.V_UNIT_KMS             # km/s, G = 1 au / Msun
    return {"x": r * np.cos(ph), "y": r * np.sin(ph), "z": z, "vx": -vk * np.sin(ph), "vy": vk * np.cos(ph), "vz": 0 * r,
            "m": np.full(N, 1e-7), "T": T0 * (r / 50.0) ** -0.5, "mstar": mstar}


def main():
    d = kepler_disc()
    n, au, incl = 128, 1.8, 30.0
    velax = np.arange(-6.0, 6.0001, 0.1)
    cube = sc.render(d, n, au, velax, incl, tau0=10.0)
    assert np.isfinite(cube).all() and cube.min() >= 0
    assert cube.max() <= d["T"].max() * 1.001, (cube.max(), d["T"].max())     # optically thick limit: I <= T
    # intensity-weighted M1 against the analytic projected Keplerian field
    w = cube.sum(0)
    m1 = (cube * velax[:, None, None]).sum(0) / np.maximum(w, 1e-30)
    X, Y = np.meshgrid((np.arange(n) + 0.5) * au - n * au / 2, (np.arange(n) + 0.5) * au - n * au / 2)
    ci, si = math.cos(math.radians(incl)), math.sin(math.radians(incl))
    r = np.hypot(X, Y / ci)
    truth = -sc.V_UNIT_KMS / np.sqrt(np.maximum(r, 1e-3)) * (X / np.maximum(r, 1e-3)) * si
    sel = (r > 25) & (r < 85) & (w > 0.2 * np.median(w[w > 0]))
    err = np.median(np.abs(m1[sel] - truth[sel])) / np.median(np.abs(truth[sel]))
    print(f"M1 median relative error vs Keplerian truth: {err:.3f} over {sel.sum()} px; peak I {cube.max():.1f} K <= T_max {d['T'].max():.1f} K")
    assert err < 0.08, err
    # optical depth controls which layer is seen: thin and thick both stay finite, thick is brighter-capped
    thin = sc.render(d, n, au, velax, incl, tau0=0.05)
    assert thin.max() < cube.max(), (thin.max(), cube.max())
    # beam: unit gain (flux kept) and smoother
    b = sc.gaussian_beam(cube, 6.0, 4.0, 30.0)
    assert abs(b.sum() / cube.sum() - 1) < 1e-3 and b.std() < cube.std()
    # projection symmetry: a flipped, PA-rotated disc renders without loss
    f = sc.render(d, n, au, velax, incl, pa_deg=120.0, phi0_deg=45.0, flip=True)
    assert abs(f.sum() / cube.sum() - 1) < 0.05, f.sum() / cube.sum()
    print("PASS")


if __name__ == "__main__":
    main()
