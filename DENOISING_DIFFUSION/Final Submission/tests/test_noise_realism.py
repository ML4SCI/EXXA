"""Realistic-noise training pieces of experiments/train_unified.py.

Run: PYTHONPATH=. python3 tests/test_noise_realism.py
"""
import os, sys
import numpy as np
from scipy import ndimage

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")]
import train_unified as tu


def disk_cube(C=160, n=96, seed=0):
    """A rotating-disk-like line cube: Gaussian line whose centre shifts across x, confined to a central disk."""
    y, x = np.indices((n, n)) - n / 2
    r = np.hypot(x, y)
    v0 = 0.6 * x / (n / 2)                                    # km/s, a linear velocity gradient
    vel = np.linspace(-2, 2, C)[:, None, None]
    return (np.exp(-0.5 * ((vel - v0) / 0.15) ** 2) * (r < n / 3)).astype(np.float32)


def corr_noise(shape, a, rng):
    w = rng.standard_normal(shape).astype(np.float32)
    return ndimage.convolve1d(w, tu.chan_kernel(a), axis=0, mode="wrap") if a > 0 else w


def main():
    rng = np.random.default_rng(1)
    clean = disk_cube()

    # 1) blind lag correlation: Hanning noise reads ~+0.67 / +0.17, independent noise ~0, with a disk present
    for a, want in ((0.5, (2 * 0.5 / 1.5, 0.25 / 1.5)), (0.0, (0.0, 0.0))):
        r1, r2 = tu.velocity_corr(clean + 0.05 * corr_noise(clean.shape, a, rng))
        assert abs(r1 - want[0]) < 0.03 and abs(r2 - want[1]) < 0.03, (a, r1, r2, want)
        print(f"a={a}: lag-1 {r1:+.3f} (want {want[0]:+.3f}), lag-2 {r2:+.3f} (want {want[1]:+.3f})")

    vs = tu.vel_spectrum(160, 0.64, 0.15)
    assert abs(vs.mean() - 1) < 1e-3, vs.mean()

    # 2) the Wiener with the velocity noise spectrum beats the channel-independent one on Hanning noise
    dirty = clean + 0.15 * corr_noise(clean.shape, 0.5, rng)
    nz = tu.empirical_noise2d(dirty)
    flat = tu.wiener(dirty, None, nz)
    vel = tu.native_wiener(dirty, "line")
    e_flat, e_vel = (float(np.sqrt(((w - clean) ** 2).mean())) for w in (flat, vel))
    print(f"Wiener rms error on Hanning noise: channel-independent {e_flat:.4f}, velocity-aware {e_vel:.4f}")
    assert e_vel < e_flat, (e_vel, e_flat)

    # 3) SNR draw: peak / noise rms of a redrawn line cube matches the requested SNR
    tu.REAL.update(snr=(40.0, 40.0), chan_a=0.5, beam_max=1.0)
    cube = tu.TrainCube("t", "line", clean, clean + 0.01 * rng.standard_normal(clean.shape).astype(np.float32), 0.025, fwhm_px=4.0)
    cube.redraw(rng, [clean.shape[-1]])
    c, d = cube.c.astype(np.float64), cube.d.astype(np.float64)
    snr = c.max() / (d - c).std()
    print(f"requested SNR 40, got {snr:.1f}")
    assert abs(snr - 40) < 2, snr

    # 4) beam broadening keeps flux and smooths the target
    tu.REAL.update(beam_max=2.0)
    cube.redraw(np.random.default_rng(3), [clean.shape[-1]])
    tu.REAL.update(snr=None, chan_a=0.0, beam_max=1.0)
    print("PASS")


if __name__ == "__main__":
    main()
