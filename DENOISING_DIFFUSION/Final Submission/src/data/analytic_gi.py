"""
Semi-analytic GI-wiggle disk (after C. Longarini's model).

A self-gravitating disk rotates faster than Keplerian (its own mass adds to the pull) and an m-armed
logarithmic spiral perturbs the radial and azimuthal velocity by an amount that scales with the
disc-to-star mass ratio q = md/ms and cooling beta^(-1/2). Projected on the line of sight this is
the "interlocking fingers" wiggle. Because the line-of-sight velocity is known in closed form, a cube
built from it has an EXACT truth for the line centre at every pixel, which no hydro cube gives us.

What is ported (the `momentoneC` path of the original, the one its plots use), and what changed:
  - physics unchanged: rotation curve `basicspeed` (Keplerian + disc term through elliptic integrals),
    perturbations `ura`/`uphC`, sky velocity = (u_phi cos(phi) + u_r sin(phi)) sin(i).
  - vectorised (no Python pixel loops); `np.interp` for the rotation curve instead of an index lookup.
  - not ported: `get_masses`, `amplitude_central_channel`, the `*eta` variants and the plot code
    (the eta variants carry no azimuthal spiral term).
Units: au, Msun, km/s.

Added here (NOT in the original, chosen and not derived): the emission model that turns the velocity
field into a cube (power-law brightness with an inner hole and outer taper, a density-spiral contrast
of `dens_contrast`, a Gaussian line of width `line_sigma`) and the beam. Treat the cubes as a test
bench with exact velocity truth, not as a hydro-quality disk.
"""
import numpy as np
from scipy import special
from scipy.integrate import quad

G = 4.30091e-3 * 206265          # au (km/s)^2 / Msun


def omega(ms, r):
    return np.sqrt(G * ms / r ** 3)


def sigma_in(p, rin, rout, md):
    return ((2 + p) * md) / (2 * np.pi * rin ** 2) / ((rout / rin) ** (2 + p) - 1)


def sigma(p, rin, rout, md, r):
    """Surface density [Msun/au^2], Sigma ~ r^p between rin and rout."""
    return sigma_in(p, rin, rout, md) * (r / rin) ** p


def _integrand(r1, r, z, md, p, rin, rout):
    zet = np.sqrt(4 * r1 * r / ((r + r1) ** 2 + z ** 2))
    K, E = special.ellipk(zet), special.ellipe(zet)
    return ((K - 0.25 * (zet ** 2 / (1 - zet ** 2)) * (r1 / r - r / r1 + z ** 2 / (r * r1)) * E)
            * np.sqrt(r1 / r) * zet * sigma(p, rin, rout, md, r1))


def rotation_curve(radii, md, p, rin, rout, ms, z=1e-3):
    """Midplane rotation speed of the self-gravitating disk [km/s] at `radii` [au]."""
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    disc = np.array([G * quad(_integrand, 0.5 * rin, 2 * rout, args=(r, z, md, p, rin, rout), limit=200)[0]
                     for r in radii])
    return np.sqrt(np.maximum(G * ms / radii + disc, 0.0))


def q_local(ms, md, p, rin, rout, r):
    """Local disc-to-star mass ratio for a Q = 1 disc."""
    return (md / ms) * (rout / rin) ** (-2 - p) * (r / rin) ** (2 + p)


def u_r_amp(ms, md, p, m, chi, beta, rin, rout, r):
    return 2 * m * chi * beta ** -0.5 * q_local(ms, md, p, rin, rout, r) ** 2 * omega(ms, r) * r


class AnalyticGI:
    """One disk. `v_los(s0, s1)` is the exact line-of-sight velocity [km/s] at sky offsets (au);
    axis 0 is the minor axis (compressed by cos i), axis 1 the major axis (v ~ cos(phi) = s1/r)."""

    def __init__(self, ms=1.0, md=0.35, p=-1.0, m=2, chi=1.0, beta=5.0, pitch_deg=13.0, incl_deg=30.0,
                 rin=10.0, rout=290.0, off=0.0, n_radii=400):
        self.__dict__.update(ms=ms, md=md, p=p, m=m, chi=chi, beta=beta, alpha=np.radians(pitch_deg),
                             incl=np.radians(incl_deg), rin=rin, rout=rout, off=off)
        self.pitch_deg, self.incl_deg = pitch_deg, incl_deg
        self.radii = np.geomspace(0.5 * rin, 1.5 * rout, n_radii)
        self.rc = rotation_curve(self.radii, md, p, rin, rout, ms)

    def _polar(self, s0, s1):
        gx, gy = s0 / np.cos(self.incl), s1             # deprojected disc-plane coordinates
        return np.hypot(gx, gy), np.arctan2(gx, gy)     # orig: grid_angle = atan2(gx, gy)

    def _phase(self, r, ang):
        return self.m * ang - self.m / np.tan(self.alpha) * np.log(r) + self.off

    def v_los(self, s0, s1, perturb=True):
        r, ang = self._polar(s0, s1)
        r = np.maximum(r, 1e-3)
        rc = np.interp(r, self.radii, self.rc)
        if not perturb:
            return rc * np.cos(ang) * np.sin(self.incl)
        ph = self._phase(r, ang)
        uph = (np.sqrt(G * self.ms / r) * self.beta ** -0.5 / 2 * (self.md / self.ms) * np.sin(ph) + rc)
        ur = -u_r_amp(self.ms, self.md, self.p, self.m, self.chi, self.beta, self.rin, self.rout, r) * np.sin(ph)
        return (uph * np.cos(ang) + ur * np.sin(ang)) * np.sin(self.incl)

    def density_modulation(self, s0, s1):
        """Original `perturbed_sigma`: beta^(-1/2) * -cos(phase). Dimensionless, order 0.4."""
        r, ang = self._polar(s0, s1)
        return self.beta ** -0.5 * -np.cos(self._phase(np.maximum(r, 1e-3), ang))

    def brightness(self, s0, s1, dens_contrast=0.5):
        r, _ = self._polar(s0, s1)
        r = np.maximum(r, 1e-3)
        I = (r / self.rin) ** -1.0 * (1 - np.exp(-(r / self.rin) ** 2)) * np.exp(-(r / (0.8 * self.rout)) ** 3)
        return I * np.clip(1 + dens_contrast * self.density_modulation(s0, s1), 0, None)


def gaussian_beam_kernel(n, fwhm_maj_px, fwhm_min_px, pa_deg):
    """Elliptical Gaussian, unit sum, centred (for FFT use with ifftshift). PA east of north as FITS BPA:
    angle of the major axis measured from +axis0 toward +axis1 is taken as pa_deg + 90 (image convention)."""
    y, x = np.indices((n, n)) - n // 2
    t = np.radians(pa_deg)
    u = x * np.sin(t) + y * np.cos(t)         # along the major axis
    w = x * np.cos(t) - y * np.sin(t)         # along the minor axis
    s1, s2 = fwhm_maj_px / 2.3548, fwhm_min_px / 2.3548
    k = np.exp(-0.5 * ((u / s1) ** 2 + (w / s2) ** 2))
    return k / k.sum()


def render_cube(disk, n=301, au_per_px=1.99, n_chan=101, dv=0.1, line_sigma=0.25, dens_contrast=0.5,
                beam_px=(16.1, 12.9, 175.7), peak=0.05, perturb=True):
    """Noise-free 'clean' cube (C, n, n) float32, v_sys = 0, channel k at velocity (k - n_chan//2) dv.
    Returns (cube, v_los_truth) with v_los_truth the exact, unbeamed line centre on the sky grid."""
    c = (np.arange(n) - n // 2) * au_per_px
    s0, s1 = np.meshgrid(c, c, indexing="ij")
    v = disk.v_los(s0, s1, perturb=perturb)                 # perturb=False: the no-wiggle twin (pure rotation, same disc)
    I = disk.brightness(s0, s1, dens_contrast if perturb else 0.0)
    vel = (np.arange(n_chan) - n_chan // 2) * dv
    cube = np.empty((n_chan, n, n), np.float32)
    k = np.fft.fft2(np.fft.ifftshift(gaussian_beam_kernel(n, *beam_px)))
    for i, vc in enumerate(vel):
        ch = I * np.exp(-0.5 * ((vc - v) / line_sigma) ** 2)
        cube[i] = np.real(np.fft.ifft2(np.fft.fft2(ch) * k))
    cube *= peak / cube.max()
    return cube, v
