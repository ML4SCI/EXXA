"""
Phase 0's discriminator decides whether DDRM and VIREO-lite get built at all, so it is run
here against synthetic cases whose true forward operator is known by construction.

Three earlier versions of this check each produced a confident WRONG verdict on the project's
real cubes, and each failure is preserved below as a case:

  1. Testing whether the raw ratio P_d/P_c dips below 1. Fails whenever the two maps are on
     different intensity scales -- a dirty map is conventionally Jy/beam and a model image
     Jy/pixel, differing by the beam area in pixels. A real convolution then sits above 1 and
     reads as no_convolution. (case 3)
  2. Normalising that ratio by its own low-k level, then taking the minimum over the whole
     band. For a monotonically RISING ratio the lowest bin is by construction below the median
     of the lowest bins, so a dip appears where nothing was suppressed and additive noise
     reads as a convolution. (case 1)
  3. Deciding by whether a Gaussian beam beats the no-beam model. A sidelobed beam is not
     Gaussian, so neither model fits, the comparison ties, and a real convolution reads as
     no_convolution. (case 4)

What survives all three is fitting the forward model
`P_d(k) = A exp(-4 pi^2 sigma^2 k^2) P_c(k) + N` and asking whether the no-beam version of it
is adequate on its own. A is a free parameter, so intensity scale cannot mislead it.

    PYTHONPATH=. python3 tests/test_forward_operator.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.evaluation.forward_operator import (
    beam_kernel_of, phase0_report, format_report, pixel_scale_arcsec,
)

SIGMA, N, C = 3.0, 256, 6
failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def power_law_field(n, rng, slope=-2.5):
    """A field with a realistic falling spectrum, so there is signal at every k."""
    ky, kx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    kr[0, 0] = 1.0
    amp = kr ** (slope / 2.0)
    amp[0, 0] = 0.0
    f = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))
    return f / f.std()


def gaussian_blur(img, sigma=SIGMA):
    n = img.shape[-1]
    ky, kx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img)
                                * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (ky ** 2 + kx ** 2))))


def sidelobed_blur(img, sigma=SIGMA):
    """Gaussian core times a radial ripple: the crudest stand-in for uv-gap sidelobes."""
    n = img.shape[-1]
    y, x = np.mgrid[-24:25, -24:25].astype(float)
    r = np.sqrt(x ** 2 + y ** 2)
    k = np.exp(-r ** 2 / (2 * sigma ** 2)) * np.cos(r / 1.6)
    k /= k.sum()
    pad = np.zeros((n, n))
    pad[:49, :49] = k
    pad = np.roll(pad, (-24, -24), axis=(0, 1))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)))


HEADER = {  # the example header recorded in the physics-informed plan
    "BMAJ": 4.2629e-5, "BMIN": 3.2923e-5, "BPA": 16.9333, "CDELT2": 0.01 / 3600.0,
}
MATCHED = {  # a header describing SIGMA exactly, at 0.01"/px
    "BMAJ": SIGMA * 2.3548 * 0.01 / 3600.0, "BMIN": SIGMA * 2.3548 * 0.01 / 3600.0,
    "BPA": 0.0, "CDELT2": 0.01 / 3600.0,
}

print("=" * 70)
print("Phase 0 forward-operator discriminator")
print("=" * 70)

rng = np.random.default_rng(0)
clean = np.stack([power_law_field(N, rng) for _ in range(C)])
clean /= clean.std()
blurred = np.stack([gaussian_blur(c) for c in clean])

# --- case 1: additive noise only. A = I, nothing for DDRM to constrain. ----------------
# Regression for failure 2: the ratio here RISES monotonically with k, and a low-k
# normalisation used to turn that into a false dip.
print("\ncase 1  dirty = clean + noise")
r = phase0_report(clean, clean + rng.normal(0, 0.15, clean.shape))
print(format_report(r))
check("additive noise reads as no_convolution", r["verdict"] == "no_convolution")
check("a rising ratio is not mistaken for suppression",
      r["residual_no_convolution"] < 0.05, f"resid {r['residual_no_convolution']:.4f}")
check("best-fit beam is sub-pixel", r["fit_sigma_px"] < 0.5, f"{r['fit_sigma_px']:.2f} px")

# --- case 2: Gaussian convolution. DDRM applies. ---------------------------------------
print("\ncase 2  dirty = clean (*) gaussian(sigma=3.0) + noise")
r = phase0_report(clean, blurred + rng.normal(0, 0.02, clean.shape), MATCHED)
print(format_report(r))
check("gaussian blur reads as gaussian_convolution", r["verdict"] == "gaussian_convolution")
check("recovered sigma within 5% of truth", abs(r["fit_sigma_px"] - SIGMA) / SIGMA < 0.05,
      f"got {r['fit_sigma_px']:.3f}, true {SIGMA}")
check("measured beam agrees with the header describing it",
      0.9 < r["fit_sigma_px"] / r["header_sigma_px"] < 1.1,
      f"{r['fit_sigma_px']:.2f} px vs header {r['header_sigma_px']:.2f} px")

# --- case 3: scale invariance. Failure 1. ----------------------------------------------
print("\ncase 3  the same blur, seen on four different intensity scales")
for amp in (1.0, 5.0, 25.0, 200.0):
    r = phase0_report(clean, amp * blurred + rng.normal(0, 0.02 * amp, clean.shape))
    check(f"convolution still found at x{amp:g} intensity scale",
          r["verdict"] == "gaussian_convolution",
          f"{r['verdict']}, sigma {r['fit_sigma_px']:.2f}, A {r['scale_factor']:.1f}")
    check(f"sigma unaffected by the x{amp:g} scale",
          abs(r["fit_sigma_px"] - SIGMA) / SIGMA < 0.05, f"{r['fit_sigma_px']:.2f} px")
for amp in (1.0, 25.0):
    r = phase0_report(clean, amp * (clean + rng.normal(0, 0.15, clean.shape)))
    check(f"no false convolution at x{amp:g} intensity scale",
          r["verdict"] == "no_convolution", r["verdict"])

# --- case 4: a beam with sidelobes. Failure 3. -----------------------------------------
# A Gaussian cannot fit this, so a "does a Gaussian beat no-beam" test ties and calls it
# no_convolution. The no-beam model must be judged on its own adequacy instead.
print("\ncase 4  dirty = clean (*) sidelobed_beam + noise")
r = phase0_report(clean, np.stack([sidelobed_blur(c) for c in clean])
                  + rng.normal(0, 0.02, clean.shape), HEADER)
print(format_report(r))
check("sidelobed beam reads as non_gaussian_convolution",
      r["verdict"] == "non_gaussian_convolution")
check("no-beam model is clearly inadequate here",
      r["residual_no_convolution"] > 0.05, f"resid {r['residual_no_convolution']:.4f}")

# --- case 5: a null result must prove a beam could have been seen ----------------------
print("\ncase 5  null results, wide band vs band too narrow to see the header's beam")
r_wide = phase0_report(clean, clean + rng.normal(0, 0.05, clean.shape), HEADER)
check("clean null with a wide band is reported as no_convolution",
      r_wide["verdict"] == "no_convolution",
      f"band to k={r_wide['band_k_max']:.3f}, beam acts from k={r_wide['k_half_from_header']:.3f}")
check("the band's reach is reported so the null can be judged",
      r_wide["band_k_max"] > r_wide["k_half_from_header"])
r_narrow = phase0_report(clean, clean + rng.normal(0, 8.0, clean.shape), HEADER)
check("a null from a band too narrow to see the beam is NOT no_convolution",
      r_narrow["verdict"] == "indeterminate", r_narrow["verdict"])

# --- case 6: neither model fits -> indeterminate, never a bogus "beam" ------------------
# What the third real run produced: best-fit sigma 0.00 px (no beam found) yet a no-beam
# residual of 0.55 to 0.90 (no-beam model inadequate either). Without a guard this fell into
# the Gaussian-vs-sidelobe branch and reported RMS values up to 7e32 as
# "non_gaussian_convolution". If neither hypothesis describes the data, say so.
print("\ncase 6  clean and dirty unrelated -> neither model fits")
unrelated = np.stack([power_law_field(N, rng) for _ in range(C)])
r = phase0_report(clean, unrelated + rng.normal(0, 0.05, clean.shape), HEADER)
check("unrelated cubes give indeterminate, not a fabricated beam",
      r["verdict"] == "indeterminate", r["verdict"])
check("no absurd Gaussian RMS is reported",
      r.get("gaussian_rms_error") is None or r["gaussian_rms_error"] < 1e6,
      str(r.get("gaussian_rms_error")))

# --- case 7: channel selection must pick line-bright channels --------------------------
# phase0_from_fits used evenly spaced channels, which includes the extreme high-velocity
# ends. Those are "mostly continuum with little signal" (2026-06-18): P_clean is
# near zero there, so the ratio explodes for reasons unrelated to any beam.
print("\ncase 7  channel selection")
import tempfile
from astropy.io import fits as _fits
from src.evaluation.forward_operator import load_pair

with tempfile.TemporaryDirectory() as td:
    # 12 channels: 4 bright in the middle, 8 near-empty at the ends
    cube = np.zeros((12, 64, 64))
    bright = [4, 5, 6, 7]
    for i in bright:
        cube[i] = power_law_field(64, rng)
    cp, dp = f"{td}/c.fits", f"{td}/d.fits"
    _fits.PrimaryHDU(cube.astype(np.float32)).writeto(cp)
    _fits.PrimaryHDU((cube + rng.normal(0, 0.01, cube.shape)).astype(np.float32)).writeto(dp)
    c_sel, d_sel, _, _ = load_pair(cp, dp, max_channels=4)
    stds = c_sel.reshape(4, -1).std(axis=1)
    check("only the line-bright channels are selected", bool((stds > 1e-6).all()),
          f"selected stds {np.round(stds, 3).tolist()}")

# --- case 8: what this project's cubes actually are ------------------------------------
# Both cubes are BUNIT=JY/BEAM, so the CLEAN map is already beam-convolved: its power falls
# thirteen orders of magnitude by k ~ 0.10 and then flattens onto the float32 floor. Past
# that point neither spectrum is physical, but their ratio settles near a plausible-looking
# 2.2, and including that region alongside the real one (where the ratio climbs 1.0 to 58)
# left no (A, N) able to fit either -- Phase 0 run 4, sigma = 0 with residual 0.42 to 0.89.
print("\ncase 8  clean already beam-convolved, dirty = clean + beam-shaped noise")
BIG, BSIG = 600, 7.83
CUBE_HDR = {"BMAJ": BSIG * 2.3548 * 0.0082982052 / 3600.0,
            "BMIN": BSIG * 2.3548 * 0.0082982052 / 3600.0,
            "BPA": 0.0, "CDELT2": 0.0082982052 / 3600.0}


def beam_convolve(img, sig):
    n = img.shape[-1]
    ky, kx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img)
                                * np.exp(-2 * np.pi ** 2 * sig ** 2 * (ky ** 2 + kx ** 2))))


sky = np.stack([power_law_field(BIG, rng, slope=-3.0) for _ in range(4)])
cl = np.stack([beam_convolve(x, BSIG) for x in sky]).astype(np.float32)   # float32 on purpose
nz = np.stack([beam_convolve(x, BSIG) for x in rng.normal(0, 1, sky.shape)])
nz = nz / nz.std() * cl.std()
di = (cl + 3e-4 * nz).astype(np.float32)

r = phase0_report(cl.astype(float), di.astype(float), CUBE_HDR)
check("an already-convolved clean cube reads as no_convolution",
      r["verdict"] == "no_convolution", f"{r['verdict']}, resid {r['fit_residual']:.4f}")
check("the float32 floor is excluded rather than fitted",
      r["fit_residual"] < 0.05, f"resid {r['fit_residual']:.4f}")

# control: the check must still see a beam applied BETWEEN the two cubes
extra = np.stack([beam_convolve(c, 4.0) for c in cl])
di2 = (extra + 3e-4 * nz / nz.std() * extra.std()).astype(np.float32)
r2 = phase0_report(cl.astype(float), di2.astype(float), CUBE_HDR)
check("an EXTRA beam between the cubes is still detected",
      r2["verdict"].endswith("convolution") and r2["verdict"] != "no_convolution",
      r2["verdict"])
check("and its sigma is recovered", abs(r2["fit_sigma_px"] - 4.0) / 4.0 < 0.10,
      f"{r2['fit_sigma_px']:.2f} px, true 4.0")

# --- case 9: is the invented structure above the beam cutoff? --------------------------
# Decides whether a VIREO-style band-limit loss is aimed at the real failure. The clean cubes
# are beam-convolved, so they are band-limited by construction; a prediction with power above
# the cutoff asserts structure the instrument could not have measured.
#
# The measurement is on the RESIDUAL, not the prediction. A global power fraction of the
# prediction cannot see a local artifact: the clean field's own out-of-band share is ~2e-4 of
# a total dominated by the lowest frequencies, so one invented blob is three orders of
# magnitude too small to move it. Measuring the prediction gave 1.02x for a sharp blob and
# 0.99x for a broad one -- no discrimination at all.
print("\ncase 9  out-of-band power of a model's error")
from src.evaluation.forward_operator import out_of_band_power, beam_cutoff_k

BS_, PX_ = 7.83, 0.0082982052
OOB_HDR = {"BMAJ": BS_ * 2.3548 * PX_ / 3600.0, "BMIN": BS_ * 2.3548 * PX_ / 3600.0,
           "BPA": 0.0, "CDELT2": PX_ / 3600.0}


def _blob(n, cx, cy, r, amp):
    y, x = np.mgrid[0:n, 0:n]
    return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r ** 2))


oob_clean = np.stack([beam_convolve(power_law_field(256, rng, slope=-3.0), BS_)
                      for _ in range(4)])
oob_clean /= oob_clean.std()
amp = 0.3 * oob_clean.max()
kc = beam_cutoff_k(OOB_HDR)
check("cutoff is below Nyquist and above zero", 0 < kc < 0.5, f"k_cut {kc:.4f}")

sharp = out_of_band_power(
    oob_clean, oob_clean + np.stack([_blob(256, 80, 80, 2.0, amp) for _ in range(4)]), OOB_HDR)
broad = out_of_band_power(
    oob_clean, oob_clean + np.stack([_blob(256, 80, 80, 15.0, amp) for _ in range(4)]), OOB_HDR)
check("a sharp invented blob is flagged as out of band", sharp["excess"] > 100,
      f"{sharp['excess']:.0f}x, {sharp['residual_out_of_band_frac']:.3f} of residual power")
check("a blob broader than the beam is NOT flagged", broad["excess"] < 3,
      f"{broad['excess']:.2f}x")
check("the two are separated by orders of magnitude",
      sharp["excess"] / max(broad["excess"], 1e-9) > 100)

# --- case 10: recover the operator from a clean/dirty pair -----------------------------
# When an operator DOES sit between two cubes, it can be measured rather than requested. On
# the SG v2 self-gravitating pair this recovered a dirty beam with peak 0.911, sum ~0 and
# a -2.8% sidelobe ring, and convolving the clean cube with it reproduced HELD-OUT channels
# at correlation 0.994. That removes "ask for the PSF image" from the DDRM critical path.
print("\ncase 10  recovering the forward operator from a pair")
from src.evaluation.forward_operator import estimate_beam_from_pair, apply_beam

_y, _x = np.mgrid[-20:21, -20:21].astype(float)
_r = np.sqrt(_x ** 2 + _y ** 2)
true_beam = np.exp(-_r ** 2 / (2 * 2.5 ** 2)) * np.cos(_r / 3.0)   # sidelobed, like a real one
true_beam /= true_beam.max()

sky = np.abs(np.stack([rng.random((256, 256)) for _ in range(8)]))
made = apply_beam(sky, true_beam) + rng.normal(0, 1e-3, (8, 256, 256))
rec = estimate_beam_from_pair(sky, made, crop=41)

check("a known sidelobed beam is recovered to 1e-3",
      np.abs(rec - true_beam).max() < 1e-3, f"max err {np.abs(rec - true_beam).max():.2e}")
check("its peak is preserved", abs(rec.max() - 1.0) < 1e-3, f"{rec.max():.4f}")
# Compare against the TRUTH's own minimum, not an invented threshold: this synthetic
# beam only dips to -0.024 inside the crop, so asserting "< -0.05" asserted something false.
check("negative sidelobes survive the recovery",
      rec.min() < 0 and abs(rec.min() - true_beam.min()) < 1e-3,
      f"recovered {rec.min():.4f} vs true {true_beam.min():.4f}")
check("re-convolving reproduces the dirty cube",
      float(np.corrcoef(apply_beam(sky, rec).ravel(), made.ravel())[0, 1]) > 0.999)

# A = I must give back a delta, not a spurious beam
flat = estimate_beam_from_pair(sky, sky + rng.normal(0, 1e-3, sky.shape), crop=41)
c = flat.shape[0] // 2
off = flat.copy(); off[c, c] = 0
check("an identity operator recovers a delta, not a beam",
      abs(flat[c, c] - 1.0) < 0.05 and np.abs(off).max() < 0.05,
      f"centre {flat[c, c]:.4f}, largest off-centre {np.abs(off).max():.4f}")

# --- case 11: a large intensity-scale factor must not break the least-squares solve ------
# 2026-08-27, the self-gravitating v2 cube: P_dirty/P_clean ran from ~1e5 at the lowest k to
# ~1e-4 by mid-band. The unscaled lstsq returned A=2.4e-5 against a naive low-k estimate of
# A~1e5 -- nine orders of magnitude off, and it fed a wrong Phase 0 verdict (A=0.000 printed,
# Gaussian RMS in the billions) that looked like a data anomaly and was actually a numerical
# one. Fixed by column-scaling the lstsq problem before solving. This is the regression case.
print("\ncase 11  a large real amplitude factor (A ~ 1e5) does not break the fit")
amp = 1.0e5              # amplitude-space scale factor applied to the dirty MAP
big_amp = amp ** 2        # `A` in fit_forward_model is a POWER-spectrum coefficient, so a
                          # map-amplitude scale of `amp` shows up as amp**2 in P_dirty/P_clean
d_big = amp * blurred + rng.normal(0, 0.02 * amp, clean.shape)
r_big = phase0_report(clean, d_big)
check("verdict is still a convolution, not broken by the huge amplitude",
      r_big["verdict"] == "gaussian_convolution", r_big["verdict"])
check("amplitude recovered within an order of magnitude of the truth (power-spectrum A = amp^2)",
      0.1 * big_amp < r_big["scale_factor"] < 10 * big_amp,
      f"fitted A={r_big['scale_factor']:.3g}, true A={big_amp:.3g}")
check("sigma still recovered correctly at this amplitude",
      abs(r_big["fit_sigma_px"] - SIGMA) / SIGMA < 0.10, f"{r_big['fit_sigma_px']:.2f} px")

# --- beam kernel from a header ---------------------------------------------------------
print("\nbeam kernel from the recorded header")
k = beam_kernel_of(HEADER)
check("pixel scale read from CDELT2", abs(pixel_scale_arcsec(HEADER) - 0.01) < 1e-9)
check("kernel normalised to unit sum", abs(k.sum() - 1.0) < 1e-9)
check("kernel is odd-sized and square",
      k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1, str(k.shape))
check("peak at the centre",
      np.unravel_index(k.argmax(), k.shape) == (k.shape[0] // 2,) * 2)
maj = HEADER["BMAJ"] * 3600.0 / 0.01 / 2.3548
mn = HEADER["BMIN"] * 3600.0 / 0.01 / 2.3548
check("major-axis sigma matches the header", abs(maj - 6.52) < 0.05, f"{maj:.2f} px")
check("kernel is elliptical, not round", abs(maj - mn) > 1.0, f"{maj:.2f} vs {mn:.2f} px")
check("no beam keys -> None", beam_kernel_of({"CDELT2": 1e-5}) is None)
check("no pixel scale -> None", beam_kernel_of({"BMAJ": 1e-5, "BMIN": 1e-5, "BPA": 0}) is None)

print("\n" + "-" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
sys.exit(0)
