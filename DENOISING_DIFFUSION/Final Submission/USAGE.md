# Running the denoiser on a cube

One script, `experiments/alma_unified.py`, takes a line-emission cube in FITS through the whole pipeline: it finds the line-free channels and the noise, decides on continuum subtraction, resamples the cube so the beam spans about 10 pixels, applies the Wiener filter and the four networks, blends with the do-no-harm gate, runs the second Wiener pass, puts the result back on the original grid, and writes the denoised cube, the moment maps, a figure and a report of checks. The same command is used for a new observation and for the 20 published disks of `results/05-real-alma/`.

## What you need

- Python 3.10 or newer and `pip install -r requirements.txt` (numpy 2 is required).
- The four checkpoints of the final model (see [`MODELS.md`](MODELS.md)). They are passed together, comma-separated, and averaged.
- A GPU for real cubes. A 301-channel exoALMA cube at 512 px took 26 to 27 minutes on a Kaggle GPU, about 70 minutes with the injection test. CPU works for small cubes (the smoke test below runs a toy cube in about 30 s) but is impractical at full size.
- A CLEANed image cube (`.fits`, axes RA, Dec, velocity or frequency) with the beam in the header (`BMAJ`, `BMIN`). Any channel width and pixel size. Raw visibilities or a dirty image are not supported.

Check the installation first; this runs every step of the pipeline on a synthetic cube with a random network and needs no data or checkpoints:

```bash
cd "DENOISING_DIFFUSION/Final Submission"
PYTHONPATH=. python3 tests/test_alma_unified_smoke.py
```

## A new cube

```bash
CKPTS=unified_r4n_k15_long.pth,unified_r4n_k15_long_s11.pth,unified_r4n_k15_long_s23.pth,unified_r8_m1_r6.pth

python experiments/alma_unified.py --cube MyDisk_13CO.fits --ckpt $CKPTS --out out/mydisk \
    --crop 512 --target-beam-px 10
```

That is enough to get a denoised cube and moment maps. Add the disk's geometry to also get the kinematic checks, the rotation-curve residual and the minor-axis PV curve:

```bash
python experiments/alma_unified.py --cube MyDisk_13CO.fits --ckpt $CKPTS --out out/mydisk \
    --crop 512 --target-beam-px 10 --incl 35 --dist 150 --pa 120
```

`--incl` (degrees) switches the kinematic checks on; the position angle and systemic velocity are fitted on the raw M1 with the inclination fixed. `--dist` (pc, default 150) converts pixels to au. `--pa` (degrees) is only recorded next to the fitted value for comparison.

## Choosing the settings

| situation | option |
|---|---|
| any cube | `--crop 512 --target-beam-px 10`: a 512 px centre crop (default 1024), resampled so the beam spans about 10 px (it only zooms in, never out). Independently, `--infer-beam-px` (default 10) runs the network on a coarser grid when the beam is wider than 12 px, so together the network always sees a beam of about 10 px. Every real-data run used `--target-beam-px 10` |
| channels coarser than 0.1 km/s (DSHARP: 0.35 km/s) | `--spec-interp 0.1`: interpolate to 0.1 km/s along velocity, denoise there, average back. On the native 0.35 km/s channels the wiggle score falls (AS 209 at SNR 12: 0.19) |
| a large field or a large beam (GPU memory) | `--max-scaled-px 1024` caps the resampled grid (default 1024); lower it if the GPU runs out of memory |
| noise rising toward the edge of the field (primary-beam corrected cubes) | `--auto-fov` crops a square around the disk automatically, or `--fov-arcsec 6.4` crops a fixed field; the output keeps the matching coordinates |
| continuum | default `--cont-sub auto` subtracts only when continuum is found on both sides of the line. Use `--cont-sub off` for cubes that are already continuum subtracted (exoALMA); `--cont-sub on` always subtracts |
| an uncertainty map for M1 | `--boot 4`: re-denoises four copies of the denoised cube with new realisations of the cube's own noise and writes the per-pixel spread of M1 (noise term only; MWC 758 with `--boot 4` took about 2 hours (125 minutes) instead of 26 minutes) |
| flux near the star | `--central-fix` rescales the flux within 2 to 4 beams of the centre to the input's. Off by default: it improves M0 on hydrodynamic disks and worsens it on smooth ones |

The model settings themselves (four networks averaged, gate `40,120,0.25`, second Wiener pass, beam at 10 px) are the defaults and should not be changed for results comparable to this project. `--gate ""` and `--no-wiener2` exist to switch the gate and the second pass off.

## The 20 published disks

The real-data results were produced with exactly these commands. The cubes are public.

**exoALMA** (13CO fiducial images, Harvard Dataverse [doi:10.7910/DVN/CFHWNH](https://doi.org/10.7910/DVN/CFHWNH), files `<disk>_13CO_fiducial.image.fits`):

```bash
python experiments/alma_unified.py --cube MWC_758_13CO_fiducial.image.fits --ckpt $CKPTS --out out/MWC_758_13CO \
    --crop 512 --target-beam-px 10 --incl 19.4 --dist 156 --pa 240.3
```

| disk | `--incl` | `--dist` | `--pa` | | disk | `--incl` | `--dist` | `--pa` |
|---|---|---|---|---|---|---|---|---|
| MWC_758 | 19.4 | 156 | 240.3 | | HD_135344B | 16.1 | 135 | 242.9 |
| DM_Tau | 38.7 | 144 | 335.7 | | HD_143006 | 16.9 | 167 | 168.9 |
| AA_Tau | 58.7 | 135 | 272.7 | | J1604 | 6.0 | 145 | 258.1 |
| LkCa_15 | 50.3 | 157 | 62.1 | | J1615 | 46.5 | 156 | 325.4 |
| HD_34282 | 58.3 | 309 | 117.4 | | V4046_Sgr | 34.1 | 72 | 255.7 |
| CQ_Tau | 36.3 | 149 | 235.1 | | J1842 | 39.4 | 151 | 205.9 |
| SY_Cha | 52.4 | 182 | 346.2 | | J1852 | 32.7 | 147 | 117.1 |
| PDS_66 | 31.9 | 98 | 189.0 | | | | | |

Geometry from exoALMA I (Teague et al. 2025): distance from Table 1, inclination and position angle from Table 2.

**DSHARP** (12CO images, `https://bulk.cv.nrao.edu/almadata/lp/DSHARP/images/<target>_CO.fits`), always with `--spec-interp 0.1` and `--max-scaled-px 1024`:

```bash
python experiments/alma_unified.py --cube AS209_CO.fits --ckpt $CKPTS --out out/AS_209 \
    --crop 768 --target-beam-px 10 --max-scaled-px 1024 --spec-interp 0.1 --incl 34.97 --dist 121 --pa 85.76
```

| target | `--crop` | `--incl` | `--dist` | `--pa` |
|---|---|---|---|---|
| HD143006 | 512 | 18.6 | 165 | 169.0 |
| AS209 | 768 | 34.97 | 121 | 85.76 |
| HD142666 | 512 | 62.22 | 148 | 162.11 |
| HD163296 | 1024 | 46.7 | 101 | 133.33 |
| IMLup | 1280 | 47.5 | 158 | 144.5 |

Geometry from Huang et al. (2018), Table 2 (continuum fits), distances from DSHARP I Table 1 (Andrews et al. 2018). Run from this branch on a Kaggle GPU with the four final checkpoints, the MWC 758 and AS 209 commands above reproduce the archived numbers to the digit ([`results/05-real-alma/final-submission-realcube_v1_2026-10-31/`](results/05-real-alma/final-submission-realcube_v1_2026-10-31/)). The archived outputs of all 20 runs, with their reports, are in [`results/05-real-alma/`](results/05-real-alma/); the Kaggle scripts that ran them are `kernels/alma_exoalma.py` and `kernels/alma_dsharp.py`.

## What comes out

For an input `MyDisk_13CO.fits`, the folder given by `--out` holds:

| file | content |
|---|---|
| `MyDisk_13CO_denoised.fits` | the denoised cube on the input's grid and header |
| `MyDisk_13CO_m0_raw.fits`, `_m1_raw`, `_m2_raw` | moment maps of the input (M1 and M2 from a quadratic fit to the line, bettermoments) |
| `MyDisk_13CO_m0_den.fits`, `_m1_den`, `_m2_den` | the same for the denoised cube |
| `MyDisk_13CO_moments.png` | raw (top) against denoised (bottom): a channel, M0, M1, M2 and a line-free channel |
| `MyDisk_13CO.png`, `MyDisk_13CO_pv.png` | with `--incl`: the rotation-curve residual maps and the minor-axis PV curves, raw against denoised |
| `MyDisk_13CO_m1_boot_std.fits` | with `--boot`: the M1 uncertainty map |
| `report.json` | every number below, plus the settings used |

The checks in `report.json` (defined in [`results/05-real-alma/CHECKS.md`](results/05-real-alma/CHECKS.md)):

- `S1`: noise in the line-free channels before and after (`noise_reduction`), the total M0 inside the disk denoised over raw (`m0_total_ratio`, should be close to 1), and how far M1 moved on the brightest pixels (`m1_shift_bright_median_ms`).
- `S2`: the fraction of line-free pixels above 4 sigma before and after. A rise means the model drew structure where there is no line.
- `S6` (with `--incl`): the fitted geometry, the rotation-curve residual of raw and denoised M1, and the PV curve magnitude of each (`pv_mag_ms`, `pv5_mag_ms` for the 5 px slit).
- `snr_est`, `gate_a`: the blind estimate of the peak SNR and the weight of the network output against the Wiener result (1 up to an estimate of 40, falling linearly to 0.25 at 120 and above; the plain network was measured to damage M0 and M2 at an estimate of 121 to 133, and the ramp between 40 and 120 is an interpolation that was not measured).
- `continuum`, `residual_continuum_peak_sigma`: the continuum decision and what was left after it.

## Testing on a cube with a known answer

A real cube has no truth, so the checks above say what the model changed, not whether it is right. To measure recovery in the noise of your own cube, inject a simulated disk whose velocity field is known into its line-free channels:

```bash
python experiments/alma_unified.py --cube MyDisk_13CO.fits --ckpt $CKPTS --out out/mydisk --crop 512 --target-beam-px 10 \
    --inject-dir <clean cubes> --inject-cases GI_r0.5,Kep_r0.5 --inject-snr 12,24 --inject-seeds 5
```

`<clean cubes>` is a folder of noise-free cubes named `<case>_clean.fits` (made by `experiments/speedie_prep.py`). Each case is scaled to each peak SNR, added to the cube's real noise, denoised and scored against its truth; the scores go to `S5` in `report.json`. `GI_r0.5` carries a wiggle and `Kep_r0.5` is its twin without one, so the second measures the spurious wiggle the model adds in this noise. For small fields add `--inject-field-frac 0.625`.

For synthetic cubes generated with their truth (`experiments/make_analytic_gi_cubes.py`), and for the held-out test disks, scoring is done by `experiments/train_unified.py --eval-only`; the commands are in [`MODELS.md`](MODELS.md).

## Before using the output for science

- Take error bars from fits to the raw cube, not the denoised one: its noise is correlated, so fitting tools report uncertainties that are too small.
- The absence of a kink or wiggle in a denoised cube is not evidence that there is none. Below peak SNR of about 6, and for wiggles below about 15 to 20 m/s, nothing is recovered.
- Amplitudes are underestimated; the denoised PV magnitude is a lower bound.
- Use flux only beyond 3 beams from the star.
- Do not force a second continuum subtraction on cubes that are already subtracted.
