# Denoising Astronomical Observations of Protoplanetary Disks

A denoiser for ALMA line-emission cubes of protoplanetary disks that keeps the small velocity perturbations astronomers look for: the "wiggle" a gravitationally unstable disk imprints on its velocity field, and the kinks of embedded planets. It is judged on moment maps and on the velocity residual after the rotation curve is removed, not on pixel error alone.

Written by Krishan Yadav for Google Summer of Code 2026 with ML4Sci (EXXA), mentored by Jason Terry, PhD and Gaurav S. The final write-up is on Medium: [Denoising Astronomical Observations of Protoplanetary Disks](https://medium.com/@kryshan753/denoising-astronomical-observations-of-protoplanetary-disks-cfec46bcaf60). The midterm write-up is [here](https://medium.com/@kryshan753/denoising-astronomical-observations-of-protoplanetary-disks-600403e036c8).

This folder is the final submission. It holds the code, the scripts that trained the final networks and produced every number below, and the results behind them, grouped by result.

## The main result

On a disk held back for a single test and evaluated once, after the model was frozen, the wiggle score (defined below) is:

| peak SNR | model | Wiener filter | noisy cube |
|---|---|---|---|
| 12 | 0.44 (0.59 ± 0.10 over four noise realisations) | 0.17 (0.18 ± 0.03) | 0.03 |
| 24 | 0.70 (0.79 ± 0.06) | 0.41 (0.42 ± 0.02) | 0.16 |

| test (never used for training or tuning) | model | Wiener filter |
|---|---|---|
| held-out line-emission cubes, PSNR at 256 to 600 px | 51.2 to 51.8 dB | 30.5 to 30.6 dB |
| held-out line-emission cubes, M0 / M1 / M2 improvement over the noisy cube | 62 / 88.5 / 90.5 % | |
| 8 fresh analytic GI disks, beam to be undone: wiggle score | 0.79 | 0.47 |
| 8 fresh hydrodynamic simulations (Young et al.): wiggle score | 0.56 | 0.33 |
| GI disk injected into real MWC 758 noise, SNR 12 / 24: wiggle score | 0.84 / 0.93 | 0.64 / 0.83 |
| 20 real disks (15 exoALMA, 5 DSHARP): position angle within 0 to 3.9 degrees of the published value | 19 of 20 | |

The evidence for each line is in [`results/`](results/), one folder per result.

## Metrics

All scores lie between 1 (identical to the noise-free truth) and 0 (no relation).

- **Wiggle score** (`rc_r` in the code): the azimuthally averaged rotation curve is subtracted from the velocity map (M1), and the Pearson correlation of the residual with the truth's residual is computed.
- **PV r:** the correlation between the denoised and true position-velocity (PV) curves along the disk's minor axis, where rotation has no line-of-sight component (1 px slice or 5 px slit, quadratic trend removed). The curve's standard deviation is the **PV magnitude** in m/s, always reported next to the correlation, since a correlation ignores scale.
- **Peak SNR:** the peak of the noise-free cube divided by the noise standard deviation.
- **Twin test:** the same disk with the spiral switched off. Its true minor-axis curve is flat, so whatever wiggle the denoiser draws on it is spurious; this sets the detection limit.
- **M0, M1, M2 and PSNR:** integrated intensity, mean velocity and line width maps, and the peak signal-to-noise ratio of the reconstruction in dB. Moment improvements are quoted over the noisy cube, signal-masked and noise-clipped.

## Why this problem is awkward

**The signal sits at the noise level.** The wiggle is tens of m/s on top of rotation of several km/s. A denoiser can fail by smoothing it away, which makes the map look better and removes the science, or by drawing one that is not there, which looks like a detection.

**Pixel metrics cannot see either failure.** A model can reach 51.7 dB and still lose the wiggle, which is why every result here is scored on the wiggle and on a twin disk with no wiggle.

**There is a hard limit.** An ideal linear filter that knows the true spectrum reaches a wiggle score of only 0.45 at SNR 12 on the selection disk; the model reaches 0.68 (mean of eight noise draws; the first draw gave 0.60). A target of 0.85 at that SNR is not reachable on these data.

## Method

A blind 3D Wiener filter, estimated from the cube itself, is the baseline. A U-Net learns a correction to it whose output layer starts at zero, so the untrained model equals the Wiener filter. Four networks are averaged; a do-no-harm gate blends the output back toward the Wiener result on cubes that need little denoising; the Wiener filter is recomputed from the first pass; and the cube is resampled so that the beam spans about 10 pixels. Settings: [`results/final/RECIPES.md`](results/final/RECIPES.md).

### Names used in the results files

The run records keep the working names they were written with:

- **Recipe B** is the final model described above. **Recipe A** is the same pipeline with networks 1 to 3 only; it was stored for comparison and is not used.
- **Candidates A, C, D, E** are the four improvement attempts in [`results/06-improvement-attempts/`](results/06-improvement-attempts/): a central-flux correction, more varied training disks, an amplitude-matching loss, and hydrodynamic simulations in training.
- **R4n-long** (seeds 3, 11, 23) are networks 1 to 3; **R6** (256 px crops) is the first stage of network 4 and **R8-from-R6** is network 4. Checkpoint names follow these, see [`MODELS.md`](MODELS.md).
- **The Speedie disk** (r1.5) is the held-out test disk of the main result; **9074** is the hydrodynamic disk used to measure amplitude loss.
- **Kaggle account** in the run records: the runs were spread over several of the author's Kaggle accounts for GPU quota, so a run that cannot see another run's output or datasets is explained by the account, not by an error.

## Limits

- Nothing is recovered below peak SNR of about 6, or for wiggles below about 15 to 20 m/s.
- The model draws about 18 to 27 m/s of spurious wiggle on disks that have none.
- Amplitudes are underestimated (57 to 72 percent of the wiggle amplitude kept on the hydrodynamic disk 9074).
- Flux within one beam of the star is over-estimated; use flux only beyond 3 beams.
- Nearly face-on disks and compact, beam-sized features are not recovered.
- On DSHARP's coarse 0.35 km/s channels the model needs interpolation to 0.1 km/s, and at SNR 12 it can lose to Wiener.
- No per-pixel uncertainty by default (`--boot` gives an empirical map of the noise term).

Four attempts to improve the model, judged by a rule fixed before they were trained, did not pass ([`results/06-improvement-attempts/`](results/06-improvement-attempts/)).

## What is where

```text
Final Submission/
├── README.md           this file
├── USAGE.md            how to run the denoiser on a new or published cube
├── MODELS.md           the four trained networks, how to obtain them, and the exact commands that trained them
├── requirements.txt
├── kernels/            the Kaggle scripts as they ran
├── src/                library: data loading, analytic GI disk model, U-Net, training, evaluation
├── experiments/        train_unified.py (training and evaluation), alma_unified.py (real cubes),
│                       pv_curve.py, dataset generators, figure and table builders
├── tests/              unit and end-to-end tests (plain scripts)
└── results/
    ├── README.md       index of the results
    ├── 01-final-test/ ... 10-training-runs/   one folder per result, each with a README
    ├── final/          ANALYSIS.md, RESULTS.md, RECIPES.md, detection_limit.md, figs/
    └── RUNS.md         every run of the final phase in launch order
```

## Setup and tests

```bash
cd "DENOISING_DIFFUSION/Final Submission"
pip install -r requirements.txt
PYTHONPATH=. python3 tests/test_infer.py                 # tests are plain scripts, one per module
PYTHONPATH=. python3 tests/test_alma_unified_smoke.py    # the real command line on a synthetic cube, CPU, about 30 s
python3 tests/test_md_links.py                           # relative links in the documents resolve
```

numpy 2 or newer is required: on numpy 1.x the four tests that compute moment maps fail inside `bettermoments`. `tests/test_unet.py` trains a small network and takes minutes on CPU.

## Using the denoiser on a cube

The input is a CLEANed ALMA line cube in FITS, any channel width or pixel size. It needs PyTorch, in practice a GPU, and the four checkpoints ([`MODELS.md`](MODELS.md)).

```bash
python experiments/alma_unified.py --cube MyDisk_13CO.fits --out out/mydisk \
    --ckpt unified_r4n_k15_long.pth,unified_r4n_k15_long_s11.pth,unified_r4n_k15_long_s23.pth,unified_r8_m1_r6.pth \
    --crop 512 --target-beam-px 10 \
    --incl 35 --dist 150      # optional: enables the rotation-curve and PV checks
```

[`USAGE.md`](USAGE.md) covers the rest: which options to use for coarse channels, noisy field edges, continuum and uncertainty maps; the exact commands and geometry for the 20 published disks; every output file and check in the report; how to test recovery in a cube's own noise by injecting a disk with a known answer; and what not to do with the output.

## Data and models

The training data (synthetic line-emission cubes and self-gravitating simulations) is not public and is not in this repository. The analytic GI disks are generated by `experiments/make_analytic_gi_cubes.py`. exoALMA and DSHARP observations are public and were used for evaluation only. The trained weights are described in [`MODELS.md`](MODELS.md) and will be released when permitted; [`MODELS.md`](MODELS.md) gives the commands to retrain them.

## Acknowledgements

Thanks to the mentors for their guidance throughout the project. Data used for evaluation: the Young et al. (2026) GI simulations, Speedie et al. (2024), the exoALMA large program (Teague et al. 2025) and DSHARP (Andrews et al. 2018). Analytic GI model: Longarini et al. (2021).
