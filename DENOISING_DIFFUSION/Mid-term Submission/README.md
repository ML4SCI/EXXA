# EXXA: denoising protoplanetary disk observations

This is a machine-learning pipeline for cleaning up synthetic ALMA observations of protoplanetary
disks. It is judged on the moment maps astronomers actually use, and not on pixel error alone.

It was written by Krishan Yadav for Google Summer of Code 2026 with ML4Sci (EXXA), mentored by
Jason Terry and Gaurav S. The midterm write-up is on Medium:
[Denoising Astronomical Observations of Protoplanetary Disks](https://medium.com/@kryshan753/denoising-astronomical-observations-of-protoplanetary-disks-600403e036c8).
It covers the background, the classical baselines, the architecture comparison and the U-Net
results.

This folder is the midterm submission. It holds the five notebooks that produced the midterm
numbers, with their outputs left in, plus the library code and the results behind them.

## The main result

A U-Net denoiser (the reference checkpoint, called V12) improves all three moment maps on every
one of the five held-out cubes, and no cube gets worse.

| | PSNR | SSIM | MSE | M0 (intensity) | M1 (velocity) | M2 (dispersion) |
|---|---|---|---|---|---|---|
| **V12** | 32.95 dB | 0.9857 | 0.000681 | **+69.8% ± 15.2%** | **+17.5% ± 7.8%** | **+20.1% ± 14.3%** |

A moment-map improvement is `100 × (1 − |denoised − clean| / |dirty − clean|)`. It is the mean
absolute difference over finite pixels, averaged across the five holdout cubes, with the
standard deviation next to it.

A later search over three seeds, using D4 augmentation, reached 39.30 dB. That result, the
classical baselines, the architecture comparison and the conditional DDPM are all in
`notebooks/`.

## Why this problem is awkward

**The noise is structured.** A "dirty" interferometric image carries correlated artifacts from
the telescope's point-spread function and sidelobes, which come from sparse *uv* coverage. It is
not independent per-pixel noise, which is what most of the denoising literature assumes.

**The data is 3D.** A line-emission cube is `(201, 600, 600)`: 201 velocity channels. What
astronomers use is the set of moment maps computed over the whole cube, not any one channel.

**Pixel metrics can mislead.** More than once here, PSNR improved while the moment maps got
worse. Beam conditioning did this, and so did the conditional DDPM, which lands within about
1 dB of the U-Net on PSNR while its M0 error is two orders of magnitude larger. That is why the
evaluation looks at moment maps first.

**Normalisation has to be reversible at inference.** Normalising the clean target by its own
statistics, which you do not have on real data, once gave an M0 of -6402% while the per-channel
SSIM still looked healthy.

## What is where

```
Mid-term Submission/
├── README.md
├── MODELS.md               which checkpoint came from which run, and where to find it
├── requirements.txt
│
├── notebooks/              the five notebooks as they ran, with outputs
│   ├── 05-unet-line-emission.ipynb         Kaggle v21
│   ├── 06-ddpm-line-emission.ipynb         Kaggle v13
│   ├── 07-classical-baselines.ipynb        v2
│   ├── 08-seeds-and-augmentation.ipynb     v2
│   └── 09-architecture-comparison.ipynb    v7
│
├── src/
│   ├── data/               loading and splitting
│   │   ├── cube_split.py           cube-level split grouped by RunID, so there is no leakage
│   │   ├── channel_sampler.py      Gaussian sampling of velocity channels, centred on channel 100
│   │   ├── fits_cube_dataset.py    the line-emission Dataset, plus continuum and beam helpers
│   │   ├── stacked_pair.py         adapter that stacks (dirty, clean) into one tensor for the DDPM
│   │   ├── patches.py              patch training and tiled inference
│   │   └── dataset.py, fits_loader.py, preprocessing.py, augmentation.py
│   ├── models/
│   │   ├── unet.py                 the U-Net, with optional beam conditioning
│   │   ├── diffusion_unet.py       the conditional DDPM backbone
│   │   ├── noise_scheduler.py      linear and cosine beta schedules
│   │   └── autoencoder.py, vae.py  baselines from the continuum phase
│   ├── training/
│   │   ├── sweep.py                early-stopping trainer and the random sweep
│   │   ├── architectures.py        one interface over the architectures so they can be compared
│   │   ├── diffusion.py            DDPM training, EMA and DDIM sampling
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── moment_maps.py          M0, M1 and M2 through bettermoments
│   │   ├── artifacts.py            overshoot, floor leak and invented-structure checks
│   │   ├── classical.py            Gaussian, median and Wiener baselines on line emission
│   │   ├── postprocess.py          spectral smoothing and ensembling
│   │   ├── sweep_analysis.py       correlation analysis of the sweep results
│   │   └── collect_outputs.py, recover_version.py
│   ├── utils/losses.py             HybridLoss (alpha*MSE + beta*(1-SSIM)) and the VAE loss
│   └── baselines.py                classical filters
│
├── tests/                  run with pytest, or run each file on its own
└── results/
    └── <notebook>/<run>/   the figures and a short README for each run
```

`results/` holds the run behind each of the five notebooks.

## Setup

```bash
git clone https://github.com/ML4SCI/EXXA.git
cd "EXXA/DENOISING_DIFFUSION/Mid-term Submission"
pip install -r requirements.txt
```

Besides the usual scientific stack you need `astropy` to read FITS files, `bettermoments` for the
moment maps, and `pytorch-msssim` for the SSIM term in the loss.

### Tests

```bash
python -m pytest tests/ -q
python tests/test_artifacts.py     # each test file also runs on its own
```

The tests use small synthetic arrays, so nothing has to be downloaded first.

### Reproducing a run

The line-emission FITS cubes (14 cubes across 11 RunIDs, about 7.6 GB) and the trained
checkpoints are not in git. [MODELS.md](MODELS.md) says what each checkpoint is and where to get
it. Each notebook clones the repository in its first cell, so a run can start from a blank Kaggle
kernel. Set the `BRANCH` variable in that cell to the branch you want it to clone.

1. Create a new Kaggle notebook with a GPU and internet turned on.
2. Use Add Input to attach the line-emission Dataset. The bootstrap cell finds it under
   `/kaggle/input/`.
3. Paste in a notebook from `notebooks/` and run it from top to bottom.

Everything here was computed on Kaggle's dual Tesla T4 with `DataParallel`.

## Working rules

Each of these is here because breaking it once cost us a result.

- Split at the cube level. Hold out whole cubes grouped by RunID, because the radiative-transfer
  variants of one simulation are near-duplicates and leak if they end up on both sides of a split.
- Holdout cubes are for inference only. Training and validation never see them.
- Report moment-map results as an average over all five holdout cubes, always with a spread.
- Normalise using only statistics that exist at inference time.
- Every number should trace back to the run that produced it. See the README inside each run
  folder.
- Use `seed=42` throughout, and load `dirty` arrays as `float32`.

## How the project got here

In short:

| Phase | What happened |
|---|---|
| Weeks 2 to 3 | Continuum baselines: classical filters, an autoencoder, a VAE and a patch U-Net (best SSIM 0.76) |
| Week 4 | The patch DDPM did worse than classical filtering, which led the mentors to move us to full-image line emission |
| Week 5 | First line-emission U-Net. We found and fixed the normalisation bug (M0 of -6402%, then recovered) |
| Weeks 5 to 6 | Continuum subtraction, suggested by the mentors, took M0 from -1672% to +84.9% |
| Weeks 6 to 7 | The variance between V7 and V9 led to the five-cube evaluation protocol, and V12 became the reference |
| Weeks 7 to 8 | Beam conditioning, a 12-run sweep that reached 37.11 dB (later traced to seed luck), and a DDPM retune |
| Weeks 8 to 9 | Seed-validated bands, classical baselines on line emission, the architecture comparison, a rebuilt and diagnosed DDPM, and the midterm write-up |
