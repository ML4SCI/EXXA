# Model checkpoints

The checkpoints themselves are not in this repository. Every one used for the midterm is over
100 MB (the DDPM's is 332 MB), and `.pth`, `.ckpt` and `.pth.tar` files are gitignored. They
live in the Kaggle Output (or a Kaggle Dataset) of the notebook run that produced them.

This file says which checkpoint came from which run, with the numbers that identify it, so you
can pick the right one on Kaggle without guessing.

One thing worth knowing if you ever move a checkpoint between Kaggle accounts: a `.pth` file is
really a zip archive, and Kaggle unpacks it when you upload it as a Dataset, after which
`torch.load` refuses it. Upload it under another extension such as `.ckpt` instead, or attach
the Output of the notebook that made it.

## U-Net (notebooks 05 and 08)

Notebook 08 trained twelve core checkpoints, and every later notebook reuses them (05 and the
moment-map tables included). The numbers below come from that run's `seed_repeats.csv`. They
were re-scored in a later version of notebook 08 (commit `1ca611f`); the version archived here
is 08 v2, which uses the same twelve checkpoints and reports the same per-run PSNRs.

| config | seed | best epoch | val loss | PSNR | SSIM |
|---|---|---|---|---|---|
| v12 | 42 | 59 | 0.001492 | 38.659 | 0.99343 |
| v12 | 43 | 17 | 0.002018 | 36.671 | 0.99121 |
| v12 | 44 | 21 | 0.001848 | 37.466 | 0.99196 |
| winner | 42 | 21 | 0.001125 | 37.230 | 0.99228 |
| winner | 43 | 23 | 0.001060 | 37.724 | 0.99264 |
| winner | 44 | 23 | 0.001023 | 38.969 | 0.99292 |
| winner_aug | 42 | 26 | 0.001077 | 38.939 | 0.99240 |
| winner_aug | 43 | 46 | 0.000914 | 39.808 | 0.99346 |
| winner_aug | 44 | 29 | 0.001020 | 39.141 | 0.99277 |
| winner_p10 | 42 | 30 | 0.000920 | 38.954 | 0.99352 |
| winner_p10 | 43 | 35 | 0.000927 | 39.034 | 0.99355 |
| winner_p10 | 44 | 54 | 0.000809 | 39.830 | 0.99427 |

Three more were trained inside notebook 05 itself: `sweep_winner` seed 49, `winner_beam` and
`winner_patch` (the last two at seed 42). Source:
`results/05-unet-line-emission/v21_2026-08-17_6f5c798/`.

| config | seed | best epoch | PSNR |
|---|---|---|---|
| sweep_winner | 49 | 11 | 36.162 |
| winner_beam | 42 | 24 | 38.710 |
| winner_patch | 42 | 21 | 33.960 |

**About `winner_beam`.** The checkpoint is fine, but the moment score printed in the v21
notebook is wrong. That run scored the beam-conditioned model without passing its beam vector
at inference, and `UNet.forward` silently ignores a missing `beam`, so the conditioning branch
never ran. Scored properly, M0 goes from -95.7% to +9.6%. The notebook keeps the old numbers
on purpose, because it is archived exactly as it ran.

**Architecture.** `base_channels=48, channel_multipliers=(1,2,4,8)` for `winner`, `winner_aug`,
`winner_p10`, `winner_beam` and `winner_patch`. The v12 models use `base_channels=32` and
`(1,2,4)`. `beam_dim=4` for `winner_beam` only, and 0 for the rest. The model is defined in
`src/models/unet.py`.

## Conditional DDPM (notebook 06)

There is one production checkpoint, `ddpm_seed42.pth` (332 MB), from Kaggle Version 13 of
notebook 06 (`results/06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/`). It was trained for 60
epochs with v-prediction, a cosine schedule and min-SNR weighting, and reaches PSNR 38.180 and
SSIM 0.9933. The code is in `src/training/diffusion.py`.

## Getting a checkpoint

Look in the Kaggle Output of the notebook that trained it, or in one of the Datasets built to
move checkpoints between accounts (`exxa-nb08-checkpoints-v4`, `exxa-nb05-checkpoints-v19`).
The restore code is in `notebooks/05-unet-line-emission.ipynb` (`_import_nb08` and
`_import_prior_nb05`).
