# Notebooks

These are the executed copies of the five notebooks, with their real cell outputs still in
place. They are the same code that produced the midterm numbers, kept as evidence of what ran
and what came out.

| notebook | run | what it shows |
|---|---|---|
| `05-unet-line-emission.ipynb` | Kaggle Version 21 | U-Net moment scores, seed bands and the artifact diagnostics |
| `06-ddpm-line-emission.ipynb` | Kaggle Version 13 | the conditional DDPM's PSNR and moment scores, and how its M0 fails |
| `07-classical-baselines.ipynb` | v2, commit `eb03589` | classical filters against the U-Net, scored on moment maps |
| `08-seeds-and-augmentation.ipynb` | v2, commit `eb03589` | the twelve checkpoints and seed bands that the other notebooks reuse |
| `09-architecture-comparison.ipynb` | v7, commit `ee491fc` | the U-Net against the autoencoder and the VAE, under one protocol |

Each notebook's first cell clones the repository from GitHub. The branch it clones is set by a
`BRANCH` variable near the top of that cell.

One known problem in 05 v21: its `winner_beam` arm was scored without feeding the trained beam
vector back in, and `UNet.forward` silently ignores a missing one. The moment scores that the
notebook prints for that arm are therefore wrong (-95.7 / +14.1 / -27.3, where the real values
are +9.6 / +63.2 / +20.9). The notebook keeps the old numbers because it is archived as it ran.

The figures from each run, and a README explaining the run, are under
[`../results/`](../results/).
