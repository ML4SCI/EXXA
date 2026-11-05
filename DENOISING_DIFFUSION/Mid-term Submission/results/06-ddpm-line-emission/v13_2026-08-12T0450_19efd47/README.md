# Notebook 06, Kaggle Version 13

Code `19efd47`, run on 2026-08-12 at 04:50 UTC. It finished with no errors, and it is the first
DDPM run that produced moment maps at all. It used about 9 hours of GPU time: 2.6 hours for the
sweep, 4.5 for training and 1.8 for the holdout.

## The result

PSNR is **38.180** at SSIM **0.9933**, within about 1 dB of the best U-Net. The moment maps, on
the signal-masked metric, are:

```
M0  -56.1% ± 152.2
M1  +15.0% ±  87.4
M2  +10.9% ±  81.7
```

For comparison, V12 scores +69.8 / +17.5 / +20.1. The gap between a competitive PSNR and a bad M0
is the main finding of this run.

## Files

The executed notebook, with its full output, is
[`../../../notebooks/06-ddpm-line-emission.ipynb`](../../../notebooks/06-ddpm-line-emission.ipynb).
It is not duplicated here.

| file | what it is |
|---|---|
| `ddpm_objective_sweep.csv` | the four objective arms |
| `ddpm_seed_repeats.csv` | the 60-epoch run |
| `moment_map_holdout_summary_ddpm.csv` | per-cube moments, masked and unmasked |
| `ddpm_moment_maps.png`, `moment_map_holdout_summary_ddpm.png`, `ddpm_line_emission_loss.png` | figures saved with `savefig` |
| `line_emission_ddpm_comparison.png` | five validation channels: dirty, DDPM output and clean truth |
| `figure_cell*.png` | copies of the same figures as they appeared inline in the notebook |
| `manifest.json` | the `collect_outputs` record: commit `19efd47`, 2026-08-12T04:50:41Z, a batch run, and a sha256 for each file |

Every file listed in the manifest is here and its hash matches, apart from the one below.

## Not in this folder

`ddpm_moment_maps.npz` is missing. It is 14 MB of M0, M1 and M2 arrays for one cube, and it would
let you redraw the moment figure without re-running anything. `*.npz` files are gitignored across
the repository. The file is in Kaggle Version 13's Output.

The five 332 MB checkpoints are gitignored too: `ddpm_seed42.pth` (epoch 58, the trained model)
and the four sweep arms. They are also in Kaggle Version 13's Output. [`../../../MODELS.md`](../../../MODELS.md)
lists every checkpoint and says how to get them.
