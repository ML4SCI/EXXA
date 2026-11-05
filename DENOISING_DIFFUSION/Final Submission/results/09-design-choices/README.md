# 09. How the inference settings were chosen

Each part of the final inference recipe was tested on its own before being kept. The chosen settings are in `../final/RECIPES.md`.

## Runs in this folder

- [`eval-ensemble_v1_2026-10-12/`](eval-ensemble_v1_2026-10-12/): averaging several networks against single networks
- [`eval-gate_v1_2026-10-13/`](eval-gate_v1_2026-10-13/): the do-no-harm gate (blend toward Wiener at high SNR)
- [`eval-ladder_v1_2026-10-11/`](eval-ladder_v1_2026-10-11/): the SNR ladder on the selection disk for early models
- [`eval-final_v1_2026-10-14/`](eval-final_v1_2026-10-14/): evaluation of three averaged networks before the last two inference changes
- [`size-test_v1_2026-10-15/`](size-test_v1_2026-10-15/): the size normalisation (beam resampled to about 10 px)
- [`wiener2-test_v1_2026-10-15/`](wiener2-test_v1_2026-10-15/): the second Wiener pass
- [`combo-a_v1_2026-10-15/`](combo-a_v1_2026-10-15/): three networks with all inference settings (stored alternative)
- [`combo-b_v1_2026-10-15/`](combo-b_v1_2026-10-15/): four networks with all inference settings (the final model)
- [`eval-young-test2_v1_2026-10-18/`](eval-young-test2_v1_2026-10-18/): extra Young test simulations

Where a run folder has its own README, it states the Kaggle run, its settings and its outcome.
