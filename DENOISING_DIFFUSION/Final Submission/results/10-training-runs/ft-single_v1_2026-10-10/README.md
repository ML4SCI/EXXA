# unified-ft-single v1 (Kaggle), COMPLETE

Fine-tune of three single-output nb16 checkpoints on the mixed data (line emission + hydro sg_synth + analytic GI disks,
sizes 256/300/320/480/600, deconvolution and noise-only tasks, p_ai 0.5), each in its own I/O format ("legacy" arm of
`experiments/train_unified.py`): lr 1e-4, 20 rounds x 300 steps = 6000 steps, ~81-85 min per model. Same recipe as `unified-ft-kin`.

| checkpoint | start | final round L1 / d1 / d2 |
|---|---|---|
| `winner_aug_res480_seed43` (nb05) | L1 0.0493 | 0.0254 / 0.0226 / 0.0341 |
| `sweep_winner_p10_600` (nb14) | L1 0.0512 | 0.0260 / 0.0231 / 0.0348 |
| `sg_k3_fresh` | L1 0.0366 | 0.0115 / 0.0087 / 0.0051 |

The d2 (second velocity difference) loss of the two single-channel models stays ~0.034 for the whole run (sg_k3_fresh
reaches 0.005): one output channel from a narrow input cannot fit the velocity structure the mixed data asks for.

Result (comparison tables of the run; line = 5 line-emission holdouts, mean over cubes, not seeds):
- res480 and p10 fine-tunes are WORSE on line emission at 600 px (PSNR 32.6 / 32.7, M0 -99 / -140 %, M2 -171 / -158 %).
  Refuted. Only SG v2 improves slightly (PSNR 27.5 / 27.4, rc_r 0.83).
- sg_k3_fresh fine-tune: line 600 px PSNR 40.9, M0 -16 %, M1 67 %, M2 -40 %, rc_r 0.55; analytic rc_r 0.79; SG v2 rc_r 0.87.
  Better than its own start, still below `unified-wiener` (line PSNR 46.3, M1 85, M2 83, rc_r 0.70).
- No model beats dirty on SG v2 rc_r by more than noise (dirty 0.85 at 600 px).

Caveats: the "before" JSONs cover 2 line cubes and 2 analytic cases, the "after" JSONs 5 and 4, so before/after
rows are not the same case set. `wiener` hydro is 9074 only (n=1), the fine-tunes add 9074_ai (n=2).

Scoring failures: the same 6 cases for every model (run_0002_00560_rt_04 at 256/300, an08 and an08_ai at 300/320),
`TypeError: expected non-empty vector for x` in `wiggle_fourier`'s polyfit. This kernel ran a code-dataset version
from before the `len(y) < 8` guard; the current code has it. Re-score those cases if they are needed.

Checkpoints: `unified_ft_nb05_winner_aug_res480_seed43.pth`, `unified_ft_nb14_sweep_winner_p10_600.pth`,
`unified_ft_sg_k3_fresh.pth` in the kernel Output, `ft_<name>/`.

Files: the six eval JSONs.
