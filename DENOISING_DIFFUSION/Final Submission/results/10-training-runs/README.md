# 10. Training runs

The four networks of the final model come from the runs marked **final** below; the exact commands are in [`MODELS.md`](../../MODELS.md) and the kernels as they ran in `../../kernels/`. Each folder holds the run's evaluation file. The other runs are the alternatives tried and not adopted.

## Runs in this folder

- [`r4n-k15-long_v1_2026-10-12/`](r4n-k15-long_v1_2026-10-12/): **final**, network 1: seed 3, 80 rounds (24,000 steps)
- [`r4n-k15-long-s11_v1_2026-10-13/`](r4n-k15-long-s11_v1_2026-10-13/): **final**, network 2: seed 11
- [`r4n-k15-long-s23_v1_2026-10-13/`](r4n-k15-long-s23_v1_2026-10-13/): **final**, network 3: seed 23 (one session limit, resumed)
- [`r6-k15-crop256_v2_2026-10-15/`](r6-k15-crop256_v2_2026-10-15/): **final**, stage 1 of network 4: 256 px training crops, batch 4, seed 31
- [`r8-m1-r6_v1_2026-10-15/`](r8-m1-r6_v1_2026-10-15/): **final**, network 4: fine-tune of the crop-256 run with the line-centre penalty, weight 0.01
- [`r6-k15-s32_v1-v2_2026-10-17/`](r6-k15-s32_v1-v2_2026-10-17/): crop-256 recipe, seed 32 (repeat; not adopted)
- [`r6-k15-s33_v1-v3_2026-10-17_stopped/`](r6-k15-s33_v1-v3_2026-10-17_stopped/): crop-256 recipe, seed 33 (stopped)
- [`r6-k15-s33b_v1_2026-10-18_round65/`](r6-k15-s33b_v1_2026-10-18_round65/): crop-256 recipe, seed 33, partial
- [`r6-k15-s33b_v2_2026-10-18/`](r6-k15-s33b_v2_2026-10-18/): crop-256 recipe, seed 33, complete (not adopted)
- [`r7-k15-snr40-s42_v1_2026-10-14/`](r7-k15-snr40-s42_v1_2026-10-14/): training SNR 3 to 40 only, seed 42 (worse)
- [`r7-k15-snr40-s43_v1_2026-10-14/`](r7-k15-snr40-s43_v1_2026-10-14/): training SNR 3 to 40 only, seed 43 (worse)
- [`r8-m1-r4n_v1_2026-10-15/`](r8-m1-r4n_v1_2026-10-15/): line-centre fine-tune from seed 3 (worse)
- [`r8-m1-r6-w003_v1_2026-10-17/`](r8-m1-r6-w003_v1_2026-10-17/): line-centre weight 0.003 (worse)
- [`r8-m1-r6-w030_v1_2026-10-17/`](r8-m1-r6-w030_v1_2026-10-17/): line-centre weight 0.03 (worse)
- [`r9-k15-s61_v1_2026-10-17/`](r9-k15-s61_v1_2026-10-17/): training at the beam scale, seed 61 (not adopted)
- [`r9-k15-s62_v2_2026-10-17/`](r9-k15-s62_v2_2026-10-17/): training at the beam scale, seed 62 (not adopted)
- [`real-k15_v1_2026-10-10/`](real-k15_v1_2026-10-10/): early realistic-noise run, K = 15
- [`real-k3-sgv2_v1_2026-10-10/`](real-k3-sgv2_v1_2026-10-10/): early realistic-noise run with the second self-gravitating cube
- [`unified-wiener_v1_2026-10-09/`](unified-wiener_v1_2026-10-09/): first Wiener-plus-correction model
- [`unified-wiener-k15_v1_2026-10-10_failed/`](unified-wiener-k15_v1_2026-10-10_failed/): failed run (memory)
- [`unified-wiener-k15_v3_2026-10-10_failed/`](unified-wiener-k15_v3_2026-10-10_failed/): failed run
- [`unified-wiener-real_v1_2026-10-10_failed/`](unified-wiener-real_v1_2026-10-10_failed/): failed run
- [`ft-kin_v1_2026-10-10/`](ft-kin_v1_2026-10-10/): fine-tune of an earlier kinematic model on mixed data
- [`ft-single_v1_2026-10-10/`](ft-single_v1_2026-10-10/): single-data fine-tune

Where a run folder has its own README, it states the Kaggle run, its settings and its outcome.
