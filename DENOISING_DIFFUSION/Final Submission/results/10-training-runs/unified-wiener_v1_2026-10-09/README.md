# unified-wiener, Kaggle Version 1 (2026-10-09 to 10): one Wiener + correction model, all data kinds, all five sizes

Private GPU script kernel `unified-wiener`; code `experiments/train_unified.py` (arm wiener, window +-3 channels, deconvolution-task data only, built BEFORE the noise-only variants),
shipped in `exxa-wiggle-code` (not committed at launch). 12,000 steps in 163 min, scored 197 min in total. Trained on 9 line-emission cubes, hydro 9015/9019/9025/9032, analytic an01-an06;
held out: 5 line cubes, 9074, SG v2, an07, an08. A first smoke version (v1 of `unified-smoke`) produced NaN losses because hydro 9019 is in different units (peak 1.4e10) and overflowed float16; fixed before this run.

The evaluation JSON stays in the kernel Output (every case at every size; 4 cases FAILED to score: run_0002_00560_rt_04 at 256 and 300, an08 at 300 and 320, an empty spine track, fixed afterwards).
Checkpoint: `unified_wiener/unified_wiener.pth` in the kernel Output.
