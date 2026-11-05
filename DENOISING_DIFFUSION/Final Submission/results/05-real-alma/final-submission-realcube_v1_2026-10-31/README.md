# final-submission-realcube_v1_2026-10-31

A Kaggle GPU kernel (script, about 41 min) that clones the branch `final-submission` at commit `06b549e` with no code overlay, attaches the four final checkpoints and runs the two commands printed in [`USAGE.md`](../../../USAGE.md) on real cubes, with no injection and no `--boot`:

- exoALMA MWC 758 13CO: `--crop 512 --target-beam-px 10 --incl 19.4 --dist 156 --pa 240.3` (26.3 min)
- DSHARP AS 209 12CO: `--crop 768 --target-beam-px 10 --max-scaled-px 1024 --spec-interp 0.1 --incl 34.97 --dist 121 --pa 85.76` (13.6 min)

Both exited 0 and reproduce the archived runs ([`exoalma-final-p0_v1_2026-10-18/`](../exoalma-final-p0_v1_2026-10-18/), [`dsharp-final-p1_v1_2026-10-18/`](../dsharp-final-p1_v1_2026-10-18/)) to the digit on every number compared:

| | blind SNR | noise reduction | M0 total ratio | M1 shift (m/s) | PA (sky, deg) | S6 rc_r | PV5 r | PV5 magnitude den / raw (m/s) | M* |
|---|---|---|---|---|---|---|---|---|---|
| MWC 758, this run | 10.7 | 195.7 | 1.037 | 72.6 | 62.0 | 0.76 | 0.79 | 36 / 54 | 1.34 |
| MWC 758, archived | 10.7 | 195.7 | 1.037 | 72.6 | 62.0 | 0.76 | 0.79 | 36 / 54 | 1.34 |
| AS 209, this run | 12.4 | 11.2 | 1.030 | 107.1 | 84.9 | 0.85 | 0.63 | 30 / 45 | 1.24 |
| AS 209, archived | 12.4 | 11.2 | 1.030 | 107.1 | 84.9 | 0.85 | 0.63 | 30 / 45 | 1.24 |

`realcube_result.json` has the environment (PyTorch 2.11 with CUDA) and timings; the two `report.json` files are the full reports. The denoised cubes and figures stay in the kernel Output.
