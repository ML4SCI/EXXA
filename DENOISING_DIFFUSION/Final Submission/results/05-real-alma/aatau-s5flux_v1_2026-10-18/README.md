# aatau-s5flux_v1_2026-10-18

Kaggle kernel `alma-aatau-s5flux` v1, 2026-10-18. Recipe B, exoALMA 13CO fiducial, 512 px field + `--target-beam-px 10`, continuum auto, PA from
exoALMA I Table 2. AA Tau with the truth-based S5 flux test at SNR 6 / 8 / 12 / 24. Denoised / moment FITS stay in the kernel Output.  Table (`experiments/alma_table.py`):

| disc | blind SNR | beam px (zoom) | flux | noise red. | S2 | S6 rc_r | PV5 r | PV5 mag den / raw (m/s) | PA sky fit / lit (diff mod 180) | M* | S5 GI rc_r | cont. sub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AA_Tau_13CO | 8 | 6.0 (1.67) | 1.090 | 82x | 1e-07 | 0.80 | 0.38 | 26 / 27 | 93.2 / 272.7 (0.5) | 0.74 | 0.16 / 0.30 / 0.72 / 0.87 | False |
