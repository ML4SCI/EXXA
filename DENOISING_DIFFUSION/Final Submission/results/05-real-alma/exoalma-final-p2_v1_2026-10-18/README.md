# exoalma-final-p2_v1_2026-10-18

Kaggle kernel `alma-exoalma-final-p2` v1, pushed about 10:25 IST 2026-10-18, done by 14:15. Recipe B, exoALMA 13CO fiducial,
512 px field + `--target-beam-px 10` (beam 6 to 10 px, scale 1.667), continuum auto (none found), PA from exoALMA I Table 2. AA Tau, SY Cha, J1604, J1852
(no S5 in this part). Denoised / moment FITS stay in the kernel Output. Table (`experiments/alma_table.py`):

| disc | blind SNR | beam px (zoom) | flux | noise red. | S2 | S6 rc_r | PV5 r | PV5 mag den / raw (m/s) | PA sky fit / lit (diff mod 180) | M* | S5 GI rc_r | cont. sub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AA_Tau_13CO | 8 | 6.0 (1.67) | 1.090 | 82x | 1e-07 | 0.80 | 0.38 | 26 / 27 | 93.2 / 272.7 (0.5) | 0.74 | - | False |
| J1604_13CO | 18 | 6.0 (1.67) | 0.992 | 224x | 0e+00 | 0.08 | 0.64 | 9 / 28 | 89.8 / 258.1 (11.7) | 0.43 | - | False |
| J1852_13CO | 13 | 6.0 (1.67) | 1.065 | 143x | 3e-08 | 0.81 | -0.32 | 9 / 17 | 118.1 / 117.1 (1.0) | 0.96 | - | False |
| SY_Cha_13CO | 6 | 6.0 (1.67) | 1.073 | 166x | 0e+00 | 0.73 | 0.24 | 18 / 50 | 163.6 / 346.2 (2.6) | 0.66 | - | False |
