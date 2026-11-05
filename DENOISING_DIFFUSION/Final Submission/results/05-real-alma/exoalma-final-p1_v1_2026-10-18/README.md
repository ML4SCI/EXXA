# exoalma-final-p1_v1_2026-10-18

Kaggle kernel `alma-exoalma-final-p1` v1, pushed about 10:25 IST 2026-10-18, done by 14:56. Recipe B, exoALMA 13CO fiducial, 512 px field +
`--target-beam-px 10` (scale 1.667), continuum auto (none found), PA from exoALMA I Table 2. DM Tau (S5), CQ Tau, HD 143006, J1842. Denoised / moment FITS stay in the kernel Output.
Table (`experiments/alma_table.py`):

| disc | blind SNR | beam px (zoom) | flux | noise red. | S2 | S6 rc_r | PV5 r | PV5 mag den / raw (m/s) | PA sky fit / lit (diff mod 180) | M* | S5 GI rc_r | cont. sub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CQ_Tau_13CO | 9 | 6.0 (1.67) | 1.141 | 100x | 0e+00 | 0.24 | 0.58 | 31 / 38 | 52.7 / 235.1 (2.4) | 1.30 | - | False |
| DM_Tau_13CO | 9 | 6.0 (1.67) | 1.059 | 51x | 2e-06 | 0.49 | -0.17 | 6 / 14 | 156.1 / 335.7 (0.4) | 0.48 | 0.68 / 0.84 / 0.90 | False |
| HD_143006_13CO | 7 | 6.0 (1.67) | 1.142 | 41x | 2e-06 | 0.98 | 0.98 | 133 / 136 | 6.4 / 168.9 (17.5) | 50.00 BOUND | - | False |
| J1842_13CO | 7 | 6.0 (1.67) | 1.147 | 76x | 7e-07 | 0.64 | 0.69 | 35 / 39 | 26.8 / 205.9 (0.9) | 1.04 | - | False |
