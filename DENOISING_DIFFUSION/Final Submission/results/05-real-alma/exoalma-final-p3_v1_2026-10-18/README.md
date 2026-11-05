# exoalma-final-p3_v1_2026-10-18

Kaggle kernel `alma-exoalma-final-p3` v1, pushed about 10:25 IST 2026-10-18, done by 14:56. Recipe B, exoALMA 13CO fiducial, 512 px field +
`--target-beam-px 10` (scale 1.667), continuum auto (none found), PA from exoALMA I Table 2. LkCa 15 (S5), PDS 66, J1615. Denoised / moment FITS stay in the kernel Output.
Table (`experiments/alma_table.py`):

| disc | blind SNR | beam px (zoom) | flux | noise red. | S2 | S6 rc_r | PV5 r | PV5 mag den / raw (m/s) | PA sky fit / lit (diff mod 180) | M* | S5 GI rc_r | cont. sub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J1615_13CO | 12 | 6.0 (1.67) | 1.022 | 80x | 1e-07 | 0.79 | 0.54 | 10 / 26 | 145.8 / 325.4 (0.4) | 1.14 | - | False |
| LkCa_15_13CO | 11 | 6.0 (1.67) | 1.018 | 124x | 0e+00 | 0.77 | 0.65 | 12 / 21 | 61.2 / 62.1 (0.9) | 1.15 | 0.75 / 0.88 / 0.92 | False |
| PDS_66_13CO | 7 | 6.0 (1.67) | 1.291 | 35x | 1e-05 | 0.97 | nan | nan / nan | 77.7 / 189.0 (68.7) | 50.00 BOUND | - | False |
