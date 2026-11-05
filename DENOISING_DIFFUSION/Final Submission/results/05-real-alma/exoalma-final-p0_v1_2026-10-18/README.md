# exoalma-final-p0_v1_2026-10-18

Kaggle kernel `alma-exoalma-final-p0` v1, 2026-10-18. Recipe B, exoALMA 13CO fiducial, 512 px field + `--target-beam-px 10`, continuum auto, PA from
exoALMA I Table 2. MWC 758 (S5), HD 34282, HD 135344B (S5), V4046 Sgr. Denoised / moment FITS stay in the kernel Output.  Table (`experiments/alma_table.py`):

| disc | blind SNR | beam px (zoom) | flux | noise red. | S2 | S6 rc_r | PV5 r | PV5 mag den / raw (m/s) | PA sky fit / lit (diff mod 180) | M* | S5 GI rc_r | cont. sub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HD_135344B_13CO | 15 | 6.0 (1.67) | 1.045 | 183x | 0e+00 | 0.77 | 0.87 | 25 / 53 | 61.7 / 242.9 (1.2) | 1.60 | 0.86 / 0.93 / 0.95 | False |
| HD_34282_13CO | 15 | 6.0 (1.67) | 1.060 | 88x | 1e-07 | 0.97 | 0.54 | 65 / 112 | 167.4 / 117.4 (50.0) | 7.16 | - | False |
| MWC_758_13CO | 11 | 6.0 (1.67) | 1.037 | 196x | 0e+00 | 0.76 | 0.79 | 36 / 54 | 62.0 / 240.3 (1.7) | 1.34 | 0.85 / 0.93 / 0.95 | False |
| V4046_Sgr_13CO | 18 | 6.0 (1.67) | 1.075 | 62x | 1e-06 | 0.77 | 0.68 | 20 / 33 | 75.7 / 255.7 (0.0) | 1.67 | - | True |
