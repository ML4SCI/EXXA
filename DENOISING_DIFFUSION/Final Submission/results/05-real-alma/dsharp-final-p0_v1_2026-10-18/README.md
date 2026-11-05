# dsharp-final-p0_v1_2026-10-18

Kaggle kernel `alma-dsharp-final-p0` v1, pushed about 10:25 IST 2026-10-18, done by 13:45. Recipe B, DSHARP 12CO, `--spec-interp 0.1`,
`--target-beam-px 10 --max-scaled-px 1024`, continuum auto (none found on any). HD 143006 (S5), HD 142666, IM Lup. Denoised / moment FITS stay in the kernel
Output (needed for the local geometry refit). Table (`experiments/alma_table.py`):

| disc | blind SNR | beam px (zoom) | flux | noise red. | S2 | S6 rc_r | PV5 r | PV5 mag den / raw (m/s) | PA sky fit / lit (diff mod 180) | M* | S5 GI rc_r | cont. sub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HD_142666_interp01 | 7 | 7.7 (1.29) | 1.030 | 10x | 1e-04 | 0.94 | 0.62 | 15 / 27 | 47.6 / 162.1 (65.5) | 10.28 | - | False |
| HD_143006_interp01 | 8 | 16.5 (1.00) | 1.006 | 2x | 4e-03 | 0.84 | 0.88 | 12 / 44 | 165.1 / 169.0 (3.9) | 1.56 | -0.20 / -0.21 / -0.19 / -0.15 | False |
| IM_Lup_interp01 | 13 | 12.2 (1.00) | 0.977 | 45x | 0e+00 | 0.93 | 0.94 | 49 / 87 | 144.4 / 144.5 (0.1) | 1.02 | - | False |

HD 142666's geometry fit failed (M* 10.3, PA 66 deg off); HD 143006's S5 rc_r is negative for every method including dirty.
