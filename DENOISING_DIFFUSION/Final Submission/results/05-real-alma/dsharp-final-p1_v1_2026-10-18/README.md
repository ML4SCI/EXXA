# dsharp-final-p1_v1_2026-10-18

Kaggle kernel `alma-dsharp-final-p1` v1, pushed about 10:25 IST 2026-10-18, done by 13:21. Recipe B, DSHARP 12CO,
`--spec-interp 0.1`, `--target-beam-px 10 --max-scaled-px 1024`, continuum auto. AS 209 complete (S5 at SNR 6 / 8 / 12 / 24); HD 163296 FAILED at
read (`KeyError: 'BMAJ'`: a CASA per-channel beam cube, `CASAMBM = T`), fixed (`beam_of`) and rerun as `alma-dsharp-final-hd163296`.
Denoised / moment FITS stay in the kernel Output. Table (`experiments/alma_table.py`):

| disc | blind SNR | beam px (zoom) | flux | noise red. | S2 | S6 rc_r | PV5 r | PV5 mag den / raw (m/s) | PA sky fit / lit (diff mod 180) | M* | S5 GI rc_r | cont. sub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AS_209_interp01 | 12 | 9.5 (1.05) | 1.030 | 11x | 1e-04 | 0.85 | 0.63 | 30 / 45 | 84.9 / 85.8 (0.9) | 1.24 | 0.26 / 0.55 / 0.86 / 0.95 | False |
