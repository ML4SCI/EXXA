# final-boot-mwc758 v1, 2026-10-23

Kaggle `final-boot-mwc758` v1 (GPU, about 2 h 35 min).  Recipe B with the
exact `alma-exoalma-final-p0` setup (512 px, `--target-beam-px 10`, continuum auto) plus `--boot 4` (`alma_unified.bootstrap_m1`, first run).
The main pass reproduces the Oct 18 run exactly (noise reduction 195.74, M1 shift 72.6 m/s). `alma_boot/MWC_758_13CO/report.json` ("U") has the numbers. The M1 uncertainty map `*_m1_boot_std.fits` (m/s, NaN outside the S1 mask) stays in the kernel Output; it is not in this repository.

