# eval-young-test2_v1_2026-10-18

Recipe B on the 8 more held-out Young et al. simulations (TEST2), 300 and 600 px, PV included. 16 cases, none failed, 52.8 min.
**It ran under the wrong kernel name:** `alma-mwc758-zoom512` (pushed 03:58 IST by the watcher script), because of the bash bug in
(macOS bash 3.2 has no associative arrays, so every queued job got the last folder, the Young test2 folder). The code it ran is
the kernel's source (checked by pulling it from Kaggle), so the results are valid Young-test2 results. A second, intended run of the same
code (`eval-young-test2`, pushed 04:18) is a duplicate and serves as a reproducibility check.
