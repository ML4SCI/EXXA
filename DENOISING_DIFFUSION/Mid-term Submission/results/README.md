# Results

There is one folder per notebook here, holding the run behind the matching file in
[`../notebooks/`](../notebooks/).

| folder | run | notebook |
|---|---|---|
| `05-unet-line-emission/v21_2026-08-17_6f5c798/` | Kaggle Version 21 | `05-unet-line-emission.ipynb` |
| `06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/` | Kaggle Version 13 | `06-ddpm-line-emission.ipynb` |
| `07-classical-baselines/v2_2026-07-31T1904_eb03589/` | v2 | `07-classical-baselines.ipynb` |
| `08-seeds-and-augmentation/v2_2026-07-31_eb03589/` | v2 | `08-seeds-and-augmentation.ipynb` |
| `09-architecture-comparison/v7_2026-08-02T1721_ee491fc/` | v7 | `09-architecture-comparison.ipynb` |

Each folder has that run's figures and a short README about what the run showed and, where it
matters, what changed since the run before. The notebook itself, with its full output, is in
`../notebooks/`, so it is not copied here a second time.

Only these five runs keep their figures here.

## Reading the numbers

Every moment-map number comes from one of three metrics: raw, 3σ-clipped, or clipped plus
signal-masked. Numbers from different metrics cannot be compared with each other; each run's
README says which metric that run used.

Some notebooks write checkpoints to `results/checkpoints/`. That is a local scratch path and is
gitignored. [`../MODELS.md`](../MODELS.md) says where the checkpoints actually are.
