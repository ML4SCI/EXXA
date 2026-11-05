# Kaggle kernels as they ran

The scripts that produced the four final networks and the main evaluations, as they ran on Kaggle. Each clones the project repository, attaches the data as Kaggle inputs, and calls `experiments/train_unified.py` or `experiments/alma_unified.py`. The settings that matter are the variables at the top of each script (`K`, `EXTRA`, `WARM`) and the argument lists passed to those two programs. [`../MODELS.md`](../MODELS.md) gives the same calls without the Kaggle setup.

| script | produced |
|---|---|
| `train_seed3.py`, `train_seed11.py`, `train_seed23.py` | networks 1 to 3 |
| `train_crop256_seed31.py` | stage 1 of network 4 |
| `finetune_m1_from_crop256.py` | network 4 |
| `eval_five_sizes.py` | the evaluation at five image sizes (`results/07-image-sizes/`) |
| `eval_final_test.py` | the one-time final test (`results/01-final-test/`) |
| `eval_fresh_set.py` | the fresh test set (`results/03-fresh-set/`) |
| `alma_exoalma.py`, `alma_dsharp.py` | the real-data runs (`results/05-real-alma/`) |

The scripts read the repository from the author's working branch, under its own folder layout, and code added after that branch's last push from a private Kaggle dataset; they are kept as the record of what ran, not as a turnkey recipe. The branch they clone (`midterm-prep`) and the private Kaggle dataset `exxa-wiggle-code` are not part of this repository: the library they ran is `src/` and `experiments/` here, and the training data is described in the top-level README.
