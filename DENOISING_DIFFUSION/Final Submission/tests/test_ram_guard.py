"""
The host-RAM watchdog in train_unet must stop an arm cleanly, keep the best-epoch weights, record
`ram_guard` in the checkpoint and the result, and do nothing when RAM is fine.

Free RAM is faked (the real leak only exists on Kaggle). Run: PYTHONPATH=. python3 tests/test_ram_guard.py
"""
import os, sys, tempfile
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.training.sweep as sw


class Toy(Dataset):
    def __init__(self, n=8, size=32):
        g = torch.Generator().manual_seed(0)
        self.c = torch.rand(n, 1, size, size, generator=g)
        self.d = (self.c + 0.2 * torch.randn(n, 1, size, size, generator=g)).clamp(0, 1)

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        return self.d[i], self.c[i]


def run(fake_free, min_free, ckpt):
    it = iter(fake_free)
    sw._free_ram_gb = lambda: next(it, fake_free[-1])
    return sw.train_unet(Toy(), Toy(), "cpu", base_channels=8, channel_multipliers=(1, 2),
                         min_epochs=1, max_epochs=10, patience=100, batch_size=4,
                         ckpt_path=ckpt, verbose=False, min_free_ram_gb=min_free)


tmp = tempfile.mkdtemp()

# leaking: 10 GB, 8, 6, 4, 2 -> loses 2 GB/epoch, threshold 3 + 1.5 * 2 -> must stop before it hits 0
res = run([10, 8, 6, 4, 2, 0.5], 1.5, os.path.join(tmp, "leak.pth"))
g = res["ram_guard"]
assert g is not None, "guard did not fire on a leaking run"
assert res["epochs_run"] < 10, res["epochs_run"]
ck = torch.load(os.path.join(tmp, "leak.pth"), weights_only=False)
assert ck["ram_guard"] == g and "model_state_dict" in ck
print(f"leaking run   : stopped at epoch {g['stopped_epoch']} with {g['free_gb']} GB free, "
      f"{g['growth_gb_per_epoch']} GB/epoch; best epoch {res['best_epoch']}; checkpoint carries the flag")

# healthy: plenty of RAM, runs all 10 epochs, no flag
res = run([20.0], 1.5, os.path.join(tmp, "ok.pth"))
assert res["ram_guard"] is None and res["epochs_run"] == 10
assert torch.load(os.path.join(tmp, "ok.pth"), weights_only=False)["ram_guard"] is None
print("healthy run   : 10/10 epochs, ram_guard None")

# disabled
res = run([0.1], 0, os.path.join(tmp, "off.pth"))
assert res["ram_guard"] is None and res["epochs_run"] == 10
print("guard off (0) : 10/10 epochs even at 0.1 GB free")

# off Linux (None) must be a no-op
sw._free_ram_gb = lambda: None
res = sw.train_unet(Toy(), Toy(), "cpu", base_channels=8, channel_multipliers=(1, 2), min_epochs=1,
                    max_epochs=3, patience=100, batch_size=4, verbose=False)
assert res["ram_guard"] is None and res["epochs_run"] == 3
print("no /proc      : guard is a no-op")
print("PASS")


# --- DataParallel rule (the actual cause of the nb14 leak) -------------
os.environ.pop("EXXA_DP", None)
assert sw._use_data_parallel(0, 600) is False and sw._use_data_parallel(1, 600) is False   # no GPUs / one GPU
assert sw._use_data_parallel(2, 600) is True and sw._use_data_parallel(2, 256) is True      # full images: as before
assert sw._use_data_parallel(2, 64) is False                                                # 64 px patches: single GPU
assert sw._use_data_parallel(2, None) is True                                               # unknown size: as before
assert sw._use_data_parallel(2, 64, override=True) is True and sw._use_data_parallel(2, 600, override=False) is False
os.environ["EXXA_DP"] = "0"; assert sw._use_data_parallel(2, 600) is False
os.environ["EXXA_DP"] = "1"; assert sw._use_data_parallel(2, 64) is True
os.environ.pop("EXXA_DP")
assert sw._sample_side(Toy(size=32)) == 32 and sw._sample_side([]) is None
print("DataParallel rule: off below 256 px, on for full images, env/arg override works")
print("PASS (dp)")
