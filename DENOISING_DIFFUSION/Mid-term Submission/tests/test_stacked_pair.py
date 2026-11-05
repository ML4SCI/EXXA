#!/usr/bin/env python
"""Tests for src.data.stacked_pair.StackedPairDataset."""

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.stacked_pair import StackedPairDataset


class _FakePairDataset(Dataset):
    """Yields (dirty, clean), each (1, H, W), like FITSChannelDataset."""

    def __init__(self, n=6, size=8):
        self.n, self.size = n, size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        dirty = torch.full((1, self.size, self.size), float(i))
        clean = torch.full((1, self.size, self.size), float(i) + 0.5)
        return dirty, clean


def test_stacked_shape_and_layout():
    ds = StackedPairDataset(_FakePairDataset(size=8))
    stacked, label = ds[3]
    assert stacked.shape == (2, 8, 8), stacked.shape
    assert label == 0
    # channel 0 = dirty (=3.0), channel 1 = clean (=3.5)
    assert torch.allclose(stacked[0], torch.full((8, 8), 3.0))
    assert torch.allclose(stacked[1], torch.full((8, 8), 3.5))


def test_batches_through_dataloader():
    ds = StackedPairDataset(_FakePairDataset(n=6, size=8))
    loader = DataLoader(ds, batch_size=3)
    x, _ = next(iter(loader))
    assert x.shape == (3, 2, 8, 8), x.shape


def test_len_matches_base():
    base = _FakePairDataset(n=11)
    assert len(StackedPairDataset(base)) == 11


if __name__ == "__main__":
    test_stacked_shape_and_layout()
    test_batches_through_dataloader()
    test_len_matches_base()
    print("all stacked_pair tests passed")
