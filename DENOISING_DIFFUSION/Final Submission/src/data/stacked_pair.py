#!/usr/bin/env python
"""
src/data/stacked_pair.py
========================
Thin adapter that turns a (dirty, clean) tuple dataset into a single stacked
tensor, for the conditional-DDPM trainer.

`FITSChannelDataset.__getitem__` returns a 2-tuple ``(dirty, clean)``, each of
shape ``(1, H, W)`` in the shared dirty-scale [0,1]/[~0,1+] range. The DDPM
trainer/sampler in ``src.training.diffusion`` instead iterates ``for x, _ in
loader`` expecting ``x`` of shape ``(B, 2, H, W)`` with channel layout
``[dirty, clean]`` (channel 0 = condition, channel 1 = clean target). This
wrapper stacks the pair on the channel axis and returns ``(stacked, 0)`` so the
proven dataset is reused unchanged (continuum subtraction + shared-scale
normalization all still happen inside the wrapped dataset).

The dummy second element mirrors the DDPM loader contract (the label is ignored).
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset


class StackedPairDataset(Dataset):
    """
    Wrap a dataset yielding ``(dirty, clean)`` -> yield ``(stacked, 0)``.

    Args:
        base: any dataset whose ``__getitem__`` returns ``(dirty, clean)``, each
            a ``(1, H, W)`` float tensor.

    Returns per item:
        stacked: ``(2, H, W)`` float tensor, channel 0 = dirty, channel 1 = clean.
        label:   ``0`` (unused; keeps the ``x, _`` loader contract).
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        dirty, clean = self.base[i]
        # each is (1, H, W); concat on channel axis -> (2, H, W) = [dirty, clean]
        stacked = torch.cat([dirty, clean], dim=0)
        return stacked, 0
