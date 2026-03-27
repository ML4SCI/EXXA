"""Tests for Inferencer: preprocess, postprocess, run, and from_checkpoint."""

import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.inference.inferencer import Inferencer
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------

class PassthroughModel(nn.Module):
    """Returns input unchanged — used to test shape preservation."""
    def forward(self, x):
        return x


class SampleModel(nn.Module):
    """Has a sample() method — tests that Inferencer prefers sample() over forward()."""
    def forward(self, x):
        return torch.zeros_like(x)          # forward returns zeros

    def sample(self, x):
        return torch.ones_like(x)           # sample returns ones


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    def training_loss(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return (self.conv(x) ** 2).mean()


def make_inferencer(model=None):
    model = model or PassthroughModel()
    return Inferencer(model, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_inferencer_instantiates():
    inf = make_inferencer()
    assert inf is not None


def test_model_set_to_eval():
    inf = make_inferencer()
    assert not inf.model.training


def test_device_stored():
    inf = make_inferencer()
    assert inf.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# from_checkpoint
# ---------------------------------------------------------------------------

def test_from_checkpoint_loads_weights():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = LinearModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(model, optimizer, torch.device("cpu"), checkpoint_dir=tmpdir)

        from torch.utils.data import DataLoader, TensorDataset
        dl = DataLoader(TensorDataset(torch.randn(16, 1, 8, 8)), batch_size=4)
        trainer.train_one_epoch(dl)
        path = trainer.save_checkpoint()

        inf = Inferencer.from_checkpoint(LinearModel(), path)
        for p1, p2 in zip(trainer.model.parameters(), inf.model.parameters()):
            assert torch.allclose(p1, p2)


def test_from_checkpoint_returns_inferencer():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = LinearModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(model, optimizer, torch.device("cpu"), checkpoint_dir=tmpdir)

        from torch.utils.data import DataLoader, TensorDataset
        dl = DataLoader(TensorDataset(torch.randn(8, 1, 8, 8)), batch_size=4)
        trainer.train_one_epoch(dl)
        path = trainer.save_checkpoint()

        inf = Inferencer.from_checkpoint(LinearModel(), path)
        assert isinstance(inf, Inferencer)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def test_preprocess_hw_adds_batch_and_channel():
    inf = make_inferencer()
    img = np.random.rand(64, 64).astype(np.float32)
    t = inf.preprocess(img)
    assert t.shape == (1, 1, 64, 64)


def test_preprocess_chw_adds_batch():
    inf = make_inferencer()
    img = np.random.rand(1, 64, 64).astype(np.float32)
    t = inf.preprocess(img)
    assert t.shape == (1, 1, 64, 64)


def test_preprocess_bchw_unchanged():
    inf = make_inferencer()
    img = np.random.rand(2, 1, 64, 64).astype(np.float32)
    t = inf.preprocess(img)
    assert t.shape == (2, 1, 64, 64)


def test_preprocess_range_minus1_to_1():
    inf = make_inferencer()
    img = np.random.rand(32, 32).astype(np.float32)
    t = inf.preprocess(img)
    assert t.min().item() >= -1.0 - 1e-5
    assert t.max().item() <= 1.0 + 1e-5


def test_preprocess_returns_tensor():
    inf = make_inferencer()
    t = inf.preprocess(np.random.rand(16, 16))
    assert isinstance(t, torch.Tensor)


def test_preprocess_constant_image_no_nan():
    inf = make_inferencer()
    img = np.ones((32, 32), dtype=np.float32)
    t = inf.preprocess(img)
    assert not torch.isnan(t).any()


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------

def test_postprocess_clamps_to_0_1():
    inf = make_inferencer()
    t = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = inf.postprocess(t.reshape(1, 1, 1, 5))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_postprocess_returns_numpy():
    inf = make_inferencer()
    t = torch.zeros(1, 1, 8, 8)
    out = inf.postprocess(t)
    assert isinstance(out, np.ndarray)


def test_postprocess_shape_preserved():
    inf = make_inferencer()
    t = torch.randn(2, 1, 16, 16)
    out = inf.postprocess(t)
    assert out.shape == (2, 1, 16, 16)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_output_shape_hw():
    inf = make_inferencer()
    img = np.random.rand(32, 32).astype(np.float32)
    out = inf.run(img)
    assert out.shape == (1, 1, 32, 32)


def test_run_output_range_0_1():
    inf = make_inferencer()
    img = np.random.rand(32, 32).astype(np.float32)
    out = inf.run(img)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_run_prefers_sample_over_forward():
    inf = make_inferencer(SampleModel())
    img = np.random.rand(16, 16).astype(np.float32)
    out = inf.run(img)
    # sample() returns ones -> postprocess maps 1.0 to 1.0
    assert np.allclose(out, 1.0)


def test_run_falls_back_to_forward():
    class ZeroForward(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    inf = make_inferencer(ZeroForward())
    img = np.random.rand(16, 16).astype(np.float32)
    out = inf.run(img)
    # forward returns zeros -> postprocess maps 0.0 to 0.5
    assert np.allclose(out, 0.5)


def test_run_no_grad_no_error():
    inf = make_inferencer()
    img = np.random.rand(16, 16).astype(np.float32)
    out = inf.run(img)
    assert out is not None
