"""Tests for Trainer: instantiation, train step, loss descent, checkpoint round-trip."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Minimal toy model: accepts a batch tensor, returns MSE toward zero
# ---------------------------------------------------------------------------

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def training_loss(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return (self.linear(x) ** 2).mean()


def make_dataloader(n=32, feat=16, batch_size=8):
    data = torch.randn(n, feat)
    return DataLoader(TensorDataset(data), batch_size=batch_size)


def make_trainer(tmpdir=None, log_fn=None):
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    return Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        log_fn=log_fn,
        checkpoint_dir=tmpdir or "checkpoints",
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_trainer_instantiates():
    trainer = make_trainer()
    assert trainer.epoch == 0


def test_trainer_model_on_device():
    trainer = make_trainer()
    for p in trainer.model.parameters():
        assert p.device == torch.device("cpu")


def test_trainer_default_epoch_zero():
    trainer = make_trainer()
    assert trainer.epoch == 0


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

def test_train_one_epoch_returns_float():
    trainer = make_trainer()
    loss = trainer.train_one_epoch(make_dataloader())
    assert isinstance(loss, float)


def test_train_one_epoch_loss_positive():
    trainer = make_trainer()
    loss = trainer.train_one_epoch(make_dataloader())
    assert loss > 0


def test_train_one_epoch_increments_epoch():
    trainer = make_trainer()
    trainer.train_one_epoch(make_dataloader())
    assert trainer.epoch == 1


def test_train_multiple_epochs_increments():
    trainer = make_trainer()
    for _ in range(3):
        trainer.train_one_epoch(make_dataloader())
    assert trainer.epoch == 3


def test_loss_decreases_over_epochs():
    trainer = make_trainer()
    dl = make_dataloader(n=64)
    losses = [trainer.train_one_epoch(dl) for _ in range(10)]
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# Logging hook
# ---------------------------------------------------------------------------

def test_log_fn_called():
    calls = []
    trainer = make_trainer(log_fn=lambda epoch, step, loss: calls.append((epoch, step, loss)))
    trainer.train_one_epoch(make_dataloader(n=32, batch_size=8))
    assert len(calls) == 4  # 32 samples / batch_size 8 = 4 steps


def test_log_fn_receives_correct_epoch():
    calls = []
    trainer = make_trainer(log_fn=lambda epoch, step, loss: calls.append(epoch))
    trainer.train_one_epoch(make_dataloader())
    assert all(e == 0 for e in calls)


def test_log_fn_loss_is_float():
    calls = []
    trainer = make_trainer(log_fn=lambda epoch, step, loss: calls.append(loss))
    trainer.train_one_epoch(make_dataloader())
    assert all(isinstance(l, float) for l in calls)


def test_no_log_fn_runs_without_error():
    trainer = make_trainer(log_fn=None)
    trainer.train_one_epoch(make_dataloader())


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def test_save_checkpoint_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = make_trainer(tmpdir=tmpdir)
        path = trainer.save_checkpoint("test")
        assert os.path.exists(path)


def test_save_checkpoint_returns_correct_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = make_trainer(tmpdir=tmpdir)
        path = trainer.save_checkpoint("v1")
        assert path.endswith("ckpt_v1.pt")


def test_load_checkpoint_restores_epoch():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = make_trainer(tmpdir=tmpdir)
        trainer.train_one_epoch(make_dataloader())
        path = trainer.save_checkpoint()

        trainer2 = make_trainer(tmpdir=tmpdir)
        trainer2.load_checkpoint(path)
        assert trainer2.epoch == 1


def test_load_checkpoint_restores_model_weights():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = make_trainer(tmpdir=tmpdir)
        trainer.train_one_epoch(make_dataloader())
        path = trainer.save_checkpoint()

        trainer2 = make_trainer(tmpdir=tmpdir)
        trainer2.load_checkpoint(path)

        for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
            assert torch.allclose(p1, p2)


def test_checkpoint_round_trip_loss_consistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = make_trainer(tmpdir=tmpdir)
        dl = make_dataloader()
        trainer.train_one_epoch(dl)
        path = trainer.save_checkpoint()

        trainer2 = make_trainer(tmpdir=tmpdir)
        trainer2.load_checkpoint(path)

        trainer.model.eval()
        trainer2.model.eval()
        x = torch.randn(8, 16)
        with torch.no_grad():
            out1 = trainer.model.linear(x)
            out2 = trainer2.model.linear(x)
        assert torch.allclose(out1, out2)


def test_multiple_checkpoints_independent():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = make_trainer(tmpdir=tmpdir)
        dl = make_dataloader()
        trainer.train_one_epoch(dl)
        path1 = trainer.save_checkpoint("epoch1")
        trainer.train_one_epoch(dl)
        path2 = trainer.save_checkpoint("epoch2")
        assert path1 != path2
        assert os.path.exists(path1)
        assert os.path.exists(path2)
