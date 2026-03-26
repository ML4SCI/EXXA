"""Minimal training loop with logging hooks and checkpoint save/load."""

import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Model-agnostic training loop for the DDPM denoising pipeline.

    Expects the model to accept a batch and return a scalar loss tensor
    via a `training_loss(batch) -> Tensor` method.

    Args:
        model: nn.Module with a `training_loss(batch) -> Tensor` method
        optimizer: PyTorch optimizer
        device: torch device
        log_fn: Optional callback called with (epoch, step, loss) after each step
        checkpoint_dir: Directory for saving/loading checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_fn: Optional[Callable[[int, int, float], None]] = None,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.log_fn = log_fn
        self.checkpoint_dir = checkpoint_dir
        self.epoch = 0

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        """Run one full epoch. Returns mean loss over all steps."""
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
            else:
                batch = batch.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.training_loss(batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if self.log_fn is not None:
                self.log_fn(self.epoch, step, loss.item())

        self.epoch += 1
        return total_loss / max(len(dataloader), 1)

    def save_checkpoint(self, tag: str = "latest") -> str:
        """Save model + optimizer state. Returns path to saved file."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"ckpt_{tag}.pt")
        torch.save(
            {
                "epoch": self.epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load model + optimizer state from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt["epoch"]
