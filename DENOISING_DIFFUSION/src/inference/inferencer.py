"""Inference pipeline: load checkpoint, preprocess, run model, postprocess."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class Inferencer:
    """
    Inference pipeline for the EXXA denoising models.

    Handles preprocessing (numpy → normalized tensor), model execution,
    and postprocessing (tensor → numpy). Compatible with any nn.Module
    that implements either `sample(x)` or a standard `forward(x)`.

    Args:
        model: Loaded nn.Module in eval mode
        device: torch device to run inference on
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> "Inferencer":
        """
        Load model weights from a checkpoint and return an Inferencer.

        Args:
            model: Uninitialised nn.Module with the correct architecture
            checkpoint_path: Path to a .pt checkpoint saved by Trainer
            device: Target device (defaults to cpu)
        """
        if device is None:
            device = torch.device("cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return cls(model, device)

    # ------------------------------------------------------------------
    # Pre / post processing
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy image to a normalised tensor ready for the model.

        Applies min-max normalisation to [0, 1] then maps to [-1, 1].
        Adds a batch dimension and a channel dimension if missing.

        Args:
            image: numpy array of shape (H, W), (C, H, W), or (B, C, H, W)

        Returns:
            Float tensor of shape (B, C, H, W) in [-1, 1]
        """
        x = image.astype(np.float32)

        vmin, vmax = x.min(), x.max()
        if vmax - vmin > 1e-8:
            x = (x - vmin) / (vmax - vmin)

        # map [0, 1] -> [-1, 1]
        x = x * 2.0 - 1.0

        t = torch.from_numpy(x)

        if t.ndim == 2:          # (H, W) -> (1, 1, H, W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:        # (C, H, W) -> (1, C, H, W)
            t = t.unsqueeze(0)
        # ndim == 4: already (B, C, H, W)

        return t.to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert model output tensor back to a numpy image in [0, 1].

        Args:
            tensor: Float tensor of shape (B, C, H, W) in [-1, 1]

        Returns:
            numpy array of shape (B, C, H, W) clamped to [0, 1]
        """
        out = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
        return out.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        End-to-end inference: preprocess → model → postprocess.

        Calls `model.sample(x)` if available, otherwise `model(x)`.

        Args:
            image: Raw numpy image (H, W), (C, H, W), or (B, C, H, W)

        Returns:
            Denoised numpy array of shape (B, C, H, W) in [0, 1]
        """
        x = self.preprocess(image)
        with torch.no_grad():
            if hasattr(self.model, "sample") and callable(self.model.sample):
                out = self.model.sample(x)
            else:
                out = self.model(x)
        return self.postprocess(out)
