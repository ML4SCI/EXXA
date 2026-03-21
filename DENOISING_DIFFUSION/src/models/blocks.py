"""Building blocks for the U-Net backbone."""

import torch
import torch.nn as nn
from typing import Optional


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timestep conditioning.

    Encodes scalar timestep t into a fixed-size vector that gets injected
    into each ResidualBlock of the U-Net.

    Args:
        dim: Embedding dimension (should match U-Net base channels * 4)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor, shape (B,)

        Returns:
            Embedding tensor, shape (B, dim)
        """
        raise NotImplementedError


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm, SiLU activation, and time conditioning.

    Used in both encoder and decoder of the U-Net. Injects the timestep
    embedding via a linear projection added to the hidden features.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_emb_dim: Dimension of the time embedding vector
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map, shape (B, in_channels, H, W)
            time_emb: Time embedding, shape (B, time_emb_dim)

        Returns:
            Output feature map, shape (B, out_channels, H, W)
        """
        raise NotImplementedError


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing global context.

    Applied at deeper U-Net levels where spatial resolution is low.
    Uses multi-head self-attention with GroupNorm and residual connection.

    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map, shape (B, channels, H, W)

        Returns:
            Output feature map, shape (B, channels, H, W)
        """
        raise NotImplementedError
