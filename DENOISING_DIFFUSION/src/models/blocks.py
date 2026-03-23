"""Building blocks for the U-Net backbone."""

import math
import torch
import torch.nn as nn


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
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        ).float()
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


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

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map, shape (B, in_channels, H, W)
            time_emb: Time embedding, shape (B, time_emb_dim)

        Returns:
            Output feature map, shape (B, out_channels, H, W)
        """
        h = self.block1(x)
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


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
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map, shape (B, channels, H, W)

        Returns:
            Output feature map, shape (B, channels, H, W)
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h
