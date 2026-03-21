"""U-Net backbone for the diffusion model."""

import torch
import torch.nn as nn
from typing import Tuple


class UNet(nn.Module):
    """
    U-Net architecture for predicting noise in the diffusion process.

    Encoder-decoder structure with skip connections and time conditioning.
    Attention is applied at deeper levels where spatial resolution is low.

    Args:
        in_channels: Number of input image channels (1 for grayscale)
        out_channels: Number of output channels (same as in_channels)
        base_channels: Base feature channels, doubled at each encoder level
        channel_multipliers: Per-level channel multipliers, e.g. (1, 2, 4, 8)
        num_res_blocks: Number of residual blocks per encoder/decoder level
        attention_levels: Which levels (0-indexed) to apply self-attention
        dropout: Dropout probability in residual blocks

    Example:
        >>> model = UNet(in_channels=1, out_channels=1, base_channels=64)
        >>> x = torch.randn(2, 1, 64, 64)
        >>> t = torch.randint(0, 1000, (2,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([2, 1, 64, 64])
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: Tuple[int, ...] = (2, 3),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.dropout = dropout

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise given noisy image and timestep.

        Args:
            x: Noisy image tensor, shape (B, in_channels, H, W)
            t: Diffusion timestep, shape (B,)

        Returns:
            Predicted noise, shape (B, out_channels, H, W)
        """
        raise NotImplementedError
