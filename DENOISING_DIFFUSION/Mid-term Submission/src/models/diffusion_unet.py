"""
src/models/diffusion_unet.py
============================
Conditional DDPM U-Net ported from the GSoC main notebook
(`GSOC_2025_EXXA_Main.ipynb`, "Model" cell) and scaled down for a 4 GB RTX 2050.

Lineage: https://github.com/ermongroup/ddim and https://github.com/bahjat-kawar/ddrm.

The network predicts the noise added to the *clean* image, conditioned on the
*dirty* observation. Input is the channel-wise concatenation ``[x_cond, x_t]``
(2 channels for single-channel astronomy data); the output is the predicted
noise (1 channel, same spatial size).

Scaled config vs. the notebook original
---------------------------------------
                      notebook          this repo (RTX 2050)
  ch                  128               64
  ch_mult             [1,1,2,2,4,4]     [1, 2, 2, 4]   (6 levels -> 4 levels)
  attn_resolutions    [16]              [16]
  num_res_blocks      2                 2
  -> params           ~110M             ~ (see __main__ printout, ~17M)

Config object
-------------
The model is config-driven, matching the notebook. Use :func:`default_diffusion_config`
to obtain the scaled-down config (a :class:`DotDict` with ``.model``, ``.data`` and
``.diffusion`` sub-dicts), or build your own.
"""

import math

import torch
import torch.nn as nn


class DotDict(dict):
    """Dictionary with dot-notation attribute access.

    Nested dicts are converted to ``DotDict`` at construction so that nested
    attribute *writes* persist (e.g. ``cfg.model.ch = 128`` mutates the stored
    config rather than a throwaway copy).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, DotDict):
                self[k] = DotDict(v)

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def default_diffusion_config(image_size: int = 64) -> DotDict:
    """
    Scaled-down conditional DDPM config for a 4 GB RTX 2050.

    ch=64 with ch_mult=[1, 2, 2, 4] gives 4 resolution levels
    (64 -> 32 -> 16 -> 8) with self-attention at the 16x16 level.

    Returns:
        DotDict with ``model``, ``data`` and ``diffusion`` sections.
    """
    return DotDict({
        "data": {
            "image_size": image_size,
            "channels": 1,
            "conditional": True,
        },
        "model": {
            "in_channels": 1,
            "out_ch": 1,
            "ch": 64,
            "ch_mult": [1, 2, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0,
            "ema_rate": 0.999,
            "ema": True,
            "resamp_with_conv": True,
        },
        "diffusion": {
            "beta_schedule": "linear",   # "cosine" recommended: see get_beta_schedule
            # Training objective. "eps" reproduces the original behaviour exactly;
            # "v" + min_snr_gamma=5.0 is the configuration the literature reports as
            # converging fastest and degrading least at low SNR.
            "prediction_type": "eps",
            "min_snr_gamma": 0.0,
            "beta_start": 1e-4,
            "beta_end": 2e-2,
            "num_diffusion_timesteps": 1000,
        },
    })


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings (DDPM / tensor2tensor style)."""
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish / SiLU activation."""
    return x * torch.sigmoid(x)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.Module:
    """GroupNorm with a group count that always divides the channel count."""
    # Scaled config keeps channels in {64, 128, 256}; 32 groups divides all of them.
    num_groups = math.gcd(num_groups, in_channels)
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # asymmetric padding done manually (no padding arg on the conv)
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)   # b, hw, c
        k = k.reshape(b, c, h * w)                    # b, c, hw
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))     # b, hw, hw
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)                       # b, hw, hw
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_


class DiffusionUNet(nn.Module):
    """
    Config-driven conditional DDPM U-Net.

    Args:
        config: object exposing ``config.model`` and ``config.data`` (e.g. the
            :class:`DotDict` returned by :func:`default_diffusion_config`).

    Forward:
        x: (B, in_channels*2, H, W) if conditional else (B, in_channels, H, W).
           Channel layout when conditional is ``[x_cond, x_t]``.
        t: (B,) timestep indices.
        returns: (B, out_ch, H, W) predicted noise.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        ch = config.model.ch
        out_ch = config.model.out_ch
        ch_mult = tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to keep consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        # The net is fully convolutional and its attention is 1x1-conv based, so any size
        # the down/up path can halve and restore works, not just the configured one.
        # Required for the patch arm (train on 64px patches, score on full 256px channels)
        # and for tiled native-resolution inference.
        div = 2 ** (self.num_resolutions - 1)
        assert x.shape[2] % div == 0 and x.shape[3] % div == 0, \
            (f"spatial dims must be multiples of {div} for {self.num_resolutions} "
             f"resolution levels, got {tuple(x.shape[2:])}")

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def create_diffusion_unet(config=None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Build the scaled-down conditional DDPM U-Net and move it to ``device``."""
    if config is None:
        config = default_diffusion_config()
    return DiffusionUNet(config).to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = default_diffusion_config()
    model = DiffusionUNet(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())

    # conditional: input is [x_cond, x_t] -> 2 channels
    x = torch.randn(2, 2, 64, 64, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    with torch.no_grad():
        out = model(x, t)

    print("\n--- DiffusionUNet (scaled: ch=64, ch_mult=[1,2,2,4], attn@16) ---")
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Params : {params:,}")
    assert out.shape == (2, 1, 64, 64), "Shape mismatch!"
    print("  Forward pass: OK")
