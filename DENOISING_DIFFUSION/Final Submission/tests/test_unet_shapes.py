#!/usr/bin/env python
"""
Tests for the U-Net's accepted input sizes.

The net used to assert that every input matched `config.data.image_size` exactly. That
silently broke notebook 06's patch arm, which trains on 64px patches with a config built at
TARGET_SIZE=256, and it blocked tiled native-resolution inference for the same reason. The
network itself is fully convolutional and its attention is 1x1-conv based, so the only real
constraint is that the down/up path can halve each dimension and restore it.

These pin that contract: the sizes the notebook actually feeds must work, and sizes the
skip connections cannot reconcile must still be rejected rather than failing deep inside a
concat with an unreadable message.
"""

import torch

from src.models.diffusion_unet import DiffusionUNet, default_diffusion_config

print("=" * 60)
print("U-Net Shape Tests")
print("=" * 60)

# Notebook 06's real config, with ch shrunk so this runs on CPU in seconds. ch does not
# affect which spatial sizes are accepted -- ch_mult does, via the number of levels.
cfg = default_diffusion_config(image_size=256)
cfg.model.ch_mult = [1, 2, 2, 2, 4]
cfg.model.ch = 32
net = DiffusionUNet(cfg).eval()

div = 2 ** (net.num_resolutions - 1)
assert div == 16, div
print(f"[1] {net.num_resolutions} resolution levels -> inputs must be multiples of {div}")


# ------------------------------------------------------------------ [2] sizes 06 feeds
# 256 = TARGET_SIZE (full channels), 64 = PATCH_SIZE (the sweep's patch arm). Both are
# driven through ONE net instance, which is the actual patch-arm scenario: train at 64,
# score the resulting weights on full 256px validation channels.
for size in (256, 64, 128):
    with torch.no_grad():
        out = net(torch.randn(1, 2, size, size), torch.zeros(1))
    assert out.shape[2:] == (size, size), (size, out.shape)
    assert torch.isfinite(out).all(), f"non-finite output at {size}px"
print("[2] 256px (TARGET_SIZE), 64px (PATCH_SIZE) and 128px all round-trip on one net")


# ------------------------------------------------------------------ [3] non-square
# tiled_denoise only emits square tiles today, but nothing in the net requires square, and
# a needless assert here is exactly what broke the patch arm.
with torch.no_grad():
    out = net(torch.randn(1, 2, 64, 128), torch.zeros(1))
assert out.shape[2:] == (64, 128), out.shape
print("[3] non-square 64x128 accepted, shape preserved")


# ------------------------------------------------------------------ [4] bad sizes rejected
# 72 and 100 are even but not multiples of 16: the decoder would upsample to a size the
# encoder skip cannot match. Fail loudly at the front door, not inside a concat.
for size in (72, 100, 31):
    try:
        with torch.no_grad():
            net(torch.randn(1, 2, size, size), torch.zeros(1))
    except AssertionError as exc:
        assert str(div) in str(exc), f"message should name the required multiple: {exc}"
    else:
        raise AssertionError(f"{size}px should have been rejected")
print(f"[4] 72, 100 and 31px rejected with a message naming the multiple ({div})")


# ------------------------------------------------------------------ [5] shallower net
# Fewer levels means a smaller required multiple -- the check must follow ch_mult, not a
# hardcoded number.
cfg2 = default_diffusion_config(image_size=64)
cfg2.model.ch_mult = [1, 2]
cfg2.model.ch = 16
net2 = DiffusionUNet(cfg2).eval()
with torch.no_grad():
    out = net2(torch.randn(1, 2, 30, 30), torch.zeros(1))
assert out.shape[2:] == (30, 30), out.shape
print(f"[5] 2-level net accepts 30px (multiple of {2 ** (net2.num_resolutions - 1)}), "
      "so the bound tracks ch_mult")

print("\n" + "=" * 60)
print("All U-Net shape tests PASSED")
print("=" * 60)
