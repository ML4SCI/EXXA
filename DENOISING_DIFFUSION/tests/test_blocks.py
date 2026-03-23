"""Shape-consistency tests for U-Net building blocks."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.blocks import ResidualBlock, AttentionBlock, SinusoidalTimeEmbedding


class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(dim=256)
        t = torch.randint(0, 1000, (4,))
        out = emb(t)
        assert out.shape == (4, 256)

    def test_output_shape_batch_1(self):
        emb = SinusoidalTimeEmbedding(dim=128)
        t = torch.tensor([42])
        out = emb(t)
        assert out.shape == (1, 128)

    def test_different_timesteps_differ(self):
        emb = SinusoidalTimeEmbedding(dim=64)
        t1 = torch.tensor([0])
        t2 = torch.tensor([500])
        assert not torch.allclose(emb(t1), emb(t2))

    def test_same_timestep_same_output(self):
        emb = SinusoidalTimeEmbedding(dim=64)
        t = torch.tensor([100])
        assert torch.allclose(emb(t), emb(t))

    def test_output_dtype_float32(self):
        emb = SinusoidalTimeEmbedding(dim=64)
        t = torch.randint(0, 1000, (2,))
        assert emb(t).dtype == torch.float32


class TestResidualBlock:
    @pytest.fixture
    def block_same(self):
        return ResidualBlock(in_channels=64, out_channels=64, time_emb_dim=256)

    @pytest.fixture
    def block_proj(self):
        return ResidualBlock(in_channels=64, out_channels=128, time_emb_dim=256)

    def test_output_shape_same_channels(self, block_same):
        x = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 256)
        out = block_same(x, t)
        assert out.shape == (2, 64, 32, 32)

    def test_output_shape_channel_projection(self, block_proj):
        x = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 256)
        out = block_proj(x, t)
        assert out.shape == (2, 128, 32, 32)

    def test_spatial_dims_preserved(self, block_same):
        for h, w in [(16, 16), (32, 32), (64, 64)]:
            x = torch.randn(2, 64, h, w)
            t = torch.randn(2, 256)
            out = block_same(x, t)
            assert out.shape == (2, 64, h, w)

    def test_skip_identity_when_same_channels(self, block_same):
        assert isinstance(block_same.skip, torch.nn.Identity)

    def test_skip_conv_when_different_channels(self, block_proj):
        assert isinstance(block_proj.skip, torch.nn.Conv2d)

    def test_is_nn_module(self, block_same):
        assert isinstance(block_same, torch.nn.Module)

    def test_output_not_equal_to_input(self, block_same):
        x = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 256)
        out = block_same(x, t)
        assert not torch.allclose(out, x)


class TestAttentionBlock:
    @pytest.fixture
    def block(self):
        return AttentionBlock(channels=64, num_heads=4)

    def test_output_shape(self, block):
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_spatial_dims_preserved(self, block):
        for h, w in [(8, 8), (16, 16), (32, 32)]:
            x = torch.randn(2, 64, h, w)
            assert block(x).shape == (2, 64, h, w)

    def test_batch_size_preserved(self, block):
        for b in [1, 4, 8]:
            x = torch.randn(b, 64, 16, 16)
            assert block(x).shape[0] == b

    def test_residual_connection(self, block):
        # zero input should not produce zero output due to residual
        x = torch.zeros(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_is_nn_module(self, block):
        assert isinstance(block, torch.nn.Module)
