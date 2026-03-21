"""Tests for model skeleton — imports, instantiation, and interface contracts."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import DDPM, UNet, NoiseScheduler, ResidualBlock, AttentionBlock, SinusoidalTimeEmbedding


class TestImports:
    def test_all_classes_importable(self):
        assert UNet is not None
        assert NoiseScheduler is not None
        assert DDPM is not None
        assert ResidualBlock is not None
        assert AttentionBlock is not None
        assert SinusoidalTimeEmbedding is not None


class TestUNet:
    def test_instantiation_defaults(self):
        model = UNet()
        assert model.in_channels == 1
        assert model.out_channels == 1
        assert model.base_channels == 64

    def test_instantiation_custom(self):
        model = UNet(in_channels=1, out_channels=1, base_channels=32, channel_multipliers=(1, 2, 4))
        assert model.base_channels == 32
        assert model.channel_multipliers == (1, 2, 4)

    def test_forward_not_implemented(self):
        model = UNet()
        x = torch.randn(2, 1, 64, 64)
        t = torch.randint(0, 1000, (2,))
        with pytest.raises(NotImplementedError):
            model(x, t)

    def test_is_nn_module(self):
        assert isinstance(UNet(), torch.nn.Module)


class TestNoiseScheduler:
    def test_instantiation_defaults(self):
        scheduler = NoiseScheduler()
        assert scheduler.timesteps == 1000
        assert scheduler.beta_schedule == "linear"

    def test_instantiation_custom(self):
        scheduler = NoiseScheduler(timesteps=500, beta_schedule="cosine")
        assert scheduler.timesteps == 500
        assert scheduler.beta_schedule == "cosine"

    def test_q_sample_not_implemented(self):
        scheduler = NoiseScheduler()
        x0 = torch.randn(2, 1, 64, 64)
        t = torch.tensor([100, 500])
        with pytest.raises(NotImplementedError):
            scheduler.q_sample(x0, t)

    def test_p_sample_not_implemented(self):
        scheduler = NoiseScheduler()
        model = UNet()
        xt = torch.randn(2, 1, 64, 64)
        t = torch.tensor([100, 500])
        with pytest.raises(NotImplementedError):
            scheduler.p_sample(model, xt, t)


class TestDDPM:
    @pytest.fixture
    def ddpm(self):
        return DDPM(unet=UNet(), scheduler=NoiseScheduler())

    def test_instantiation(self, ddpm):
        assert ddpm.loss_type == "l2"
        assert isinstance(ddpm.unet, UNet)
        assert isinstance(ddpm.scheduler, NoiseScheduler)

    def test_is_nn_module(self, ddpm):
        assert isinstance(ddpm, torch.nn.Module)

    def test_training_loss_not_implemented(self, ddpm):
        x0 = torch.randn(2, 1, 64, 64)
        with pytest.raises(NotImplementedError):
            ddpm.training_loss(x0)

    def test_sample_not_implemented(self, ddpm):
        with pytest.raises(NotImplementedError):
            ddpm.sample(shape=(2, 1, 64, 64), device=torch.device("cpu"))

    def test_forward_delegates_to_training_loss(self, ddpm):
        x0 = torch.randn(2, 1, 64, 64)
        with pytest.raises(NotImplementedError):
            ddpm(x0)


class TestBlocks:
    def test_residual_block_instantiation(self):
        block = ResidualBlock(in_channels=64, out_channels=64, time_emb_dim=256)
        assert block.in_channels == 64
        assert block.out_channels == 64

    def test_residual_block_forward_not_implemented(self):
        block = ResidualBlock(in_channels=64, out_channels=64, time_emb_dim=256)
        x = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 256)
        with pytest.raises(NotImplementedError):
            block(x, t)

    def test_attention_block_instantiation(self):
        block = AttentionBlock(channels=64, num_heads=4)
        assert block.channels == 64
        assert block.num_heads == 4

    def test_attention_block_forward_not_implemented(self):
        block = AttentionBlock(channels=64)
        x = torch.randn(2, 64, 16, 16)
        with pytest.raises(NotImplementedError):
            block(x)

    def test_time_embedding_instantiation(self):
        emb = SinusoidalTimeEmbedding(dim=256)
        assert emb.dim == 256

    def test_time_embedding_forward_not_implemented(self):
        emb = SinusoidalTimeEmbedding(dim=256)
        t = torch.randint(0, 1000, (2,))
        with pytest.raises(NotImplementedError):
            emb(t)
