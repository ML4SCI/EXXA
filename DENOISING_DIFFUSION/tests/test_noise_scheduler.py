"""Tests for NoiseScheduler — schedule shapes, ranges, monotonicity, and q_sample."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.noise_scheduler import NoiseScheduler


@pytest.fixture(params=["linear", "cosine"])
def scheduler(request):
    return NoiseScheduler(timesteps=1000, beta_schedule=request.param)


class TestScheduleShapes:
    def test_betas_shape(self, scheduler):
        assert scheduler.betas.shape == (1000,)

    def test_alphas_shape(self, scheduler):
        assert scheduler.alphas.shape == (1000,)

    def test_alphas_cumprod_shape(self, scheduler):
        assert scheduler.alphas_cumprod.shape == (1000,)

    def test_sqrt_alphas_cumprod_shape(self, scheduler):
        assert scheduler.sqrt_alphas_cumprod.shape == (1000,)

    def test_sqrt_one_minus_alphas_cumprod_shape(self, scheduler):
        assert scheduler.sqrt_one_minus_alphas_cumprod.shape == (1000,)


class TestScheduleRanges:
    def test_betas_in_range(self, scheduler):
        assert scheduler.betas.min() >= 0.0
        assert scheduler.betas.max() <= 1.0

    def test_alphas_in_range(self, scheduler):
        assert scheduler.alphas.min() >= 0.0
        assert scheduler.alphas.max() <= 1.0

    def test_alphas_cumprod_in_range(self, scheduler):
        assert scheduler.alphas_cumprod.min() >= 0.0
        assert scheduler.alphas_cumprod.max() <= 1.0

    def test_linear_beta_start(self):
        scheduler = NoiseScheduler(timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2)
        assert torch.isclose(scheduler.betas[0], torch.tensor(1e-4, dtype=torch.float64), atol=1e-6)

    def test_linear_beta_end(self):
        scheduler = NoiseScheduler(timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2)
        assert torch.isclose(scheduler.betas[-1], torch.tensor(2e-2, dtype=torch.float64), atol=1e-6)


class TestScheduleMonotonicity:
    def test_betas_monotonically_increasing(self, scheduler):
        # betas should increase over time (more noise added later)
        assert (scheduler.betas[1:] >= scheduler.betas[:-1]).all()

    def test_alphas_cumprod_monotonically_decreasing(self, scheduler):
        # signal is progressively destroyed
        assert (scheduler.alphas_cumprod[1:] <= scheduler.alphas_cumprod[:-1]).all()

    def test_alphas_cumprod_starts_near_one(self, scheduler):
        assert scheduler.alphas_cumprod[0] > 0.99

    def test_alphas_cumprod_ends_near_zero(self, scheduler):
        assert scheduler.alphas_cumprod[-1] < 0.02


class TestQSample:
    @pytest.fixture
    def x0(self):
        torch.manual_seed(0)
        return torch.randn(4, 1, 64, 64)

    def test_output_shape(self, scheduler, x0):
        t = torch.randint(0, 1000, (4,))
        xt, noise = scheduler.q_sample(x0, t)
        assert xt.shape == x0.shape
        assert noise.shape == x0.shape

    def test_output_dtype(self, scheduler, x0):
        t = torch.randint(0, 1000, (4,))
        xt, noise = scheduler.q_sample(x0, t)
        assert xt.dtype == torch.float32
        assert noise.dtype == torch.float32

    def test_custom_noise_used(self, scheduler, x0):
        t = torch.randint(0, 1000, (4,))
        noise = torch.zeros_like(x0)
        xt, returned_noise = scheduler.q_sample(x0, t, noise=noise)
        assert torch.equal(returned_noise, noise)

    def test_t0_close_to_x0(self, x0):
        # at t=0, very little noise should be added
        scheduler = NoiseScheduler(timesteps=1000, beta_schedule="linear")
        t = torch.zeros(4, dtype=torch.long)
        noise = torch.zeros_like(x0)
        xt, _ = scheduler.q_sample(x0, t, noise=noise)
        assert torch.allclose(xt, x0, atol=1e-3)

    def test_large_t_is_noisy(self, scheduler, x0):
        # at t=T-1, image should be close to pure noise
        t = torch.full((4,), 999, dtype=torch.long)
        xt, _ = scheduler.q_sample(x0, t)
        # correlation with original should be low
        assert not torch.allclose(xt, x0, atol=0.1)


class TestInvalidSchedule:
    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown beta_schedule"):
            NoiseScheduler(beta_schedule="invalid")
