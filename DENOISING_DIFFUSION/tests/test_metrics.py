"""Tests for metrics: mse, psnr, ssim, compare_denoisers."""

import math

import numpy as np
import pytest

from src.utils.metrics import mse, psnr, ssim, compare_denoisers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

def clean_hw():
    return RNG.random((64, 64), dtype=np.float32)

def noisy_hw(img, sigma=0.1):
    return np.clip(img + RNG.normal(0, sigma, img.shape).astype(np.float32), 0, 1)


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

def test_mse_identical_images_is_zero():
    img = clean_hw()
    assert mse(img, img) == 0.0


def test_mse_positive_for_different_images():
    img = clean_hw()
    assert mse(noisy_hw(img), img) > 0.0


def test_mse_symmetric():
    a = clean_hw()
    b = noisy_hw(a)
    assert math.isclose(mse(a, b), mse(b, a), rel_tol=1e-6)


def test_mse_returns_float():
    img = clean_hw()
    assert isinstance(mse(img, img), float)


def test_mse_known_value():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    assert math.isclose(mse(a, b), 1.0)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def test_psnr_identical_images_is_inf():
    img = clean_hw()
    assert math.isinf(psnr(img, img))


def test_psnr_noisy_is_finite_positive():
    img = clean_hw()
    val = psnr(noisy_hw(img), img)
    assert math.isfinite(val) and val > 0


def test_psnr_higher_for_less_noise():
    img = clean_hw()
    low_noise = psnr(noisy_hw(img, sigma=0.01), img)
    high_noise = psnr(noisy_hw(img, sigma=0.2), img)
    assert low_noise > high_noise


def test_psnr_returns_float():
    img = clean_hw()
    assert isinstance(psnr(noisy_hw(img), img), float)


def test_psnr_data_range_affects_result():
    img = clean_hw()
    noisy = noisy_hw(img)
    assert psnr(noisy, img, data_range=1.0) != psnr(noisy, img, data_range=255.0)


def test_psnr_chw_input():
    img = RNG.random((1, 32, 32), dtype=np.float32)
    noisy = np.clip(img + 0.05, 0, 1).astype(np.float32)
    val = psnr(noisy, img)
    assert math.isfinite(val)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def test_ssim_identical_images_is_one():
    img = clean_hw()
    assert math.isclose(ssim(img, img), 1.0, abs_tol=1e-5)


def test_ssim_noisy_less_than_one():
    img = clean_hw()
    assert ssim(noisy_hw(img), img) < 1.0


def test_ssim_in_valid_range():
    img = clean_hw()
    val = ssim(noisy_hw(img), img)
    assert -1.0 <= val <= 1.0


def test_ssim_returns_float():
    img = clean_hw()
    assert isinstance(ssim(img, img), float)


def test_ssim_higher_for_less_noise():
    img = clean_hw()
    low = ssim(noisy_hw(img, sigma=0.01), img)
    high = ssim(noisy_hw(img, sigma=0.3), img)
    assert low > high


def test_ssim_chw_input():
    img = RNG.random((1, 32, 32), dtype=np.float32)
    val = ssim(img, img)
    assert math.isclose(val, 1.0, abs_tol=1e-5)


def test_ssim_bchw_input():
    img = RNG.random((2, 1, 32, 32), dtype=np.float32)
    val = ssim(img, img)
    assert math.isclose(val, 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# compare_denoisers
# ---------------------------------------------------------------------------

def test_compare_denoisers_returns_dict():
    img = clean_hw()
    noisy = noisy_hw(img)
    result = compare_denoisers(noisy, img, {"method_a": noisy_hw(img, 0.05)})
    assert isinstance(result, dict)


def test_compare_denoisers_includes_noisy_baseline():
    img = clean_hw()
    noisy = noisy_hw(img)
    result = compare_denoisers(noisy, img, {})
    assert "noisy" in result


def test_compare_denoisers_includes_all_keys():
    img = clean_hw()
    noisy = noisy_hw(img)
    result = compare_denoisers(noisy, img, {"gaussian": noisy_hw(img, 0.05), "ddpm": noisy_hw(img, 0.02)})
    assert "gaussian" in result and "ddpm" in result


def test_compare_denoisers_metric_keys():
    img = clean_hw()
    noisy = noisy_hw(img)
    result = compare_denoisers(noisy, img, {"a": noisy_hw(img)})
    for entry in result.values():
        assert set(entry.keys()) == {"psnr", "ssim", "mse"}


def test_compare_denoisers_perfect_output():
    img = clean_hw()
    noisy = noisy_hw(img)
    result = compare_denoisers(noisy, img, {"perfect": img})
    assert math.isinf(result["perfect"]["psnr"])
    assert math.isclose(result["perfect"]["ssim"], 1.0, abs_tol=1e-5)
    assert result["perfect"]["mse"] == 0.0


def test_compare_denoisers_better_method_higher_psnr():
    img = clean_hw()
    noisy = noisy_hw(img, sigma=0.2)
    result = compare_denoisers(noisy, img, {
        "weak": noisy_hw(img, sigma=0.15),
        "strong": noisy_hw(img, sigma=0.01),
    })
    assert result["strong"]["psnr"] > result["weak"]["psnr"]
