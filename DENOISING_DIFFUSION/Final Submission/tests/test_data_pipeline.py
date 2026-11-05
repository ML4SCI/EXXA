"""
Unit tests for data loading and preprocessing utilities.

Run with: pytest tests/test_data_pipeline.py -v
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from astropy.io import fits

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import AstroDataset, create_dataloaders
from data.preprocessing import (
    normalize_image,
    denormalize_image,
    data_transform,
    inverse_data_transform,
    pad_to_multiple,
    unpad_image,
)
from data.augmentation import (
    RandomRotation90,
    AddGaussianNoise,
    RandomIntensityScale,
    get_augmentation_transforms,
)


class TestAstroDataset:
    """Test suite for AstroDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dirty = np.random.randn(10, 64, 64).astype(np.float32)
        clean = np.random.randn(10, 64, 64).astype(np.float32)
        return dirty, clean
    
    def test_dataset_creation(self, sample_data):
        """Test creating a dataset."""
        dirty, clean = sample_data
        dataset = AstroDataset(
            dirty, clean, patch_size=64, n=4, parse_patches=True
        )
        assert len(dataset) == 10
    
    def test_dataset_getitem_with_patches(self, sample_data):
        """Test loading an item with patch extraction."""
        dirty, clean = sample_data
        dataset = AstroDataset(
            dirty, clean, patch_size=32, n=4, parse_patches=True
        )
        
        img, img_id = dataset[0]
        
        # Should return (n_patches, 2_channels, H, W)
        assert img.shape == (4, 2, 32, 32)
        assert img_id == "00000"
        assert isinstance(img, torch.Tensor)
    
    def test_dataset_getitem_without_patches(self, sample_data):
        """Test loading an item without patch extraction."""
        dirty, clean = sample_data
        dataset = AstroDataset(
            dirty, clean, patch_size=64, n=1, parse_patches=False
        )
        
        img, img_id = dataset[0]
        
        # Should return (2_channels, H, W) - resized to multiple of 16
        assert img.shape[0] == 2  # 2 channels (input + gt)
        assert img.shape[1] % 16 == 0  # Height multiple of 16
        assert img.shape[2] % 16 == 0  # Width multiple of 16
    
    def test_dataset_shape_mismatch(self):
        """Test that mismatched shapes raise error."""
        dirty = np.random.randn(10, 64, 64)
        clean = np.random.randn(10, 32, 32)  # Different shape!
        
        with pytest.raises(AssertionError):
            AstroDataset(dirty, clean, patch_size=64, n=4)
    
    def test_random_crops(self):
        """Test random crop generation."""
        i_list, j_list, th, tw = AstroDataset.get_params(128, 128, (64, 64), n=4)
        
        assert len(i_list) == 4
        assert len(j_list) == 4
        assert th == 64
        assert tw == 64
        assert all(0 <= i <= 128 - 64 for i in i_list)
        assert all(0 <= j <= 128 - 64 for j in j_list)


class TestDataloaders:
    """Test suite for dataloader creation."""
    
    @pytest.fixture
    def sample_split_data(self):
        """Create sample train/val split."""
        np.random.seed(42)
        X_train = np.random.randn(80, 64, 64).astype(np.float32)
        y_train = np.random.randn(80, 64, 64).astype(np.float32)
        X_val = np.random.randn(20, 64, 64).astype(np.float32)
        y_val = np.random.randn(20, 64, 64).astype(np.float32)
        return X_train, y_train, X_val, y_val
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "training": {"patch_n": 4, "batch_size": 4},
            "data": {"image_size": 64, "num_workers": 0},  # 0 workers for testing
        }
    
    def test_create_dataloaders(self, sample_split_data, sample_config):
        """Test creating train and validation dataloaders."""
        X_train, y_train, X_val, y_val = sample_split_data
        
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, sample_config, parse_patches=True
        )
        
        assert len(train_loader.dataset) == 80
        assert len(val_loader.dataset) == 20
        
        # Test loading a batch
        batch, ids = next(iter(train_loader))
        assert batch.shape[0] == 4  # Batch size
        assert batch.shape[1] == 4  # n patches
        assert batch.shape[2] == 2  # 2 channels (input + gt)


class TestPreprocessing:
    """Test suite for preprocessing functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        np.random.seed(42)
        return np.random.randn(80, 80).astype(np.float32)
    
    def test_normalize_minmax(self, sample_image):
        """Test min-max normalization."""
        normalized = normalize_image(sample_image, method="minmax")
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert np.isclose(normalized.min(), 0.0, atol=1e-6)
        assert np.isclose(normalized.max(), 1.0, atol=1e-6)
    
    def test_normalize_percentile(self, sample_image):
        """Test percentile-based normalization."""
        normalized = normalize_image(sample_image, method="percentile")
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_normalize_zscore(self, sample_image):
        """Test z-score normalization."""
        normalized = normalize_image(sample_image, method="zscore")
        
        assert np.isclose(normalized.mean(), 0.0, atol=1e-6)
        assert np.isclose(normalized.std(), 1.0, atol=1e-6)
    
    def test_denormalize(self, sample_image):
        """Test denormalization reverses normalization."""
        vmin, vmax = sample_image.min(), sample_image.max()
        normalized = normalize_image(sample_image, method="minmax")
        denormalized = denormalize_image(normalized, vmin, vmax)
        
        assert np.allclose(sample_image, denormalized, atol=1e-5)
    
    def test_data_transform(self):
        """Test [0,1] to [-1,1] transformation."""
        X = torch.tensor([0.0, 0.5, 1.0])
        X_transformed = data_transform(X)
        
        expected = torch.tensor([-1.0, 0.0, 1.0])
        assert torch.allclose(X_transformed, expected)
    
    def test_inverse_data_transform(self):
        """Test [-1,1] to [0,1] transformation."""
        X = torch.tensor([-1.0, 0.0, 1.0])
        X_transformed = inverse_data_transform(X)
        
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(X_transformed, expected)
    
    def test_pad_to_multiple(self, sample_image):
        """Test padding to multiple of 16."""
        # Create image not multiple of 16
        img = sample_image[:63, :67]  # (63, 67)
        
        padded, padding = pad_to_multiple(img, multiple=16)
        
        # Check dimensions are multiples of 16
        assert padded.shape[0] % 16 == 0
        assert padded.shape[1] % 16 == 0
        assert padded.shape == (64, 80)
        
        # Check padding amounts
        pad_top, pad_bottom, pad_left, pad_right = padding
        assert pad_top + pad_bottom == 1
        assert pad_left + pad_right == 13
    
    def test_unpad_image(self, sample_image):
        """Test unpadding reverses padding."""
        img = sample_image[:63, :67]
        padded, padding = pad_to_multiple(img, multiple=16)
        unpadded = unpad_image(padded, padding)
        
        assert np.array_equal(img, unpadded)


class TestAugmentation:
    """Test suite for augmentation functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        np.random.seed(42)
        return np.random.randn(64, 64).astype(np.float32)
    
    def test_random_rotation_90(self, sample_image):
        """Test 90-degree rotation."""
        transform = RandomRotation90()
        
        # Set seed for reproducibility
        np.random.seed(42)
        rotated = transform(sample_image)
        
        # Check shape is preserved
        assert rotated.shape == sample_image.shape
        
        # Check different rotations produce different results
        results = [transform(sample_image) for _ in range(10)]
        # At least some should be different
        differences = sum(not np.array_equal(r, sample_image) for r in results)
        assert differences > 0
    
    def test_add_gaussian_noise(self, sample_image):
        """Test Gaussian noise addition."""
        transform = AddGaussianNoise(sigma_range=(0.1, 0.1))  # Fixed sigma
        
        noisy = transform(sample_image)
        
        # Check shape is preserved
        assert noisy.shape == sample_image.shape
        
        # Check noise was added
        assert not np.array_equal(noisy, sample_image)
        
        # Check noise level is reasonable
        noise = noisy - sample_image
        assert np.abs(noise.std() - 0.1) < 0.05  # Should be close to sigma
    
    def test_random_intensity_scale(self, sample_image):
        """Test intensity scaling."""
        transform = RandomIntensityScale(scale_range=(1.5, 1.5))  # Fixed scale
        
        scaled = transform(sample_image)
        
        # Check shape is preserved
        assert scaled.shape == sample_image.shape
        
        # Check scaling was applied
        ratio = scaled / (sample_image + 1e-8)
        assert np.isclose(ratio.mean(), 1.5, atol=0.1)
    
    def test_augmentation_pipeline(self):
        """Test complete augmentation pipeline."""
        transform = get_augmentation_transforms(augment=True, image_size=64)
        
        img = np.random.randn(64, 64).astype(np.float32)
        img_aug = transform(img)
        
        # Check output is tensor
        assert isinstance(img_aug, torch.Tensor)
        
        # Check shape
        assert img_aug.shape == (64, 64)


class TestFITSLoader:
    """Test suite for FITS file loading."""
    
    @pytest.fixture
    def temp_fits_file(self):
        """Create temporary FITS file for testing."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        filepath = Path(temp_dir) / "test.fits"
        
        # Create simple FITS file
        data = np.random.randn(64, 64).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(filepath)
        
        yield filepath
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_load_fits_file(self, temp_fits_file):
        """Test loading a FITS file."""
        from data.fits_loader import load_fits_file
        
        data, header = load_fits_file(temp_fits_file)
        
        assert isinstance(data, np.ndarray)
        assert isinstance(header, fits.Header)
        assert data.shape == (64, 64)
        assert data.dtype == np.float32
    
    def test_fits_loader_normalization(self, temp_fits_file):
        """Test FITS loader with normalization."""
        from data.fits_loader import FITSLoader
        
        loader = FITSLoader(normalize=True)
        data, header = loader.load(temp_fits_file)
        
        # Should be normalized to [0, 1]
        assert data.min() >= 0.0
        assert data.max() <= 1.0
    
    def test_fits_loader_resize(self, temp_fits_file):
        """Test FITS loader with resizing."""
        from data.fits_loader import FITSLoader
        
        loader = FITSLoader(target_shape=(128, 128))
        data, header = loader.load(temp_fits_file)
        
        assert data.shape == (128, 128)
    
    def test_fits_nonexistent_file(self):
        """Test loading nonexistent FITS file raises error."""
        from data.fits_loader import load_fits_file
        
        with pytest.raises(FileNotFoundError):
            load_fits_file("nonexistent.fits")


# Performance tests
class TestPerformance:
    """Test suite for performance and edge cases."""
    
    def test_large_dataset(self):
        """Test handling large dataset."""
        # Create moderately large dataset
        dirty = np.random.randn(1000, 64, 64).astype(np.float32)
        clean = np.random.randn(1000, 64, 64).astype(np.float32)
        
        dataset = AstroDataset(
            dirty, clean, patch_size=64, n=4, parse_patches=True
        )
        
        assert len(dataset) == 1000
        
        # Test accessing random indices
        for idx in [0, 500, 999]:
            img, img_id = dataset[idx]
            assert img.shape == (4, 2, 64, 64)
    
    def test_edge_case_single_value(self):
        """Test handling constant images."""
        # Create constant images
        dirty = np.ones((10, 64, 64), dtype=np.float32)
        clean = np.ones((10, 64, 64), dtype=np.float32)
        
        normalized = normalize_image(dirty[0], method="minmax")
        
        # Should return zeros (or handle gracefully)
        assert np.all(normalized == 0.0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
