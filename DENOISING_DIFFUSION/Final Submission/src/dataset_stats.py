import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    # Setup paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')
    stats_dir = os.path.join(root_dir, 'results', 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    clean_path = os.path.join(data_dir, 'clean.npy')
    dirty_path = os.path.join(data_dir, 'dirty.npy')

    print("Loading data...")
    clean = np.load(clean_path)
    dirty = np.load(dirty_path).astype(np.float32)

    # Print stats
    print("-" * 40)
    print("Dataset Statistics")
    print("-" * 40)
    for name, arr in [("Clean", clean), ("Dirty", dirty)]:
        print(f"{name}:")
        print(f"  Samples: {arr.shape[0]}")
        print(f"  Size:    {arr.shape[1]}x{arr.shape[2]}")
        print(f"  dtype:   {arr.dtype}")
        print(f"  Min:     {arr.min():.4f}")
        print(f"  Max:     {arr.max():.4f}")
        print(f"  Mean:    {arr.mean():.4f}")
        print(f"  Std:     {arr.std():.4f}")
        print("-" * 40)

    # 1. sample_grid.png
    print("Generating sample_grid.png...")
    np.random.seed(42)
    sample_ids = np.random.choice(len(clean), 3, replace=False)
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    for i, idx in enumerate(sample_ids):
        axes[i, 0].imshow(clean[idx], cmap='inferno')
        axes[i, 0].set_title(f'Clean #{idx}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(dirty[idx], cmap='inferno')
        axes[i, 1].set_title(f'Dirty #{idx}')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'sample_grid.png'), dpi=150)
    plt.close()

    # 2. pixel_distribution.png
    print("Generating pixel_distribution.png...")
    dist_ids = np.random.choice(len(clean), 100, replace=False)
    clean_sample = clean[dist_ids].flatten()
    dirty_sample = dirty[dist_ids].flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(clean_sample, bins=100, alpha=0.5, label='Clean', density=True)
    plt.hist(dirty_sample, bins=100, alpha=0.5, label='Dirty', density=True)
    plt.title('Pixel Value Distribution (100 random samples)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(stats_dir, 'pixel_distribution.png'), dpi=150)
    plt.close()

    # 3. mean_images.png
    print("Generating mean_images.png...")
    mean_clean = np.mean(clean, axis=0)
    mean_dirty = np.mean(dirty, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].imshow(mean_clean, cmap='inferno')
    axes[0].set_title('Mean Clean Image')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mean_dirty, cmap='inferno')
    axes[1].set_title('Mean Dirty Image')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'mean_images.png'), dpi=150)
    plt.close()

    # 4. noise_difference.png
    print("Generating noise_difference.png...")
    mean_diff = np.mean(np.abs(dirty - clean), axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(mean_diff, cmap='inferno')
    plt.title('Mean Absolute Difference (Dirty - Clean)')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(stats_dir, 'noise_difference.png'), dpi=150)
    plt.close()

    print("All 4 images saved to results/stats/")

if __name__ == "__main__":
    main()
