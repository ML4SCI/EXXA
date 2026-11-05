import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener
import os

def visualize_baselines(clean, dirty, idx=0, save_path="results/baseline_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    d = dirty[idx].astype(np.float32)
    c = clean[idx]

    methods = {
        "Clean (GT)": c,
        "Noisy Input": d,
        "Gaussian σ1": gaussian_filter(d, 1.0),
        "Gaussian σ2": gaussian_filter(d, 2.0),
        "Median 3x3": median_filter(d, 3),
        "Wiener": wiener(d).astype(np.float32),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (title, img) in zip(axes, methods.items()):
        ax.imshow(img, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(title, fontsize=13)
        ax.axis('off')

    plt.suptitle("Classical Denoising Baselines: Sample #{}".format(idx), fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    clean = np.load("data/clean.npy")
    dirty = np.load("data/dirty.npy")
    visualize_baselines(clean, dirty, idx=0)
