import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener
import sys

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')
    stats_dir = os.path.join(root_dir, 'results', 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    from src.baselines import run_baselines

    clean_path = os.path.join(data_dir, 'clean.npy')
    dirty_path = os.path.join(data_dir, 'dirty.npy')

    print("Loading data...")
    clean = np.load(clean_path)
    dirty = np.load(dirty_path).astype(np.float32)

    idx = 42
    print(f"Processing sample index {idx}...")
    c = clean[idx]
    d = dirty[idx]

    # Apply filters
    g = gaussian_filter(d, sigma=2)
    m = median_filter(d, size=3)
    w = wiener(d, (5, 5))
    diff = np.abs(c - g)

    # 1. baseline_visual.png
    print("Generating baseline_visual.png...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    images = [
        (c, 'Clean', 'gray'),
        (d, 'Noisy', 'gray'),
        (g, 'Gaussian (sigma=2)', 'gray'),
        (m, 'Median (3x3)', 'gray'),
        (w, 'Wiener (5x5)', 'gray'),
        (diff, 'Difference (Clean - Gaussian)', 'inferno')
    ]

    for ax, (img, title, cmap) in zip(axes, images):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        if cmap == 'inferno':
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'baseline_visual.png'), dpi=150)
    plt.close()

    # 2. baseline_metrics_chart.png
    print("Computing metrics for chart...")
    # Get metrics from run_baselines for 50 samples
    results = run_baselines(clean, dirty, n_samples=50)
    
    methods = list(results.keys())
    psnr_vals = [float(np.mean(results[m]['PSNR'])) for m in methods]
    ssim_vals = [float(np.mean(results[m]['SSIM'])) for m in methods]

    print("Generating baseline_metrics_chart.png...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.35

    color1 = '#4C9EEB'
    color2 = '#E8715A'

    rects1 = ax1.bar(x - width/2, psnr_vals, width, label='PSNR (dB)', color=color1)
    ax1.set_ylabel('PSNR (dB)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, ssim_vals, width, label='SSIM', color=color2)
    ax2.set_ylabel('SSIM', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Baseline Methods Comparison (Higher is better)')
    fig.tight_layout()
    
    # Add legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.savefig(os.path.join(stats_dir, 'baseline_metrics_chart.png'), dpi=150)
    plt.close()

    print("Both images saved to results/stats/")

if __name__ == "__main__":
    main()
