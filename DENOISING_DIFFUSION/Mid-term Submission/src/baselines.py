import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def gaussian_denoise(img, sigma=1.0):
    return gaussian_filter(img, sigma=sigma)


def median_denoise(img, size=3):
    return median_filter(img, size=size)


def wiener_denoise(img, noise=None):
    return wiener(img, noise=noise)


def compute_metrics(clean, denoised):
    p = psnr(clean, denoised, data_range=1.0)
    s = ssim(clean, denoised, data_range=1.0)
    m = np.mean((clean - denoised) ** 2)
    return {"PSNR": round(p, 4), "SSIM": round(s, 4), "MSE": round(float(m), 6)}


def run_baselines(clean, dirty, n_samples=50):
    dirty = dirty.astype(np.float32)
    results = {
        "Noisy":      {"PSNR": [], "SSIM": [], "MSE": []},
        "Gaussian_s1": {"PSNR": [], "SSIM": [], "MSE": []},
        "Gaussian_s2": {"PSNR": [], "SSIM": [], "MSE": []},
        "Median_s3":   {"PSNR": [], "SSIM": [], "MSE": []},
        "Wiener":      {"PSNR": [], "SSIM": [], "MSE": []},
    }

    for i in range(n_samples):
        c = clean[i]
        d = dirty[i]

        for key, val in compute_metrics(c, d).items():
            results["Noisy"][key].append(val)
        for key, val in compute_metrics(c, gaussian_denoise(d, 1.0)).items():
            results["Gaussian_s1"][key].append(val)
        for key, val in compute_metrics(c, gaussian_denoise(d, 2.0)).items():
            results["Gaussian_s2"][key].append(val)
        for key, val in compute_metrics(c, median_denoise(d, 3)).items():
            results["Median_s3"][key].append(val)
        for key, val in compute_metrics(c, wiener_denoise(d)).items():
            results["Wiener"][key].append(val)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    print("\n--- Baseline Results (averaged over {} samples) ---".format(n_samples))
    print(f"{'Method':<15} {'PSNR':>10} {'SSIM':>10} {'MSE':>12}")
    print("-" * 50)
    for method, metrics in results.items():
        p = round(np.mean(metrics["PSNR"]), 4)
        s = round(np.mean(metrics["SSIM"]), 4)
        m = round(np.mean(metrics["MSE"]), 6)
        print(f"{method:<15} {p:>10} {s:>10} {m:>12}")

    return results


if __name__ == "__main__":
    import os
    import numpy as np

    # Resolve data/ relative to the DENOISING_DIFFUSION root (one level up from src/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # DENOISING_DIFFUSION/
    CLEAN_PATH = os.path.join(ROOT_DIR, "data", "clean.npy")
    DIRTY_PATH  = os.path.join(ROOT_DIR, "data", "dirty.npy")

    print(f"Loading data from: {ROOT_DIR}/data/")
    clean = np.load(CLEAN_PATH)
    dirty = np.load(DIRTY_PATH)
    print(f"clean: {clean.shape}  dirty: {dirty.shape}")
    print("Running baselines on 50 samples...")
    run_baselines(clean, dirty, n_samples=50)
