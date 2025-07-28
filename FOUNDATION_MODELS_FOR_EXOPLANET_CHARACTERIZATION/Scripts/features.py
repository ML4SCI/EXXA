import numpy as np
from astropy.io import fits
from image_augmentations import transform_mae
from tqdm.auto import tqdm
from glob import glob
import os

def get_patchwise_radial_feats(image, 
                               patch_size=16, 
                               dr=5, 
                               normalize=True):
    
    if hasattr(image, 'detach'):
        img_np = image.squeeze().detach().cpu().numpy()
    else:
        img_np = image.squeeze()
    H, W = img_np.shape
    assert H == W, "This assumes square images"
    cx, cy = W//2, H//2
    
   
    y_idx, x_idx = np.indices((H, W))
    r_array     = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    theta_array = np.arctan2(y_idx - cy, x_idx - cx)
    
    
    r_max     = min(cx, cy)
    r_values  = np.arange(0, r_max, dr)
    n_bins    = len(r_values)
    
    
    annulus_masks = [
        (r_array >= r) & (r_array < r + dr)
        for r in r_values
    ]
    

    n_patches_x = W // patch_size
    n_patches_y = H // patch_size
    feats = []
    
    for py in range(n_patches_y):
        for px in range(n_patches_x):
            
            y0, y1 = py*patch_size, (py+1)*patch_size
            x0, x1 = px*patch_size, (px+1)*patch_size
            patch_mask = np.zeros_like(img_np, dtype=bool)
            patch_mask[y0:y1, x0:x1] = True
            
           
            mass_list, density_list, grad_list = [], [], []
            
            for bin_idx, ann_mask in enumerate(annulus_masks):
                
                mask = ann_mask & patch_mask
                if not mask.any():
                    
                    mass_list.append(0.)
                    density_list.append(0.)
                    grad_list.append(0.)
                    continue
                
                vals = img_np[mask]
                mass = vals.sum()
                mass_list.append(mass)
                
                
                ideal_area = np.pi * ((r_values[bin_idx] + dr)**2
                                     - r_values[bin_idx]**2)
                density_list.append(mass / (ideal_area + 1e-8))
                
                
                angles = theta_array[mask]
                X = np.sum(vals * np.cos(angles))
                Y = np.sum(vals * np.sin(angles))
                grad_list.append((np.hypot(X, Y) / (mass + 1e-8)))
            
            
            patch_feats = np.concatenate([mass_list,
                                          density_list,
                                          grad_list])
            if normalize:
                norm = np.linalg.norm(patch_feats) + 1e-8
                patch_feats = patch_feats / norm
            feats.append(patch_feats)
    
    feats = np.stack(feats, axis=0)
    #(n_patches_y * n_patches_x, 3 * n_bins)
    return feats


def get_patchwise_elliptical_feats(
    image,
    patch_size=16,
    dr=5,
    normalize=True
):
    if hasattr(image, 'detach'):
        img = image.squeeze().detach().cpu().numpy()
    else:
        img = image.squeeze()
    H, W = img.shape
    assert H == W, "This assumes square images"
    
    cx, cy = W//2, H//2
    a, b = W/2.0, H/2.0
    
    y_idx, x_idx = np.indices((H, W))
    
    re = np.sqrt(((x_idx - cx)/a)**2 + ((y_idx - cy)/b)**2)
    theta = np.arctan2(y_idx - cy, x_idx - cx)
    
    dr_norm = dr / min(a, b)
    r_vals = np.arange(0.0, 1.0, dr_norm)
    n_bins = len(r_vals)
    
    ann_masks = [
        (re >= r) & (re < r + dr_norm)
        for r in r_vals
    ]
    
    n_px = W // patch_size
    n_py = H // patch_size
    
    feats = []
    for py in range(n_py):
        for px in range(n_px):
            y0, y1 = py*patch_size, (py+1)*patch_size
            x0, x1 = px*patch_size, (px+1)*patch_size
            
            # patch mask
            pm = np.zeros_like(img, dtype=bool)
            pm[y0:y1, x0:x1] = True
            
            mass_list, dens_list, grad_list = [], [], []
            
            for i, ann in enumerate(ann_masks):
                m = ann & pm
                if not m.any():
                    mass_list.append(0.)
                    dens_list.append(0.)
                    grad_list.append(0.)
                    continue
                
                vals = img[m]
                mass = vals.sum()
                mass_list.append(mass)
                
                # area of elliptical ring = π a b [(r+dr)^2 - r^2]
                area = np.pi * a * b * (
                    (r_vals[i] + dr_norm)**2 - r_vals[i]**2
                )
                dens_list.append(mass / (area + 1e-8))
                
                angs = theta[m]
                X = np.sum(vals * np.cos(angs))
                Y = np.sum(vals * np.sin(angs))
                grad_list.append(np.hypot(X, Y) / (mass + 1e-8))
            
            pf = np.concatenate([mass_list, dens_list, grad_list])
            if normalize:
                pf = pf / (np.linalg.norm(pf) + 1e-8)
            feats.append(pf)
    
    return np.stack(feats, axis=0)

def extract_feats(image_folder_path):
    image_paths = sorted(glob(os.path.join(image_folder_path, "*.fits")))
    combined_feats = {}
    for p in tqdm(image_paths):
        img = fits.getdata(p)
        img = np.array(img).astype(np.float32)
        img = img.squeeze()
        #plt.imshow(img)
        lower_percentile = np.percentile(img, 1)  # 1st percentile
        upper_percentile = np.percentile(img, 99)  # 99th percentile
        img = np.clip(img, lower_percentile, upper_percentile)
        img = transform_mae(image=img)['image']
        #plt.imshow(img.squeeze())
        
        rad_feats = get_patchwise_radial_feats(img, dr=2)
        eli_feats = get_patchwise_elliptical_feats(img, dr=2)
        feats = np.concatenate([rad_feats, eli_feats], axis=-1)
        feats = feats.astype(np.float32)
        #print(feats.shape)
        combined_feats[p] = feats
    return combined_feats

