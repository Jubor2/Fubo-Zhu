# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 14:02:02 2025

@author: zhufu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated_analysis.py

Integrated workflow:
1) Batch-process TIFF EL images to save raw brightness arrays and quick "temperature-like" previews.
2) Build a full-image Rse map with per-pixel CJ (C/J0) estimation.
3) Smooth the Rse map, plot histogram and pseudocolor heatmap.
4) Do regional UT*(dφ/dU)/φ vs φ fits on 4 preset ROIs.
5) Save original TIFFs with coordinate grids.

Deps: numpy, scipy, tifffile, matplotlib, re
"""

import os
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ---------- Parameters ----------
# Physical constants & experiment settings
pixel_size = 120e-6       # [m]
pixel_area = pixel_size**2
T = 298.15                # [K]
kB = 1.380649e-23         # [J/K]
q = 1.602176634e-19       # [C]
UT = kB * T / q           # ~25.7 mV

# Four ROI boxes: (x0, y0, x1, y1)
region_coords = [
    (100,  50, 200, 150),  # ROI 1
    (300,  50, 400, 150),  # ROI 2
    (100, 200, 200, 300),  # ROI 3
    (300, 200, 400, 300),  # ROI 4 (φ will be ×5 in plotting)
]
region_labels = ['Region A', 'Region B', 'Region C', 'Region D (×5)']

# Output paths
folder       = os.path.dirname(os.path.abspath(__file__))
input_dir    = folder
out_bright   = os.path.join(folder, 'panel_brightness')
out_imgs     = os.path.join(folder, 'panel_temp_maps')
out_rse_tif  = os.path.join(folder, 'panel_Rse_map.tif')
out_rse_npy  = os.path.join(folder, 'panel_Rse_map.npy')
out_hist     = os.path.join(folder, 'Rse_analysis')
out_region   = os.path.join(folder, 'region_analysis')
out_coords   = os.path.join(folder, 'coordinate_plots')

for d in [out_bright, out_imgs, out_hist, out_region, out_coords]:
    os.makedirs(d, exist_ok=True)

# ---------- STEP 1: Read all TIFFs & stack ----------
# 1.1 List TIFF files (natural sort by filename)
file_list = sorted(
    f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))
)
if not file_list:
    raise RuntimeError("No .tif/.tiff files found in current directory.")

# 1.2 Load each frame; save brightness npy and quick pseudo-color PNG
frames = []
for fn in file_list:
    path = os.path.join(input_dir, fn)
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim > 2:
        # For multi-channel or multi-page, take the first channel/page by default
        img = img[..., 0]
    frames.append(img)

    name = os.path.splitext(fn)[0]
    np.save(os.path.join(out_bright, f'{name}.npy'), img)

    # Pseudo-color preview
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    norm = (img - vmin) / (vmax - vmin + 1e-12)
    plt.figure(figsize=(4, 4))
    plt.imshow(norm, cmap='hot', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(out_imgs, f'{name}.png'), dpi=200)
    plt.close()

stack = np.stack(frames, axis=0)  # shape = (K, H, W)
K, H, W = stack.shape

# ---------- STEP 2: Parse U/I from filenames & sort by U ----------
U_list, I_list = [], []
for fn in file_list:
    name = os.path.splitext(fn)[0]
    parts = name.split()
    if len(parts) >= 2:
        # Filename like: "<U> <I>" (e.g., "0.618 1.23")
        u = float(parts[0])
        i = float(parts[1])
    else:
        # Only voltage is present; set I as NaN (no CJ update from this frame)
        u = float(parts[0])
        i = np.nan
    U_list.append(u)
    I_list.append(i)

U = np.array(U_list, dtype=np.float64)
I = np.array(I_list, dtype=np.float64) * 0.9 * 0.9  # optional correction factor

order = np.argsort(U)
imgs = stack[order]       # (K, H, W), sorted by increasing U
U    = U[order]
I    = I[order]

# ---------- STEP 3: dφ/dU and per-pixel CJ, then Rse_map ----------
# 3.1 Central-difference dφ/dU for each frame (pixel-wise)
dphi = np.zeros_like(imgs, dtype=np.float32)
for k in range(K):
    if k == 0:
        dphi[k] = (imgs[1] - imgs[0]) / (U[1] - U[0] + 1e-12)
    elif k == K - 1:
        dphi[k] = (imgs[-1] - imgs[-2]) / (U[-1] - U[-2] + 1e-12)
    else:
        dphi[k] = (imgs[k + 1] - imgs[k - 1]) / (U[k + 1] - U[k - 1] + 1e-12)

# 3.2 Compute per-pixel CJ (C/J0) using all frames with valid current:
#     CJ_{k,i,j} = phi_{k,i,j} * pixel_area / I_k
#     Aggregate across k by median to be robust to outliers.
CJ_map = np.full((H, W), np.nan, dtype=np.float32)

valid_k = np.where(np.isfinite(I) & (I != 0))[0]
if valid_k.size == 0:
    raise RuntimeError("No valid current values parsed from filenames; cannot build per-pixel CJ.")

# Build a stack of CJ_k for valid frames only to save memory
# shape -> (K_valid, H, W)
CJ_stack = np.empty((len(valid_k), H, W), dtype=np.float32)
for idx, k in enumerate(valid_k):
    CJ_stack[idx] = (imgs[k] * pixel_area) / (I[k] + 1e-20)

# Median across frames; you can change to np.mean if preferred
CJ_map = np.median(CJ_stack, axis=0)

# Optional: mask obviously non-physical values (e.g., negative or inf)
CJ_map[~np.isfinite(CJ_map)] = np.nan
CJ_map[CJ_map <= 0] = np.nan

# 3.3 Pixel-wise linear regression to obtain Rse_map:
#     For each pixel (i,j), regress:
#       x = dφ/dU
#       y = UT * ( (dφ/dU) / φ )
#     linregress returns slope (m) and intercept (a).
#     Use formula: Rse(i,j) = CJ_map(i,j) * m / a
Rse_map = np.full((H, W), np.nan, dtype=np.float32)

for i in range(H):
    # Vectorize along columns for a small speed-up if desired; here keep it explicit/clear
    for j in range(W):
        phi_ij  = imgs[:, i, j]
        phip_ij = dphi[:, i, j]
        # Valid samples: positive phi and finite derivative
        mask = (phi_ij > 0) & np.isfinite(phi_ij) & np.isfinite(phip_ij)
        if np.count_nonzero(mask) < 3:
            continue

        x = phip_ij[mask]
        y = UT * (phip_ij[mask] / (phi_ij[mask] + 1e-20))
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue

        m, a, _, _, _ = linregress(x, y)
        if not (np.isfinite(m) and np.isfinite(a)) or a == 0:
            continue

        cj = CJ_map[i, j]
        if not np.isfinite(cj):
            continue

        Rse_map[i, j] = (cj * m) / a  # [Ω·m²]

# 3.4 Unit conversion: Ω·m² → Ω·cm²
Rse_map *= 1e4

# Save Rse outputs
tifffile.imsave(out_rse_tif, Rse_map.astype(np.float32))
np.save(out_rse_npy, Rse_map)
print("✔ Rse_map computed and saved.")

# ---------- STEP 4: Smoothing & plotting ----------
# 4.1 Simple 8-neighborhood mean smoothing (reflect padding)
offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
rse_pad = np.pad(Rse_map, 1, mode='reflect')
nbr_stack = np.empty((len(offsets), H, W), dtype=np.float64)
for idx, (dx, dy) in enumerate(offsets):
    nbr_stack[idx] = rse_pad[1+dx:1+dx+H, 1+dy:1+dy+W]
rse_smooth = np.nanmean(nbr_stack, axis=0)

# 4.2 Averages
mean_all   = np.nanmean(rse_smooth)
rse_flat   = rse_smooth[~np.isnan(rse_smooth)]
rse_focus  = rse_flat[(rse_flat >= 0) & (rse_flat <= 5)]
mean_focus = np.mean(rse_focus)
print(f"Average Rse (all): {mean_all:.4f} Ω·cm²")
print(f"Average Rse [0,5]:  {mean_focus:.4f} Ω·cm²")

# 4.3 Histogram (focus range)
plt.figure(figsize=(8, 5))
plt.hist(rse_focus, bins=100, edgecolor='black')
plt.xlabel('R$_{se}$ (Ω·cm²)')
plt.ylabel('Pixel Count')
plt.title('Histogram of Smoothed R$_{se}$ in [0,5]')
plt.tight_layout()
plt.savefig(os.path.join(out_hist, 'Rse_histogram.png'), dpi=300)
plt.close()

# 4.4 Pseudocolor heatmap (clip negatives to 0.6 as requested)
rse_heat = rse_smooth.copy()
rse_heat[~np.isfinite(rse_heat)] = 0.6
rse_heat[rse_heat < 0] = 0.6
vmin, vmax = 0.6, 3.0
rse_heat = np.clip(rse_heat, vmin, vmax)

plt.figure(figsize=(10, 8))
im = plt.imshow(rse_heat, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.title('Pseudocolor Heatmap of Smoothed R$_{se}$')
cb = plt.colorbar(im, label='R$_{se}$ (Ω·cm²)')
plt.tight_layout()
plt.savefig(os.path.join(out_hist, 'Rse_heatmap.png'), dpi=300)
plt.close()

# ---------- STEP 5: Regional analysis ----------
n_reg = len(region_coords)
phi   = np.zeros((n_reg, K))
for idx, (x0, y0, x1, y1) in enumerate(region_coords):
    sub = imgs[:, y0:y1, x0:x1]  # (K, h, w)
    phi[idx] = np.nanmean(sub.reshape(K, -1), axis=1)

# Central-difference on regional means
dphi_dU = np.gradient(phi, U, axis=1)
y_mat   = UT * (dphi_dU / (phi + 1e-20))

plt.figure(figsize=(8, 6))
markers = ['o', 's', '^', 'D']
for idx in range(n_reg):
    xi = phi[idx].copy()
    if idx == 3:
        xi *= 5  # scale φ for the 4th ROI to improve readability
    yi = y_mat[idx]
    mask = np.isfinite(xi) & np.isfinite(yi)
    if np.count_nonzero(mask) < 3:
        continue

    slope, intercept, _, _, _ = linregress(xi[mask], yi[mask])
    plt.scatter(xi, yi, marker=markers[idx], s=50, label=f"{region_labels[idx]}")
    xfit = np.linspace(np.nanmin(xi[mask]), np.nanmax(xi[mask]), 100)
    plt.plot(xfit, slope * xfit + intercept, '-', label=f"Fit {region_labels[idx]}: slope={slope:.2e}")

plt.xlabel(r'$\phi$ (mean EL intensity)')
plt.ylabel(r'$U_T\,\frac{d\phi/dU}{\phi}$')
plt.title('Local R$_{se}$ Extraction: $UT(\phi\'/\phi)$ vs $\phi$')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_region, 'region_Rse_fits.png'), dpi=300)
plt.close()
print("✔ Regional analysis plots saved.")

# ---------- STEP 6: Save TIFFs with coordinate grids ----------
for fn in file_list:
    path = os.path.join(input_dir, fn)
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim > 2:
        img = img[..., 0]
    h, w = img.shape

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray', origin='upper')
    ax.set_title(f"{fn} — {w}×{h} px")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    # 10-division grid ticks
    xt = list(range(0, w + 1, max(1, w // 10)))
    yt = list(range(0, h + 1, max(1, h // 10)))
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.grid(True, linestyle='--', linewidth=0.5, color='white')

    plt.tight_layout()
    fig.savefig(os.path.join(out_coords, f"{os.path.splitext(fn)[0]}_coords.png"), dpi=200)
    plt.close(fig)

print("✔ All steps finished. Outputs:")
print(f"  • Brightness npy / previews: {out_bright} / {out_imgs}")
print(f"  • Rse TIFF / NPY:            {out_rse_tif} / {out_rse_npy}")
print(f"  • Rse analysis plots:        {out_hist}")
print(f"  • Regional analysis plots:   {out_region}")
print(f"  • Coordinate images:         {out_coords}")
