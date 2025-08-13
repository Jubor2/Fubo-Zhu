#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tifffile
import matplotlib.pyplot as plt

def show_image_with_coords(path):
    """
    Read a TIFF image, display it with plt.imshow,
    and show pixel coordinates and grid lines on the axes.
    """
    img = tifffile.imread(path)
    # If multi-channel or 3D, select one plane (e.g., img[...,0])
    if img.ndim > 2:
        img = img[..., 0]
    h, w = img.shape

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray', origin='upper')
    plt.title(f"{os.path.basename(path)}  —  size: {w}×{h} px")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    # Automatically generate ticks dividing the axis into 10 segments
    x_ticks = list(range(0, w + 1, max(1, w // 10)))
    y_ticks = list(range(0, h + 1, max(1, h // 10)))
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.grid(True, linestyle='--', linewidth=0.5, color='white')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    # Iterate over all TIFF files in the folder
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.tif', '.tiff')):
            path = os.path.join(folder, fname)
            show_image_with_coords(path)
