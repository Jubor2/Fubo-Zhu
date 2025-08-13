#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PIL import Image

def crop_manual(img, left, top, right, bottom):
    """
    Crop the image using manually specified coordinates.
    The parameters are in pixels, based on the original image size.
    """
    return img.crop((left, top, right, bottom))

if __name__ == "__main__":
    # Define the crop box — modify as needed for manual coordinates
    # Example: crop area from left=100 px, top=50 px, to right=600 px, bottom=450 px
    CROP_LEFT = 580
    CROP_TOP = 109
    CROP_RIGHT = 1351
    CROP_BOTTOM = 982

    folder = os.path.dirname(os.path.abspath(__file__))
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.tif', '.tiff')):
            path = os.path.join(folder, fname)
            img  = Image.open(path)
            cropped = crop_manual(img, CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)
            out_name = f"cropped_{fname}"
            cropped.save(os.path.join(folder, out_name))
            print("Saved:", out_name)
