# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:30:26 2025

@author: zhufu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from glob import glob

def extract_and_rename_tifs(root_dir='.', output_dir='extracted_tifs'):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Traverse all entries in the current directory
    for entry in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, entry)
        # Only process subdirectories
        if os.path.isdir(dir_path):
            # Find all tif/tiff files in the subfolder
            tifs = glob(os.path.join(dir_path, '*.tif')) + glob(os.path.join(dir_path, '*.tiff'))
            if not tifs:
                print(f"[Skip] No tif file found in folder '{entry}'")
                continue
            # Take the first tif found
            src_path = tifs[0]
            # Build the destination file name, removing any problematic characters
            safe_name = entry.replace(os.sep, '_').strip()
            dst_name = f"{safe_name}.tif"
            dst_path = os.path.join(output_dir, dst_name)
            # Copy and rename
            shutil.copy2(src_path, dst_path)
            print(f"[Copied] {src_path} -> {dst_path}")

if __name__ == "__main__":
    # Script usage example: run in current directory, results stored in ./extracted_tifs
    extract_and_rename_tifs()
