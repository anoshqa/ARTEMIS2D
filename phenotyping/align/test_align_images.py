"""

For each (MIP, combined_mask) pair:
    1. Read the combined mask (semantic labels, e.g. 0=bg, 1=cell, 2=nucleus,
       3=nucleolus, 4=lipid) and the companion MIP intensity image.
    2. Binarize the combined mask (mask > 0 -> 1) -- this binary version is
       ONLY used to find the largest connected component / its centroid.
"""

import os
import numpy as np
import tifffile

import phenotyping.align.align_image as align

# --- Folders (edit these paths for your setup) ---
mip_folder = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\MIP"
combined_mask_folder = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Combined_mask"

save_path_mip = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\MIP_aligned"
save_path_mask = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Combined_mask_aligned"

os.makedirs(save_path_mip, exist_ok=True)
os.makedirs(save_path_mask, exist_ok=True)

mip_files = sorted(os.listdir(mip_folder))
mask_files = sorted(os.listdir(combined_mask_folder))


for jj, (mip_name, mask_name) in enumerate(zip(mip_files, mask_files)):

    mip = tifffile.imread(os.path.join(mip_folder, mip_name))
    combined_mask = tifffile.imread(os.path.join(combined_mask_folder, mask_name))

    # Binary mask used only to locate the largest component / centroid.
    binary_mask = (combined_mask > 0).astype(np.uint8)

    # Align
    aligned_combined_mask, aligned_mip = align.align_image_org(binary_mask, combined_mask, mip)

    tifffile.imwrite(
        os.path.join(save_path_mip, mip_name),
        aligned_mip.astype(mip.dtype),
    )
    tifffile.imwrite(
        os.path.join(save_path_mask, mask_name),
        aligned_combined_mask.astype(combined_mask.dtype),
    )