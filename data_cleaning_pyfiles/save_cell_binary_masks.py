from scipy.ndimage import binary_fill_holes, label as ndi_label
import numpy as np
import os
import skimage
import tifffile
import torchvision.transforms.v2 as transforms_v2
import qpi_seg.train.split_mask_5_channels as split
import torch

combined_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'
cell_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Cell_mask_cleaned'
mask_files = sorted(os.listdir(combined_mask_folder))

from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))
for idx in range(len(mask_files)):
    combined_mask=skimage.io.imread(os.path.join(combined_mask_folder,mask_files[idx]))
    cell_mask_cleaned=combined_mask>0
    tifffile.imwrite(
        os.path.join(cell_mask_folder, mask_files[idx]),
        cell_mask_cleaned.astype(bool)
    )