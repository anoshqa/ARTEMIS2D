from scipy.ndimage import binary_fill_holes, label as ndi_label
import numpy as np
import os
import skimage
import tifffile
import torchvision.transforms.v2 as transforms_v2
import qpi_seg.train.split_mask_5_channels as split
import torch

combined_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'
nucleus_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Nucleus_mask_cleaned'
mask_files = sorted(os.listdir(combined_mask_folder))

from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))
os.makedirs(nucleus_mask_folder, exist_ok=True)
for idx in range(len(mask_files)):
    combined_mask = skimage.io.imread(os.path.join(combined_mask_folder, mask_files[idx]))
    channeled_mask = split.split_into_channels(from_np(combined_mask))
    unstacked_masks = torch.unbind(channeled_mask, dim=0)
    nucleus_mask = unstacked_masks[2].squeeze().numpy()
    nucleus_mask_cleaned = binary_fill_holes(nucleus_mask)
    nucleolus_mask = unstacked_masks[3].squeeze().numpy()
    
    combined_binary_mask = np.logical_or(
        nucleus_mask_cleaned.astype(bool),
        nucleolus_mask.astype(bool)
    )

    tifffile.imwrite(
        os.path.join(nucleus_mask_folder, mask_files[idx]),
        combined_binary_mask.astype(bool)
    )