#this function will just use centercrop and tifffile to write images in 1344 x 1344 size
import tifffile
import torchvision.transforms.v2 as transforms_v2
import os
import torch

big_mip_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\MIP_stitched_align'
big_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\Mask_stitched_align'

output_mip_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\MIP_ALL'
output_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\MASK_ALL'


transform = transforms_v2.CenterCrop((1344,1344))
from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))
mip_files=os.listdir(big_mip_folder)
mask_files=os.listdir(big_mask_folder)
mips=[from_np(tifffile.imread(os.path.join(big_mip_folder, file))) for file in mip_files]
masks=[from_np(tifffile.imread(os.path.join(big_mask_folder, file))) for file in mask_files]

masks_transformed=[transform(torch.unsqueeze(mask, dim=0)) for mask in masks]
mips_transformed=[transform(torch.unsqueeze(mip, dim=0)) for mip in mips]

output_mip_files=[os.path.join(output_mip_folder, file) for file in mip_files]
output_mask_files=[os.path.join(output_mask_folder,file) for file in mask_files]
for i in range(len(mip_files)):
    tifffile.imwrite(output_mip_files[i],mips_transformed[i].squeeze().numpy())
    tifffile.imwrite(output_mask_files[i], masks_transformed[i].squeeze().numpy())