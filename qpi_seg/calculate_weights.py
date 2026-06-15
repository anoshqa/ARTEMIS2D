import torchvision.transforms.v2 as transforms_v2
import torch
import tifffile
from tqdm import tqdm
import os
import qpi_seg.split_mask_5_channels as split
import numpy as np
#this function will pass the training folder's mask values
def calculate_weights(mask_folder, mask_files):
    num_samples=len(mask_files)
    loaded_imgs = [None]*num_samples
   
    from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x)) #TODO change name
    sum_of_bgs=torch.tensor(0.0)
    sum_of_cells=torch.tensor(0.0)
    sum_of_nucleus=torch.tensor(0.0)
    sum_of_nucleolus=torch.tensor(0.0)
    sum_of_lipid_droplets=torch.tensor(0.0)
    
    for sample_ind in tqdm(range(num_samples),desc="Reading images"):
        mask_path = os.path.join(mask_folder, mask_files[sample_ind])
        mask = from_np(tifffile.imread(mask_path))
        channel_mask = split.split_into_channels(mask)
        channel_bg= channel_mask[0,:,:]
        channel_cell=channel_mask[1,:,:]
        channel_nucleus=channel_mask[2,:,:]
        channel_nucleolus=channel_mask[3,:,:]
        channel_lipids=channel_mask[4,:,:]
        sum_of_bgs += (channel_bg ==1).sum().item()
        sum_of_cells+=(channel_cell==1).sum().item()
        sum_of_nucleus+=(channel_nucleus==1).sum().item()
        sum_of_nucleolus+=(channel_nucleolus==1).sum().item()
        sum_of_lipid_droplets+=(channel_lipids==1).sum().item()
    h= mask.size(dim=0)
    w=mask.size(dim=1)
    count = num_samples*w*h
    weights = 1-torch.tensor([sum_of_bgs/count, sum_of_cells/count, sum_of_nucleus/count, sum_of_nucleolus/count, sum_of_lipid_droplets/count])
    return weights