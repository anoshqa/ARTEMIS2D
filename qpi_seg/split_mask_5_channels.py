import numpy as np
import torch
def split_into_channels(mask):
    mask_h=mask.shape[0]
    mask_w=mask.shape[1]
    background_mask=mask==0
    cell_mask=mask==1
    nucleus_mask=mask==2
    nucleolus_mask=mask==3
    lipid_droplet=mask==4
    ch_1=background_mask.to(dtype=torch.int64)
    ch_2=cell_mask.to(dtype=torch.int64)
    ch_3=nucleus_mask.to(dtype=torch.int64)
    ch_4=nucleolus_mask.to(dtype=torch.int64)
    ch_5=lipid_droplet.to(dtype=torch.int64)
    channeled_mask =torch.stack((ch_1,ch_2,ch_3,ch_4,ch_5),dim=0)
    #print(channeled_mask.shape)
    return channeled_mask