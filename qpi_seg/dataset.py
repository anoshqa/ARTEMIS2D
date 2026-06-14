import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import file_charactersmatch as filetest
import shape_match as shape_match
import visualize
import tifffile
from tqdm import tqdm
import torchvision.transforms.v2 as transforms_v2
import qpi_seg.split_mask_5_channels as split
    
class MIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, mask_folder, image_files, mask_files,transform=None,norm_setting="Dataset",norm_mean=None, norm_std=None):
        self.image_folder=image_folder
        self.mask_folder=mask_folder
        self.image_files=image_files
        self.mask_files=mask_files
        self.num_samples=len(image_files)
        #self.shapetest=shape_match.matchtest(self.images,self.masks)
        self.transform=(transform)
        self.loaded_imgs = [None]*self.num_samples
        self.loaded_masks=[None]*self.num_samples
        self.norm_mean=norm_mean
        self.norm_std=norm_std
        #norm_setting varies between dataset or per image
        self.from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))
        for sample_ind in tqdm(range(self.num_samples),desc="Reading images"):
            img_path = os.path.join(self.image_folder, self.image_files[sample_ind])
            image = self.from_np(tifffile.imread(img_path))
            image=image.float()
            if norm_setting =="Dataset":
                image = (image - self.norm_mean)/(self.norm_std)
            else:
                image_mean = torch.mean(image)
                image_std = torch.std(image)
                image = (image - image_mean)/(image_std) 
            self.loaded_imgs[sample_ind] = torch.unsqueeze(image, dim=0)
            mask_path = os.path.join(self.mask_folder, self.mask_files[sample_ind])
            mask = self.from_np(tifffile.imread(mask_path))
            channeled_mask = split.split_into_channels(mask)
            self.loaded_masks[sample_ind] = channeled_mask
            
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        image=self.loaded_imgs[idx]
        mask=self.loaded_masks[idx]
        if self.transform is not None:
            seed=torch.seed()
            torch.manual_seed(seed)
            image=self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
            #print(mask.dtype)
            #print(mask.shape)
            mask=mask.float()
            #print(mask.squeeze(axis=0).shape)
        return image,mask

#saves a demoimage in main folder
#visualize.visualize(images[120],masks[120])
