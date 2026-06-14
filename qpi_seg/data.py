import os
import torch
import torchvision.transforms as transforms
import skimage
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import file_charactersmatch as filetest
import shape_match as shape_match
import visualize

    
class MIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, mask_folder, image_files, mask_files,transform):
        self.image_folder=image_folder
        self.mask_folder=mask_folder
        self.image_files=image_files
        self.mask_files=mask_files
        
        self.image_filenames= sorted(os.listdir(self.image_folder))
        self.mask_filenames=sorted(os.listdir(self.mask_folder))
        self.image_full_filenames= [os.path.join(self.image_folder, f) for f in self.image_filenames]
        self.mask_full_filenames = [os.path.join(self.mask_folder,f) for f in self.mask_filenames]
        self.images = [skimage.io.imread(file) for file in self.image_full_filenames]
        self.masks=[skimage.io.imread(file) for file in self.mask_full_filenames]
        self.shapetest=shape_match.matchtest(self.images,self.masks)
        self.transform=transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        if self.shapetest==True:
            print("File names (first 30 characters) and shapes match throughout")
            image = self.images[idx]
            mask=self.masks[idx]
            if self.transform:
                image=self.transform(image)
                mask=self.transform(mask)
            res={"image": image, "mask": mask}
            return res
        else:
            print("Check files again as there is a mismatch")
            return False

#saves a demoimage in main folder
#visualize.visualize(images[120],masks[120])
