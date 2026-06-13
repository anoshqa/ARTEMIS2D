import os
import torch
import torchvision.transforms as transforms
import skimage
from torch.utils.data import DataLoader

mipfolder=r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_padded_final'
maskfolder=r'/mnt/efs/dl_jrc/student_data/S-DC/Masks_padded_final'
mipfiles=[]
for files in os.listdir(mipfolder):
    f=os.path.join(mipfolder,files)
    mipfiles.append(f)

maskfiles=[]
for files in os.listdir(maskfolder):
    f=os.path.join(maskfolder,files)
    maskfiles.append(f)
    
print(len(mipfiles))

class MIPDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.images=images
        self.masks=masks
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        mask=self.masks[idx]
        res={"image": image, "mask": mask}
        return res

images = [skimage.io.imread(file) for file in mipfiles]
masks= [skimage.io.imread(file) for file in maskfiles]
myQPIdataset=MIPDataset(images,masks)
batch_size = 5
data_loader = DataLoader(myQPIdataset, batch_size=batch_size,shuffle=True)
for batch_images, batch_masks in data_loader:
    print(f"Loaded a batch of {len(batch_images)} images and {len(batch_masks)} masks.")
    break  # Just to check the first batch