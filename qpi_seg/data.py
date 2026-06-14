import os
import torch
import torchvision.transforms as transforms
import skimage
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import file_charactersmatch as filetest
import shape_match as shape_match
import visualize

mipfolder=r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_padded_final'
maskfolder=r'/mnt/efs/dl_jrc/student_data/S-DC/Masks_padded_final'
mipfiles=[]
mip_justnames=sorted(os.listdir(mipfolder))
mask_justnames=sorted(os.listdir(maskfolder))

for files in mip_justnames:
    f=os.path.join(mipfolder,files)
    mipfiles.append(f)
    

maskfiles=[]
for files in mask_justnames:
    f=os.path.join(maskfolder,files)
    maskfiles.append(f)

images = [skimage.io.imread(file) for file in mipfiles]
masks= [skimage.io.imread(file) for file in maskfiles]
if filetest.filetest(mip_justnames, mask_justnames)==True and shape_match.matchtest(images,masks)==True:
    print("File names (first 30 characters) and shapes match throughout")
else:
    print("Check files again as there is a mismatch")
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

#saves a demoimage in main folder
#visualize.visualize(images[120],masks[120])

myQPIdataset=MIPDataset(images,masks)
batch_size = 5
data_loader = DataLoader(myQPIdataset, batch_size=batch_size,shuffle=True)
for batch_images, batch_masks in data_loader:
    print(f"Loaded a batch of {len(batch_images)} images and {len(batch_masks)} masks.")
    break  # Just to check the first batch