import visualize
import qpi_seg.dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models.unet import UNet
import torch.nn as nn
import os
import torchvision.transforms.v2 as transforms_v2
import qpi_seg.file_charactersmatch as filetest
import torch.nn.functional as F
import qpi_seg.train_model as train_model
import qpi_seg.validate as validate
import qpi_seg.mean_fn as mean
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
assert torch.cuda.is_available()
#call Dataset

image_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_padded_final'
mask_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/Masks_padded_final'
train_files_percentage = 0.80
image_filenames=sorted(os.listdir(image_folder))
mask_filenames=sorted(os.listdir(mask_folder))
npimage_files=np.array(image_filenames)
npmask_files=np.array(mask_filenames)

assert filetest.filetest(image_filenames,mask_filenames)


shuffled_indices = np.random.permutation(len(npimage_files))
image_files = npimage_files[shuffled_indices] # YOUR CODE HERE -> Hint: first transform to np.array
mask_files = npmask_files[shuffled_indices]
tot_num_image_files = len(npimage_files)
tot_num_mask_files = len(npmask_files)
num_train_image_files = int(train_files_percentage*tot_num_image_files) 
num_train_mask_files = int(train_files_percentage*tot_num_mask_files)
train_image_files = image_files[:num_train_image_files ] # YOUR CODE HERE
train_mask_files = mask_files[:num_train_mask_files ] # YOUR CODE HERE
val_image_files = image_files[num_train_image_files:] # YOUR CODE HERE
val_mask_files = mask_files[num_train_mask_files:] # YOUR CODE HERE


train_mean, train_std= mean.mean_fn(image_folder,train_image_files)

print(train_mean,train_std)
trainQPIdataset=qpi_seg.dataset.MIPDataset(image_folder,mask_folder,train_image_files, train_mask_files,transform=transforms_v2.Resize((832,832),interpolation=transforms_v2.InterpolationMode.NEAREST),norm_setting="Dataset",norm_mean=train_mean, norm_std=train_std) #original image is 836,836
validationQPIdataset=qpi_seg.dataset.MIPDataset(image_folder,mask_folder,val_image_files, val_mask_files,transform=transforms_v2.Resize((832,832),interpolation=transforms_v2.InterpolationMode.NEAREST),norm_setting="Dataset",norm_mean=train_mean, norm_std=train_std)

train_loader=DataLoader(trainQPIdataset, batch_size=1, shuffle=True)
val_loader=DataLoader(validationQPIdataset, batch_size=1,shuffle=True)
batch_image,batch_mask=next(iter(train_loader))
print(batch_image.shape)
print(batch_mask.shape)
#visualize.visualize(batch_image.squeeze(),batch_mask.squeeze())


myUnet = UNet(depth=4,in_channels=1,out_channels=5, num_fmaps=4).to(device)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(myUnet.parameters())

for epoch in range(3):
    train_model.train_model(myUnet, train_loader, optimizer, loss, epoch, device=device)
#get a validation metric
