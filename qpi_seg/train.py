import visualize
import data
import torch
from torch.utils.data import DataLoader
import skimage
import numpy as np
import matplotlib.pyplot as plt
from models.unet import UNet
import torch.nn as nn
import os
import torchvision.transforms.v2 as transforms_v2
import file_charactersmatch as filetest
import shape_match as shape_match
import torch.nn.functional as F


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


trainQPIdataset=data.MIPDataset(image_folder,mask_folder,train_image_files, train_mask_files,transform=transforms_v2.Resize((832,832))) #original image is 836,836
validationQPIdataset=data.MIPDataset(image_folder,mask_folder,val_image_files, val_mask_files,transform=transforms_v2.Resize((832,832)))

train_loader=DataLoader(trainQPIdataset, batch_size=8, shuffle=True)
val_loader=DataLoader(validationQPIdataset, batch_size=8,shuffle=True)
batch=next(iter(train_loader))


myUnet = UNet(depth=4,in_channels=1,out_channels=1, num_fmaps=4).to(device)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(myUnet.parameters())
