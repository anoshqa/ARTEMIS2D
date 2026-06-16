import torch
import tifffile
import os
import torchvision.transforms.v2 as transforms_v2
from models.unet import UNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import qpi_seg.visualize as visual_test

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
assert torch.cuda.is_available()
model=UNet(depth=6,in_channels=1,out_channels=5, num_fmaps=32,final_activation=nn.Softmax()).to(device)
model_save=torch.load('/home/S-DC/ARTEMIS2D/checkpoints_unet1/checkpoint_epoch_190.pt')
model.load_state_dict(model_save['model_state_dict'])
model=model.to(device)

test_images_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_padded_final'
test_masks_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/Masks_padded_final'

#TODO: Export masks to be usable by Fiji
unet_masks_output_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/output_masks'

test_files= os.listdir(test_images_folder)
test_image = tifffile.imread(os.path.join(test_images_folder,test_files[10]))
test_mask_files= os.listdir(test_images_folder)
test_mask = tifffile.imread(os.path.join(test_masks_folder,test_mask_files[10]))

from_np=transforms_v2.Lambda(lambda x: torch.from_numpy(x))
torch_test_image = from_np(test_image)
norm_min=13300
norm_max=14100
torch_test_image=torch_test_image.float()
img_norm= (torch_test_image - norm_min) / (norm_max - norm_min)
print(img_norm.shape)
img_norm=img_norm.unsqueeze(dim=0).to(device)
img_norm2=img_norm.unsqueeze(dim=0).to(device)
model.eval()
transform = transforms_v2.CenterCrop((832,832))
img_norm2=transform(img_norm2)
torch_prediction=model(img_norm2)
clustermap=np.zeros((832,832))
torch_prediction=torch_prediction.reshape(5,832,832).detach().cpu()
ch1_prediction = torch_prediction[0,:,:]
ch2_prediction = torch_prediction[1,:,:]
ch3_prediction= torch_prediction[2,:,:]
ch4_prediction=torch_prediction[3,:,:]
ch5_prediction=torch_prediction[4,:,:]

clustermap[ch1_prediction>=0.5]=0
clustermap[ch2_prediction>=0.5]=1
clustermap[ch3_prediction>=0.5]=2
clustermap[ch4_prediction>=0.5]=3
clustermap[ch5_prediction>=0.5]=4

#pad clustermap with 4 pixels 
padded_clustermap=np.pad(clustermap,((2,2),(2,2)),'constant',constant_values=(0,0))
visual_test.visualize(test_image, test_mask,padded_clustermap)
plt.savefig('test_predict10.png')

#the targets here are to export a merged mask of 5 channels


