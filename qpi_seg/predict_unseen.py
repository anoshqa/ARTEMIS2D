#an improvised version of predict which predicts on completely unseen images :) and saves the masks
import torch
import tifffile
import os
import torchvision.transforms.v2 as transforms_v2
from models.unet import UNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import qpi_seg.visualize as visual_test
import qpi_seg.plot_grids as pg

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
assert torch.cuda.is_available()
model=UNet(depth=6,in_channels=1,out_channels=5, num_fmaps=32,final_activation=nn.Softmax()).to(device)
model_save=torch.load('/home/S-DC/ARTEMIS2D/checkpoints_unet1/checkpoint_epoch_190.pt')
model.load_state_dict(model_save['model_state_dict'])
model=model.to(device)

test_images_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_unseen_padded'

#TODO: Export masks to be usable by Fiji
unet_masks_output_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/Masks_unseen_unet'

test_files= os.listdir(test_images_folder)
test_images = [tifffile.imread(os.path.join(test_images_folder,file)) for file in test_files]



out_file_name_stems=[os.path.splitext(file)[0]+'_unet_masks.tiff'for file in test_files ]

out_file_name_masks=[os.path.join(unet_masks_output_folder, file) for file in out_file_name_stems]

from_np=transforms_v2.Lambda(lambda x: torch.from_numpy(x))
transform = transforms_v2.CenterCrop((832,832))
norm_min=13300
norm_max=14100
model.eval()
clustermaps=[]
for i in range(len(test_images)):
    torch_test_image = from_np(test_images[i])
    torch_test_image=torch_test_image.float()
    img_norm= (torch_test_image - norm_min) / (norm_max - norm_min)
    img_norm=img_norm.unsqueeze(dim=0).to(device)
    img_norm2=img_norm.unsqueeze(dim=0).to(device)
    img_norm2=transform(img_norm2)
    
    torch_prediction=model(img_norm2) #prediction
    clustermap=np.zeros((832,832))
    torch_prediction=torch_prediction.reshape(5,832,832).detach().cpu() #reshape to 5 channels and detach and cpu
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
    clustermaps.append(padded_clustermap)

#for i in range(len(out_file_name_masks)):
#    tifffile.imwrite(
#        out_file_name_masks[i],
#        clustermaps[i]
#    )


pg.plot_grids(test_images[0:5],  clustermaps[0:5])

#visual_test.visualize(test_image, test_mask,padded_clustermap)
#plt.savefig('test_predict10.png')

#the targets here are to export a merged mask of 5 channels


